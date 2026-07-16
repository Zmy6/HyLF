#!/usr/bin/env python

import copy
import os
from collections import namedtuple

import torch
import torch.nn.functional as F
from iopath.common.file_io import NativePathHandler, PathManager
from omegaconf import ListConfig
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from datasets import dataset_registry
from losses import loss_registry
from metrics import (
    average_step_outputs,
    compute_psnr,
    compute_ssim,
    compute_tensor_psnr,
)
from utils import (
    build_learning_rate_scheduler,
    build_optimizer,
    float_image_to_uint8,
    initialization_registry,
)
from utils.config_utils import apply_to_matching_config_keys
from utils.planar_voxel_utils import OccupancyGridMask

from .models import model_registry
from .regularizers import regularizer_registry
from .rendering import render_chunked, renderer_registry


class HyLFTrainer(Trainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(
            devices=cfg.training.num_gpus,
            max_epochs=cfg.training.num_epochs,
            max_steps=-1,
            log_every_n_steps=cfg.training.flush_logs,
            **kwargs,
        )

    def save_checkpoint(self, *args, **kwargs):
        if self.is_global_zero:
            super().save_checkpoint(*args, **kwargs)


class HyLFDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.current_epoch = 0
        self.test_every = cfg.training.test_every
        self.test_only = getattr(cfg.params, "test_only", False)
        self.current_batch_size = cfg.training.batch_size
        self.prepared = False

    def prepare_data(self):
        if self.prepared:
            return
        self.prepared = True
        dataset_class = dataset_registry[self.cfg.dataset.name]
        self.train_dataset = dataset_class(self.cfg, split="train")
        self.val_dataset = dataset_class(self.cfg, split="val")
        self.test_dataset = dataset_class(self.cfg, split="test")
        self.render_dataset = dataset_class(self.cfg, split="render")
        self.original_dataset_size = len(self.train_dataset)

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        num_workers = self.cfg.training.num_workers
        if self.cfg.training.sample_with_replacement:
            sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=self.cfg.training.num_iters * self.current_batch_size,
            )
            return DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.current_batch_size,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=True,
            )

        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.current_batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        should_test = (self.current_epoch + 1) % self.test_every == 0 or self.test_only
        dataset = self.test_dataset if should_test else self.val_dataset
        num_workers = self.cfg.training.num_workers
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )


class HyLFSystem(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.training_started = False
        self.rendering = False
        self.render_only = getattr(cfg.params, "render_only", False)
        self.test_only = getattr(cfg.params, "test_only", False)
        self.loss = loss_registry[cfg.training.loss.type](cfg.training.loss)

        self.path_manager = PathManager()
        self.path_manager.register_handler(NativePathHandler())

        self._convert_epoch_config_to_iterations()
        hylf_model = model_registry[cfg.model.type](cfg.model)
        if getattr(cfg.params, "reference_rng_compatibility", False):
            # The reference startup profiled one random ray batch and one
            # training-mode render before constructing the data sampler.
            torch.randn(1024, 6)
            torch.rand(1)
        self.render_fn = renderer_registry[cfg.model.render.type](hylf_model)

        self.camera_plane_scale = nn.Parameter(
            torch.tensor(0.125, dtype=torch.float32),
            requires_grad=True,
        )
        self.optimizer_configs = {
            key: copy.deepcopy(value)
            for key, value in cfg.training.optimizers.items()
        }
        self.optimizer_groups = self._collect_optimizer_groups()
        self.optimizer_groups["camera_geometry"] = [self.camera_plane_scale]
        self.loaded_checkpoint_epoch = None

        self.regularizers = []
        for regularizer_cfg in cfg.regularizers.values():
            regularizer = regularizer_registry[regularizer_cfg.type](
                self,
                regularizer_cfg,
            )
            self.regularizers.append(regularizer)

        self.apply(
            initialization_registry[cfg.training.weight_init.type](
                cfg.training.weight_init
            )
        )

    def on_load_checkpoint(self, checkpoint):
        checkpoint_epoch = checkpoint.get("epoch")
        if checkpoint_epoch is not None:
            self.loaded_checkpoint_epoch = int(checkpoint_epoch) + 1

    def _convert_epoch_config_to_iterations(self):
        def add_iteration_config(config, key):
            value = config[key]
            output_key = key.replace("epoch", "iter")
            if isinstance(value, ListConfig):
                config[output_key] = [
                    item * self.cfg.training.iters_per_epoch for item in value
                ]
            else:
                config[output_key] = value * self.cfg.training.iters_per_epoch

        for prefix in ["max_freq", "wait", "stop", "window", "decay", "warmup"]:
            apply_to_matching_config_keys(
                self.cfg,
                f"{prefix}_epoch",
                add_iteration_config,
            )
            apply_to_matching_config_keys(
                self.cfg,
                f"{prefix}_epochs",
                add_iteration_config,
            )

    def _collect_optimizer_groups(self):
        groups = {}
        for module in self.render_fn.modules():
            if not hasattr(module, "opt_group"):
                continue
            if isinstance(module.opt_group, str):
                groups.setdefault(module.opt_group, []).append(module)
            else:
                for key, values in module.opt_group.items():
                    groups.setdefault(key, []).extend(list(values))
        return groups

    def parameterize_hybrid_rays(self, rays):
        origins = rays[..., :3].clone()
        origins = torch.cat(
            [origins[..., :2] * self.camera_plane_scale, origins[..., 2:3]],
            dim=-1,
        )
        directions = torch.cat(
            [rays[..., 3:5] - origins[..., :2], rays[..., 5:6]],
            dim=-1,
        )
        directions = F.normalize(directions, p=2.0, dim=-1)
        return torch.cat([origins, directions], dim=-1)

    def load_state_dict(self, state_dict, strict=False):
        grid = self.render_fn.model.color_model.net
        if not hasattr(grid, "grid_size"):
            return super().load_state_dict(state_dict, strict=strict)

        device = grid.aabb.device
        grid_key = next(
            (
                key
                for key in state_dict
                if key.endswith("color_model.net.grid_size")
            ),
            None,
        )
        if grid_key is not None:
            grid.grid_size = state_dict[grid_key].to(device)
            grid.initialize_factors()

        mask_aabb = next(
            (value for key, value in state_dict.items() if key.endswith("mask_aabb")),
            None,
        )
        occupancy_volume = next(
            (
                value
                for key, value in state_dict.items()
                if key.endswith("occupancy_volume")
            ),
            None,
        )

        current_state = self.state_dict()
        compatible_state = dict(state_dict)
        if mask_aabb is not None and occupancy_volume is not None:
            compatible_state = {
                key: value
                for key, value in compatible_state.items()
                if ".occupancy_mask." not in key
            }
        for key, value in state_dict.items():
            current_value = current_state.get(key)
            if (
                current_value is not None
                and value.numel() == 0
                and current_value.numel() == 0
                and value.shape != current_value.shape
            ):
                compatible_state[key] = value.reshape(current_value.shape)

        load_result = super().load_state_dict(compatible_state, strict=False)
        grid.update_grid_geometry(grid.grid_size)
        if mask_aabb is not None and occupancy_volume is not None:
            grid.occupancy_mask = OccupancyGridMask(
                mask_aabb.to(device),
                occupancy_volume.to(device),
            )

        self.optimizer_groups = self._collect_optimizer_groups()
        self.optimizer_groups["camera_geometry"] = [self.camera_plane_scale]
        return load_result

    def forward(self, rays):
        chunk = (
            self.cfg.training.render_ray_chunk
            if self.rendering
            else self.cfg.training.ray_chunk
        )
        return render_chunked(rays, self.render_fn, chunk_size=chunk)

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        for key, modules in self.optimizer_groups.items():
            config = copy.deepcopy(self.optimizer_configs[key])
            optimizer = build_optimizer(config, modules)
            optimizers.append(optimizer)
            schedulers.append(
                build_learning_rate_scheduler(
                    config,
                    optimizer,
                    self.cfg.training.iters_per_epoch,
                )
            )
        return optimizers, schedulers

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def _needs_optimizer_reset(self, training_iteration):
        return training_iteration == 0 or any(
            training_iteration
            in getattr(self.optimizer_configs[key], "reset_opt_list", [])
            for key in self.optimizer_groups
        )

    def _reset_optimizers(self, training_iteration):
        optimizers = []
        schedulers = []
        for index, (key, modules) in enumerate(self.optimizer_groups.items()):
            config = copy.deepcopy(self.optimizer_configs[key])
            if training_iteration in getattr(config, "reset_opt_list", []):
                optimizer = build_optimizer(config, modules)
                scheduler = build_learning_rate_scheduler(
                    config,
                    optimizer,
                    self.cfg.training.iters_per_epoch,
                )
            else:
                optimizer = self.trainer.optimizers[index]
                scheduler = self.trainer.lr_scheduler_configs[index].scheduler
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        self.trainer.optimizers = optimizers
        self.trainer.strategy.lr_scheduler_configs = [
            namedtuple("SchedulerConfig", ("scheduler",))(scheduler)
            for scheduler in schedulers
        ]

    def get_training_iteration(self, epoch, batch_idx):
        if self.render_only or self.test_only:
            epoch = 10_000_000
        elif self.cfg.params.load_from_weights:
            epoch = self.cfg.params.start_epoch
        return self.cfg.training.num_iters * epoch + batch_idx * self.cfg.training.num_gpus

    def set_training_iteration(self, training_iteration):
        self.render_fn.model.set_iteration(training_iteration)
        for regularizer in self.regularizers:
            regularizer.set_iteration(training_iteration)

    def training_step(self, batch, batch_idx):
        if self.render_only or self.test_only:
            return {}

        self.training_started = True
        training_iteration = self.get_training_iteration(self.current_epoch, batch_idx)
        self.set_training_iteration(training_iteration)
        if self._needs_optimizer_reset(training_iteration):
            self._reset_optimizers(training_iteration)

        batch = self.trainer.datamodule.train_dataset.format_training_batch(batch)
        rays = self.parameterize_hybrid_rays(batch["rays"])
        results = self(rays)
        image_loss = self.loss(
            results["rgb"] * batch["weight"],
            batch["rgb"] * batch["weight"],
            **batch,
        )
        loss = image_loss
        regularizer_loss = 0.0
        for regularizer in self.regularizers:
            current_loss = (
                regularizer.compute_loss()
                * regularizer.compute_weight()
            )
            regularizer_loss += current_loss
            loss += current_loss

        train_psnr = compute_tensor_psnr(results["rgb"], batch["rgb"]).detach()
        if self.cfg.params.print_loss:
            print(
                f"Iter {training_iteration}: image_loss={image_loss:.4f}, "
                f"regularizer_loss={regularizer_loss:.4f}, psnr={train_psnr:.4f}"
            )

        optimizers = self.optimizers(use_pl_optimizer=True)
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            optimizer.zero_grad()
        self.manual_backward(loss)
        for optimizer in optimizers:
            optimizer.step()

        return {
            "train/loss": loss.detach(),
            "train/psnr": train_psnr,
        }

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.cfg.training.log_every == 0:
            for key, value in average_step_outputs(outputs).items():
                self.log(key, value, on_epoch=True, on_step=False, sync_dist=True)

        if self.training_started:
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()

        self.trainer.datamodule.current_epoch = self.current_epoch + 1
        if (self.current_epoch + 1) % self.trainer.datamodule.test_every in {0, 1}:
            self.trainer.reset_val_dataloader(self)

    def validation_output_epoch(self):
        if self.test_only and self.loaded_checkpoint_epoch is not None:
            return self.loaded_checkpoint_epoch
        configured_epoch = getattr(self.cfg.params, "evaluation_epoch", None)
        if self.test_only and configured_epoch is not None:
            return int(configured_epoch)
        return self.current_epoch + 1

    def render_validation_video(self):
        if not self.trainer.is_global_zero:
            return
        save_dir = os.path.join(
            self.cfg.params.save_video_dir,
            str(self.validation_output_epoch()),
            "rgb",
        )
        self.path_manager.mkdirs(save_dir)

        self.rendering = True
        for index, batch in enumerate(self.trainer.datamodule.render_dataset):
            rays = batch["rays"].to(self.device)
            results = self(self.parameterize_hybrid_rays(rays))
            image = (
                results["rgb"]
                .reshape(batch["height"], batch["width"], 3)
                .cpu()
                .numpy()
            )
            if self.cfg.params.save_results:
                with self.path_manager.open(
                    os.path.join(save_dir, f"{index:04d}.png"),
                    "wb",
                ) as file:
                    Image.fromarray(float_image_to_uint8(image)).save(file)
        self.rendering = False

    def evaluate_validation_image(self, batch, batch_idx):
        batch_idx = batch_idx * self.cfg.training.num_gpus + self.global_rank
        rays = batch["rays"].reshape(-1, batch["rays"].shape[-1])
        rgb = batch["rgb"].reshape(-1, 3)
        results = self(self.parameterize_hybrid_rays(rays))

        width, height = int(batch["width"]), int(batch["height"])
        prediction = results["rgb"].reshape(height, width, 3).cpu().numpy()
        target = rgb.reshape(height, width, 3).cpu().numpy()
        depth = results["depth"].reshape(height, width).cpu().numpy()

        if self.cfg.params.save_results:
            save_dir = os.path.join(
                self.cfg.params.save_image_dir,
                str(self.validation_output_epoch()),
            )
            self.path_manager.mkdirs(save_dir)
            for name, image in {
                "pred": prediction,
                "gt": target,
                "depth": depth,
            }.items():
                with self.path_manager.open(
                    os.path.join(save_dir, f"{batch_idx:04d}_{name}.png"),
                    "wb",
                ) as file:
                    Image.fromarray(float_image_to_uint8(image)).save(file)

        return {
            "val/loss": self.loss(results["rgb"], rgb, **batch).detach().cpu().numpy(),
            "val/psnr": compute_psnr(prediction, target),
            "val/ssim": compute_ssim(prediction, target),
        }

    def validation_step(self, batch, batch_idx):
        self.render_fn.eval()
        with torch.no_grad():
            training_iteration = self.get_training_iteration(self.current_epoch + 1, 0)
            self.set_training_iteration(max(training_iteration, 0))
            if batch_idx == 0 and (
                (self.current_epoch + 1) % self.cfg.training.render_every == 0
                or self.render_only
            ):
                self.render_validation_video()
            output = self.evaluate_validation_image(batch, batch_idx)
        self.render_fn.train()
        return output

    def validation_epoch_end(self, outputs):
        mean_metrics = average_step_outputs(outputs, cpu=True)
        for key, value in mean_metrics.items():
            self.log(key, value, on_epoch=True, on_step=False, sync_dist=True)
        if not self.cfg.params.save_results:
            return {}

        save_dir = os.path.join(
            self.cfg.params.save_image_dir,
            str(self.validation_output_epoch()),
        )
        self.path_manager.mkdirs(save_dir)
        with self.path_manager.open(
            os.path.join(save_dir, "metrics.txt"),
            "w",
        ) as file:
            for key, value in mean_metrics.items():
                file.write(f"{key}: {float(value)}\n")
        return {}
