#!/usr/bin/env python

import os
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from iopath.common.file_io import NativePathHandler, PathManager
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from HyLF import HyLFDataModule, HyLFSystem, HyLFTrainer


class HyLFModelCheckpoint(ModelCheckpoint):
    @property
    def state_key(self):
        return f"{super().state_key}.save_weights_only={self.save_weights_only}"

    def _save_last_checkpoint(
        self,
        trainer,
        monitor_candidates: Dict[str, torch.Tensor],
    ) -> None:
        if self._every_n_epochs >= 1 and (
            trainer.current_epoch + 1
        ) % self._every_n_epochs == 0:
            super()._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_epoch_end(self, trainer, pl_module):
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, monitor_candidates)


def run_training(root_config: DictConfig) -> None:
    print(OmegaConf.to_yaml(root_config))
    OmegaConf.set_struct(root_config, False)
    experiment_config = root_config.experiment

    disable_validation = (
        bool(getattr(experiment_config.training, "disable_validation", False))
        and not bool(getattr(experiment_config.params, "test_only", False))
        and not bool(getattr(experiment_config.params, "render_only", False))
    )
    disable_checkpointing = bool(
        getattr(experiment_config.training, "disable_checkpointing", False)
    )

    seed_everything(experiment_config.params.seed, workers=True)
    os.chdir(Path(__file__).resolve().parent)

    path_manager = PathManager()
    path_manager.register_handler(NativePathHandler())

    log_dir = os.path.join(
        os.path.expanduser(experiment_config.params.log_dir),
        experiment_config.params.name,
    )
    checkpoint_dir = os.path.join(
        os.path.expanduser(experiment_config.params.ckpt_dir),
        experiment_config.params.name,
    )
    if experiment_config.params.tensorboard or experiment_config.params.save_results:
        path_manager.mkdirs(log_dir)
    if not disable_checkpointing:
        path_manager.mkdirs(checkpoint_dir)

    if experiment_config.params.save_results:
        experiment_config.params.save_video_dir = os.path.join(
            log_dir,
            experiment_config.params.save_video_dir,
        )
        experiment_config.params.save_image_dir = os.path.join(
            log_dir,
            experiment_config.params.save_image_dir,
        )
        path_manager.mkdirs(experiment_config.params.save_video_dir)
        path_manager.mkdirs(experiment_config.params.save_image_dir)

    data_module = HyLFDataModule(experiment_config)
    data_module.prepare_data()
    if experiment_config.training.sample_with_replacement:
        experiment_config.training.iters_per_epoch = experiment_config.training.num_iters
    else:
        experiment_config.training.iters_per_epoch = int(
            np.ceil(
                data_module.original_dataset_size
                / experiment_config.training.batch_size
            )
        )

    resume_checkpoint_name = (
        "last-weights" if experiment_config.params.load_from_weights else "last"
    )
    resume_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{resume_checkpoint_name}.ckpt",
    )
    if not path_manager.exists(resume_checkpoint_path):
        resume_checkpoint_path = None
    if disable_checkpointing and not experiment_config.params.load_from_weights:
        resume_checkpoint_path = None

    callbacks = []
    if not disable_checkpointing:
        full_checkpoint_callback = HyLFModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:d}",
            monitor="val/loss",
            mode="min",
            save_top_k=0,
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=experiment_config.training.ckpt_every,
        )
        full_checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
        callbacks.append(full_checkpoint_callback)

    if (
        resume_checkpoint_path is not None
        and experiment_config.params.load_from_weights
    ):
        hylf_system = HyLFSystem.load_from_checkpoint(
            resume_checkpoint_path,
            cfg=experiment_config,
        )
    else:
        hylf_system = HyLFSystem(experiment_config)

    if experiment_config.params.render_only:
        experiment_config.training.render_every = 1
        experiment_config.training.val_every = 1
    if experiment_config.params.test_only:
        experiment_config.training.test_every = 1
        experiment_config.training.val_every = 1

    logger = (
        TensorBoardLogger(save_dir=log_dir, name=experiment_config.params.name)
        if experiment_config.params.tensorboard
        else False
    )
    trainer = HyLFTrainer(
        experiment_config,
        callbacks=callbacks,
        enable_checkpointing=not disable_checkpointing,
        resume_from_checkpoint=(
            resume_checkpoint_path
            if not experiment_config.params.load_from_weights
            else None
        ),
        logger=logger,
        accelerator="gpu",
        strategy="ddp" if experiment_config.training.num_gpus > 1 else None,
        check_val_every_n_epoch=(
            None if disable_validation else experiment_config.training.val_every
        ),
        limit_val_batches=0 if disable_validation else None,
        num_sanity_val_steps=0 if disable_validation else 2,
        enable_progress_bar=False,
        benchmark=False,
        profiler=None,
    )
    if experiment_config.params.test_only:
        if resume_checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint available for test-only evaluation: {checkpoint_dir}"
            )
        trainer.validate(
            model=hylf_system,
            datamodule=data_module,
            ckpt_path=resume_checkpoint_path,
        )
        return

    trainer.fit(hylf_system, datamodule=data_module)


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(root_config: DictConfig):
    run_training(root_config)


if __name__ == "__main__":
    main()
