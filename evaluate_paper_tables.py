#!/usr/bin/env python

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm


PAPER_TABLES = {
    "HCI": {
        "number": 1,
        "scenes": [
            "bedroom",
            "bicycle",
            "herbs",
            "boxes",
            "cotton",
            "dino",
            "origami",
            "sideboard",
        ],
    },
    "INRIA": {
        "number": 2,
        "scenes": ["Bee", "Buildings", "Hublais", "MessyDesk", "Sculpture"],
    },
    "Stanford": {
        "number": 3,
        "scenes": ["Cars", "Flower1", "Flower2", "Rock", "Seahorse"],
    },
}
DATASET_ALIASES = {"HCI_new": "HCI", "30scene": "Stanford"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate HyLF predictions with Y-PSNR, Y-SSIM, and RGB LPIPS."
    )
    parser.add_argument(
        "--log-root",
        "--logs-root",
        dest="log_root",
        type=Path,
        default=Path("logs"),
    )
    parser.add_argument(
        "--dataset",
        "--suite",
        dest="dataset",
        choices=["all", *PAPER_TABLES, *DATASET_ALIASES],
        default="all",
        help="Paper dataset to evaluate. HCI, INRIA, and Stanford map to Tables 1-3.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional explicit scene list; overrides --suite.",
    )
    parser.add_argument(
        "--angular-step",
        "--step",
        dest="angular_step",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--experiment-prefix",
        "--model-prefix",
        dest="experiment_prefix",
        default="hylf",
    )
    parser.add_argument(
        "--validation-epoch",
        "--epoch",
        dest="validation_epoch",
        default="latest",
        help="Validation epoch directory, or 'latest'.",
    )
    parser.add_argument(
        "--lpips-network",
        "--lpips-net",
        dest="lpips_network",
        choices=["alex", "vgg"],
        default="vgg",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-directory",
        "--output-dir",
        dest="output_directory",
        type=Path,
        default=Path("metrics"),
    )
    return parser.parse_args()


def load_rgb_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def rgb_to_bt601_luma(rgb):
    return (
        65.481 * rgb[..., 0]
        + 128.553 * rgb[..., 1]
        + 24.966 * rgb[..., 2]
        + 16.0
    ) / 255.0


def compute_luma_psnr(prediction, target):
    mean_squared_error = np.mean(
        np.square(rgb_to_bt601_luma(prediction) - rgb_to_bt601_luma(target))
    )
    return (
        float("inf")
        if mean_squared_error == 0
        else float(-10.0 * np.log10(mean_squared_error))
    )


def compute_luma_ssim(prediction, target):
    target_luma = rgb_to_bt601_luma(target)
    return float(
        structural_similarity(
            rgb_to_bt601_luma(prediction),
            target_luma,
            data_range=float(target_luma.max() - target_luma.min()),
        )
    )


def compute_rgb_lpips(prediction, target, lpips_model, device):
    prediction_tensor = (
        torch.from_numpy(prediction.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        * 2.0
        - 1.0
    )
    target_tensor = (
        torch.from_numpy(target.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        * 2.0
        - 1.0
    )
    with torch.inference_mode():
        return float(lpips_model(prediction_tensor, target_tensor).item())


def resolve_prediction_directory(arguments, scene):
    experiment_directory = (
        arguments.log_root
        / f"{arguments.experiment_prefix}_{scene}_step_{arguments.angular_step}"
    )
    validation_root = experiment_directory / "val_images"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"Missing validation output: {validation_root}")

    if arguments.validation_epoch == "latest":
        epochs = sorted(
            (
                path
                for path in validation_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )
        if not epochs:
            raise FileNotFoundError(f"No epoch directories in {validation_root}")
        prediction_directory = epochs[-1]
    else:
        prediction_directory = validation_root / str(arguments.validation_epoch)

    evaluation_directory = prediction_directory / "eval"
    return (
        evaluation_directory
        if evaluation_directory.is_dir()
        else prediction_directory
    )


def collect_prediction_target_pairs(prediction_directory):
    image_pairs = []
    for prediction_path in sorted(prediction_directory.glob("*_pred.png")):
        target_path = prediction_path.with_name(
            prediction_path.name.replace("_pred.png", "_gt.png")
        )
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target for {prediction_path.name}")
        image_pairs.append((prediction_path, target_path))
    if not image_pairs:
        raise FileNotFoundError(f"No *_pred.png files in {prediction_directory}")
    return image_pairs


def find_paper_dataset(scene):
    for dataset_name, table_metadata in PAPER_TABLES.items():
        if scene in table_metadata["scenes"]:
            return dataset_name, table_metadata["number"]
    return "custom", None


def evaluate_scene(arguments, scene, lpips_model):
    prediction_directory = resolve_prediction_directory(arguments, scene)
    image_pairs = collect_prediction_target_pairs(prediction_directory)
    metric_values = {"psnr_y": [], "ssim_y": [], "lpips": []}

    for prediction_path, target_path in tqdm(image_pairs, desc=scene, leave=False):
        prediction = load_rgb_image(prediction_path)
        target = load_rgb_image(target_path)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {scene}: {prediction.shape} != {target.shape}"
            )
        metric_values["psnr_y"].append(compute_luma_psnr(prediction, target))
        metric_values["ssim_y"].append(compute_luma_ssim(prediction, target))
        metric_values["lpips"].append(
            compute_rgb_lpips(prediction, target, lpips_model, arguments.device)
        )

    dataset_name, table_number = find_paper_dataset(scene)
    scene_result = {
        "table": table_number,
        "dataset": dataset_name,
        "scene": scene,
        "views": len(image_pairs),
        "prediction_directory": str(prediction_directory),
    }
    scene_result.update(
        {key: float(np.mean(values)) for key, values in metric_values.items()}
    )
    return scene_result


def average_metrics(scene_results):
    return {
        key: float(np.mean([result[key] for result in scene_results]))
        for key in ("psnr_y", "ssim_y", "lpips")
    }


def write_results(arguments, scene_results):
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{arguments.experiment_prefix}_2x2_to_"
        f"{arguments.angular_step + 1}x{arguments.angular_step + 1}"
    )
    csv_output_path = arguments.output_directory / f"{stem}.csv"
    json_output_path = arguments.output_directory / f"{stem}.json"

    with csv_output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "table",
                "dataset",
                "scene",
                "views",
                "psnr_y",
                "ssim_y",
                "lpips",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames}
            for row in scene_results
        )

    dataset_means = {}
    for dataset_name in dict.fromkeys(
        result["dataset"] for result in scene_results
    ):
        dataset_means[dataset_name] = average_metrics(
            [
                result
                for result in scene_results
                if result["dataset"] == dataset_name
            ]
        )
    payload = {
        "task": (
            f"2x2->{arguments.angular_step + 1}x{arguments.angular_step + 1}"
        ),
        "validation_epoch": arguments.validation_epoch,
        "lpips_network": arguments.lpips_network,
        "scenes": scene_results,
        "dataset_means": dataset_means,
        "overall_mean": average_metrics(scene_results),
    }
    dataset_coverage = {}
    for dataset_name, table_metadata in PAPER_TABLES.items():
        evaluated_scenes = [
            result["scene"]
            for result in scene_results
            if result["dataset"] == dataset_name
        ]
        if not evaluated_scenes:
            continue
        missing_scenes = [
            scene
            for scene in table_metadata["scenes"]
            if scene not in evaluated_scenes
        ]
        dataset_coverage[dataset_name] = {
            "evaluated": evaluated_scenes,
            "missing": missing_scenes,
            "complete": not missing_scenes,
        }
    payload["dataset_coverage"] = dataset_coverage
    payload["complete"] = all(
        coverage["complete"] for coverage in dataset_coverage.values()
    )
    with json_output_path.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return csv_output_path, json_output_path, payload


def main():
    arguments = parse_arguments()
    scenes = arguments.scenes
    if not scenes:
        selected_dataset = DATASET_ALIASES.get(arguments.dataset, arguments.dataset)
        if selected_dataset == "all":
            scenes = [
                scene
                for table_metadata in PAPER_TABLES.values()
                for scene in table_metadata["scenes"]
            ]
        else:
            scenes = PAPER_TABLES[selected_dataset]["scenes"]

    lpips_model = (
        lpips.LPIPS(net=arguments.lpips_network).to(arguments.device).eval()
    )
    scene_results = []
    for scene in scenes:
        result = evaluate_scene(arguments, scene, lpips_model)
        scene_results.append(result)
        print(
            f"Table {result['table']} {result['dataset']}/{scene}: "
            f"views={result['views']} "
            f"PSNR-Y={result['psnr_y']:.4f} "
            f"SSIM-Y={result['ssim_y']:.6f} "
            f"LPIPS={result['lpips']:.6f}"
        )

    csv_output_path, json_output_path, payload = write_results(
        arguments,
        scene_results,
    )
    print(json.dumps(payload["dataset_means"], indent=2))
    print("Overall:", json.dumps(payload["overall_mean"], indent=2))
    print(f"CSV: {csv_output_path}")
    print(f"JSON: {json_output_path}")


if __name__ == "__main__":
    main()
