from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import make_loaders
from models.resnet_finetune import build_resnet_finetune
from models.ssf_resnet import SSFResNet
from utils import CONFIG_DIR, RESULTS_DIR, count_params, load_config, seed_everything, summarize_results, write_csv, write_json


DEFAULT_CONFIGS = [
    "linear_probe.yaml",
    "partial_ft.yaml",
    "full_ft.yaml",
    "ssf.yaml",
    "ssf_l4.yaml",
    "ssf_l34.yaml",
    "fewshot_25.yaml",
    "fewshot_50.yaml",
    "lora.yaml",
    "adapter.yaml",
]


def build_model(method: str, config: dict[str, Any]) -> torch.nn.Module:
    if method.startswith("ssf_"):
        layers = config.get("model", {}).get("ssf_layers", ["layer1", "layer2", "layer3", "layer4"])
        return SSFResNet(layers)
    return build_resnet_finetune(method, config)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(y.numel())
    return loss_sum / total, correct / total


def train_one_method(
    config: dict[str, Any],
    method: str,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    train_loader, test_loader = make_loaders(config, seed)
    model = build_model(method, config).to(device)
    total_params, trainable_params = count_params(model)
    train_cfg = config["training"]
    lr = float(train_cfg["lr_full"] if method == "full_ft" else train_cfg["lr_head"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    epochs = int(train_cfg.get("epochs", 15))
    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"{config['experiment_id']} {method} seed={seed}",
        leave=False,
    )
    for epoch in epoch_iter:
        model.train()
        batch_iter = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
        )
        for x, y in batch_iter:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_iter.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        test_loss, test_acc = evaluate_model(model, test_loader, device)
        epoch_iter.set_postfix(test_acc=f"{test_acc:.4f}")
        history.append(
            {
                "experiment_id": config["experiment_id"],
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    elapsed = time.perf_counter() - start
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    return history, {
        "experiment_id": config["experiment_id"],
        "method": method,
        "seed": seed,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "time_sec": elapsed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_params / total_params * 100,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


def plot_results(history: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4), dpi=160)
    for method in sorted({row["method"] for row in history}):
        series: dict[int, list[float]] = {}
        for row in history:
            if row["method"] == method:
                series.setdefault(int(row["epoch"]), []).append(float(row["test_acc"]))
        xs = sorted(series)
        ys = [np.mean(series[x]) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png")
    plt.close()

    labels = [f'{row["experiment_id"]}-{row["method"]}' for row in summary]
    acc = [float(row["test_acc_mean"]) * 100 for row in summary]
    plt.figure(figsize=(10, 4.8), dpi=160)
    plt.bar(labels, acc, color="#4775b8")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png")
    plt.close()

    params = [float(row["trainable_percent"]) for row in summary]
    plt.figure(figsize=(10, 4.8), dpi=160)
    plt.bar(labels, params, color="#b85f47")
    plt.ylabel("Trainable Parameters (%)")
    plt.yscale("log")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parameter_comparison.png")
    plt.close()


def expand_methods(config: dict[str, Any]) -> list[str]:
    if "methods" in config:
        return list(config["methods"])
    return [str(config["method"])]


def run_configs(config_paths: list[Path]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full experiment. Use existing results for documentation-only review.")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    all_history: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for path in config_paths:
        config = load_config(path)
        for method in expand_methods(config):
            for seed in config["training"].get("seeds", [2026, 2027]):
                history, result = train_one_method(config, method, int(seed), device)
                all_history.extend(history)
                seed_rows.append(result)
                print(
                    f'{config["experiment_id"]} {method} seed={seed} '
                    f'acc={result["test_acc"]:.4f} trainable={result["trainable_percent"]:.4f}%'
                )

    summary = summarize_results(seed_rows)
    write_csv(RESULTS_DIR / "history.csv", all_history)
    write_csv(RESULTS_DIR / "seed_results.csv", seed_rows)
    write_csv(RESULTS_DIR / "metrics.csv", summary)
    write_json(RESULTS_DIR / "metrics.json", summary)
    plot_results(all_history, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run efficient fine-tuning experiments.")
    parser.add_argument("--config", type=str, help="Path to one YAML config. Omit with --all to run the full list.")
    parser.add_argument("--all", action="store_true", help="Run E1-E8 plus LoRA/Adapter extension configs.")
    args = parser.parse_args()

    if args.all:
        config_paths = [CONFIG_DIR / name for name in DEFAULT_CONFIGS]
    elif args.config:
        config_paths = [Path(args.config)]
    else:
        config_paths = [CONFIG_DIR / "ssf.yaml"]
    run_configs(config_paths)


if __name__ == "__main__":
    main()
