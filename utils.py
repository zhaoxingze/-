from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def count_params(model: Any) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["experiment_id"]), str(row["method"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (experiment_id, method), selected in sorted(grouped.items()):
        acc = np.array([float(r["test_acc"]) for r in selected])
        loss = np.array([float(r["test_loss"]) for r in selected])
        times = np.array([float(r["time_sec"]) for r in selected])
        trainable = int(selected[0]["trainable_params"])
        total = int(selected[0]["total_params"])
        summary.append(
            {
                "experiment_id": experiment_id,
                "method": method,
                "test_acc_mean": round(float(acc.mean()), 4),
                "test_acc_std": round(float(acc.std(ddof=1)), 4) if len(acc) > 1 else 0.0,
                "test_loss_mean": round(float(loss.mean()), 4),
                "time_sec_mean": round(float(times.mean()), 2),
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percent": round(trainable / total * 100, 4),
            }
        )
    return summary
