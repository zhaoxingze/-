from __future__ import annotations

import argparse
import csv
from pathlib import Path

from utils import RESULTS_DIR


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def format_metric_row(row: dict[str, str]) -> str:
    experiment_id = row.get("experiment_id") or "--"
    acc = float(row["test_acc_mean"]) * 100
    params = float(row["trainable_percent"])
    return f'{experiment_id:>2} {row["method"]:<14} acc={acc:6.2f}% trainable={params:8.4f}%'


def main() -> None:
    parser = argparse.ArgumentParser(description="Print saved experiment metrics.")
    parser.add_argument("--metrics", type=Path, default=RESULTS_DIR / "metrics.csv")
    args = parser.parse_args()

    rows = read_metrics(args.metrics)
    if not rows:
        raise SystemExit(f"No rows found in {args.metrics}")
    for row in rows:
        print(format_metric_row(row))


if __name__ == "__main__":
    main()
