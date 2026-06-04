from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(r"F:\Exploration")
OUT = ROOT / "outputs" / "finetune"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    train_per_class: int = 100
    test_per_class: int = 100
    epochs: int = 4
    batch_size: int = 128
    image_size: int = 128
    lr_head: float = 2e-3
    lr_full: float = 5e-5
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (2026, 2027)
    lora_rank: int = 8
    adapter_dim: int = 64


class FeatureResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FeatureResNet()
        self.classifier = nn.Linear(self.backbone.out_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class SSFProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FeatureResNet()
        dim = self.backbone.out_dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.classifier = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = z * self.scale + self.shift
        return self.classifier(z)


class LoRAClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.a = nn.Parameter(torch.empty(rank, in_dim))
        self.b = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
        self.scaling = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.b @ self.a * self.scaling
        return F.linear(x, self.weight + delta, self.bias)


class LoRAProbe(nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.backbone = FeatureResNet()
        self.classifier = LoRAClassifier(self.backbone.out_dim, 10, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class AdapterProbe(nn.Module):
    def __init__(self, adapter_dim: int) -> None:
        super().__init__()
        self.backbone = FeatureResNet()
        dim = self.backbone.out_dim
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.classifier = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = z + self.adapter(z)
        return self.classifier(z)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def class_balanced_indices(dataset: CIFAR10, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    indices: list[int] = []
    for c in range(10):
        cls = np.where(targets == c)[0]
        chosen = rng.choice(cls, size=per_class, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices


def make_loaders(cfg: Config, seed: int) -> tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_set = CIFAR10(root=str(ROOT / "data"), train=True, download=False, transform=train_tf)
    test_set = CIFAR10(root=str(ROOT / "data"), train=False, download=False, transform=test_tf)
    train_idx = class_balanced_indices(train_set, cfg.train_per_class, seed)
    test_idx = class_balanced_indices(test_set, cfg.test_per_class, seed + 1000)
    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_set, test_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def configure_trainable(model: nn.Module, method: str) -> None:
    if method == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    freeze_all(model)
    if method == "linear":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif method == "bitfit":
        for name, p in model.named_parameters():
            if name.endswith("bias") or name.startswith("classifier."):
                p.requires_grad = True
    elif method == "ssf":
        for name, p in model.named_parameters():
            if name in {"scale", "shift"} or name.startswith("classifier."):
                p.requires_grad = True
    elif method == "lora":
        for name, p in model.named_parameters():
            if name.startswith("classifier.a") or name.startswith("classifier.b") or name.startswith("classifier.bias"):
                p.requires_grad = True
    elif method == "adapter":
        for name, p in model.named_parameters():
            if name.startswith("adapter.") or name.startswith("classifier."):
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown method: {method}")


def build_model(method: str, cfg: Config) -> nn.Module:
    if method in {"full", "linear", "bitfit"}:
        model = LinearProbe()
    elif method == "ssf":
        model = SSFProbe()
    elif method == "lora":
        model = LoRAProbe(cfg.lora_rank)
    elif method == "adapter":
        model = AdapterProbe(cfg.adapter_dim)
    else:
        raise ValueError(method)
    configure_trainable(model, method)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
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


def train_method(method: str, seed: int, cfg: Config, device: torch.device) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed_everything(seed)
    train_loader, test_loader = make_loaders(cfg, seed)
    model = build_model(method, cfg).to(device)
    total_params, trainable_params = count_params(model)
    lr = cfg.lr_full if method == "full" else cfg.lr_head
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.append(
            {
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

    elapsed = time.perf_counter() - start
    test_loss, test_acc = evaluate(model, test_loader, device)
    return history, {
        "method": method,
        "seed": seed,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "time_sec": elapsed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_params / total_params * 100,
        "device": torch.cuda.get_device_name(0),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for method in ["linear", "bitfit", "ssf", "lora", "adapter", "full"]:
        selected = [r for r in rows if r["method"] == method]
        acc = np.array([float(r["test_acc"]) for r in selected])
        loss = np.array([float(r["test_loss"]) for r in selected])
        times = np.array([float(r["time_sec"]) for r in selected])
        trainable = float(selected[0]["trainable_params"])
        total = float(selected[0]["total_params"])
        out.append(
            {
                "method": method.upper(),
                "test_acc_mean": round(float(acc.mean()), 4),
                "test_acc_std": round(float(acc.std(ddof=1)), 4) if len(acc) > 1 else 0.0,
                "test_loss_mean": round(float(loss.mean()), 4),
                "time_sec_mean": round(float(times.mean()), 2),
                "trainable_params": int(trainable),
                "total_params": int(total),
                "trainable_percent": round(trainable / total * 100, 4),
            }
        )
    return out


def plot(history: list[dict[str, object]]) -> None:
    plt.figure(figsize=(9, 4), dpi=160)
    for method in ["linear", "bitfit", "ssf", "lora", "adapter", "full"]:
        series = {}
        for row in history:
            if row["method"] == method:
                series.setdefault(int(row["epoch"]), []).append(float(row["test_acc"]))
        xs = sorted(series)
        ys = [np.mean(series[x]) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=method.upper())
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "finetune_accuracy_curves.png")
    plt.close()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment, but torch.cuda.is_available() is False.")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cfg = Config()
    methods = ["linear", "bitfit", "ssf", "lora", "adapter", "full"]
    all_history: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for method in methods:
        for seed in cfg.seeds:
            history, result = train_method(method, seed, cfg, device)
            all_history.extend(history)
            seed_rows.append(result)
            print(
                f"{method.upper()} seed={seed} acc={result['test_acc']:.4f} "
                f"trainable={result['trainable_percent']:.3f}% time={result['time_sec']:.1f}s"
            )

    summary = summarize(seed_rows)
    write_csv(OUT / "finetune_history.csv", all_history)
    write_csv(OUT / "finetune_seed_results.csv", seed_rows)
    write_csv(OUT / "finetune_summary_results.csv", summary)
    (OUT / "finetune_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "finetune_summary_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(all_history)
    print(json.dumps({"device": torch.cuda.get_device_name(0), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
