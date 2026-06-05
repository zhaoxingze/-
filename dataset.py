from __future__ import annotations

import os
import pickle
import tarfile
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from utils import ROOT


class CIFAR10TarDataset(Dataset):
    train_batches = [f"cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_batches = ["cifar-10-batches-py/test_batch"]

    def __init__(self, archive_path: Path, train: bool, transform: Any = None) -> None:
        self.transform = transform
        batch_names = self.train_batches if train else self.test_batches
        images: list[np.ndarray] = []
        targets: list[int] = []
        with tarfile.open(archive_path, "r:gz") as archive:
            for batch_name in batch_names:
                member = archive.getmember(batch_name)
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    raise RuntimeError(f"Missing CIFAR-10 member: {batch_name}")
                payload = pickle.load(file_obj, encoding="latin1")
                data = payload["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                images.append(data)
                targets.extend(payload["labels"])
        self.data = np.concatenate(images, axis=0)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = Image.fromarray(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def class_balanced_indices(dataset: CIFAR10, per_class: int, seed: int, fraction: float = 1.0) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    actual_per_class = max(1, int(round(per_class * fraction)))
    indices: list[int] = []
    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]
        chosen = rng.choice(class_indices, size=actual_per_class, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices


def make_loaders(config: dict[str, Any], seed: int) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    image_size = int(data_cfg.get("image_size", 128))
    root = Path(data_cfg.get("root", ROOT / "data"))

    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    try:
        train_set = CIFAR10(root=str(root), train=True, download=bool(data_cfg.get("download", False)), transform=train_tf)
        test_set = CIFAR10(root=str(root), train=False, download=bool(data_cfg.get("download", False)), transform=test_tf)
    except RuntimeError:
        archive_path = root / "cifar-10-python.tar.gz"
        if not archive_path.exists():
            raise
        train_set = CIFAR10TarDataset(archive_path, train=True, transform=train_tf)
        test_set = CIFAR10TarDataset(archive_path, train=False, transform=test_tf)

    train_idx = class_balanced_indices(
        train_set,
        int(data_cfg.get("train_per_class", 100)),
        seed,
        float(data_cfg.get("train_fraction", 1.0)),
    )
    test_idx = class_balanced_indices(test_set, int(data_cfg.get("test_per_class", 100)), seed + 1000, 1.0)

    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_set, test_idx),
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=True,
    )
    return train_loader, test_loader
