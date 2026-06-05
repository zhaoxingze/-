from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNetFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = base.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class FineTuneResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNetFeatureExtractor()
        self.classifier = nn.Linear(self.backbone.out_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


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
    def __init__(self, rank: int = 8) -> None:
        super().__init__()
        self.backbone = ResNetFeatureExtractor()
        self.classifier = LoRAClassifier(self.backbone.out_dim, 10, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class AdapterProbe(nn.Module):
    def __init__(self, adapter_dim: int = 64) -> None:
        super().__init__()
        self.backbone = ResNetFeatureExtractor()
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


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def configure_trainable(model: nn.Module, method: str) -> None:
    if method == "full_ft":
        for param in model.parameters():
            param.requires_grad = True
        return

    freeze_all(model)
    if method == "linear_probe":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif method == "partial_ft":
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif method == "lora":
        for name, param in model.named_parameters():
            if name.startswith("classifier.a") or name.startswith("classifier.b") or name.startswith("classifier.bias"):
                param.requires_grad = True
    elif method == "adapter":
        for name, param in model.named_parameters():
            if name.startswith("adapter.") or name.startswith("classifier."):
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown ResNet fine-tuning method: {method}")


def build_resnet_finetune(method: str, config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    if method == "lora":
        model: nn.Module = LoRAProbe(rank=int(model_cfg.get("lora_rank", 8)))
    elif method == "adapter":
        model = AdapterProbe(adapter_dim=int(model_cfg.get("adapter_dim", 64)))
    else:
        model = FineTuneResNet()
    configure_trainable(model, method)
    return model
