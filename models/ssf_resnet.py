from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / "models" / "torch"))

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SSF2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class SSF1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class SSFResNet(nn.Module):
    layer_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(self, ssf_layers: Iterable[str]) -> None:
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

        layer_set = set(ssf_layers)
        self.ssf2d = nn.ModuleDict(
            {name: SSF2d(channels) for name, channels in self.layer_channels.items() if name in layer_set}
        )
        self.ssf_feature = SSF1d(self.out_dim)
        self.classifier = nn.Linear(self.out_dim, 10)
        self.configure_trainable()

    def configure_trainable(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        for param in self.ssf2d.parameters():
            param.requires_grad = True
        for param in self.ssf_feature.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _apply_ssf(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name in self.ssf2d:
            return self.ssf2d[name](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self._apply_ssf("layer1", self.layer1(x))
        x = self._apply_ssf("layer2", self.layer2(x))
        x = self._apply_ssf("layer3", self.layer3(x))
        x = self._apply_ssf("layer4", self.layer4(x))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.ssf_feature(x)
        return self.classifier(x)
