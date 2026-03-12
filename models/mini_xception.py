from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.sep_conv1(x))
        x = self.sep_conv2(x)
        x = self.pool(x)
        residual = self.skip(residual)
        return F.relu(x + residual)


class MiniXception(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(8, 16)
        self.block2 = ResidualBlock(16, 32)
        self.block3 = ResidualBlock(32, 64)
        self.block4 = ResidualBlock(64, 128)

        self.conv_final = nn.Sequential(
            SeparableConv2d(128, 256),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_final(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv_final(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)


def get_model(
    num_classes: int,
    in_channels: int = 3,
    dropout: Optional[float] = None,
) -> MiniXception:
    model_dropout = dropout if dropout is not None else 0.5
    return MiniXception(num_classes=num_classes, in_channels=in_channels, dropout=model_dropout)
