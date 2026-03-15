"""
models/resnet.py - ResNet-18 Model (Transfer Learning)
========================================================
Facial expression recognition using ImageNet pretrained ResNet-18.

- Transfer Learning with pretrained weights
- Automatic Grayscale (1 channel) -> RGB (3 channel) conversion
- Final layer adapted for emotion classification
- ~11.2M parameters (majority frozen, ~1.3M trainable)

Reference:
    He, K. et al. (2016). Deep Residual Learning for Image Recognition
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
import config


class ResNet18(nn.Module):
    """
    ResNet-18 based emotion recognition model.

    Transfer Learning strategy:
        1. Load ImageNet pretrained ResNet-18
        2. Freeze initial layers (general features preserved)
        3. Fine-tune final residual layers (specialized for FER)
        4. Classifier adapted for num_classes

    Input: [batch, 1, 224, 224] (grayscale) or [batch, 3, 224, 224] (RGB)
    Output: [batch, num_classes] logit output
    """

    def __init__(self, num_classes=5, in_channels=3,
                 freeze_backbone=True, unfreeze_last_n=2):
        """
        Args:
            num_classes (int): Number of output classes (default: 5)
            in_channels (int): Input channels (1=grayscale, 3=RGB)
            freeze_backbone (bool): Freeze backbone (Transfer Learning)
            unfreeze_last_n (int): Number of last residual layers
                to unfreeze for fine-tuning
        """
        super(ResNet18, self).__init__()

        self.in_channels = in_channels

        # Grayscale -> RGB conversion layer
        if in_channels == 1:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3)
            )
        else:
            self.channel_adapter = None

        # Load ImageNet pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)

        # Replace final FC layer with custom classifier
        in_features = self.backbone.fc.in_features  # 512

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        # Freeze backbone (optional)
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n)

    def _freeze_backbone(self, unfreeze_last_n=2):
        """
        Freeze backbone layers, unfreeze last N residual layers.

        ResNet-18 structure:
            - conv1, bn1: Initial convolution
            - layer1: Residual block 1 (2 BasicBlocks)
            - layer2: Residual block 2 (2 BasicBlocks)
            - layer3: Residual block 3 (2 BasicBlocks)
            - layer4: Residual block 4 (2 BasicBlocks)
        """
        # First freeze all backbone layers
        layers = [
            self.backbone.conv1, self.backbone.bn1,
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4
        ]

        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

        # Unfreeze last N residual layers
        residual_layers = [
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4
        ]
        for i in range(
            max(0, len(residual_layers) - unfreeze_last_n),
            len(residual_layers)
        ):
            for param in residual_layers[i].parameters():
                param.requires_grad = True

        # Classifier is always trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Make all layers trainable (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): [batch, in_channels, H, W]

        Returns:
            Tensor: [batch, num_classes] logit output
        """
        # Grayscale -> RGB conversion
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        return self.backbone(x)

    def get_feature_vector(self, x):
        """
        Returns the feature vector before the final FC layer.
        Useful for transfer learning or t-SNE visualization.

        Returns:
            Tensor: Feature vector of shape [batch, 256]
        """
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Pass through backbone (without FC)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through first 3 layers of FC
        # (Dropout -> Linear -> ReLU)
        for i in range(3):
            x = self.backbone.fc[i](x)

        return x


def get_resnet_model(num_classes=None, in_channels=3,
                     pretrained_path=None, freeze_backbone=True,
                     unfreeze_last_n=2):
    """
    ResNet-18 model factory function.

    Args:
        num_classes (int): Number of classes
        in_channels (int): Input channels (1=grayscale, 3=RGB)
        pretrained_path (str): Path to previously trained model
        freeze_backbone (bool): Freeze backbone
        unfreeze_last_n (int): Number of last residual layers
            to unfreeze

    Returns:
        ResNet18: Model instance
    """
    if num_classes is None:
        num_classes = config.DATA_CONFIG["num_classes"]

    model = ResNet18(
        num_classes=num_classes,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n=unfreeze_last_n
    )

    # Load previously trained model if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] Loading ResNet model: {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path, map_location="cpu"
        )

        if isinstance(checkpoint, dict) \
                and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] ResNet model loaded successfully.")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(f"\n[MODEL] ResNet-18 (Transfer Learning)")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Number of classes:    {num_classes}")
    print(f"  Input channels:       {in_channels}")
    print(f"  Backbone frozen:      {freeze_backbone}")

    return model
