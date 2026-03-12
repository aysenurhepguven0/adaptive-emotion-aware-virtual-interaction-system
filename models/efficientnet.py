"""
models/efficientnet.py - EfficientNet-B0 Model (Transfer Learning)
====================================================================
Facial expression recognition using ImageNet pretrained EfficientNet-B0.

- Transfer Learning with pretrained weights
- Automatic Grayscale (1 channel) -> RGB (3 channel) conversion
- Final layer adapted for 5 classes (5 emotions)
- ~4.34M parameters (majority frozen, ~741K trainable)

Reference:
    Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 based emotion recognition model.

    Transfer Learning strategy:
        1. Load ImageNet pretrained EfficientNet-B0
        2. Freeze initial layers (frozen) -- general features preserved
        3. Fine-tune final layers -- specialized for emotion recognition
        4. Classifier adapted for num_classes

    Input: [batch, 1, 128, 128] (grayscale) -> auto [batch, 3, 128, 128] (RGB)
    Output: [batch, num_classes] logit output
    """

    def __init__(self, num_classes=5, in_channels=1,
                 freeze_backbone=True, unfreeze_last_n=2):
        """
        Args:
            num_classes (int): Number of classes (default: 5)
            in_channels (int): Input channels (1=grayscale, 3=RGB)
            freeze_backbone (bool): Freeze backbone (Transfer Learning)
            unfreeze_last_n (int): Number of last blocks to unfreeze
        """
        super(EfficientNetB0, self).__init__()

        self.in_channels = in_channels

        # Grayscale -> RGB conversion layer
        if in_channels == 1:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3)
            )
        else:
            self.channel_adapter = None

        # Load ImageNet pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # Remove original classifier
        in_features = self.backbone.classifier[1].in_features  # 1280

        # New classifier: Dropout -> FC -> ReLU -> Dropout -> FC
        self.backbone.classifier = nn.Sequential(
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
        Freeze backbone layers, unfreeze last N blocks.

        EfficientNet-B0 features structure:
            - features[0]: Stem (Conv + BN)
            - features[1-8]: MBConv blocks
            - features[8]: Head (Conv + BN)
        """
        # First freeze all
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks (fine-tuning)
        total_blocks = len(self.backbone.features)
        for i in range(max(0, total_blocks - unfreeze_last_n), total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

        # Classifier is always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Make all layers trainable (full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): [batch, 1, 48, 48] or [batch, 3, 48, 48]

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

        # Pass through features
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through first 3 layers of classifier (Dropout -> Linear -> ReLU)
        for i in range(3):
            x = self.backbone.classifier[i](x)

        return x


def get_efficientnet_model(num_classes=None, in_channels=None,
                           pretrained_path=None, freeze_backbone=True,
                           unfreeze_last_n=2):
    """
    EfficientNet-B0 model factory function.

    Args:
        num_classes (int): Number of classes
        in_channels (int): Input channels
        pretrained_path (str): Path to previously trained model
        freeze_backbone (bool): Freeze backbone
        unfreeze_last_n (int): Number of last blocks to unfreeze

    Returns:
        EfficientNetB0: Model instance
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if in_channels is None:
        in_channels = config.MODEL_CONFIGS["efficientnet"]["num_channels"]

    model = EfficientNetB0(
        num_classes=num_classes,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n=unfreeze_last_n
    )

    # Load previously trained model if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[INFO] Loading EfficientNet model: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=config.DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] EfficientNet model loaded successfully.")

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(f"\n[MODEL] EfficientNet-B0 (Transfer Learning)")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Number of classes:    {num_classes}")
    print(f"  Input channels:       {in_channels}")
    print(f"  Backbone frozen:      {freeze_backbone}")

    return model
