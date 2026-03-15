"""
models/hsemotion_model.py - HSEmotion Model (AffectNet Pretrained)
===================================================================
Facial expression recognition using EfficientNet-B0 with
AffectNet-pretrained weights from the HSEmotion project.

Unlike the standard EfficientNet-B0 in this repository (which uses
ImageNet pretrained weights from torchvision), HSEmotion leverages
weights trained specifically on facial expression data (AffectNet).
This domain-specific pretraining provides feature representations
that are more relevant for emotion recognition tasks.

Architecture:
    - EfficientNet-B0 backbone via timm library
    - AffectNet-pretrained weights (8-class emotion recognition)
    - Custom 2-layer classifier head for FERPlus classes
    - Automatic Grayscale (1 channel) -> RGB (3 channel) conversion
    - ~4.0M parameters

Dependencies:
    - timm (required): pip install timm
    - hsemotion (optional, for AffectNet weights): pip install hsemotion

Reference:
    Savchenko, A.V. (2022). HSEmotion: High-Speed face Emotion
    recognition library. SoftwareX, 18, 101065.
    Savchenko, A.V. (2021). Facial expression and attributes
    recognition based on multi-task learning of lightweight
    neural networks. ICPR 2022.
"""

import os
import sys

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for HSEmotion. "
        "Install it with: pip install timm"
    )

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
import config


# AffectNet 8-class label mapping used by HSEmotion weights
AFFECTNET_CLASSES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]


class HSEmotion(nn.Module):
    """
    HSEmotion-based emotion recognition model.

    Uses EfficientNet-B0 backbone with optional AffectNet-pretrained
    weights from the HSEmotion project (Savchenko, 2022). When
    AffectNet weights are loaded, the backbone starts with emotion-
    domain features rather than generic ImageNet features.

    Transfer Learning strategy:
        1. Load EfficientNet-B0 backbone (AffectNet or ImageNet)
        2. Freeze initial layers (domain features preserved)
        3. Fine-tune final blocks for FERPlus adaptation
        4. Custom classifier head for target classes

    Input: [batch, 1, 224, 224] (grayscale) or [batch, 3, 224, 224]
    Output: [batch, num_classes] logit output
    """

    def __init__(self, num_classes=5, in_channels=3,
                 freeze_backbone=True, unfreeze_last_n=2,
                 affectnet_pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (default: 5)
            in_channels (int): Input channels (1=grayscale, 3=RGB)
            freeze_backbone (bool): Freeze backbone for transfer
                learning
            unfreeze_last_n (int): Number of last feature blocks to
                unfreeze for fine-tuning
            affectnet_pretrained (bool): Use AffectNet-pretrained
                weights from the hsemotion package. Falls back to
                ImageNet pretrained weights if unavailable.
        """
        super(HSEmotion, self).__init__()

        self.in_channels = in_channels
        self.affectnet_pretrained = affectnet_pretrained

        # Grayscale -> RGB conversion layer
        if in_channels == 1:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3)
            )
        else:
            self.channel_adapter = None

        # Load backbone with pretrained weights
        self.backbone = self._create_backbone(
            affectnet_pretrained
        )

        # Feature dimension before classifier (1280 for B0)
        in_features = self.backbone.num_features

        # Custom classifier: Dropout -> FC -> ReLU -> Dropout -> FC
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

        # Disable timm's built-in dropout before classifier
        # (custom classifier already has its own dropout layers)
        self.backbone.drop_rate = 0.0

        # Freeze backbone layers (optional)
        if freeze_backbone:
            self._freeze_backbone(unfreeze_last_n)

    @staticmethod
    def _create_backbone(affectnet_pretrained):
        """
        Create EfficientNet-B0 backbone with pretrained weights.

        When affectnet_pretrained is True, attempts to load
        AffectNet-pretrained weights via the hsemotion package.
        Falls back to ImageNet pretrained weights if hsemotion
        is not installed or weight loading fails.

        Args:
            affectnet_pretrained (bool): Whether to use AffectNet
                pretrained weights.

        Returns:
            nn.Module: EfficientNet-B0 backbone.
        """
        if affectnet_pretrained:
            try:
                from hsemotion.facial_emotions import (
                    HSEmotionRecognizer,
                )
                recognizer = HSEmotionRecognizer(
                    model_name='enet_b0_8_best_vgaf',
                    device='cpu'
                )
                backbone = recognizer.model
                print(
                    "[INFO] AffectNet-pretrained weights loaded "
                    "via hsemotion package."
                )
                return backbone
            except ImportError:
                print(
                    "[WARNING] hsemotion package not installed. "
                    "Install with: pip install hsemotion"
                )
            except Exception as e:
                print(
                    f"[WARNING] Could not load AffectNet "
                    f"weights: {e}"
                )

            print(
                "[INFO] Falling back to ImageNet pretrained "
                "weights."
            )

        backbone = timm.create_model(
            'efficientnet_b0', pretrained=True
        )
        print(
            "[INFO] ImageNet-pretrained EfficientNet-B0 "
            "loaded via timm."
        )
        return backbone

    def _freeze_backbone(self, unfreeze_last_n=2):
        """
        Freeze backbone feature layers, unfreeze last N blocks.

        timm EfficientNet-B0 structure:
            - conv_stem, bn1: Stem convolution and batch norm
            - blocks[0-6]: 7 MBConv feature blocks
            - conv_head, bn2: Head convolution (1x1, to 1280)

        Args:
            unfreeze_last_n (int): Number of last feature blocks
                to keep trainable for fine-tuning.
        """
        # Freeze stem layers
        for param in self.backbone.conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False

        # Freeze all feature blocks first
        for block in self.backbone.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Unfreeze last N feature blocks
        total_blocks = len(self.backbone.blocks)
        for i in range(
            max(0, total_blocks - unfreeze_last_n),
            total_blocks
        ):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Head convolution and BN are always trainable
        if hasattr(self.backbone, 'conv_head'):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'bn2'):
            for param in self.backbone.bn2.parameters():
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
        Extract feature vector before the final classifier layer.
        Useful for transfer learning analysis or t-SNE visualization.

        Args:
            x (Tensor): [batch, in_channels, H, W]

        Returns:
            Tensor: Feature vector of shape [batch, 256]
        """
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        # Extract features through backbone (without classifier)
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        if x.dim() > 2:
            x = x.flatten(1)

        # Pass through first 3 layers of custom classifier
        # (Dropout -> Linear(1280, 256) -> ReLU)
        for i in range(3):
            x = self.backbone.classifier[i](x)

        return x


def get_hsemotion_model(num_classes=None, in_channels=3,
                        pretrained_path=None,
                        freeze_backbone=True,
                        unfreeze_last_n=2,
                        affectnet_pretrained=True):
    """
    HSEmotion model factory function.

    Args:
        num_classes (int): Number of output classes
        in_channels (int): Input channels (1=grayscale, 3=RGB)
        pretrained_path (str): Path to a previously trained
            checkpoint
        freeze_backbone (bool): Freeze backbone layers
        unfreeze_last_n (int): Number of last blocks to unfreeze
        affectnet_pretrained (bool): Use AffectNet pretrained
            weights

    Returns:
        HSEmotion: Model instance
    """
    if num_classes is None:
        num_classes = config.DATA_CONFIG["num_classes"]

    model = HSEmotion(
        num_classes=num_classes,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n=unfreeze_last_n,
        affectnet_pretrained=affectnet_pretrained
    )

    # Load previously trained checkpoint if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(
            f"[INFO] Loading HSEmotion checkpoint: "
            f"{pretrained_path}"
        )
        checkpoint = torch.load(
            pretrained_path, map_location="cpu"
        )

        if isinstance(checkpoint, dict) \
                and "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        else:
            model.load_state_dict(checkpoint)

        print("[INFO] HSEmotion checkpoint loaded.")

    # Print parameter count
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(
        f"\n[MODEL] HSEmotion "
        f"(EfficientNet-B0 + AffectNet)"
    )
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Number of classes:    {num_classes}")
    print(f"  Input channels:       {in_channels}")
    print(f"  Backbone frozen:      {freeze_backbone}")
    print(
        f"  AffectNet pretrained: {affectnet_pretrained}"
    )

    return model
