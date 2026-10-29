"""
src/models/resnet_extractor.py
------------------------------
ResNet-18 CNN backbone for feature extraction.

Loads a pretrained ResNet-18 from torchvision, removes the final
Average-Pool and Fully-Connected layers, and returns spatial feature
maps of shape (B, 512, 7, 7) for a 224×224 input.

These 49 spatial locations (7×7 grid) become the token sequence fed
into the ViT encoder.

Usage:
    from src.models.resnet_extractor import ResNetExtractor
    extractor = ResNetExtractor(pretrained=True, frozen_layers=6)
    features = extractor(images)   # (B, 512, 7, 7)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class ResNetExtractor(nn.Module):
    """
    ResNet-18 feature extractor.

    Removes ``avgpool`` and ``fc`` from the standard ResNet-18 and
    exposes the final conv feature maps: **(B, 512, H/32, W/32)**.
    For a 224×224 input this is **(B, 512, 7, 7)**.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet pretrained weights (recommended for faster convergence).
    frozen_layers : int
        Number of initial layer *groups* to freeze (0–8).
        ResNet-18 layer groups: [conv1, bn1, relu, maxpool,
                                  layer1, layer2, layer3, layer4]
        Freezing the first 6 groups (conv1 → layer3) is a good default
        when fine-tuning on a small dataset.
    """

    # Named layer groups in order — used for selective freezing
    _LAYER_GROUPS = [
        "conv1", "bn1", "relu", "maxpool",
        "layer1", "layer2", "layer3", "layer4",
    ]

    def __init__(
        self,
        pretrained: bool = True,
        frozen_layers: int = 0,
    ) -> None:
        super().__init__()

        # ── Load ResNet-18 ────────────────────────────────────────────
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
            logger.info("ResNetExtractor: loaded ImageNet pretrained weights.")
        else:
            backbone = models.resnet18(weights=None)
            logger.info("ResNetExtractor: initialized with random weights.")

        # ── Remove avgpool + fc ───────────────────────────────────────
        # Keep everything up to (and including) layer4
        self.conv1   = backbone.conv1     # 112×112
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool   # 56×56
        self.layer1  = backbone.layer1   # 56×56,  64 ch
        self.layer2  = backbone.layer2   # 28×28, 128 ch
        self.layer3  = backbone.layer3   # 14×14, 256 ch
        self.layer4  = backbone.layer4   #  7×7,  512 ch

        self.out_channels: int = 512
        self.out_spatial: int = 7   # for 224×224 input

        # ── Selective layer freezing ──────────────────────────────────
        if frozen_layers > 0:
            self._freeze_layers(frozen_layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor, shape (B, 512, 7, 7)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x  # (B, 512, 7, 7)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_layers(self, n: int) -> None:
        """Freeze the first ``n`` layer groups."""
        n = min(n, len(self._LAYER_GROUPS))
        for name in self._LAYER_GROUPS[:n]:
            layer = getattr(self, name)
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(
            "ResNetExtractor: froze first %d layer groups: %s",
            n, self._LAYER_GROUPS[:n],
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call during fine-tuning stage)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("ResNetExtractor: all layers unfrozen.")

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover
        trainable = self.count_trainable_params()
        total = sum(p.numel() for p in self.parameters())
        return (
            f"ResNetExtractor("
            f"out_channels={self.out_channels}, "
            f"out_spatial={self.out_spatial}×{self.out_spatial}, "
            f"trainable={trainable:,}/{total:,} params)"
        )

