"""
src/models/hybrid_model.py
--------------------------
Full Hybrid ResNet-18 + Vision Transformer model for paddy disease detection.

Architecture:
    Input (B, 3, 224, 224)
        │
    ResNetExtractor          → (B, 512, 7, 7)   CNN spatial features
        │
    ViTEncoder               → (B, embed_dim)   global feature vector
        │
    ClassificationHead       → (B, num_classes) logits
        │
    [Softmax at inference — CrossEntropy handles it during training]

The ``num_classes`` and ``embed_dim`` are read dynamically from config.yaml,
so Phase B (extended classes) only requires updating config + retraining.
No architecture code changes needed.

Usage:
    from src.models.hybrid_model import HybridViTModel
    model = HybridViTModel(config=cfg)
    logits = model(images)   # (B, num_classes)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet_extractor import ResNetExtractor
from src.models.vit_encoder import ViTEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    MLP classification head.

    Architecture:
        Linear(embed_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)

    Parameters
    ----------
    embed_dim   : input feature dimension (from ViTEncoder)
    hidden_dim  : intermediate MLP width
    num_classes : number of output classes (3 for Phase A)
    dropout     : dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, embed_dim) → (B, num_classes)"""
        return self.head(x)


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridViTModel(nn.Module):
    """
    Hybrid ResNet-18 + Vision Transformer model.

    All hyperparameters are read from ``config['model']`` and
    ``config['num_classes']``, making the model fully config-driven.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dict.

    Attributes
    ----------
    backbone  : ResNetExtractor
    encoder   : ViTEncoder
    head      : ClassificationHead
    classes   : list[str]   — class label list from config
    num_classes : int
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        model_cfg = config.get("model", {})
        self.num_classes: int  = config["num_classes"]
        self.classes: List[str] = config["classes"]
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        # ── CNN Backbone ──────────────────────────────────────────────
        self.backbone = ResNetExtractor(
            pretrained=model_cfg.get("pretrained", True),
            frozen_layers=model_cfg.get("frozen_layers", 0),
        )

        # ── ViT Encoder ───────────────────────────────────────────────
        embed_dim: int = model_cfg.get("embed_dim", 256)
        self.encoder = ViTEncoder(
            in_channels=self.backbone.out_channels,         # 512
            num_patches=self.backbone.out_spatial ** 2,     # 49
            embed_dim=embed_dim,
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_transformer_layers", 4),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            dropout=model_cfg.get("dropout", 0.1),
            use_cls_token=model_cfg.get("cls_token", True),
        )

        # ── Classification Head ───────────────────────────────────────
        self.head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=model_cfg.get("head_hidden_dim", 128),
            num_classes=self.num_classes,
            dropout=model_cfg.get("head_dropout", 0.3),
        )

        # Weight init for non-pretrained layers
        self._init_weights()

        logger.info(
            "HybridViTModel: num_classes=%d (%s), total_params=%s",
            self.num_classes,
            ", ".join(self.classes),
            f"{self.count_total_params():,}",
        )

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
        logits : torch.Tensor, shape (B, num_classes)
            Raw logits (no softmax). Use CrossEntropyLoss during training.
            Apply softmax at inference for probabilities.
        """
        # 1. CNN feature extraction: (B, 3, 224, 224) → (B, 512, 7, 7)
        features = self.backbone(x)

        # 2. ViT encoding: (B, 512, 7, 7) → (B, embed_dim)
        global_feat = self.encoder(features)

        # 3. Classification: (B, embed_dim) → (B, num_classes)
        logits = self.head(global_feat)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-mode forward pass returning probabilities and class indices.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 224, 224)

        Returns
        -------
        probs       : torch.Tensor, shape (B, num_classes)  — softmax probabilities
        class_ids   : torch.Tensor, shape (B,)              — predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            class_ids = probs.argmax(dim=-1)
        return probs, class_ids

    def predict_single(
        self, x: torch.Tensor
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Convenience method for single-image inference.

        Parameters
        ----------
        x : torch.Tensor, shape (1, 3, 224, 224) or (3, 224, 224)

        Returns
        -------
        predicted_class : str           — e.g. "Leaf Blast"
        confidence      : float         — max probability, e.g. 0.87
        all_scores      : dict          — {class_name: probability}
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        probs, class_ids = self.predict(x)
        probs_np   = probs[0].cpu().tolist()
        class_id   = class_ids[0].item()
        all_scores = {cls: round(probs_np[i], 6) for i, cls in enumerate(self.classes)}
        return self.idx_to_class[class_id], round(probs_np[class_id], 6), all_scores

    # ------------------------------------------------------------------
    # Phase-awareness helpers
    # ------------------------------------------------------------------

    def get_phase_info(self) -> Dict:
        """Return current phase metadata (stored in checkpoint)."""
        return {
            "num_classes": self.num_classes,
            "class_labels": self.classes,
            "class_to_idx": self.class_to_idx,
        }

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self, n_layers: int = 8) -> None:
        """Freeze backbone layers (useful for initial training warm-up)."""
        self.backbone._freeze_layers(n_layers)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning stage)."""
        self.backbone.unfreeze_all()

    def _init_weights(self) -> None:
        """Initialize ViT and head weights (backbone uses pretrained)."""
        for module in [self.encoder.patch_embed.projection,
                       *self.head.head]:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def __repr__(self) -> str:  # pragma: no cover
        total = self.count_total_params()
        trainable = self.count_trainable_params()
        return (
            f"HybridViTModel(\n"
            f"  backbone  : ResNet-18 (out=512×7×7)\n"
            f"  encoder   : ViTEncoder (embed={self.encoder.embed_dim}, "
            f"blocks={len(self.encoder.blocks)})\n"
            f"  head      : ClassificationHead → {self.num_classes} classes "
            f"({', '.join(self.classes)})\n"
            f"  params    : {trainable:,} trainable / {total:,} total\n"
            f")"
        )

