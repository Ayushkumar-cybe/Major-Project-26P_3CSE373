"""
src/models/vit_encoder.py
-------------------------
Vision Transformer encoder that operates on the spatial feature map
produced by ResNetExtractor.

Pipeline inside this module:
    (B, 512, 7, 7)          ← input from ResNet
        │
    PatchEmbedding          → (B, 49, embed_dim)   flatten spatial grid
        │
    + CLS token             → (B, 50, embed_dim)
        │
    + Positional Encoding   → (B, 50, embed_dim)
        │
    TransformerEncoderBlock × N layers
        ├── LayerNorm
        ├── Multi-Head Self-Attention
        ├── Residual + Dropout
        ├── LayerNorm
        ├── MLP (Feed-Forward)
        └── Residual + Dropout
        │
    CLS token extraction (or mean pool)
        │
    (B, embed_dim)          ← global feature vector

Usage:
    from src.models.vit_encoder import ViTEncoder
    encoder = ViTEncoder(in_channels=512, num_patches=49,
                         embed_dim=256, num_heads=8,
                         num_layers=4, dropout=0.1)
    feat = encoder(feature_map)   # (B, embed_dim)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Flatten ResNet spatial feature map → token sequence + linear projection.

    Input : (B, C, H, W)  e.g. (B, 512, 7, 7)
    Output: (B, H*W, embed_dim)  e.g. (B, 49, 256)
    """

    def __init__(self, in_channels: int, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        Returns (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, C, H*W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)          # (B, 49, 512)
        x = self.projection(x)                     # (B, 49, embed_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
        x → LN → MHSA → residual + dropout
          → LN → MLP  → residual + dropout

    Parameters
    ----------
    embed_dim   : token dimension D
    num_heads   : number of attention heads (embed_dim must be divisible)
    mlp_ratio   : MLP hidden dim = embed_dim × mlp_ratio
    dropout     : dropout rate applied after attention and MLP
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Pre-norm (more stable than post-norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, embed_dim)
        Returns same shape.
        """
        # MHSA with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop1(attn_out)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT Encoder (full)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder that converts a CNN feature map to a
    fixed-size global feature vector.

    Parameters
    ----------
    in_channels : int
        Number of channels from the CNN backbone (512 for ResNet-18).
    num_patches : int
        Number of spatial locations (H × W = 49 for a 7×7 feature map).
    embed_dim : int
        Token embedding dimension D (default 256).
    num_heads : int
        Number of attention heads (embed_dim must be divisible by num_heads).
    num_layers : int
        Number of stacked TransformerEncoderBlocks.
    mlp_ratio : float
        MLP hidden dim multiplier (default 4.0).
    dropout : float
        Dropout applied inside attention and MLP (default 0.1).
    use_cls_token : bool
        If True (default), prepends a learnable [CLS] token and uses its
        output as the global feature.
        If False, uses mean-pooling over all patch tokens.
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_patches: int = 49,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim     = embed_dim
        self.num_patches   = num_patches
        self.use_cls_token = use_cls_token

        # ── Patch Embedding ───────────────────────────────────────────
        self.patch_embed = PatchEmbedding(in_channels, num_patches, embed_dim)

        # ── CLS token ─────────────────────────────────────────────────
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            seq_len = num_patches + 1   # patches + CLS
        else:
            self.cls_token = None
            seq_len = num_patches

        # ── Learnable Positional Encoding ─────────────────────────────
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── Dropout after embedding + positional enc ──────────────────
        self.embed_dropout = nn.Dropout(dropout)

        # ── Transformer Encoder Blocks ────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Final Layer Norm ──────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        logger.info(
            "ViTEncoder: embed_dim=%d, heads=%d, layers=%d, "
            "patches=%d, cls_token=%s",
            embed_dim, num_heads, num_layers, num_patches, use_cls_token,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feature_map : torch.Tensor, shape (B, 512, 7, 7)

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Global feature vector (CLS token output or mean-pooled patches).
        """
        B = feature_map.shape[0]

        # 1. Patch embedding: (B, 512, 7, 7) → (B, 49, embed_dim)
        x = self.patch_embed(feature_map)

        # 2. Prepend CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
            x = torch.cat([cls, x], dim=1)            # (B, 50, embed_dim)

        # 3. Add positional encoding
        x = x + self.pos_embedding

        # 4. Dropout
        x = self.embed_dropout(x)

        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 6. Final norm
        x = self.norm(x)

        # 7. Extract global feature
        if self.use_cls_token:
            return x[:, 0]    # CLS token  → (B, embed_dim)
        else:
            return x.mean(1)  # mean pool  → (B, embed_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ViTEncoder("
            f"embed_dim={self.embed_dim}, "
            f"blocks={len(self.blocks)}, "
            f"params={self.count_trainable_params():,})"
        )

