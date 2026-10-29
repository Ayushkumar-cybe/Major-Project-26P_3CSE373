"""
src/models/model_factory.py
---------------------------
Factory functions to build, load, and save the HybridViTModel.

Provides a single interface for:
  - Building a fresh model from config
  - Loading a model from a Phase A/B checkpoint
  - Saving a checkpoint with full metadata (num_classes, class_labels, epoch…)
  - Counting and printing model statistics

Usage:
    from src.models.model_factory import build_model, load_checkpoint, save_checkpoint

    # Build fresh
    model = build_model(config)

    # Load from checkpoint
    model, meta = load_checkpoint("checkpoints/phase_a_best.pth", config)

    # Save
    save_checkpoint(model, optimizer, epoch=10, val_acc=0.94,
                    path="checkpoints/phase_a_best.pth", config=config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.hybrid_model import HybridViTModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_model(config: dict, device: Optional[torch.device] = None) -> HybridViTModel:
    """
    Instantiate a fresh HybridViTModel from config and move to device.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dict.
    device : torch.device, optional
        Target device. Auto-detected (CUDA → MPS → CPU) if not provided.

    Returns
    -------
    HybridViTModel
        Model on the specified device, ready for training.
    """
    if device is None:
        device = _auto_device()

    model = HybridViTModel(config)
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Model built → device=%s | total_params=%s | trainable=%s",
        device,
        f"{total:,}",
        f"{trainable:,}",
    )
    return model


# ---------------------------------------------------------------------------
# Save Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: HybridViTModel,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_accuracy: float,
    path: str | Path,
    config: dict,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict] = None,
) -> None:
    """
    Save a full checkpoint with model weights, optimizer state, and metadata.

    Checkpoint structure:
        {
            "model_state_dict"   : ...,
            "optimizer_state_dict": ...,
            "scheduler_state_dict": ...,
            "epoch"              : int,
            "val_accuracy"       : float,
            "num_classes"        : int,
            "class_labels"       : list[str],
            "phase"              : str,
            "model_version"      : str,
            "config"             : dict,   (subset)
            ...extra
        }

    Parameters
    ----------
    model       : HybridViTModel
    optimizer   : Optimizer (or None to skip)
    epoch       : Completed epoch number
    val_accuracy: Validation accuracy at this checkpoint
    path        : Save path (e.g. "checkpoints/phase_a_best.pth")
    config      : Full config dict — phase, version, class list stored in metadata
    scheduler   : LR scheduler (optional)
    extra       : Additional metadata dict (e.g. {"val_f1": 0.93})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch"               : epoch,
        "val_accuracy"        : val_accuracy,
        # Phase metadata — critical for safe Phase B upgrade
        "num_classes"         : model.num_classes,
        "class_labels"        : model.classes,
        "class_to_idx"        : model.class_to_idx,
        "phase"               : config.get("project", {}).get("phase", "phase_a"),
        "model_version"       : config.get("app", {}).get("model_version", "phase_a_v1"),
        # Snapshot of key config params
        "config_snapshot"     : {
            "embed_dim"              : config.get("model", {}).get("embed_dim", 256),
            "num_transformer_layers" : config.get("model", {}).get("num_transformer_layers", 4),
            "num_heads"              : config.get("model", {}).get("num_heads", 8),
            "image_size"             : config.get("data", {}).get("image_size", 224),
        },
    }

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(
        "Checkpoint saved → %s  (epoch=%d, val_acc=%.4f, classes=%s)",
        path, epoch, val_accuracy, model.classes,
    )


# ---------------------------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    config: dict,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[HybridViTModel, Dict]:
    """
    Load a HybridViTModel from a saved checkpoint.

    Verifies that ``num_classes`` and ``class_labels`` stored in the
    checkpoint match the current config — raises ValueError if they differ,
    preventing silent Phase B/A mismatches.

    Parameters
    ----------
    path   : path to .pth checkpoint file
    config : current config dict
    device : target device (auto-detected if None)
    strict : passed to ``load_state_dict`` (True = no missing/unexpected keys)

    Returns
    -------
    model    : HybridViTModel  (on device, in eval mode)
    metadata : dict  — checkpoint metadata (epoch, val_accuracy, phase, etc.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = _auto_device()

    checkpoint = torch.load(path, map_location=device)

    # ── Metadata validation ───────────────────────────────────────────
    ckpt_classes = checkpoint.get("class_labels", [])
    cfg_classes  = config.get("classes", [])
    if ckpt_classes and ckpt_classes != cfg_classes:
        raise ValueError(
            f"Class mismatch between checkpoint and config!\n"
            f"  Checkpoint : {ckpt_classes}\n"
            f"  Config     : {cfg_classes}\n"
            f"Update config.yaml to match the checkpoint before loading."
        )

    # ── Build model from current config and load weights ─────────────
    model = HybridViTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model = model.to(device)
    model.eval()

    meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    logger.info(
        "Checkpoint loaded ← %s  (epoch=%d, val_acc=%.4f, phase=%s)",
        path,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_accuracy", 0.0),
        checkpoint.get("phase", "unknown"),
    )
    return model, meta


def load_optimizer_state(
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Restore optimizer (and scheduler) state from a checkpoint.

    Returns
    -------
    int : epoch number stored in checkpoint (resume from next epoch)
    """
    device = device or _auto_device()
    ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Scheduler state restored.")

    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Model Summary
# ---------------------------------------------------------------------------

def print_model_summary(model: HybridViTModel) -> None:
    """Print a human-readable model summary to stdout."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print("\n" + "=" * 60)
    print("  HYBRID ResNet-18 + ViT MODEL SUMMARY")
    print("=" * 60)
    print(f"  Classes      : {model.num_classes} → {model.classes}")
    print(f"  Backbone     : ResNet-18  (out: 512 × 7 × 7)")
    print(f"  ViT encoder  : embed_dim={model.encoder.embed_dim}, "
          f"blocks={len(model.encoder.blocks)}, "
          f"heads={model.encoder.blocks[0].attn.num_heads}")
    print(f"  Head         : {model.encoder.embed_dim} → "
          f"{model.head.head[0].out_features} → {model.num_classes}")
    print(f"  Total params : {total:>12,}")
    print(f"  Trainable    : {trainable:>12,}")
    print(f"  Frozen       : {frozen:>12,}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Auto-select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Auto device: %s", device)
    return device

