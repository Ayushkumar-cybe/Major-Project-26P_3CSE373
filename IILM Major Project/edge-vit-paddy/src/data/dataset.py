"""
src/data/dataset.py
-------------------
Custom PyTorch Dataset for paddy leaf disease detection.

Phase A classes (from config.yaml):
    0 → Leaf Blast
    1 → Brown Spot
    2 → Healthy

Expects the dataset directory to follow ImageFolder-style layout:
    root/
    ├── Leaf Blast/
    │   ├── img001.jpg
    │   └── ...
    ├── Brown Spot/
    │   └── ...
    └── Healthy/
        └── ...

Usage:
    from src.data.dataset import PaddyDiseaseDataset
    dataset = PaddyDiseaseDataset(root_dir="data/train", config=cfg)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config from disk."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PaddyDiseaseDataset(Dataset):
    """
    Phase-aware paddy disease dataset.

    Parameters
    ----------
    root_dir : str or Path
        Path to dataset split directory (e.g. ``data/train/``).
        Must contain sub-folders named after each class defined in config.
    config : dict
        Parsed config dict (from config.yaml).
    transform : callable, optional
        Torchvision transforms applied to each image.
    phase_classes : list[str], optional
        Override the class list from config (useful for unit tests).
        Defaults to ``config['classes']``.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    )

    def __init__(
        self,
        root_dir: str | Path,
        config: dict,
        transform: Optional[Callable] = None,
        phase_classes: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.config = config
        self.transform = transform

        # ── Class definitions ────────────────────────────────────────────
        # Read class list from config (Phase A = 3 classes).
        # Providing `phase_classes` overrides the config (used in tests).
        self.classes: List[str] = phase_classes or config["classes"]
        self.num_classes: int = len(self.classes)

        # Bidirectional lookup maps
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls for cls, idx in self.class_to_idx.items()
        }

        # ── Discover samples ─────────────────────────────────────────────
        self.samples: List[Tuple[Path, int]] = []
        self._discover_samples()

        if len(self.samples) == 0:
            logger.warning(
                "No samples found in '%s' for classes %s. "
                "Ensure subdirectories exist and contain images.",
                self.root_dir,
                self.classes,
            )
        else:
            logger.info(
                "PaddyDiseaseDataset loaded: %d samples, %d classes from '%s'",
                len(self.samples),
                self.num_classes,
                self.root_dir,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> None:
        """
        Walk root_dir and collect (image_path, label_index) pairs.

        Only sub-folders whose names appear in ``self.classes`` are
        included — this is the Phase A filter that ignores other-disease
        folders that may exist in the raw dataset.
        """
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                logger.warning(
                    "Class directory not found: '%s'. Skipping.", class_dir
                )
                continue

            label = self.class_to_idx[class_name]
            count = 0
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((file_path, label))
                    count += 1

            logger.debug("  [%s] → %d images (label=%d)", class_name, count, label)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns
        -------
        image : PIL.Image or Tensor
            Loaded image, optionally transformed.
        label : int
            Class index (0 = Leaf Blast, 1 = Brown Spot, 2 = Healthy).
        """
        image_path, label = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to load image '%s': %s", image_path, exc)
            raise

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_class_counts(self) -> Dict[str, int]:
        """Return per-class sample count (useful for computing class weights)."""
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return counts

    def get_class_weights(self) -> List[float]:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.

        Returns a list of floats aligned with ``self.classes`` order,
        suitable for passing directly to ``torch.nn.CrossEntropyLoss(weight=...)``.
        """
        import torch

        counts = self.get_class_counts()
        total = sum(counts.values())
        weights = [
            total / (self.num_classes * counts[cls]) if counts[cls] > 0 else 0.0
            for cls in self.classes
        ]
        logger.info("Class weights: %s", dict(zip(self.classes, weights)))
        return weights

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        lines = [
            f"PaddyDiseaseDataset(root='{self.root_dir}', n={len(self.samples)})",
        ]
        for cls, cnt in counts.items():
            lines.append(f"  {self.class_to_idx[cls]} | {cls:<20} | {cnt} samples")
        return "\n".join(lines)

