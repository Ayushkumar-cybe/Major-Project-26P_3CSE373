"""
src/data/dataloader.py
----------------------
DataLoader factory for paddy disease detection.

Provides:
    build_dataloaders()         — returns (train_loader, val_loader)
    build_single_dataloader()   — returns a DataLoader for one split
    split_dataset()             — random train/val split from a single directory

Usage:
    from src.data.dataloader import build_dataloaders
    train_loader, val_loader = build_dataloaders(config)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset

from src.data.dataset import PaddyDiseaseDataset
from src.data.preprocess import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary Builder — pre-split directories (recommended)
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: dict,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from pre-split directories.

    If ``train_dir`` / ``val_dir`` are not provided, they are read from
    ``config['paths']['train_dir']`` and ``config['paths']['val_dir']``.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dict.
    train_dir : str, optional
        Override path to training images.
    val_dir : str, optional
        Override path to validation images.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
    paths = config.get("paths", {})
    train_path = Path(train_dir or paths.get("train_dir", "data/train"))
    val_path = Path(val_dir or paths.get("val_dir", "data/val"))

    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    batch_size: int = train_cfg.get("batch_size", 32)
    # num_workers: set to 0 on Windows if multiprocessing causes issues
    num_workers: int = train_cfg.get("num_workers", 0)
    seed: int = train_cfg.get("seed", 42)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = PaddyDiseaseDataset(
        root_dir=train_path,
        config=config,
        transform=get_train_transforms(config),
    )
    val_dataset = PaddyDiseaseDataset(
        root_dir=val_path,
        config=config,
        transform=get_val_transforms(config),
    )

    logger.info(
        "Train dataset: %d samples | Val dataset: %d samples",
        len(train_dataset),
        len(val_dataset),
    )

    # ── Class weights (for imbalanced data) ──────────────────────────────
    class_weights = train_dataset.get_class_weights()
    logger.info("Class weights for CrossEntropyLoss: %s", class_weights)

    # ── DataLoaders ───────────────────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=True,    # Drop incomplete last batch during training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    _log_dataloader_info(train_loader, "Train")
    _log_dataloader_info(val_loader, "Val")

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Alternative Builder — single directory with automatic split
# ---------------------------------------------------------------------------

def build_dataloaders_from_single_dir(
    config: dict,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders by randomly splitting a single directory.

    Useful when the dataset isn't pre-split. Uses ``config['data']['train_split']``
    (default 0.8) to determine the ratio.

    Parameters
    ----------
    config : dict
        Parsed config.yaml dict.
    data_dir : str, optional
        Override data directory. Defaults to ``config['paths']['raw_dir']``.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
    paths = config.get("paths", {})
    dir_path = Path(data_dir or paths.get("raw_dir", "data/raw"))

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    train_split: float = data_cfg.get("train_split", 0.8)
    seed: int = train_cfg.get("seed", 42)
    batch_size: int = train_cfg.get("batch_size", 32)
    num_workers: int = train_cfg.get("num_workers", 0)

    # Load full dataset without transform first (transforms applied per subset)
    full_dataset = PaddyDiseaseDataset(
        root_dir=dir_path,
        config=config,
        transform=None,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    logger.info(
        "Splitting %d total samples → %d train / %d val (%.0f%%/%.0f%%)",
        n_total, n_train, n_val,
        train_split * 100, (1 - train_split) * 100,
    )

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Wrap subsets with transform-aware dataset
    train_dataset = _TransformSubset(train_subset, get_train_transforms(config))
    val_dataset = _TransformSubset(val_subset, get_val_transforms(config))

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    _log_dataloader_info(train_loader, "Train")
    _log_dataloader_info(val_loader, "Val")

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Single-loader builder (for inference or test)
# ---------------------------------------------------------------------------

def build_single_dataloader(
    config: dict,
    data_dir: str,
    is_train: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a single directory (e.g. test set or inference batch).

    Parameters
    ----------
    config : dict
    data_dir : str
    is_train : bool
        If True, applies augmentation. Typically False for inference.
    """
    transform = get_train_transforms(config) if is_train else get_val_transforms(config)
    train_cfg = config.get("training", {})

    dataset = PaddyDiseaseDataset(
        root_dir=data_dir,
        config=config,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=is_train,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset with an independent transform."""

    def __init__(self, subset: Subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _log_dataloader_info(loader: DataLoader, name: str) -> None:
    """Log DataLoader statistics."""
    logger.info(
        "[%s DataLoader] batches=%d | batch_size=%d | total_samples=%d",
        name,
        len(loader),
        loader.batch_size,
        len(loader.dataset),
    )

