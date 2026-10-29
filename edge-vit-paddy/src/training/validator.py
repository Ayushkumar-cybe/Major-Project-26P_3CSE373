"""
src/training/validator.py
-------------------------
Validation loop — runs inference over the validation DataLoader
for a single epoch and returns loss + accuracy.

Used by Trainer at the end of every training epoch.

Usage:
    from src.training.validator import Validator
    validator = Validator(model, val_loader, criterion, device)
    val_loss, val_acc = validator.run(epoch=1)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Validator:
    """
    Runs one full pass over the validation set.

    Parameters
    ----------
    model      : nn.Module — the hybrid model
    val_loader : DataLoader — validation DataLoader (shuffle=False)
    criterion  : loss function (CrossEntropyLoss)
    device     : torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        self.model      = model
        self.val_loader = val_loader
        self.criterion  = criterion
        self.device     = device

    def run(self, epoch: int = 0) -> Tuple[float, float]:
        """
        Run validation for one epoch.

        Parameters
        ----------
        epoch : int — current epoch number (for logging only)

        Returns
        -------
        val_loss : float — mean loss over all batches
        val_acc  : float — top-1 accuracy in [0, 1]
        """
        self.model.eval()
        total_loss   = 0.0
        correct      = 0
        total        = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"[Val   Epoch {epoch:03d}]",
                leave=False,
                ncols=90,
            )
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds   = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += images.size(0)

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = total_loss / max(total, 1)
        val_acc  = correct   / max(total, 1)

        logger.info(
            "[Val   Epoch %03d] loss=%.4f | acc=%.4f",
            epoch, val_loss, val_acc,
        )
        return val_loss, val_acc

    def collect_predictions(self) -> Tuple[List[int], List[int]]:
        """
        Collect all ground-truth labels and predicted labels for the
        full validation set (used by metrics.py after training).

        Returns
        -------
        all_labels : list[int]
        all_preds  : list[int]
        """
        self.model.eval()
        all_labels: List[int] = []
        all_preds:  List[int] = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                preds  = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return all_labels, all_preds

