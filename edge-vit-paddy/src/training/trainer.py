"""
src/training/trainer.py
-----------------------
Full training loop for the Hybrid ResNet-18 + ViT model.

Features:
    ✓ CrossEntropyLoss with per-class weights (handles class imbalance)
    ✓ AdamW optimizer with configurable learning rate and weight decay
    ✓ CosineAnnealingLR scheduler with linear warm-up
    ✓ Gradient clipping (prevents exploding gradients in Transformer)
    ✓ Early stopping (patience from config)
    ✓ Best checkpoint saving (phase_a_best.pth) with full metadata
    ✓ Per-epoch metrics logging
    ✓ Returns full training history dict for visualizer

Usage:
    from src.training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.hybrid_model import HybridViTModel
from src.models.model_factory import save_checkpoint
from src.training.validator import Validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors validation loss and stops training when no improvement
    is seen for ``patience`` consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d/%d epochs.",
                self.counter, self.patience,
            )
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the full train → validate → checkpoint loop.

    Parameters
    ----------
    model         : HybridViTModel
    train_loader  : DataLoader (shuffle=True, augmented)
    val_loader    : DataLoader (shuffle=False, clean)
    config        : parsed config.yaml dict
    device        : torch.device
    class_weights : optional Tensor for CrossEntropyLoss (shape: [num_classes])
                    if None, computed automatically from the dataset.
    """

    def __init__(
        self,
        model:         HybridViTModel,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        config:        dict,
        device:        torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        train_cfg = config.get("training", {})
        self.epochs         : int   = train_cfg.get("epochs",                  50)
        self.lr             : float = train_cfg.get("learning_rate",           3e-4)
        self.weight_decay   : float = train_cfg.get("weight_decay",            1e-4)
        self.grad_clip      : float = train_cfg.get("grad_clip",               1.0)
        self.warmup_epochs  : int   = train_cfg.get("warmup_epochs",           5)
        self.patience       : int   = train_cfg.get("early_stopping_patience", 10)
        self.save_every     : int   = config.get("checkpoint", {}).get("save_every_n_epochs", 5)
        self.best_ckpt_path : str   = config.get("checkpoint", {}).get(
            "best_model_path", "checkpoints/phase_a_best.pth"
        )

        # ── Loss ──────────────────────────────────────────────────────
        if class_weights is None:
            class_weights = self._compute_class_weights()
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

        # ── Optimizer ─────────────────────────────────────────────────
        opt_name = train_cfg.get("optimizer", "adamw").lower()
        OptCls   = AdamW if opt_name == "adamw" else Adam
        self.optimizer = OptCls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ── Scheduler: linear warm-up then cosine annealing ───────────
        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs - self.warmup_epochs, 1),
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[self.warmup_epochs],
        )

        # ── Validator + Early Stopping ────────────────────────────────
        self.validator      = Validator(model, val_loader, self.criterion, device)
        self.early_stopping = EarlyStopping(patience=self.patience)

        # ── History ───────────────────────────────────────────────────
        self.best_val_acc  : float = 0.0
        self.best_val_loss : float = float("inf")
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss":  [],
            "train_acc":  [], "val_acc":   [],
            "lr":         [],
        }

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(self, resume_epoch: int = 0) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Parameters
        ----------
        resume_epoch : int
            Set to the epoch stored in a checkpoint to resume mid-training.

        Returns
        -------
        history : dict with keys train_loss, val_loss, train_acc, val_acc, lr.
        """
        logger.info(
            "Starting training | epochs=%d | device=%s | optimizer=%s | lr=%.2e",
            self.epochs, self.device,
            self.optimizer.__class__.__name__, self.lr,
        )

        t0 = time.time()

        for epoch in range(resume_epoch + 1, self.epochs + 1):

            # ── Train one epoch ───────────────────────────────────────
            train_loss, train_acc = self._train_epoch(epoch)

            # ── Validate ──────────────────────────────────────────────
            val_loss, val_acc = self.validator.run(epoch)

            # ── LR step ───────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # ── Record history ────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # ── Summary line ──────────────────────────────────────────
            logger.info(
                "Epoch %03d/%03d | train loss=%.4f acc=%.4f | "
                "val loss=%.4f acc=%.4f | lr=%.2e",
                epoch, self.epochs,
                train_loss, train_acc,
                val_loss,   val_acc,
                current_lr,
            )

            # ── Best checkpoint ───────────────────────────────────────
            if val_acc > self.best_val_acc:
                self.best_val_acc  = val_acc
                self.best_val_loss = val_loss
                save_checkpoint(
                    model        = self.model,
                    optimizer    = self.optimizer,
                    epoch        = epoch,
                    val_accuracy = val_acc,
                    path         = self.best_ckpt_path,
                    config       = self.config,
                    scheduler    = self.scheduler,
                    extra        = {"val_loss": val_loss},
                )
                logger.info("  ⭐ New best saved (val_acc=%.4f)", val_acc)

            # ── Periodic checkpoint ────────────────────────────────────
            if epoch % self.save_every == 0:
                periodic = (
                    Path(self.best_ckpt_path).parent / f"epoch_{epoch:03d}.pth"
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc,
                    periodic, self.config,
                )

            # ── Early stopping ────────────────────────────────────────
            if self.early_stopping.step(val_loss):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch, self.patience,
                )
                break

        logger.info(
            "Training complete in %.1f s | best val_acc=%.4f",
            time.time() - t0, self.best_val_acc,
        )
        return self.history

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """One full pass over train_loader. Returns (mean_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Train Epoch {epoch:03d}]",
            leave=False,
            ncols=90,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping — essential for Transformer stability
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """Auto-derive inverse-frequency class weights from the training dataset."""
        dataset = self.train_loader.dataset
        # Direct PaddyDiseaseDataset
        if hasattr(dataset, "get_class_weights"):
            w = dataset.get_class_weights()
            logger.info("Class weights: %s", w)
            return torch.tensor(w, dtype=torch.float32)
        # Wrapped in _TransformSubset
        inner = getattr(getattr(dataset, "subset", None), "dataset", None)
        if inner and hasattr(inner, "get_class_weights"):
            w = inner.get_class_weights()
            return torch.tensor(w, dtype=torch.float32)
        logger.warning("Could not compute class weights — using uniform weights.")
        return None

    @property
    def history_summary(self) -> str:
        if not self.history["val_acc"]:
            return "No training history yet."
        best_epoch = int(
            max(range(len(self.history["val_acc"])),
                key=lambda i: self.history["val_acc"][i])
        ) + 1
        return (
            f"Epochs run : {len(self.history['val_acc'])}\n"
            f"Best epoch : {best_epoch}\n"
            f"Best val acc : {self.best_val_acc:.4f}\n"
            f"Best val loss: {self.best_val_loss:.4f}"
        )

