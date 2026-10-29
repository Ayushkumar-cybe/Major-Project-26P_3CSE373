"""
src/evaluation/visualizer.py
-----------------------------
Visualization utilities for training curves, confusion matrices,
per-class confidence plots, and sample prediction grids.

All plots are saved to ``outputs/plots/`` by default.

Usage:
    from src.evaluation.visualizer import Visualizer
    viz = Visualizer(config, save_dir="outputs/plots")
    viz.plot_training_curves(history)
    viz.plot_confusion_matrix(results)
    viz.plot_class_f1_bars(results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for servers & edge devices
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# Colour palette aligned with project theme
PALETTE = {
    "train"       : "#2196F3",   # blue
    "val"         : "#FF5722",   # deep orange
    "healthy"     : "#4CAF50",   # green
    "leaf_blast"  : "#F44336",   # red
    "brown_spot"  : "#795548",   # brown
    "grid"        : "#E0E0E0",
    "bg"          : "#FAFAFA",
}

CLASS_COLOURS = ["#F44336", "#795548", "#4CAF50",   # Phase A
                 "#9C27B0", "#FF9800", "#00BCD4"]    # Phase B (future)


class Visualizer:
    """
    Saves all evaluation plots to ``save_dir``.

    Parameters
    ----------
    config   : parsed config dict (used for class names, phase info)
    save_dir : directory to write PNG files (created if missing)
    dpi      : image resolution (default 150)
    """

    def __init__(
        self,
        config: dict,
        save_dir: str = "outputs/plots",
        dpi: int = 150,
    ) -> None:
        self.config      = config
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi         = dpi
        self.class_names = config.get("classes", [])

    # ------------------------------------------------------------------
    # 1. Training & Validation Curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_curves.png",
    ) -> Path:
        """
        Plot loss and accuracy curves side-by-side.

        Parameters
        ----------
        history : dict with keys:
            "train_loss", "val_loss", "train_acc", "val_acc"
            Each value is a list of per-epoch floats.

        Returns
        -------
        Path to saved PNG.
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(PALETTE["bg"])

        # ── Loss ────────────────────────────────────────────────────
        ax1.plot(epochs, history["train_loss"], color=PALETTE["train"],
                 lw=2, label="Train Loss", marker="o", ms=3)
        ax1.plot(epochs, history["val_loss"], color=PALETTE["val"],
                 lw=2, label="Val Loss",   marker="s", ms=3)
        ax1.set_title("Loss per Epoch", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("CrossEntropy Loss")
        ax1.legend(); ax1.grid(color=PALETTE["grid"], linestyle="--", lw=0.8)
        ax1.set_facecolor(PALETTE["bg"])

        # ── Accuracy ─────────────────────────────────────────────────
        ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
                 color=PALETTE["train"], lw=2, label="Train Acc", marker="o", ms=3)
        ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
                 color=PALETTE["val"],   lw=2, label="Val Acc",   marker="s", ms=3)
        # Phase A target line
        ax2.axhline(92, color="gray", linestyle=":", lw=1.5, label="Target 92%")
        ax2.set_title("Accuracy per Epoch", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax2.legend(); ax2.grid(color=PALETTE["grid"], linestyle="--", lw=0.8)
        ax2.set_facecolor(PALETTE["bg"])

        # Best val epoch marker
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        ax1.axvline(best_epoch, color="green", linestyle="--", lw=1,
                    label=f"Best epoch {best_epoch}")
        ax1.legend()

        plt.suptitle(
            f"Phase A Training — {self.config.get('project', {}).get('name', '')}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        out = self._save(fig, filename)
        logger.info("Training curves saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 2. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        results: Dict,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
    ) -> Path:
        """
        Plot a colour-coded confusion matrix heatmap.

        Parameters
        ----------
        results   : dict returned by metrics.evaluate()
        normalize : if True, show row-normalised proportions (clearer for
                    imbalanced datasets); raw counts shown in cell text.
        """
        cm          = results["confusion_matrix"].astype(float)
        class_names = results.get("class_names", self.class_names)
        n           = len(class_names)
        raw_cm      = cm.astype(int)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, cm / row_sums, 0.0)
        else:
            cm_display = cm

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor(PALETTE["bg"])

        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1 if normalize else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells with raw counts + normalised %
        thresh = cm_display.max() / 2.0
        for i in range(n):
            for j in range(n):
                norm_val = f"{cm_display[i, j]:.2f}" if normalize else ""
                raw_val  = str(raw_cm[i, j])
                cell_txt = f"{raw_val}\n({norm_val})" if normalize else raw_val
                ax.text(
                    j, i, cell_txt,
                    ha="center", va="center",
                    fontsize=11,
                    color="white" if cm_display[i, j] > thresh else "black",
                )

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — Phase A (n={raw_cm.sum()})",
            fontsize=13, fontweight="bold",
        )
        ax.set_facecolor(PALETTE["bg"])
        plt.tight_layout()

        out = self._save(fig, filename)
        logger.info("Confusion matrix saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 3. Per-class F1 bar chart
    # ------------------------------------------------------------------

    def plot_class_f1_bars(
        self,
        results: Dict,
        filename: str = "class_f1_bars.png",
    ) -> Path:
        """
        Horizontal bar chart of per-class Precision / Recall / F1.
        Phase A threshold (0.90) shown as a reference line.
        """
        class_names = results.get("class_names", self.class_names)
        per_class   = results["per_class"]

        precision = [per_class[c]["precision"] for c in class_names]
        recall    = [per_class[c]["recall"]    for c in class_names]
        f1        = [per_class[c]["f1"]        for c in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax.bar(x - width, precision, width, label="Precision", color="#42A5F5", alpha=0.85)
        ax.bar(x,         recall,    width, label="Recall",    color="#FF7043", alpha=0.85)
        ax.bar(x + width, f1,        width, label="F1-Score",  color="#66BB6A", alpha=0.85)

        ax.axhline(0.90, color="red", linestyle="--", lw=1.5, label="Phase A target (0.90)")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Metrics — Phase A", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", lw=0.8)
        ax.set_facecolor(PALETTE["bg"])

        # Annotate F1 values
        for xi, fi in zip(x + width, f1):
            ax.text(xi, fi + 0.01, f"{fi:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        out = self._save(fig, filename)
        logger.info("Class F1 bars saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # 4. Phase B readiness dashboard (single summary figure)
    # ------------------------------------------------------------------

    def plot_phase_b_readiness(
        self,
        results: Dict,
        config: Optional[dict] = None,
        filename: str = "phase_b_readiness.png",
    ) -> Path:
        """
        Gauge-style summary showing how close Phase A is to the Phase B gate.
        """
        cfg = config or self.config
        thr = cfg.get("phase_a_thresholds", {})
        per_class = results["per_class"]

        metrics_info = [
            ("Leaf Blast F1",    per_class.get("Leaf Blast",  {}).get("f1", 0.0), thr.get("leaf_blast_f1",   0.90)),
            ("Brown Spot F1",    per_class.get("Brown Spot",  {}).get("f1", 0.0), thr.get("brown_spot_f1",   0.90)),
            ("Overall Accuracy", results.get("accuracy", 0.0),                    thr.get("overall_accuracy", 0.92)),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        fig.patch.set_facecolor(PALETTE["bg"])

        for ax, (label, value, target) in zip(axes, metrics_info):
            colour = "#4CAF50" if value >= target else "#F44336"
            ax.barh([0], [value], color=colour, height=0.4, alpha=0.85)
            ax.barh([0], [target], color="none", height=0.4,
                    edgecolor="gray", linestyle="--", linewidth=1.5)
            ax.set_xlim(0, 1.0)
            ax.set_yticks([])
            ax.set_xlabel("Score", fontsize=10)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.text(value + 0.01, 0, f"{value:.3f}", va="center", fontsize=11,
                    color=colour, fontweight="bold")
            ax.axvline(target, color="red", linestyle="--", lw=1.5,
                       label=f"Target {target:.2f}")
            ax.legend(fontsize=8)
            ax.set_facecolor(PALETTE["bg"])

        ready = results.get("phase_b_ready", False)
        status = "🌟 PHASE B READY" if ready else "⏳ Phase B Not Ready"
        colour = "#4CAF50" if ready else "#F44336"
        fig.suptitle(status, fontsize=14, fontweight="bold", color=colour)
        plt.tight_layout()

        out = self._save(fig, filename)
        logger.info("Phase B readiness plot saved: %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        out = self.save_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return out

    def save_all(self, history: Dict, results: Dict) -> List[Path]:
        """Convenience: save all standard plots in one call."""
        return [
            self.plot_training_curves(history),
            self.plot_confusion_matrix(results),
            self.plot_class_f1_bars(results),
            self.plot_phase_b_readiness(results),
        ]

