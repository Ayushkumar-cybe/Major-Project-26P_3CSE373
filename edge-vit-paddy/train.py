"""
train.py
--------
Main training entrypoint for Phase A paddy disease detection.

Usage:
    # Standard training
    python train.py --config config/config.yaml

    # Dry-run (1 epoch, small batch — sanity check without dataset)
    python train.py --config config/config.yaml --epochs 1 --dry-run

    # Resume from checkpoint
    python train.py --config config/config.yaml --resume checkpoints/epoch_010.pth

    # Override hyperparameters from CLI
    python train.py --config config/config.yaml --epochs 30 --lr 1e-4 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Hybrid ResNet-18 + ViT paddy disease detector"
    )
    p.add_argument("--config",     default="config/config.yaml", help="Path to config.yaml")
    p.add_argument("--epochs",     type=int,   default=None,  help="Override training epochs")
    p.add_argument("--lr",         type=float, default=None,  help="Override learning rate")
    p.add_argument("--batch-size", type=int,   default=None,  help="Override batch size")
    p.add_argument("--resume",     default=None, help="Path to checkpoint to resume from")
    p.add_argument("--dry-run",    action="store_true",
                   help="Run 1 epoch with synthetic data (no dataset needed)")
    p.add_argument("--no-eval",    action="store_true",
                   help="Skip final evaluation after training")
    p.add_argument("--device",     default=None,
                   help="Force device: cuda | mps | cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Dry-run DataLoader (no real dataset needed)
# ---------------------------------------------------------------------------

def _make_dry_run_loaders(config: dict, device: torch.device):
    """Return tiny synthetic train/val loaders for sanity-checking the pipeline."""
    from torch.utils.data import DataLoader, TensorDataset

    num_classes = config["num_classes"]
    img_size    = config["data"]["image_size"]
    bs          = config["training"]["batch_size"]

    n_train, n_val = 2 * bs, bs
    X_train = torch.randn(n_train, 3, img_size, img_size)
    y_train = torch.randint(0, num_classes, (n_train,))
    X_val   = torch.randn(n_val,   3, img_size, img_size)
    y_val   = torch.randint(0, num_classes, (n_val,))

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    # Attach class-weight helper so Trainer can auto-compute weights
    class_weights = [1.0] * num_classes
    train_ds.get_class_weights = lambda: class_weights

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs:
        config["training"]["epochs"]         = args.epochs
    if args.lr:
        config["training"]["learning_rate"]  = args.lr
    if args.batch_size:
        config["training"]["batch_size"]     = args.batch_size
    if args.dry_run:
        config["training"]["epochs"]         = 1

    # ── Logger ────────────────────────────────────────────────────────
    from src.utils.logger import get_logger_from_config
    logger = get_logger_from_config("train", config)

    logger.info("=" * 60)
    logger.info(" PHASE A TRAINING — %s", config["project"]["name"])
    logger.info("=" * 60)
    logger.info("Config : %s", args.config)
    logger.info("Classes: %s", config["classes"])
    logger.info("Phase  : %s", config["project"]["phase"])

    # ── Reproducibility ───────────────────────────────────────────────
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Seed   : %d", seed)

    # ── Device ────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device : %s", device)

    # ── DataLoaders ───────────────────────────────────────────────────
    if args.dry_run:
        logger.warning("DRY-RUN mode: using synthetic data. No dataset needed.")
        train_loader, val_loader = _make_dry_run_loaders(config, device)
    else:
        from src.data.dataloader import build_dataloaders
        train_loader, val_loader = build_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────
    from src.models.model_factory import (
        build_model, load_checkpoint, load_optimizer_state, print_model_summary
    )
    model = build_model(config, device=device)
    print_model_summary(model)

    # ── Resume from checkpoint ────────────────────────────────────────
    resume_epoch = 0
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model, meta = load_checkpoint(args.resume, config, device=device)

        from src.training.trainer import Trainer
        trainer = Trainer(model, train_loader, val_loader, config, device)
        resume_epoch = load_optimizer_state(
            args.resume, trainer.optimizer, trainer.scheduler, device
        )
        logger.info("Resuming from epoch %d", resume_epoch)
    else:
        from src.training.trainer import Trainer
        trainer = Trainer(model, train_loader, val_loader, config, device)

    # ── Train ─────────────────────────────────────────────────────────
    history = trainer.train(resume_epoch=resume_epoch)

    # ── Save training history ─────────────────────────────────────────
    import json
    history_path = Path(config["paths"].get("outputs_dir", "outputs")) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    logger.info("Training history saved: %s", history_path)

    # ── Final Evaluation ──────────────────────────────────────────────
    if not args.no_eval and not args.dry_run:
        logger.info("Running final evaluation on validation set...")

        from src.models.model_factory import load_checkpoint as lc
        best_model, _ = lc(
            config["checkpoint"]["best_model_path"], config, device=device
        )

        from src.training.validator import Validator
        validator = Validator(
            best_model, val_loader,
            torch.nn.CrossEntropyLoss(), device
        )
        all_labels, all_preds = validator.collect_predictions()

        from src.evaluation.metrics import evaluate, print_classification_report
        results = evaluate(all_labels, all_preds, config["classes"], config)
        print_classification_report(results)

        # Save plots
        from src.evaluation.visualizer import Visualizer
        viz = Visualizer(config, save_dir=config["paths"].get("plots_dir", "outputs/plots"))
        saved = viz.save_all(history, results)
        logger.info("Plots saved:")
        for p in saved:
            logger.info("  %s", p)

    elif args.dry_run:
        logger.info("Dry-run complete — pipeline verified end-to-end. ✅")

    logger.info("=" * 60)
    logger.info(" Training finished. Best val acc: %.4f", trainer.best_val_acc)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

