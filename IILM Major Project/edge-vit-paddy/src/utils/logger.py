"""
src/utils/logger.py
-------------------
Centralized logging setup for the project.

Usage:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console (and optionally file) handlers.

    Parameters
    ----------
    name : str
        Logger name (use ``__name__`` from the calling module).
    level : str
        Logging level: DEBUG | INFO | WARNING | ERROR.
    log_file : str, optional
        If provided, also writes to this file path.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # Already configured

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if _HAS_COLORLOG:
        color_fmt = (
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | "
            "%(name)s | %(message)s"
        )
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                color_fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    return logger


def get_logger_from_config(name: str, config: dict) -> logging.Logger:
    """Build a logger using settings from config.yaml."""
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO")
    log_file = None
    if log_cfg.get("log_to_file", False):
        log_file = log_cfg.get("log_file", "outputs/logs/run.log")
    return setup_logger(name, level=level, log_file=log_file)

