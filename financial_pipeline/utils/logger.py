"""Centralized structured logging configuration."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=fmt,
        datefmt=datefmt,
        force=True,
    )
    # Quiet noisy third-party loggers
    for noisy in ("yfinance", "urllib3", "requests", "peewee", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
