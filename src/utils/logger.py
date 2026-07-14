"""
logger.py
---------
Centralized logging setup for the Cardiac Digital Twin project.

Every module in this project imports `get_logger()` instead of using
`print()`. This gives us:
    1. Timestamps on every message.
    2. Severity levels (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    3. Simultaneous output to console AND a persistent log file.
    4. One consistent format across the entire codebase.
"""

import logging
import os
from pathlib import Path


def get_logger(
    name: str,
    log_dir: str = "logs",
    log_filename: str = "project.log",
    level: str = "INFO",
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    Args:
        name: Usually pass __name__ from the calling module, so log
              messages show exactly which file they came from.
        log_dir: Directory where the log file will be written.
        log_filename: Name of the log file.
        level: Minimum severity level to record (e.g. "INFO").
        log_to_file: If True, also write logs to disk, not just console.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers if this function is
    # called multiple times for the same logger name (common in
    # notebooks or repeated imports).
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler: always on, so you see logs live ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler: optional, persists logs across runs ---
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
