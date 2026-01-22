"""
Logging setup for DesktopWebcamHandtracker.

Configures logging to both console and file in %APPDATA%/AROverlay/logs/.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import LOG_FILENAME, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def get_log_directory() -> Path:
    """
    Get the log directory path in %APPDATA%/AROverlay/logs/.

    Returns:
        Path to the log directory, created if it doesn't exist.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        log_dir = Path(appdata) / "AROverlay" / "logs"
    else:
        # Fallback to user home directory
        log_dir = Path.home() / ".aroverlay" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    debug: bool = False,
    log_to_file: bool = True,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return the application logger.

    Args:
        debug: Enable debug-level logging if True.
        log_to_file: Write logs to file if True.
        log_filename: Override default log filename.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("DesktopWebcamHandtracker")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Log format
    detailed_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(simple_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = get_log_directory()
        filename = log_filename or LOG_FILENAME
        log_path = log_dir / filename

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_format)
        logger.addHandler(file_handler)

        logger.debug(f"Logging to file: {log_path}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a child logger with the given name.

    Args:
        name: Optional name for the child logger.

    Returns:
        Logger instance (child of main logger or main logger if no name).
    """
    base_logger = logging.getLogger("DesktopWebcamHandtracker")
    if name:
        return base_logger.getChild(name)
    return base_logger
