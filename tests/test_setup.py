"""
test_setup.py
-------------
Phase 1 tests: verify the config loader and logger behave correctly.

Run with:
    pytest tests/test_setup.py -v
"""

import logging

import pytest

from src.utils.config_loader import ConfigError, load_config
from src.utils.logger import get_logger


def test_config_loads_successfully():
    """The real config.yaml should load without raising."""
    config = load_config("configs/config.yaml")
    assert config.name == "Cardiac Digital Twin (PINN)"
    assert config.phase >= 1  # bumps up each phase, so just check it's valid


def test_config_missing_file_raises_error():
    """Loading a non-existent config file should raise ConfigError."""
    with pytest.raises(ConfigError):
        load_config("configs/does_not_exist.yaml")


def test_logger_returns_logger_instance():
    """get_logger should return a standard logging.Logger."""
    logger = get_logger("test_logger", log_to_file=False)
    assert isinstance(logger, logging.Logger)


def test_logger_no_duplicate_handlers():
    """Calling get_logger twice with the same name must not duplicate handlers."""
    logger1 = get_logger("dup_test", log_to_file=False)
    logger2 = get_logger("dup_test", log_to_file=False)
    assert logger1 is logger2
    assert len(logger1.handlers) == 1