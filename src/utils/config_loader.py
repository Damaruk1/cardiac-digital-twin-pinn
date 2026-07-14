"""
config_loader.py
-----------------
Loads and validates the project's YAML configuration file.

Why a class instead of a raw dict?
    - We get IDE autocompletion errors caught early if a key is missing.
    - We can add validation logic in one place as the config grows.
    - `Path` objects instead of raw strings prevent OS-specific bugs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when the config file is missing or malformed."""


@dataclass
class ProjectConfig:
    """Typed, structured representation of configs/config.yaml."""

    name: str
    version: str
    phase: int
    data_raw: Path
    data_processed: Path
    logs: Path
    log_level: str
    log_to_file: bool
    log_filename: str


def load_config(config_path: str = "configs/config.yaml") -> ProjectConfig:
    """
    Read the YAML config file and return a validated ProjectConfig.

    Args:
        config_path: Path to the YAML config file, relative to the
                      project root.

    Returns:
        A populated ProjectConfig dataclass instance.

    Raises:
        ConfigError: If the file is missing or required keys are absent.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found at: {path.resolve()}")

    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    try:
        return ProjectConfig(
            name=raw["project"]["name"],
            version=raw["project"]["version"],
            phase=raw["project"]["phase"],
            data_raw=Path(raw["paths"]["data_raw"]),
            data_processed=Path(raw["paths"]["data_processed"]),
            logs=Path(raw["paths"]["logs"]),
            log_level=raw["logging"]["level"],
            log_to_file=raw["logging"]["log_to_file"],
            log_filename=raw["logging"]["log_filename"],
        )
    except KeyError as missing_key:
        raise ConfigError(
            f"Missing required config key: {missing_key} in {path}"
        ) from missing_key
