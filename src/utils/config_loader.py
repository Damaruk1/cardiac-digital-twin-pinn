"""config_loader.py — Loads and validates YAML config."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

class ConfigError(Exception):
    pass

@dataclass
class ProjectConfig:
    name: str
    version: str
    phase: int
    data_raw: Path
    data_processed: Path
    logs: Path
    log_level: str
    log_to_file: bool
    log_filename: str
    sampling_rate_hz: int
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    powerline_freq_hz: float
    max_heart_rate_bpm: int
    database: str
    sample_records: list
    native_sampling_rate_hz: int

def load_config(config_path="configs/config.yaml") -> ProjectConfig:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found at: {path.resolve()}")
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    try:
        return ProjectConfig(
            name=raw["project"]["name"], version=raw["project"]["version"], phase=raw["project"]["phase"],
            data_raw=Path(raw["paths"]["data_raw"]), data_processed=Path(raw["paths"]["data_processed"]), logs=Path(raw["paths"]["logs"]),
            log_level=raw["logging"]["level"], log_to_file=raw["logging"]["log_to_file"], log_filename=raw["logging"]["log_filename"],
            sampling_rate_hz=raw["signal_processing"]["sampling_rate_hz"], bandpass_low_hz=raw["signal_processing"]["bandpass_low_hz"],
            bandpass_high_hz=raw["signal_processing"]["bandpass_high_hz"], filter_order=raw["signal_processing"]["filter_order"],
            powerline_freq_hz=raw["signal_processing"]["powerline_freq_hz"], max_heart_rate_bpm=raw["signal_processing"]["max_heart_rate_bpm"],
            database=raw["dataset"]["database"], sample_records=raw["dataset"]["sample_records"],
            native_sampling_rate_hz=raw["dataset"]["native_sampling_rate_hz"],
        )
    except KeyError as missing_key:
        raise ConfigError(f"Missing required config key: {missing_key} in {path}") from missing_key
