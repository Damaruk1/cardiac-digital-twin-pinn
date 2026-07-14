"""
main.py
-------
Phase 1 entry point.

This script does not do any ECG/ML work yet — its only job is to
prove that the project scaffolding works:
    1. Config loads correctly from configs/config.yaml.
    2. Logger writes to both console and logs/project.log.
    3. Required data folders exist.

Run with:
    python -m src.main
"""

from src.utils.config_loader import load_config, ConfigError
from src.utils.logger import get_logger


def main() -> None:
    # Step 1: Load configuration
    try:
        config = load_config("configs/config.yaml")
    except ConfigError as e:
        # Config isn't loaded yet, so we can't use our logger's file
        # handler (it needs config.logs path) -- fall back to a bare
        # print for this one specific failure case.
        print(f"[FATAL] Could not start project: {e}")
        raise SystemExit(1)

    # Step 2: Initialize logger using settings FROM the config
    logger = get_logger(
        name=__name__,
        log_dir=str(config.logs),
        log_filename=config.log_filename,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    # Step 3: Sanity-check the setup
    logger.info("=" * 60)
    logger.info(f"Project: {config.name}  (v{config.version})")
    logger.info(f"Phase:   {config.phase}")
    logger.info("=" * 60)

    logger.debug("This DEBUG message will NOT show at INFO level.")
    logger.info("Config loaded successfully.")
    logger.info(f"Raw data path:       {config.data_raw.resolve()}")
    logger.info(f"Processed data path: {config.data_processed.resolve()}")

    for required_dir in [config.data_raw, config.data_processed, config.logs]:
        if required_dir.exists():
            logger.info(f"[OK] Directory exists: {required_dir}")
        else:
            logger.warning(f"[MISSING] Directory not found: {required_dir}")

    logger.info("Phase 1 setup verification complete.")


if __name__ == "__main__":
    main()
