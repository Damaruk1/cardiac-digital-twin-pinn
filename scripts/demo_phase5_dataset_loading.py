"""
demo_phase5_dataset_loading.py
--------------------------------
Downloads (if needed) and loads one real record from the MIT-BIH
Arrhythmia Database, then reports its beat class distribution --
making the class imbalance problem visible before we ever train
a model on it.

Run with:
    python -m scripts.demo_phase5_dataset_loading

NOTE: requires internet access to physionet.org. If that's blocked
(e.g. in a restricted sandbox), this will log a clear error rather
than crash silently.
"""

from src.data.dataset_loader import ECGDatasetLoader
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def main() -> None:
    config = load_config("configs/config.yaml")
    logger = get_logger(
        name=__name__,
        log_dir=str(config.logs),
        log_filename=config.log_filename,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    loader = ECGDatasetLoader(data_dir=config.data_raw, database=config.database)
    record_name = config.sample_records[0]

    try:
        logger.info(f"Downloading record '{record_name}' from {config.database}...")
        loader.download_record(record_name)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(
            f"Could not download from PhysioNet (network issue?): {e}. "
            "This step requires internet access to physionet.org."
        )
        return

    record = loader.load_record(record_name)
    logger.info(
        f"Loaded record '{record.record_name}': "
        f"{record.duration_sec:.1f}s, {record.sampling_rate}Hz, "
        f"leads={record.lead_names}"
    )

    annotations = loader.load_annotations(record_name)
    logger.info(f"Total annotated beats: {len(annotations.sample_indices)}")

    distribution = loader.get_class_distribution(annotations)
    logger.info("AAMI class distribution:")
    total = sum(distribution.values())
    for aami_class, count in sorted(distribution.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        logger.info(f"  {aami_class}: {count:6d} beats ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
