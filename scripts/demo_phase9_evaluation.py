"""
demo_phase9_evaluation.py
----------------------------
Loads the trained CNN checkpoint from Phase 8, evaluates it on the
held-out TEST split (never touched during training), and reports
per-class precision/recall/F1 plus a confusion matrix.

Also computes a "dummy baseline" that always predicts Normal, to make
the accuracy-is-misleading point concrete with real numbers side by side.

Run with:
    python -m scripts.demo_phase9_evaluation

NOTE: requires Phase 8's training demo to have been run first
(checkpoints/cnn_best.pt must exist).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.beat_dataset import BeatDataset, INDEX_TO_AAMI_CLASS
from src.data.dataset_loader import ECGDatasetLoader
from src.data.split import stratified_split
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import evaluate_predictions, format_report
from src.models.cnn import ECG1DCNN
from src.visualization.confusion_matrix_plot import plot_confusion_matrix
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

CLASS_NAMES = [INDEX_TO_AAMI_CLASS[i] for i in range(5)]  # ["N","S","V","F","Q"]


def main() -> None:
    config = load_config("configs/config.yaml")
    logger = get_logger(
        name=__name__,
        log_dir=str(config.logs),
        log_filename=config.log_filename,
        level=config.log_level,
        log_to_file=config.log_to_file,
    )

    checkpoint_path = Path(config.checkpoint_dir) / "cnn_best.pt"
    if not checkpoint_path.exists():
        logger.error(
            f"No checkpoint found at {checkpoint_path}. "
            "Run 'python -m scripts.demo_phase8_training' first."
        )
        return

    loader = ECGDatasetLoader(data_dir=config.data_raw, database=config.database)
    record_name = config.sample_records[0]
    record = loader.load_record(record_name)
    annotations = loader.load_annotations(record_name)

    dataset = BeatDataset(
        record, annotations,
        window_before=config.window_before,
        window_after=config.window_after,
    )

    # IMPORTANT: same split fractions and random_seed as Phase 8, so we
    # reconstruct the EXACT same test set the model never trained on.
    splits = stratified_split(dataset, train_frac=config.train_frac, val_frac=config.val_frac)
    test_loader = DataLoader(splits.test, batch_size=config.batch_size, shuffle=False)
    logger.info(f"Evaluating on held-out test set: {len(splits.test)} beats")

    # --- Load the trained model ---
    model = ECG1DCNN(
        in_channels=len(record.lead_names),
        window_length=dataset.window_length,
        num_classes=5,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    evaluator = Evaluator(model, class_names=CLASS_NAMES)
    y_true, y_pred = evaluator.collect_predictions(test_loader)
    report = evaluate_predictions(y_true, y_pred, CLASS_NAMES)

    logger.info("=== Trained CNN -- Test Set Performance ===")
    logger.info("\n" + format_report(report))

    # --- Baseline: always predict the majority class (Normal, index 0) ---
    baseline_pred = [0] * len(y_true)
    baseline_report = evaluate_predictions(y_true, baseline_pred, CLASS_NAMES)

    logger.info("=== Baseline (always predicts 'N') -- Test Set Performance ===")
    logger.info("\n" + format_report(baseline_report))

    logger.info(
        f"\nComparison -- Accuracy: model={report.accuracy:.3f} vs "
        f"baseline={baseline_report.accuracy:.3f} "
        f"(often nearly identical!)"
    )
    logger.info(
        f"Comparison -- Macro F1: model={report.macro_f1:.3f} vs "
        f"baseline={baseline_report.macro_f1:.3f} "
        f"(THIS is where the real difference shows up)"
    )

    # --- Confusion matrix plot ---
    fig = plot_confusion_matrix(report.confusion, CLASS_NAMES, title="CNN - Test Set Confusion Matrix")
    fig.savefig("logs/phase9_confusion_matrix.png", dpi=150)
    logger.info("Saved logs/phase9_confusion_matrix.png")


if __name__ == "__main__":
    main()
