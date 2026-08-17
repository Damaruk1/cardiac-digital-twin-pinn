"""
demo_phase10_explainability.py
---------------------------------
Loads the trained CNN checkpoint and generates Grad-CAM explanations
for a few real test-set beats, showing which timesteps most drove
each prediction.

Run with:
    python -m scripts.demo_phase10_explainability

NOTE: requires Phase 8's training demo to have been run first
(checkpoints/cnn_best.pt must exist).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.beat_dataset import BeatDataset, INDEX_TO_AAMI_CLASS
from src.data.dataset_loader import ECGDatasetLoader
from src.data.split import stratified_split
from src.explainability.gradcam import GradCAM1D
from src.models.cnn import ECG1DCNN
from src.visualization.saliency_plot import plot_saliency_overlay
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
    splits = stratified_split(dataset, train_frac=config.train_frac, val_frac=config.val_frac)

    model = ECG1DCNN(
        in_channels=len(record.lead_names),
        window_length=dataset.window_length,
        num_classes=5,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Explain the LAST conv block -- most semantically meaningful features,
    # while still being localized enough in time to be interpretable.
    gradcam = GradCAM1D(model, target_layer=model.conv_block3)

    # Grab a handful of real test examples to explain
    num_examples = min(3, len(splits.test))
    logger.info(f"Generating Grad-CAM explanations for {num_examples} test beats...")

    for i in range(num_examples):
        signal_tensor, true_label = splits.test[i]
        input_batch = signal_tensor.unsqueeze(0)  # add batch dimension: (1, channels, length)
        input_batch.requires_grad_(True)

        cam, predicted_class = gradcam.generate(input_batch)

        true_class_name = INDEX_TO_AAMI_CLASS[true_label.item()]
        pred_class_name = INDEX_TO_AAMI_CLASS[predicted_class]

        # Plot only the first lead for clarity
        first_lead_signal = signal_tensor[0].numpy()
        title = f"Beat {i}: true={true_class_name}, predicted={pred_class_name}"
        fig = plot_saliency_overlay(first_lead_signal, cam, title=title)
        output_path = f"logs/phase10_gradcam_beat{i}.png"
        fig.savefig(output_path, dpi=150)
        logger.info(f"  Beat {i} ({true_class_name} -> {pred_class_name}): saved {output_path}")

    gradcam.remove_hooks()
    logger.info("Grad-CAM explanations complete.")


if __name__ == "__main__":
    main()
