"""
demo_phase11_anatomical_mapping.py
--------------------------------------
Connects a real model prediction to the anatomical heart mesh:
    1. Loads the trained CNN and a real test beat.
    2. Determines which input LEAD contributed most to the prediction
       (via input gradient magnitude per channel).
    3. Maps that dominant lead to its anatomical wall region(s).
    4. Visualizes the heart mesh with those regions highlighted.

Run with:
    python -m scripts.demo_phase11_anatomical_mapping

NOTE: requires Phase 8's training demo to have been run first.
"""

from pathlib import Path

import torch

from src.anatomy.heart_mesh import generate_heart_mesh
from src.anatomy.lead_mapping import get_regions_for_leads
from src.data.beat_dataset import BeatDataset, INDEX_TO_AAMI_CLASS
from src.data.dataset_loader import ECGDatasetLoader
from src.data.split import stratified_split
from src.models.cnn import ECG1DCNN
from src.visualization.heart_mesh_plot import plot_heart_mesh
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

    logger.info(f"Record leads: {record.lead_names}")
    all_regions = get_regions_for_leads(record.lead_names)
    logger.info(f"Anatomical regions covered by this record's leads: {all_regions}")

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

    # --- Pick one test beat and find which LEAD drove its prediction ---
    signal_tensor, true_label = splits.test[0]
    input_batch = signal_tensor.unsqueeze(0)
    input_batch.requires_grad_(True)

    logits = model(input_batch)
    predicted_class = logits.argmax(dim=1).item()
    logits[0, predicted_class].backward()

    # Sum absolute gradient magnitude per channel (per lead) -- the lead
    # with the largest total gradient contributed most to this prediction.
    per_lead_importance = input_batch.grad.abs().sum(dim=2).squeeze(0)  # shape: (n_leads,)
    dominant_lead_idx = per_lead_importance.argmax().item()
    dominant_lead_name = record.lead_names[dominant_lead_idx]

    true_class_name = INDEX_TO_AAMI_CLASS[true_label.item()]
    pred_class_name = INDEX_TO_AAMI_CLASS[predicted_class]

    logger.info(f"Beat: true={true_class_name}, predicted={pred_class_name}")
    logger.info(f"Per-lead importance: "
                f"{dict(zip(record.lead_names, per_lead_importance.tolist()))}")
    logger.info(f"Dominant lead: {dominant_lead_name}")

    dominant_regions = get_regions_for_leads([dominant_lead_name])
    logger.info(f"Anatomically implicated region(s): {dominant_regions}")

    # --- Build and visualize the heart mesh ---
    mesh = generate_heart_mesh()
    fig = plot_heart_mesh(
        mesh,
        highlighted_regions=dominant_regions,
        title=f"Beat prediction ({true_class_name}->{pred_class_name}): "
              f"dominant lead {dominant_lead_name} -> {dominant_regions}",
    )
    output_path = "logs/phase11_anatomical_mapping.png"
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved {output_path}")


if __name__ == "__main__":
    main()
