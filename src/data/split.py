"""
split.py
--------
Stratified train/validation/test splitting for a BeatDataset.

Ordinary random splitting risks leaving rare AAMI classes (S, V, F, Q)
entirely out of one split by chance, given how imbalanced real ECG
beat data is. Stratified splitting preserves each class's proportion
across all three splits.

IMPORTANT LIMITATION: stratification requires enough examples of every
class to appear in all three splits. A single MIT-BIH record can have
as few as 1 example of a rare class (e.g. only 1 'V' beat) -- in that
case, true stratification is mathematically impossible for that class.
We detect this and fall back to a random (non-stratified) split with
a clear warning, rather than crashing. This limitation naturally goes
away once multiple records are combined (a future extension), since
rare classes accumulate more total examples across records.
"""

from dataclasses import dataclass
from collections import Counter

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from src.data.beat_dataset import BeatDataset


@dataclass
class DatasetSplits:
    """Container for the three stratified splits."""

    train: Subset
    val: Subset
    test: Subset


def _can_stratify(labels: list, min_per_class: int = 3) -> bool:
    """Every class needs at least `min_per_class` examples to guarantee
    at least 1 example lands in each of train/val/test."""
    counts = Counter(labels)
    return min(counts.values()) >= min_per_class


def stratified_split(
    dataset: BeatDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = 42,
    logger=None,
) -> DatasetSplits:
    """
    Splits a BeatDataset into train/val/test subsets, preserving the
    class distribution in each split where mathematically possible.

    Args:
        dataset: A BeatDataset with a populated `.labels` list.
        train_frac: Fraction of data for training.
        val_frac: Fraction of data for validation.
                  (test_frac is implicitly 1 - train_frac - val_frac)
        random_seed: For reproducibility.
        logger: Optional logger to record a warning if stratification
                had to be skipped due to insufficient class examples.

    Returns:
        DatasetSplits with three torch Subset objects.
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0"
        )

    all_indices = list(range(len(dataset)))
    labels = dataset.labels

    use_stratify = _can_stratify(labels)
    if not use_stratify and logger is not None:
        counts = Counter(labels)
        logger.warning(
            f"Cannot fully stratify: some class has only "
            f"{min(counts.values())} example(s), fewer than the 3 needed "
            f"for train/val/test. Falling back to a random split -- rare "
            f"classes may be missing from one or more splits. This "
            f"resolves itself once more records are combined."
        )

    stratify_arg = labels if use_stratify else None
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_frac,
        stratify=stratify_arg,
        random_state=random_seed,
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    use_stratify_2 = _can_stratify(train_val_labels, min_per_class=2)
    stratify_arg_2 = train_val_labels if use_stratify_2 else None

    relative_val_frac = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_frac,
        stratify=stratify_arg_2,
        random_state=random_seed,
    )

    return DatasetSplits(
        train=Subset(dataset, train_idx),
        val=Subset(dataset, val_idx),
        test=Subset(dataset, test_idx),
    )
