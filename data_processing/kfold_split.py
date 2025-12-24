import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import defaultdict

# Paths
BASE_PATH = Path("/kaggle/working/data/datav2")
OUTPUT_PATH = Path("/kaggle/working/data/datav2-folded")
TRAIN_IMAGES = BASE_PATH / "train/images"
TRAIN_LABELS = BASE_PATH / "train/labels"
VAL_IMAGES = BASE_PATH / "valid/images"
VAL_LABELS = BASE_PATH / "valid/labels"
TEST_IMAGES = BASE_PATH / "test/images"
TEST_LABELS = BASE_PATH / "test/labels"

# Class names
CLASS_NAMES = [
    "dentigeroz kist",
    "keratokist",
    "radikuler kist",
    "ameloblastoma",
    "odontoma",
]


def get_image_class(label_path):
    """Extract the primary class from a YOLO label file"""
    if not label_path.exists():
        return -1

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return -1

    # Get the first object's class (for stratification)
    first_line = lines[0].strip()
    if first_line:
        class_id = int(first_line.split()[0])
        return class_id
    return -1


def collect_dataset_info():
    """Combine TRAIN and VAL sets and collect their information"""
    print("Collecting dataset information...")

    combined_data = []

    # Collect TRAIN images
    for img_path in TRAIN_IMAGES.glob("*.*"):
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        class_id = get_image_class(label_path)
        combined_data.append(
            {
                "image": img_path,
                "label": label_path,
                "class": class_id,
                "source": "train",
            }
        )

    # Collect VAL images
    for img_path in VAL_IMAGES.glob("*.*"):
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        class_id = get_image_class(label_path)
        combined_data.append(
            {"image": img_path, "label": label_path, "class": class_id, "source": "val"}
        )

    print(f"Total images collected: {len(combined_data)}")

    # Print class distribution
    class_counts = defaultdict(int)
    for item in combined_data:
        class_counts[item["class"]] += 1

    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        if class_id >= 0:
            print(f"  {CLASS_NAMES[class_id]}: {count}")

    return combined_data


def create_fold_structure(fold_idx):
    """Create directory structure for a specific fold"""
    fold_path = OUTPUT_PATH / f"fold_{fold_idx}"

    # Create directories
    (fold_path / "train/images").mkdir(parents=True, exist_ok=True)
    (fold_path / "train/labels").mkdir(parents=True, exist_ok=True)
    (fold_path / "val/images").mkdir(parents=True, exist_ok=True)
    (fold_path / "val/labels").mkdir(parents=True, exist_ok=True)

    return fold_path


def create_data_yaml(fold_path, fold_idx):
    """Create data.yaml file for a specific fold"""
    data_yaml = {
        "path": str(fold_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": str((BASE_PATH / "test/images").absolute()),
        "nc": 5,
        "names": CLASS_NAMES,
    }

    yaml_path = fold_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"  Created: {yaml_path}")
    return yaml_path


def copy_files(data_subset, dest_folder):
    """Copy images and labels to destination folder"""
    images_dest = dest_folder / "images"
    labels_dest = dest_folder / "labels"

    for item in data_subset:
        # Copy image
        img_dest = images_dest / item["image"].name
        shutil.copy2(item["image"], img_dest)

        # Copy label if exists
        if item["label"].exists():
            lbl_dest = labels_dest / item["label"].name
            shutil.copy2(item["label"], lbl_dest)


def create_5fold_splits():
    """Main function to create 5-fold cross-validation splits"""
    print("=" * 60)
    print("Creating 5-Fold Cross-Validation Dataset")
    print("=" * 60)

    # Collect combined dataset
    combined_data = collect_dataset_info()

    # Prepare data for stratified split
    X = np.arange(len(combined_data))
    y = np.array([item["class"] for item in combined_data])

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Creating folds...")
    print(f"{'=' * 60}\n")

    fold_stats = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Processing Fold {fold_idx}...")

        # Create fold structure
        fold_path = create_fold_structure(fold_idx)

        # Split data
        train_data = [combined_data[i] for i in train_idx]
        val_data = [combined_data[i] for i in val_idx]

        # Copy files
        print(f"  Copying {len(train_data)} training images...")
        copy_files(train_data, fold_path / "train")

        print(f"  Copying {len(val_data)} validation images...")
        copy_files(val_data, fold_path / "val")

        # Create data.yaml
        create_data_yaml(fold_path, fold_idx)

        # Calculate statistics
        train_classes = defaultdict(int)
        val_classes = defaultdict(int)

        for item in train_data:
            if item["class"] >= 0:
                train_classes[item["class"]] += 1

        for item in val_data:
            if item["class"] >= 0:
                val_classes[item["class"]] += 1

        fold_stats.append(
            {
                "fold": fold_idx,
                "train_count": len(train_data),
                "val_count": len(val_data),
                "train_classes": dict(train_classes),
                "val_classes": dict(val_classes),
            }
        )

        print(f"  ✓ Fold {fold_idx} completed\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    for stats in fold_stats:
        print(f"Fold {stats['fold']}:")
        print(f"  Training:   {stats['train_count']} images")
        print(f"  Validation: {stats['val_count']} images")
        print(f"  Train class distribution:")
        for class_id, count in sorted(stats["train_classes"].items()):
            print(
                f"    {CLASS_NAMES[class_id]}: {count} ({count/stats['train_count']*100:.1f}%)"
            )
        print(f"  Val class distribution:")
        for class_id, count in sorted(stats["val_classes"].items()):
            print(
                f"    {CLASS_NAMES[class_id]}: {count} ({count/stats['val_count']*100:.1f}%)"
            )
        print()

    print(f"✓ All folds created successfully at: {OUTPUT_PATH}")
    print(f"✓ TEST set remains at: {BASE_PATH / 'test'}")

    return fold_stats


if __name__ == "__main__":
    fold_stats = create_5fold_splits()
