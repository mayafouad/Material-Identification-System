import shutil
from pathlib import Path
import cv2

from data_augmentation import DataAugmentor
from utils import CLASSES


def test_augmentor():
    # Real dataset path (from your project structure)
    real_dataset_path = Path(__file__).resolve().parents[1] / "dataset"

    # Output path - save to project directory permanently
    output_path = Path(__file__).resolve().parents[1] / "augmented_dataset"

    assert real_dataset_path.exists(), "Dataset folder does NOT exist."

    print("\n[TEST] Initializing DataAugmentor...")
    print(f"Input:  {real_dataset_path}")
    print(f"Output: {output_path}")

    # Check if output already exists
    if output_path.exists():
        response = input("\n⚠️  Output directory already exists. Delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            print("Removing existing output directory...")
            shutil.rmtree(output_path)
        else:
            print("Aborted. Please remove the directory manually or choose a different location.")
            return

    augmentor = DataAugmentor(
        input_dir=real_dataset_path,
        output_dir=output_path,
        target_samples=1000
    )

    print("\n[TEST] Running augmentation...")
    augmentor.augment_dataset()

    print("\n[TEST] Verifying output...")
    for class_name in CLASSES:
        class_out = output_path / class_name
        assert class_out.exists(), f"Missing folder for {class_name}"

        imgs = list(class_out.glob("*.jpg"))

        # Count original vs augmented
        original_imgs = [img for img in imgs if '_aug_' not in img.name]
        augmented_imgs = [img for img in imgs if '_aug_' in img.name]

        assert len(imgs) >= 10, f"Class {class_name} has {len(imgs)} images (expected >= 10)"

        # smoke test: try loading 1 image
        img = cv2.imread(str(imgs[0]))
        assert img is not None, f"Cannot load output image: {imgs[0]}"

        print(
            f"  ✔ {class_name} OK — {len(imgs)} total ({len(original_imgs)} original + {len(augmented_imgs)} augmented)")

    print("\n[TEST PASSED] Augmentation pipeline works correctly!")
    print(f"\n✅ Augmented dataset saved to: {output_path}")
    print("\nNaming convention:")
    print("  - Original images: same filename (e.g., image001.jpg)")
    print("  - Augmented images: filename_aug_N.jpg (e.g., image001_aug_0.jpg)")


if __name__ == "__main__":
    test_augmentor()