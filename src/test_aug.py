import shutil
from pathlib import Path

from data_augmentation import DataAugmentor
from src.utils import CLASSES


def test_augmentor():
    dataset_path = Path(__file__).resolve().parents[1] / "dataset"
    output_path = Path(__file__).resolve().parents[1] / "augmented_dataset"

    assert dataset_path.exists(), "Dataset folder not found"

    print(f"\nInput:  {dataset_path}")
    print(f"Output: {output_path}")

    if output_path.exists():
        response = input("\nOutput exists. Delete and recreate? (yes/no): ")
        if response.lower() == 'yes':
            shutil.rmtree(output_path)
        else:
            print("Aborted.")
            return

    augmentor = DataAugmentor(
        input_dir=dataset_path,
        output_dir=output_path,
        increase_percent=40
    )

    augmentor.augment_dataset()

    print("\nVerifying output...")
    for class_name in CLASSES:
        class_out = output_path / class_name
        if not class_out.exists():
            print(f"  ⚠ {class_name}: no folder")
            continue

        imgs = list(class_out.glob("*.jpg")) + list(class_out.glob("*.png"))
        original = [i for i in imgs if '_aug_' not in i.name]
        augmented = [i for i in imgs if '_aug_' in i.name]

        print(f"  ✔ {class_name}: {len(original)} original + {len(augmented)} augmented = {len(imgs)} total")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    test_augmentor()
