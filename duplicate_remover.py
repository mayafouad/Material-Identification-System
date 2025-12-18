from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

DATASET_DIR = Path("dataset")   # root dataset folder
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
HASH_THRESHOLD = 0  # 0 = exact duplicates, increase (1–5) for near-duplicates

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']


def compute_phash(image_path):
    try:
        with Image.open(image_path).convert("RGB") as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"[WARN] Could not process {image_path}: {e}")
        return None


def count_images_per_class(dataset_dir):
    counts = {}
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if cls_dir.exists():
            counts[cls] = len([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            ])
        else:
            counts[cls] = 0
    return counts


def remove_duplicates_and_report(dataset_dir):
    hash_to_path = {}
    removed = []
    total_images = 0

    print(f"\n[INFO] Scanning dataset: {dataset_dir}\n")

    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total_images += 1
        phash = compute_phash(img_path)
        if phash is None:
            continue

        duplicate_found = False
        for existing_hash, kept_path in hash_to_path.items():
            if phash - existing_hash <= HASH_THRESHOLD:
                duplicate_found = True
                removed.append(img_path)
                try:
                    img_path.unlink()
                    print(f"[DUPLICATE REMOVED] {img_path}  (duplicate of {kept_path})")
                except Exception as e:
                    print(f"[ERROR] Could not delete {img_path}: {e}")
                break

        if not duplicate_found:
            hash_to_path[phash] = img_path

    final_counts = count_images_per_class(dataset_dir)

    print("\n======================================")
    print(" DUPLICATE REMOVAL SUMMARY")
    print("======================================")
    print(f"Total images scanned   : {total_images}")
    print(f"Duplicates removed     : {len(removed)}")
    print(f"Unique images retained : {total_images - len(removed)}\n")

    print("Final image count per class:")
    for cls, count in final_counts.items():
        print(f"  {cls:<10}: {count}")

    return final_counts


if __name__ == "__main__":
    remove_duplicates_and_report(DATASET_DIR)
