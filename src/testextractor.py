import os
import sys
from src.FeatureExtractor import FeatureExtractor
import numpy as np
from collections import defaultdict

def test_all_dataset():
    print("=" * 80)
    print("TESTING FEATURE EXTRACTOR ON ENTIRE DATASET")
    print("=" * 80)

    extractor = FeatureExtractor(img_size=(512, 384), use_hog=True, use_lbp=True, use_hist=True, use_haralick=False)

    dataset_path = "../dataset"
    categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset directory '{dataset_path}' not found!")
        return False

    print(f"\nDataset path: {dataset_path}\n")

    print("=" * 80)
    print("SAMPLE FEATURE EXTRACTION - ONE IMAGE PER CLASS")
    print("=" * 80)

    for category in categories:
        category_path = os.path.join(dataset_path, category)

        if not os.path.exists(category_path):
            continue

        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        if not images:
            continue

        sample_img = images[0]
        sample_path = os.path.join(category_path, sample_img)

        try:
            features = extractor.extract_all(sample_path)

            print(f"\n{category.upper()}")
            print(f"  Image: {sample_img}")
            print(f"  Feature vector shape: {features.shape}")
            print(f"  Feature stats:")
            print(f"    Min: {features.min():.6f}")
            print(f"    Max: {features.max():.6f}")
            print(f"    Mean: {features.mean():.6f}")
            print(f"    Std: {features.std():.6f}")
            print(f"    Norm: {np.linalg.norm(features):.6f}")
            print(f"  First 10 features: {features[:10]}")

        except Exception as e:
            print(f"\n{category.upper()}")
            print(f"  Image: {sample_img}")
            print(f"  ERROR: {str(e)}")

    print("\n" + "=" * 80)
    print("TESTING ALL IMAGES")
    print("=" * 80 + "\n")

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'categories': defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0, 'failed_files': []})
    }


    for category in categories:
        category_path = os.path.join(dataset_path, category)

        if not os.path.exists(category_path):
            print(f"WARNING: Category '{category}' not found, skipping...")
            continue

        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        if not images:
            print(f"WARNING: No images in '{category}', skipping...")
            continue

        print(f"Testing {category.upper()}: {len(images)} images")
        stats['categories'][category]['total'] = len(images)

        success_count = 0
        for i, img_name in enumerate(images, 1):
            img_path = os.path.join(category_path, img_name)
            stats['total'] += 1

            try:
                features = extractor.extract_all(img_path)

                if features is None or len(features) == 0:
                    raise ValueError("Empty feature vector")

                if np.isnan(features).any():
                    raise ValueError("NaN values in features")

                if np.isinf(features).any():
                    raise ValueError("Inf values in features")

                stats['success'] += 1
                stats['categories'][category]['success'] += 1
                success_count += 1

                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(images)} images...")

            except Exception as e:
                stats['failed'] += 1
                stats['categories'][category]['failed'] += 1
                stats['categories'][category]['failed_files'].append((img_name, str(e)))
                print(f"  FAILED [{i}/{len(images)}]: {img_name[:50]} - {str(e)[:60]}")

        success_rate = (success_count / len(images) * 100) if len(images) > 0 else 0
        print(f"  Result: {success_count}/{len(images)} ({success_rate:.1f}%)\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal images: {stats['total']}")
    print(f"Successful: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.2f}%)")

    print(f"\nBy Category:")
    for category in categories:
        if category in stats['categories']:
            cat = stats['categories'][category]
            rate = (cat['success'] / cat['total'] * 100) if cat['total'] > 0 else 0
            status = "PASS" if cat['failed'] == 0 else "WARN"
            print(f"  [{status}] {category:10s}: {cat['success']:4d}/{cat['total']:4d} ({rate:5.1f}%)")

    if stats['failed'] > 0:
        print(f"\n{stats['failed']} images failed - see details above")

        save_report = input("\nSave failed images report? (y/n): ").lower().strip()
        if save_report == 'y':
            with open('failed_images_report.txt', 'w') as f:
                f.write(f"Failed Images Report\n")
                f.write(f"Total Failed: {stats['failed']}\n\n")
                for category in categories:
                    if category in stats['categories']:
                        cat = stats['categories'][category]
                        if cat['failed_files']:
                            f.write(f"\n{category.upper()}:\n")
                            for fname, error in cat['failed_files']:
                                f.write(f"  {fname} - {error}\n")
            print("Report saved to: failed_images_report.txt")
    else:
        print("\nALL IMAGES PROCESSED SUCCESSFULLY!")

    print("=" * 80)

    return stats['failed'] == 0


if __name__ == "__main__":
    try:
        success = test_all_dataset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

