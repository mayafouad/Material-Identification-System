from pathlib import Path
import albumentations as A
import cv2
from tqdm import tqdm

from utils import CLASSES, load_image


class DataAugmentor:
    def __init__(self, input_dir, output_dir, increase_percent=40):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.increase_percent = increase_percent
        self.transform = self._create_transform()

    def _create_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # Mirror horizontally
            A.VerticalFlip(p=0.3),  # Mirror vertically
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),  # Rotate ±30° (camera tilt)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),  # Simulate lighting variations
            A.GaussNoise(
                std_range=(0.05, 0.2),  # 5-20% noise (≈13 to 51 for 8-bit images)
                mean_range=(0.0, 0.0),
                per_channel=True,
                p=0.3
            ),  # Add sensor noise
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Simulate motion/focus blur
            A.CLAHE(clip_limit=2.0, p=0.3),  # Enhance local contrast in dark regions
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),  # Vary color temperature (different light sources)
            A.RandomScale(scale_limit=0.2, p=0.3),  # Zoom in/out ±20%
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),  # Combined shift, scale, rotate (general robustness)
        ])

    def augment_image(self, image):
        return self.transform(image=image)["image"]

    def augment_dataset(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nAugmenting dataset (+{self.increase_percent}% per class)")
        print("=" * 50)

        for class_name in CLASSES:
            class_input = self.input_dir / class_name
            class_output = self.output_dir / class_name
            class_output.mkdir(parents=True, exist_ok=True)

            if not class_input.exists():
                print(f"[WARN] {class_name}: folder not found, skipping")
                continue

            image_paths = list(class_input.glob("*.jpg")) + list(class_input.glob("*.png"))
            
            # copy originals and count valid images
            valid_count = 0
            valid_paths = []
            
            for img_path in image_paths:
                img = load_image(img_path, color_mode='rgb')
                if img is None:
                    continue
                valid_count += 1
                valid_paths.append((img_path, img))
                cv2.imwrite(str(class_output / img_path.name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # calculate augmentations needed (40% of valid originals)
            needed = int(valid_count * self.increase_percent / 100)
            
            print(f"{class_name}: {valid_count} valid images → +{needed} augmented")

            if needed == 0:
                continue

            # generate augmentations
            aug_count = 0
            for img_path, img in tqdm(valid_paths * (needed // valid_count + 1), desc=f"  {class_name}", total=needed):
                if aug_count >= needed:
                    break
                aug_img = self.augment_image(img)
                aug_name = f"{img_path.stem}_aug_{aug_count}.jpg"
                cv2.imwrite(str(class_output / aug_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                aug_count += 1

        print("=" * 50)
        print("Done!")
