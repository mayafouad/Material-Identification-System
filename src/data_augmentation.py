from pathlib import Path
import albumentations as A
import cv2

from utils import (
    CLASSES,
    get_class_counts,
    load_image,
)


class DataAugmentor:
    def __init__(self, input_dir, output_dir, target_samples=500):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_samples = target_samples
        self.transform = self._generate_heavy_transform()
        self.light_transform = self._generate_light_transform()
        self.class_counts = get_class_counts(self.input_dir)

    def _generate_heavy_transform(self):
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

    def _generate_light_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # Mirror horizontally
            A.Rotate(limit=15, p=0.5),  # Small rotation
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),  # Minor lighting changes
        ])

    def augment_image(self, image, light_transform=False):
        transform = self.light_transform if light_transform else self.transform
        augmented = transform(image=image)
        return augmented["image"]

    def augment_dataset(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for class_name in CLASSES:
            class_input_path = self.input_dir / class_name
            class_output_path = self.output_dir / class_name
            class_output_path.mkdir(parents=True, exist_ok=True)

            images_paths = list(class_input_path.glob("*.jpg"))
            count = len(images_paths)

            print(f"Class: {class_name}, count: {count}")

            for image_path in images_paths:
                image = load_image(image_path, color_mode='rgb')
                if image is None:
                    continue
                cv2.imwrite(str(class_output_path / image_path.name), image)

            if count >= self.target_samples:
                continue

            i = 0
            needed = self.target_samples - count
            while i < needed:
                for image_path in images_paths:
                    if i >= needed:
                        break
                    image = load_image(image_path, color_mode='rgb')
                    if image is None:
                        continue
                    use_light = count > self.target_samples * 0.6
                    augmented_image = self.augment_image(image, light_transform=use_light)

                    new_name = f"{image_path.stem}_aug_{i}.jpg"

                    cv2.imwrite(str(class_output_path / new_name), augmented_image)
                    i += 1
            print(f"→ Done augmenting {class_name}")
