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
        self.class_counts = get_class_counts(CLASSES)

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
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add sensor noise
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
        pass

    def augment_dataset(self):
        pass
