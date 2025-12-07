from pathlib import Path
import albumentations as A

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
        self.transform = A.compose([])
        self.light_transform = A.compose([])
        self.class_counts = get_class_counts(CLASSES)

    def augment_image(self, image, light_transform=False):
        pass

    def augment_dataset(self):
        pass
