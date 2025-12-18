import albumentations as A
import cv2


class DataAugmentor:
    """
    Image-only data augmentation.
    Does NOT touch disk.
    Does NOT extract features.
    """

    def __init__(self, increase_percent=40, output_size=(224, 224)):
        self.increase_percent = increase_percent
        self.output_size = output_size
        self.transform = self._create_transform()

    def _create_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_REFLECT),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),

            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=0.3
            ),

            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),

            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),

            A.Resize(
                height=self.output_size[0],
                width=self.output_size[1]
            ),
        ])

    def augment(self, image):
        return self.transform(image=image)["image"]
