import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import mahotas


class FeatureExtractor:
    def __init__(self, img_size=(512, 384), use_hog=True, use_lbp=True, use_hist=True, use_haralick=False):
        self.img_size = img_size
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        self.use_hist = use_hist
        self.use_haralick = use_haralick

    def load_and_preprocess(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image from path: {path}")
        img = cv2.resize(img, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def extract_hog(self, gray):
        return hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys',
            feature_vector=True
        )

    def extract_lbp(self, gray):
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist

    def extract_color_hist(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_haralick(self, gray):
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        features = mahotas.features.haralick(gray).mean(axis=0)
        return features

    def extract_all(self, path):
        img, gray = self.load_and_preprocess(path)
        vectors = []
        if self.use_hog:
            vectors.append(self.extract_hog(gray))
        if self.use_lbp:
            vectors.append(self.extract_lbp(gray))
        if self.use_hist:
            vectors.append(self.extract_color_hist(img))
        if self.use_haralick:
            vectors.append(self.extract_haralick(gray))

        if len(vectors) == 0:
            raise ValueError("No feature extraction methods enabled. Enable at least one feature type.")

        final_vec = np.concatenate(vectors)
        final_vec = final_vec.astype(np.float32)
        final_vec /= (np.linalg.norm(final_vec) + 1e-6)
        return final_vec


