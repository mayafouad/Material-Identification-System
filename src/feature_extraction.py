import cv2
import numpy as np
from rembg import remove
from skimage.feature import local_binary_pattern, hog

class FeatureExtractor:
    def __init__(self,
                 hist_bins=(8, 8, 8),
                 hist_ranges=[0, 180, 0, 256, 0, 256],
                 lbp_points=8,
                 lbp_radius=1,
                 hog_orientations=9,
                 hog_pixels_per_cell=(8, 8),
                 hog_cells_per_block=(2, 2)):

        self.hist_bins = hist_bins
        self.hist_ranges = hist_ranges
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

    def remove_background(self, image):
        image = remove(image)
        foreground = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
        return foreground

    def extract_color_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None,
                            self.hist_bins, self.hist_ranges)
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_lbp(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method="uniform")
        (lbphist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 2), 
                                    range=(0, self.lbp_points + 1))
        lbphist = lbphist.astype("float")
        lbphist /= (lbphist.sum() + 1e-6)
        return lbphist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features, _ = hog(gray, orientations=self.hog_orientations,
                              pixels_per_cell=self.hog_pixels_per_cell,
                              cells_per_block=self.hog_cells_per_block,
                              block_norm="L2-Hys", visualize=True)
        return hog_features

    def extract_features(self, image):
        foreground = self.remove_background(image)
        color_features = self.extract_color_histogram(foreground)
        texture_features = self.extract_lbp(image)
        hog_features = self.extract_hog(image)
        final_features = np.hstack([color_features, texture_features, hog_features])
        return final_features
