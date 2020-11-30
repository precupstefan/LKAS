from abc import ABC

import cv2
import numpy as np


class ImageUtil(ABC):

    @staticmethod
    def apply_threshold(frame, threshold=128):
        """
        Replace each pixel in an image with a black pixel if the image intensity is less than some fixed constant T,
        or a white pixel if the image intensity is greater than that constant
        """
        ret, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        return thresh

    @staticmethod
    def convert_to_grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_lines_using_hough_transform(frame, ):
        # TODO: Tune this parameters for better line detection
        return cv2.HoughLinesP(frame, rho=2, theta=np.pi / 180, threshold=40, minLineLength=25, maxLineGap=70)

    @staticmethod
    def apply_gaussian_blur(frame, kernel_size=(5, 5), deviation=0):
        return cv2.GaussianBlur(frame, kernel_size, deviation)

    @staticmethod
    def apply_canny(frame, low_threshold=50, high_threshold=150):
        return cv2.Canny(frame, low_threshold, high_threshold)

    @staticmethod
    def fill_poly(frame, polygon, intensity=255):
        return cv2.fillPoly(frame, polygon, intensity)

    @staticmethod
    def apply_bitwise_and(img1, img2):
        return cv2.bitwise_and(img1, img2)
