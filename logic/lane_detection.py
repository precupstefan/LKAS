import cv2
import numpy as np

from config import config
from utils.image_util import ImageUtil


class LaneDetection:
    # https://www.youtube.com/watch?v=eLTLtUVuuy4
    def __init__(self):
        pass

    def detect_lanes(self, frame):
        """
        DETECTS LANES
        :param frame: GRAYSCALE FRAME
        :return:
        """
        frame = self.get_edges_in_frame(frame)
        cv2.imshow('testing_lane', frame)
        frame = self.get_region_of_interest(frame)
        lanes = ImageUtil.get_lines_using_hough_transform(frame)
        if lanes is None:
            return np.array([], [])
        smoothed_lanes = self.get_smoothed_lanes(frame, lanes)
        return smoothed_lanes, lanes

    def get_edges_in_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, tuple(config["LKAS"]["lanes_detection"]["gaussian_blur"]["kernel_size"]),
                                 config["LKAS"]["lanes_detection"]["gaussian_blur"]["deviation"])
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                      config["LKAS"]["lanes_detection"]["threshold"]["block_size"],
                                      config["LKAS"]["lanes_detection"]["threshold"]["constant"])
        frame = cv2.Canny(frame, config["LKAS"]["lanes_detection"]["canny"]["low_threshold"],
                          config["LKAS"]["lanes_detection"]["canny"]["high_threshold"])
        return frame

    def get_region_of_interest(self, frame):
        polygons = np.array([
            config["LKAS"]["lanes_detection"]["region_of_interest"]
        ])
        mask = np.zeros_like(frame)
        masked_image = ImageUtil.fill_poly(mask, polygons)
        return ImageUtil.apply_bitwise_and(frame, masked_image)

    def get_smoothed_lanes(self, frame, lines):
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        lanes = []

        if len(left_fit) is not 0:
            left_lane = self.make_line_coordinates(frame, left_fit_average)
            lanes.append(left_lane)

        if len(right_fit) is not 0:
            right_lane = self.make_line_coordinates(frame, right_fit_average)
            lanes.append(right_lane)

        return np.array(lanes)

    def make_line_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * 3 / 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
