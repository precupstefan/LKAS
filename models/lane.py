import numpy as np

from LKAS.config import config
from LKAS.models.line import Line


class Lane:

    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.middle_lane_x = None

    def set_middle_lane_x(self, middle_lane_x):
        self.middle_lane_x = middle_lane_x

    def get_off_center_of_lane(self):
        middle_point = self.get_middle_point_for_y(np.array([0]))
        xm_per_pix = config["video"]["x_meters_per_pixel"]
        frame_width = config["video"]["size"][0]
        pixel_deviation = frame_width / 2 - abs(middle_point)
        deviation = pixel_deviation * xm_per_pix
        return deviation[0] * -1

    def get_middle_point_for_y(self, y):
        """
        @param y: numpy array
        @return: np.array
        """
        left_x = self.left_line.get_fit_x(y)
        right_x = self.left_line.get_fit_x(y)
        return np.mean([left_x, right_x], axis=0)
