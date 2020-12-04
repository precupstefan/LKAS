import numpy as np

from LKAS.config import config


class Line:
    is_present = False
    is_continuous = False
    line_fit = None
    line_fit_m = None
    curvature_radius = 0
    orientation = 0

    def set_parameters(self, line_fit, line_fit_m):
        self.line_fit = line_fit
        self.line_fit_m = line_fit_m
        self.calculate_curvature_radius()

    def calculate_curvature_radius(self):
        """
        left_fit and right_fit are assumed to have already been converted to meters
        """

        # meters per pixel in y dimension
        ym_per_pix = config["video"]["y_meters_per_pixel"]
        frame_height = config["video"]["size"][1]

        # y_eval is where we want to evaluate the fits for the line radius calcuation
        # for us it's at the bottom of the image for us, and because we know
        # the size of our video/images we can just hardcode it
        y_eval = frame_height * ym_per_pix
        fit = self.line_fit_m

        # https://stackoverflow.com/a/40021903
        if fit.size != 0:
            curve_rad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        else:
            curve_rad = None
        self.curvature_radius = curve_rad

    def get_x_given_y(self, y):
        if self.line_fit_m.size ==0:
            return None
        return self.line_fit_m[0] * y ** 2 + self.line_fit_m[1] * y + self.line_fit_m[2]

    def get_orientation(self):
        if self.line_fit_m.size == 0:
            return None
        y1 = 0
        x1 = self.get_x_given_y(y1)
        y2 = config["video"]["size"][1] - 1
        x2 = self.get_x_given_y(y2)
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        return slope

    def get_angle(self, x):
        # see https://www.youtube.com/watch?v=uVLWZCPwTh8 for more info
        m = self.get_slope(x)
        return np.arctan(m) if m else None

    def get_slope(self, x):
        if self.line_fit_m.size == 0:
            return None

        f = lambda y: self.line_fit[0] * y ** 2 + self.line_fit[1] * y + self.line_fit[2]
        y1 = 240
        x1 = f(y1)
        x = x1
        return 2 * self.line_fit_m[0] * x + self.line_fit_m[1]
