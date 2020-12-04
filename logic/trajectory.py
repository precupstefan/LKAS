import numpy as np

from LKAS.config import load_config
from LKAS.models.direction import Direction

config = load_config()


class Trajectory:
    def __init__(self):
        pass

    def compute_trajectory(self, lane):
        """
        Returns the curvature in degrees
        """
        left_line = lane.left_line
        right_line = lane.right_line
        left_curvature = left_line.curvature_radius
        right_curvature = right_line.curvature_radius

        if left_curvature is None and right_curvature is None:
            return None

        point_of_interest = 0
        if left_curvature is None and right_curvature is not None:
            avg_curvature_radius = right_curvature
            m_avg = right_line.get_slope_m(point_of_interest)

        if left_curvature is not None and right_curvature is None:
            avg_curvature_radius = left_curvature
            m_avg = left_line.get_slope_m(point_of_interest)

        if left_curvature is not None and right_curvature is not None:
            avg_curvature_radius = np.average([left_curvature, right_curvature])
            left_orientation = left_line.get_slope_m(point_of_interest)
            right_orientation = right_line.get_slope_m(point_of_interest)
            m_avg = (left_orientation + right_orientation) / 2

        off_center = lane.get_off_center_of_lane()

        print("curvature", avg_curvature_radius)
        print("deviation", off_center)

        real_curvature = avg_curvature_radius + off_center
        print("real curvature", real_curvature)

        degrees = np.degrees(real_curvature)

        print("degrees", degrees % 90)
        print("m_avg", m_avg)
        print("theta", np.arctan(m_avg), np.rad2deg(np.arctan(m_avg)))
        return np.rad2deg(np.arctan(m_avg))*-1
        if m_avg > 0:
            orientation = Direction.RIGHT.value
        elif m_avg < 0:
            orientation = Direction.LEFT.value
        else:
            orientation = 0
        return degrees * orientation

        # tan(f_steerAngle_deg / 180.0 * PI)

        ######
        # V2
        ######

        X = avg_curvature_radius
        if avg_curvature_radius < 15:
            angle = 0.17198 * (X ** 2) - 4.844 * X + 29.1965
        else:
            angle = 0

        slope_threshold = config["LKAS"]["lanes_detection"]["others"]["slope_difference"]

        if m_avg > 0:
            orientation = Direction.RIGHT.value
        elif m_avg < 0:
            orientation = Direction.LEFT.value
        else:
            orientation = 0
        return angle * orientation

        x_top_left = left_line.get_x_given_y(0)
        x_top_right = right_line.get_x_given_y(0)
        x_bottom_left = left_line.get_x_given_y(480)
        x_bottom_right = right_line.get_x_given_y(480)

        # https://math.stackexchange.com/questions/185829/how-do-you-find-an-angle-between-two-points-on-the-edge-of-a-circle

        x_top = (x_top_left + x_top_right) / 2
        x_bot = (x_bottom_left + x_bottom_right) / 2

        point_top = np.array([x_top, 0])
        point_bot = np.array([x_bot, 480])

        dist = np.linalg.norm(point_top - point_bot)
        ym_per_pix = config["video"]["y_meters_per_pixel"]
        distance_in_meters = dist * ym_per_pix
        angle = 2 * np.arcsin(0.5 * distance_in_meters / avg_curvature_radius)
        return angle

    def get_deviation(self, left_lane, right_lane):
        return 0

    def compute_orientation_from_curvature(self, radius):
        x1 = np.sqrt(radius ** 2)
        frame_height = config["video"]["size"][1]
        x2 = np.sqrt(radius ** 2 - (frame_height / 2) ** 2)
        x3 = np.sqrt(radius ** 2 - frame_height ** 2)
        1 == 1
