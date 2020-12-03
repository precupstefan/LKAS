import numpy as np

from LKAS.config import load_config

config = load_config()


class Trajectory:
    def __init__(self):
        pass

    def compute_trajectory_correction(self, lanes):
        middle_lane = config["video"]["size"][0] / 2
        # left_lane_means = np.average(left_lane, axis=0)
        # right_lane_means = np.average(right_lane, axis=0)
        # if len(lanes) == 2:
        #     x1, y1, x2, y2 = lanes[0]
        #     left_X = x1 if y1 > y2 else x2
        #     x1, y1, x2, y2 = lanes[1]
        #     right_X = x1 if y1 > y2 else x2
        #     pixel_deviation = (left_X + right_X) / 2 - middle_lane
        # else:
        #     x1, y1, x2, y2 = lanes[0]
        #     slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        #     pixel_deviation = slope * -1

        equations = []
        for line in lanes:
            x1, y1, x2, y2 = line
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            equations.append((slope, intercept))
        if len(equations) == 2:
            error = (equations[1][1] - equations[0][1]) / (equations[0][0] - equations[1][0])
            if -10 <= error - middle_lane <= 10:
                return 0
            return (error - middle_lane) *700* config["video"]["x_meters_per_pixel"]
        else:
            error = np.average(np.array(equations), axis=0)[0]
            deviation = error * config["video"]["x_meters_per_pixel"]
            return deviation * 700 * -1
