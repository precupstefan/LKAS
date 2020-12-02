import cv2
import numpy as np

from LKAS.config import get_config
from LKAS.utils.image_util import ImageUtil

config = get_config()


class LaneDetection:
    # https://www.youtube.com/watch?v=eLTLtUVuuy4
    def __init__(self):
        self.original_image = None
        self.canny_image = None
        self.region_of_interest = None
        self.lines = None
        self.smoothed_lanes = None
        self.m_inv = None
        self.warped_perspective = None
        global config
        config = get_config()

    def detect_lanes(self, frame):
        """
        DETECTS LANES
        :param frame: GRAYSCALE FRAME
        :return:
        """
        self.original_image = frame.copy()
        self.warped_perspective = self.warp_perspective(frame)
        self.canny_image = self.get_edges_in_frame(self.warped_perspective)
        # self.region_of_interest = self.get_region_of_interest(self.canny_image)
        self.lines = cv2.HoughLinesP(self.canny_image, rho=2, theta=np.pi / 180, threshold=40, minLineLength=25,
                                     maxLineGap=70)
        if self.lines is None:
            return np.array([], [])
        self.smoothed_lanes = self.get_smoothed_lanes(frame, self.lines)
        return self.smoothed_lanes

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

    def warp_perspective(self, frame):
        img_size = (frame.shape[1], frame.shape[0])
        src = np.float32([
            config["LKAS"]["lanes_detection"]["warp_perspective"]["src"]
        ])
        dst = np.float32([
            config["LKAS"]["lanes_detection"]["warp_perspective"]["dst"]
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        self.m_inv = cv2.getPerspectiveTransform(dst, src)
        return cv2.warpPerspective(frame, matrix, img_size)

    def get_region_of_interest(self, frame):
        polygons = np.array([
            config["LKAS"]["lanes_detection"]["region_of_interest"]
        ])
        mask = np.zeros_like(frame)
        masked_image = ImageUtil.fill_poly(mask, polygons)
        return ImageUtil.apply_bitwise_and(frame, masked_image)

    def get_smoothed_lanes(self, frame, lines):
        half_screen = frame.shape[1] / 2
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

    def build_status_image(self, size=(720, 1600)):
        layout = np.array((2, 4))
        size = np.array(size)
        frame_dimension = size / layout
        frame_dimension = (frame_dimension[1], frame_dimension[0])
        image = np.zeros(np.append(size, 3), np.uint8)

        region_of_interest_color = cv2.resize(self.get_region_of_interest(self.original_image),
                                              tuple(frame_dimension))  # Resize image)
        image[0:frame_dimension[1], frame_dimension[0] * 0:frame_dimension[0] * 1] = region_of_interest_color

        warped_image = cv2.resize(self.warped_perspective, tuple(frame_dimension))  # Resize image)
        # warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)
        image[0:frame_dimension[1], frame_dimension[0] * 1:frame_dimension[0] * 2] = warped_image

        canny_image = cv2.resize(self.canny_image, tuple(frame_dimension))  # Resize image)
        canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)
        image[0:frame_dimension[1], frame_dimension[0] * 2:frame_dimension[0] * 3] = canny_image

        # region_of_interest = cv2.resize(self.region_of_interest, tuple(frame_dimension))  # Resize image)
        # region_of_interest = cv2.cvtColor(region_of_interest, cv2.COLOR_GRAY2BGR)
        # image[0:frame_dimension[1], frame_dimension[0] * 2:frame_dimension[0] * 3] = region_of_interest

        # ## BOTTOM ROW
        detected_lines = cv2.resize(self.draw_lines(self.lines), tuple(frame_dimension))  # Resize image)
        image[frame_dimension[1] * 1:frame_dimension[1] * 2,
        frame_dimension[0] * 0:frame_dimension[0] * 1] = detected_lines
        smoothed_lanes = cv2.resize(self.draw_lines(self.smoothed_lanes), tuple(frame_dimension))  # Resize image)
        image[frame_dimension[1] * 1:frame_dimension[1] * 2, frame_dimension[0]:frame_dimension[0] * 2] = smoothed_lanes
        return image

    def draw_lines(self, lines):
        image = np.zeros(self.original_image.shape)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                try:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                except:
                    pass
        return image
