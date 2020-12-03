import cv2
import numpy as np
import warnings

from LKAS.config import config
from LKAS.utils.image_util import apply_bitwise_and, fill_poly, warp_perspective


class LaneDetection:
    # https://www.youtube.com/watch?v=eLTLtUVuuy4
    def __init__(self):
        self.original_image = None
        self.canny_image = None
        self.region_of_interest = None
        self.lines = None
        self.smoothed_lanes = None
        self.inverse_matrix = None
        self.warped_perspective = None
        self.image_bottom_half_histogram = None
        self.sliding_window_image = None

    def detect_lanes(self, frame):
        """
        DETECTS LANES
        :param frame: GRAYSCALE FRAME
        :return:
        """
        self.original_image = frame.copy()
        self.warped_perspective, self.inverse_matrix = warp_perspective(frame)
        hls_frame, gray_frame, blurred_frame, thresh_frame, cannyframe = self.get_edges_in_frame(
            self.warped_perspective)
        left_lane, right_lane, self.sliding_window_image = self.determine_lines_using_sliding_window(
            thresh_frame)
        detected_line_on_image = self.draw_detected_lines_on_image(self.warped_perspective, (left_lane, right_lane))
        cv2.imshow("perspective", self.warped_perspective)
        cv2.imshow("sliding_window_image", self.sliding_window_image)
        cv2.imshow("detected_line_on_image", detected_line_on_image)

        # self.canny_image = self.get_edges_in_frame(self.warped_perspective)
        # # self.region_of_interest = self.get_region_of_interest(self.canny_image)
        # self.lines = cv2.HoughLinesP(self.canny_image, rho=2, theta=np.pi / 180, threshold=40, minLineLength=25,
        #                              maxLineGap=70)
        # if self.lines is None:
        #     return np.array([], [])
        # self.smoothed_lanes = self.get_smoothed_lanes(frame, self.lines)
        return self.smoothed_lanes

    def get_edges_in_frame(self, frame):
        hls_frame = gray_frame = blurred_frame = thresh_frame = canny_frame = None
        if config["LKAS"]["lanes_detection"]["colour_filtering"]["active"]:
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            lower_white = np.array(config["LKAS"]["lanes_detection"]["colour_filtering"]["lower_white"])
            upper_white = np.array(config["LKAS"]["lanes_detection"]["colour_filtering"]["upper_white"])
            mask = cv2.inRange(hls, lower_white, upper_white)
            hls_frame = cv2.bitwise_and(frame, frame, mask=mask)
            gray_frame = cv2.cvtColor(hls_frame, cv2.COLOR_BGR2GRAY)
            frame = gray_frame
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = gray_frame
        if config["LKAS"]["lanes_detection"]["gaussian_blur"]["active"]:
            blurred_frame = cv2.GaussianBlur(frame,
                                             tuple(config["LKAS"]["lanes_detection"]["gaussian_blur"]["kernel_size"]),
                                             config["LKAS"]["lanes_detection"]["gaussian_blur"]["deviation"])
            frame = blurred_frame
        if config["LKAS"]["lanes_detection"]["threshold"]["active"]:
            if config["LKAS"]["lanes_detection"]["threshold"]["adaptive"]:
                thresh_frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                     config["LKAS"]["lanes_detection"]["threshold"]["block_size"],
                                                     config["LKAS"]["lanes_detection"]["threshold"]["constant"])
            else:
                _, thresh_frame = cv2.threshold(frame, config["LKAS"]["lanes_detection"]["threshold"]["threshold"],
                                                255, cv2.THRESH_BINARY)
            frame = thresh_frame
        canny_frame = cv2.Canny(frame, config["LKAS"]["lanes_detection"]["canny"]["low_threshold"],
                                config["LKAS"]["lanes_detection"]["canny"]["high_threshold"])
        return hls_frame, gray_frame, blurred_frame, thresh_frame, canny_frame

    @staticmethod
    def determine_lines_using_sliding_window(img):

        ### Settings
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

        # plt.figure()
        # plt.plot(histogram)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit a second order polynomial to each
        # left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        # right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return left_fit, right_fit, out_img

    def draw_detected_lines_on_image(self, frame, lines):
        left_fit, right_fit = lines
        ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0]).astype("int")
        left_fitx = np.array(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype("int")
        right_fitx = np.array(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype("int")
        left_lane = np.vstack((ploty, left_fitx))
        left_lane = left_lane[:, left_lane[-1, :] < frame.shape[1]]
        right_lane = np.vstack((ploty, right_fitx))
        right_lane = right_lane[:, right_lane[-1, :] < frame.shape[1]]
        left_lane = np.transpose(left_lane)[:, ::-1]
        right_lane = np.transpose(right_lane)[:, ::-1]

        poly = np.vstack((left_lane, np.flipud(right_lane)))
        cv2.fillPoly(frame, [poly], (0, 255, 0))
        # cv2.fillPoly(frame, [right_lane], (0, 255, 0))

        # frame[left_lane[0], left_lane[1]] = [0, 255, 0]
        # frame[right_lane[0], right_lane[1]] = [255, 0, 0]
        return frame

    # def get_edges_in_frame(self, frame):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     frame = cv2.GaussianBlur(frame, tuple(config["LKAS"]["lanes_detection"]["gaussian_blur"]["kernel_size"]),
    #                              config["LKAS"]["lanes_detection"]["gaussian_blur"]["deviation"])
    #     frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #                                   config["LKAS"]["lanes_detection"]["threshold"]["block_size"],
    #                                   config["LKAS"]["lanes_detection"]["threshold"]["constant"])
    #     frame = cv2.Canny(frame, config["LKAS"]["lanes_detection"]["canny"]["low_threshold"],
    #                       config["LKAS"]["lanes_detection"]["canny"]["high_threshold"])
    #     return frame

    def get_region_of_interest(self, frame):
        polygons = np.array([
            config["LKAS"]["lanes_detection"]["region_of_interest"]
        ])
        mask = np.zeros_like(frame)
        masked_image = fill_poly(mask, polygons)
        return apply_bitwise_and(frame, masked_image)

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
            left_lane = self.make_line_coordinates(left_fit_average, frame.shape[0])
            lanes.append(left_lane)

        if len(right_fit) is not 0:
            right_lane = self.make_line_coordinates(right_fit_average, frame.shape[0])
            lanes.append(right_lane)

        return np.array(lanes)

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