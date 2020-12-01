import cv2
import numpy as np

from config import config


class LaneDetection:

    ym_per_pix = config["video"]["y_meters_per_pixel"]
    xm_per_pix = config["video"]["x_meters_per_pixel"]

    def __init__(self):
        self.original_frame = None
        self.sliding_window_frame = None
        self.im_pillpoly = None
        self.im_newwarp = None
        self.m_inv = None

    def detect_lanes(self, frame):
        self.original_frame = frame
        stuff = self.warp_perspective(frame)
        hls_frame, gray_frame, blurred_frame, thresh_frame, canny_frame = self.get_edges_in_frame(stuff[0])
        histogram, left_lane, right_lane = self.compute_histogram(thresh_frame)
        ploty, left_fit, right_fit, left_fitx, right_fitx = self.slide_window_search(thresh_frame, histogram)
        draw_info = self.general_search(thresh_frame, left_fit, right_fit)
        curveRad, curveDir = self.measure_lane_curvature(ploty, left_fitx, right_fitx)
        meanPts, result = self.draw_lane_lines(frame, thresh_frame, draw_info)
        deviation, directionDev = self.offCenter(meanPts, frame)
        return curveRad, curveDir, deviation, directionDev

    def warp_perspective(self, frame):
        img_size = (frame.shape[1], frame.shape[0])
        src = np.float32([
            config["LKAS"]["lanes_detection"]["warp_perspective"]["src"]
        ])

        # Window to be shown
        dst = np.float32([
            config["LKAS"]["lanes_detection"]["warp_perspective"]["dst"]
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Inverse matrix to unwarp the image for final window
        self.m_inv = cv2.getPerspectiveTransform(dst, src)
        birdseye = cv2.warpPerspective(frame, matrix, img_size)

        # Get the birdseye window dimensions
        height, width = birdseye.shape[:2]

        # Divide the birdseye view into 2 halves to separate left & right lanes
        birdseyeLeft = birdseye[0:height, 0:width // 2]
        birdseyeRight = birdseye[0:height, width // 2:width]
        return birdseye, birdseyeLeft, birdseyeRight

    def _detect_lines(self, warped_perspective):
        return cv2.Canny(warped_perspective, 50, 150)

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

    def compute_histogram(self, frame):
        histogram = np.sum(frame[frame.shape[0] // 2:, :], axis=0)

        midpoint = np.int(histogram.shape[0] / 2)
        left_lane = np.argmax(histogram[:midpoint])
        right_lane = np.argmax(histogram[midpoint:]) + midpoint

        # Return histogram and x-coordinates of left & right lanes to calculate
        # lane width in pixels
        return histogram, left_lane, right_lane

    # TODO: shamelessly copied, wtf is here
    def slide_window_search(self, binary_warped, histogram):

        # Find the start of left and right lane lines using histogram info
        self.sliding_window_frame = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # A total of 9 windows will be used
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []

        #### START - Loop to iterate through windows and search for lane lines #####
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(self.sliding_window_frame, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(self.sliding_window_frame, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        #### END - Loop to iterate through windows and search for lane lines #######

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Apply 2nd degree polynomial fit to fit curves
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        ltx = np.trunc(left_fitx)
        rtx = np.trunc(right_fitx)
        # plt.show()

        self.sliding_window_frame[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        self.sliding_window_frame[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return ploty, left_fit, right_fit, ltx, rtx

    def general_search(self, binary_warped, left_fit, right_fit):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        ## VISUALIZATION ###########################################################

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        self.im_pillpoly = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty

        return ret

    def measure_lane_curvature(self, ploty, leftx, rightx):

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x, y in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        # Decide if it is a left or a right curve
        if leftx[0] - leftx[-1] > 60:
            curve_direction = 'Left Curve'

        elif leftx[-1] - leftx[0] > 60:
            curve_direction = 'Right Curve'
        else:
            curve_direction = 'Straight'

        return (left_curverad + right_curverad) / 2.0, curve_direction

    def draw_lane_lines(self, original_image, warped_image, draw_info):

        leftx = draw_info['leftx']
        rightx = draw_info['rightx']
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

        newwarp = cv2.warpPerspective(color_warp, self.m_inv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        self.im_newwarp = cv2.warpPerspective(color_warp, self.m_inv,
                                              (original_image.shape[1], original_image.shape[0]))
        return pts_mean, result

    def offCenter(self, meanPts, frame):
        # Calculating deviation in meters
        mpts = meanPts[-1][-1][-2].astype(int)
        pixelDeviation = frame.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * self.xm_per_pix
        direction = "left" if deviation < 0 else "right"
        return deviation, direction
