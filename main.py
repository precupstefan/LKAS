import cv2

from logic.lane_detection import LaneDetection
from utils.feed_streamer import FeedStreamer
from utils.image_util import ImageUtil

def ceva(frame):
    ceva= LaneDetection().detect_lanes(frame)
    if len(ceva) == 0:
        return frame
    smoothed_lanes, lanes = ceva
    # create a copy of the original frame
    processed_frame = frame[:, ].copy()
    testing = frame[:, ].copy()

    if smoothed_lanes.shape != ():
        # draw Hough lines
        for line in smoothed_lanes:
            x1, y1, x2, y2 = line.reshape(4)
            try:
                cv2.line(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            except:
                pass

    if lanes.shape != ():
        # draw Hough lines
        for line in lanes:
            x1, y1, x2, y2 = line.reshape(4)
            try:
                cv2.line(testing, (x1, y1), (x2, y2), (255, 0, 0), 2)
            except:
                pass
    cv2.imshow('testing', testing)
    return processed_frame

FeedStreamer().play(ceva)

