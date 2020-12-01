import cv2
import pafy

from config import config

feed = "rbe-cT12ljs"
feed = "n7zuQM6aPqg"
feed = "vjdrU_WOg54" #feed 4k destul de bun
# feed = "ydvsyAZbuSQ"

# feed ="3y5aRJLDeB4" # un video bunicel da e groaznic
feed = "c3Q_vHeZT1w"

class FeedStreamer:

    def __init__(self, video_url=feed, size=(640, 480)):
        vPafy = pafy.new(video_url)
        self.video = vPafy.getbest(preftype="any")

    def play(self, callback):
        capture_device = cv2.VideoCapture(self.video.url)

        skip_frames = 3000
        for _ in range(skip_frames):
            capture_device.read()
        while (True):
            ret, frame = capture_device.read()
            frame = cv2.resize(frame, tuple(config["video"]["size"]))
            processed_frame = callback(frame)
            cv2.imshow('frame', frame)
            cv2.imshow("processed_frame", processed_frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        capture_device.release()
        cv2.destroyAllWindows()
