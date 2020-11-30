import cv2
import pafy


class FeedStreamer:

    def __init__(self, video_url="eoXguTDnnHM", size=(640, 480)):
        vPafy = pafy.new(video_url)
        self.video = vPafy.getbest(preftype="any")

    def play(self, callback):
        capture_device = cv2.VideoCapture(self.video.url)

        while (True):
            ret, frame = capture_device.read()
            frame = cv2.resize(frame, (640, 480))
            callback()
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        capture_device.release()
        cv2.destroyAllWindows()
