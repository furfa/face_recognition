
import cv2
import numpy as np
from result_visualizer import ResultVisualizer
from face_detection.face_detection_manager import FaceDetectionManager
import mediapipe as mp

FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

class ApplicationManager:
    def __init__(self) -> None:
        self.face_detection_manager = FaceDetectionManager(on_detect_callback=self.visualize_result)
        self.result_visualizer = ResultVisualizer()
        self.cap = cv2.VideoCapture(0)

    def start(self):
        prev_timestamp_ms = 0.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = self.cap.read()

            self.frame = frame
            timestamp_ms = prev_timestamp_ms + 1000 / fps
            self.face_detection_manager.detect(frame, round(timestamp_ms))
            

            # show async generated result
            if hasattr(self, "result_frame"):
                cv2.imshow('video feed', self.result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_timestamp_ms = timestamp_ms
    
    def visualize_result(self, result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        image_copy = np.copy(output_image.numpy_view())
        res = self.result_visualizer.visualize(image_copy, result)
        rgb_annotated_image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        self.result_frame = rgb_annotated_image


    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

ApplicationManager().start()