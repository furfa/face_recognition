
import time
import cv2
import numpy as np
from face_embedding.face_embedding_manager import FaceEmbeddingManager
from people_manager import PeopleManager
from result_visualizer import ResultVisualizer
from face_detection.face_detection_manager import FaceDetectionManager
import mediapipe as mp

FaceDetectorResult = mp.tasks.vision.FaceDetectorResult

class ApplicationManager:
    def __init__(self) -> None:
        self.face_embedding_manager = FaceEmbeddingManager()
        self.face_detection_manager = FaceDetectionManager()
        self.result_visualizer = ResultVisualizer()
        self.people_manager = PeopleManager()
        self.cap = cv2.VideoCapture(0)

    def predict_person_id(self, image, detection_result):
        for i, detection in enumerate(detection_result.detections):
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
            cropped = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

            resized = cv2.resize(cropped, (224, 224))

            person_id = self.people_manager.get_face_id(resized)

            detection.person_label = f"Person {person_id}"

    def start(self):
        prev_timestamp_ms = 0.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, image = self.cap.read()

            detection_result = self.face_detection_manager.detect_sync(image)

            self.predict_person_id(image, detection_result)

            visualized = self.result_visualizer.visualize(image=image, detection_result=detection_result)

            cv2.imshow('video feed', visualized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

ApplicationManager().start()