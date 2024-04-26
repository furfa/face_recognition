
import os
import time
import cv2
import numpy as np

from people_manager import PeopleManager
from result_visualizer import ResultVisualizer
from face_detection.face_detection_manager import FaceDetectionManager
import mediapipe as mp

def get_cropped_faces(image, detection_result):
    for i, detection in enumerate(detection_result.detections):
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        
        cropped = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        resized = cv2.resize(cropped, (224, 224))

        person_id = people_manager.get_face_id(resized)

        detection.person_label = f"Person {person_id}"
        

fd = FaceDetectionManager()
result_visualizer = ResultVisualizer()
people_manager = PeopleManager()

files = os.listdir("test_images")
files.sort()

index = 0

while True:
    filename = "test_images/" + files[index]
    print(f"{index=} {filename=}")

    image = cv2.imread(filename)

    detection_result = fd.detect_sync(image)

    get_cropped_faces(image, detection_result)

    visualized = result_visualizer.visualize(image=image, detection_result=detection_result)

    cv2.imshow("visualized", visualized)

    key = cv2.waitKey(0)

    if key & 0xFF == ord('w'):
        index += 1
    
    if key & 0xFF == ord('s'):
        index -= 1

    if key & 0xFF == ord('q'):
        break



