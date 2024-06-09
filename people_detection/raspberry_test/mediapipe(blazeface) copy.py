import cv2
from base import run

import mediapipe.solutions.face_detection
import mediapipe.solutions.drawing_utils


# Инициализация MediaPipe Face Detection с более высокой конфиденциальностью
# mp_drawing = mediapipe.python.solutions.drawing_utils


print(1)
face_detection = mediapipe.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)
print(2)

def detector(frame):
    # Преобразование изображения в RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка изображения и детекция лиц
    results = face_detection.process(image)

    if results.detections:
        for detection in results.detections:
            mediapipe.solutions.drawing_utils.draw_detection(frame, detection)



if __name__ == "__main__":
    run(detector)
    face_detection.close()