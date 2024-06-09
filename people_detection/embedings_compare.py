import cv2
import numpy as np
from yoloface import face_analysis
import face_recognition
import time

class Timer():
    def __init__(self):
        self.start_time = time.time()
        self.end_time = time.time()
    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def get_duration(self):
        if not self.end_time:
            return 0
        if not self.start_time:
            return 0
        return f"{self.end_time - self.start_time:3f}"

# Инициализация детектора лиц YOLO
face_analyzer = face_analysis()

# Инициализация видео
cap = cv2.VideoCapture("raspberry_test/test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_embedengs_compare.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Переменные для хранения эмбеддингов лиц
unique_faces = []

# Переменные для расчета FPS
fps_counter = 0
fps_start_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_timer = Timer()
    detection_timer = Timer()
    embedding_timer = Timer()
    compare_embedding_timer = Timer()
    
    frame_timer.start()

    # Обнаружение лиц с помощью YOLOFace

    detection_timer.start()
    _,faces,conf = face_analyzer.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    detection_timer.stop()
    

    face_crops = []
    # Получение эмбеддингов для каждого обнаруженного лица
    for face in faces:
        x, y, w, h = face
        face_crop = frame[y:y+w, x:x+h]


        # Отображение прямоугольника вокруг лица
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)
        face_crops.append(face_crop)

        

    # Получение эмбеддингов лица
    embedding_timer.start()
    embeddings = []
    for face_crop in face_crops:
        # Конвертация лица в формат RGB
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face, model="small")
        embeddings.append(encodings)

    embedding_timer.stop()

    compare_embedding_timer.start()
    for encodings in embeddings:
        if len(encodings) > 0:
            encoding = encodings[0]
            # Проверка, является ли это лицо новым
            
            matches = face_recognition.compare_faces(unique_faces, encoding, tolerance=0.6)
            
            if not any(matches):
                unique_faces.append(encoding)
    compare_embedding_timer.stop()

    # Отображение количества уникальных лиц
    cv2.putText(frame, f"Unique Faces: {len(unique_faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Подсчет и отображение FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    frame_timer.stop()
    cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"frame_timer: {frame_timer.get_duration()} detection_timer: {detection_timer.get_duration()} embedding_timer: {embedding_timer.get_duration()} compare_embedding_timer: {compare_embedding_timer.get_duration()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)
    
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()