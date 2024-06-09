
import cv2
import mediapipe as mp
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


# Инициализация MediaPipe Face Detection с более высокой конфиденциальностью
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Инициализация Deep SORT
tracker = DeepSort(max_age=30, nn_budget=70)

# Захват видео с камеры или видеофайла
cap = cv2.VideoCapture("марияра_касса.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (400-150, 400-150))

# Список для хранения уникальных идентификаторов треков
unique_ids = set()

# Переменные для подсчета FPS
fps = 0
frame_count = 0
start_time = time.time()


with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:  # Увеличение конфиденциальности
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[150:400, 150:400]
        # Преобразование изображения в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обработка изображения и детекция лиц
        results = face_detection.process(image)

        detections = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                # Масштабирование координат обратно к исходному размеру изображения
                bbox = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                detections.append((bbox, detection.score[0], 'person'))

        # Обновление трекеров
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            track_id = track.track_id
            unique_ids.add(track_id)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отображение счетчика уникальных людей
        cv2.putText(frame, f"Unique Persons: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Подсчет и отображение FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Отображение изображения
        cv2.imshow('Face Detection & Tracking', frame)
        out.write(frame)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
