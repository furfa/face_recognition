import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Загрузка предобученного каскада Хаара для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализация Deep SORT
tracker = DeepSort(max_age=30, nn_budget=70)

# Захват видео с камеры или видеофайла
cap = cv2.VideoCapture("марияра_касса.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(3))   # float `width`
frame_height = int(cap.get(4))  # float `height`
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Список для хранения уникальных идентификаторов треков
unique_ids = set()

# Переменные для подсчета FPS
fps = 0
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения в серый цвет для каскадов Хаара
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))

    detections = []
    for (x, y, w, h) in faces:
        bbox = (x, y, w, h)
        detections.append((bbox, 1.0, 'person'))

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
cv2.destroyAllWindows()
