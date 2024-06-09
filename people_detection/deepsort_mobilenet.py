import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Путь к конфигурационному файлу и файлу весов
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"

# Загрузка модели
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


cap = cv2.VideoCapture("марияра_касса.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(3))   # float `width`
frame_height = int(cap.get(4))  # float `height`
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Переменные для расчета FPS
fps_counter = 0
fps_start_time = time.time()

# Инициализация Deep SORT трекера
tracker = DeepSort(max_age=30, n_init=3)

total_person_count = 0
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Класс 15 это "человек" в MobileNet SSD
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                rects.append(([startX, startY, endX - startX, endY - startY], confidence, "people"))  # x, y, w, h формат
    
    # Обновление трекера с новыми координатами
    tracks = tracker.update_tracks(rects, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()  # Получение координат бокса в формате top-left bottom-right
        unique_ids.add(track_id)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Обновление общего количества уникальных людей
    total_person_count = len(unique_ids)

    cv2.putText(frame, f"Unique Persons: {total_person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Подсчет и отображение FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

