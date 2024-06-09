import cv2
import numpy as np
from sort.sort import Sort

# Путь к конфигурационному файлу и файлу весов
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"

# Загрузка модели
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture("улицаоколопятерочки_trim.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(3))   # float `width`
frame_height = int(cap.get(4))  # float `height`
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

# Инициализация SORT трекера
tracker = Sort()
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
                rects.append([startX, startY, endX, endY])

    # Обновление трекера с новыми координатами
    trackers = tracker.update(np.array(rects))

    for d in trackers:
        startX, startY, endX, endY, track_id = d.astype(int)
        unique_ids.add(track_id)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Обновление общего количества уникальных людей
    total_person_count = len(unique_ids)

    cv2.putText(frame, f"Unique Persons: {total_person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()