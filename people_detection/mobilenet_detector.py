import cv2
import numpy as np

# Загрузка модели и конфигурации
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

cap = cv2.VideoCapture("video_crop.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(3))   # float `width`
frame_height = int(cap.get(4))  # float `height`

out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения для модели
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_count = 0

    # Обработка детекций
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Класс 15 это "человек" в MobileNet SSD
                person_count += 1
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение счетчика
    cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
