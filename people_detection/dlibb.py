import cv2
import time
import dlib

# Инициализация детектора лиц и трекера dlib
face_detector = dlib.get_frontal_face_detector()
trackers = []

# Инициализация видео
cap = cv2.VideoCapture("./raspberry_test/test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('outputttt.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Переменные для расчета FPS
fps_counter = 0
fps_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обновление трекеров
    updated_trackers = []
    for tracker in trackers:
        tracker.update(frame)
        pos = tracker.get_position()
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        updated_trackers.append(tracker)
    trackers = updated_trackers

    # Выполнение детекции лиц каждые 10 кадров
    if fps_counter % 10 == 0:
        # Обнаружение лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        # Инициализация трекеров для каждого обнаруженного лица
        for face in faces:
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(face.left(), face.top(), face.right(), face.bottom())
            tracker.start_track(frame, rect)
            trackers.append(tracker)

    # Подсчет и отображение FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
