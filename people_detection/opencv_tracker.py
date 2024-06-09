import cv2

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Создание объекта трекера
tracker = cv2.TrackerKCF()

cap = cv2.VideoCapture("video_crop.mp4")

# Флаг для обозначения, что лицо было обнаружено и трекер инициализирован
face_detected = False

total_person_count = 0
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not face_detected:
        # Обнаружение лиц с помощью каскада Хаара
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Если обнаружено лицо, инициализируем трекер
        if len(faces) > 0:
            face_detected = True
            bbox = tuple(faces[0])  # Возьмем первое обнаруженное лицо для простоты
            tracker.init(frame, bbox)

    else:
        # Обновление трекера и получение новых координат объекта
        ret, bbox = tracker.update(frame)

        if ret:  # Если трекер успешно обновлен
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            unique_ids.add(1)  # Предполагаем, что треки одинаковые для простоты
        else:  # Если трекер потерял объект, снова обнаружим лицо
            face_detected = False

    total_person_count = len(unique_ids)
    cv2.putText(frame, f"Unique Persons: {total_person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
