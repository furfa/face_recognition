import cv2
from base import run

from yoloface import face_analysis
from deep_sort_realtime.deepsort_tracker import DeepSort

face_analyzer=face_analysis()
# Инициализация трекера DeepSORT
tracker = DeepSort(max_age=100, max_cosine_distance=0.4)

total_person_count = 0
unique_ids = set()

def detector(frame):
    # Обнаружение лиц с помощью YOLOFace
    _,faces,conf = face_analyzer.face_detection(frame_arr=frame,frame_status=True,model='tiny')

    # Преобразование обнаруженных лиц для DeepSORT
    rects = []
    for i in range(len(faces)):
        box=faces[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        
        cv2.rectangle(frame,(x,y), (x+h,y+w), (255,255,255), 3)
        rects.append(([x, y, h, w], conf[i], "people"))

    
    # Обновление трекера DeepSORT
    tracks = tracker.update_tracks(rects, frame=frame)

    # Отображение треков
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        unique_ids.add(track_id)


    # Обновление общего количества уникальных людей
    total_person_count = len(unique_ids)

    cv2.putText(frame, f"Unique Persons: {total_person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



if __name__ == "__main__":
    run(detector)