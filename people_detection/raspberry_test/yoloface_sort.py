import cv2
from base import run

from yoloface import face_analysis

from sort.sort import Sort
import numpy as np

face_analyzer=face_analysis()
# Инициализация трекера SORT
tracker = Sort(max_age=100, iou_threshold=0.6)

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
        rects.append([int(x), int(y), int(x+h), int(y+w)])

    if len(rects) == 0:
        rects.append([0,0,0,0,0])
    
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



if __name__ == "__main__":
    run(detector)