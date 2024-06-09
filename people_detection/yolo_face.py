from yoloface import face_analysis
import numpy
import cv2
import time

face=face_analysis()
# Переменные для расчета FPS
fps_counter = 0
fps_start_time = time.time()
fps = 0

# example 3
cap = cv2.VideoCapture("марияра_касса.mp4")
while True: 
    _, frame = cap.read()
    _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    output_frame=face.show_output(frame,box,frame_status=True)

    # Подсчет и отображение FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('frame',output_frame)
    key=cv2.waitKey(1)
    if key ==ord('q'): 
        break 



cap.release()
cv2.destroyAllWindows()