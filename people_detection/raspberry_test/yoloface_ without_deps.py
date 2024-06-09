import cv2
from base import run

# Загрузка YOLO модели
yolo_net = cv2.dnn.readNetFromDarknet("yolov3-face.cfg", "yolov3-wider_16000.weights")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detector(frame):
    _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    face.show_output(frame,box,frame_status=True)



if __name__ == "__main__":
    run(detector)