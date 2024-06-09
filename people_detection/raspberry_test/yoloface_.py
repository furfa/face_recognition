import cv2
from base import run

from yoloface import face_analysis

face=face_analysis()

def detector(frame):
    _,box,conf=face.face_detection(frame_arr=frame,frame_status=True,model='tiny')
    face.show_output(frame,box,frame_status=True)



if __name__ == "__main__":
    run(detector)