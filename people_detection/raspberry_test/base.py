import time 
import cv2
import argparse
import numpy as np


def run(on_next_frame):
    print("run")
    parser = argparse.ArgumentParser(description='Face detection')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, help='Path to output video file')
    parser.add_argument('--no_preview', action='store_true')
    parser.set_defaults(no_preview=False)
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(3))
    frame_height = int(cap.get(4))

    if args.output:
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    else:
        out = None

    # Переменные для расчета FPS
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        on_next_frame(frame)

        # Подсчет и отображение FPS
        fps_counter += 1
        
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(frame, f"FPS: {current_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if args.no_preview:
            print("fps", fps)
        else:
            cv2.imshow("Frame", frame)

        if out:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


    print("processing time:", time.time() - start_time)
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda x: False)