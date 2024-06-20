import numpy as np
import cv2
import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time
from collections import deque

# Constants
TIME_WINDOW = 10 # 5 * 60  # 5 minutes in seconds
BUFFER_SIZE = 60 * 5  # Assuming we check the state once per second

# Global variables
lock = threading.Lock()
status_buffer = deque(maxlen=BUFFER_SIZE)


def crop_screen_area(frame):
    # Define the bounding box dimensions
    bbox_width = 1200
    bbox_height = 270

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Calculate the center position for the bounding box
    center_x = frame_width // 2
    center_y = frame_height - (bbox_height // 2) - 10  # 10 pixels from the bottom

    # Calculate the top-left and bottom-right coordinates of the bounding box
    top_left = (center_x - bbox_width // 2, center_y - bbox_height // 2)
    bottom_right = (center_x + bbox_width // 2, center_y + bbox_height // 2)

    # Draw the bounding box (optional for visualization)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Crop the area inside the bounding box
    cropped_area = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_area

def get_screen_brightness(image):
    if image is None:
        raise ValueError("Ошибка при загрузке изображения")

    # Преобразовать изображение в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размыть изображение для устранения шума
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Определить среднюю яркость изображения
    mean_brightness = np.mean(blurred_image)

    return mean_brightness

def process_frame(frame, threshold=55):
    cropped_area = crop_screen_area(frame)
    mean_brightness = get_screen_brightness(cropped_area)
    with lock:
        is_screen_on = mean_brightness > threshold
        timestamp = time.time()
        status_buffer.append((timestamp, is_screen_on))

    cv2.putText(frame, f"mean_brightness={mean_brightness:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    on_duration, off_duration, is_screen_on = calculate_statistics()
    cv2.putText(frame, f"is_screen_on={is_screen_on}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"on_duration={on_duration:.2f}s", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"off_duration={off_duration:.2f}s", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def video_processing(video_path, threshold=55, show=True):
    # Create a VideoCapture object
    try:
        video_path = int(video_path)
    except ValueError:
        pass

    cap = cv2.VideoCapture(video_path)

    # # Get frame dimensions
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))


    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Read until the video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        
        process_frame(frame, threshold)
        
        if show:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press 'q' on the keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # out.write(frame)

    # Release the video capture object
    cap.release()
    # out.release()

    # Close all OpenCV windows
    if show:
        cv2.destroyAllWindows()

def calculate_statistics():
    with lock:
        current_time = time.time()
        # Remove old entries
        while status_buffer and status_buffer[0][0] < current_time - TIME_WINDOW:
            status_buffer.popleft()

        # Calculate on and off durations
        on_duration = sum(t1 - t0 for (t0, s0), (t1, s1) in zip(status_buffer, list(status_buffer)[1:]) if s0)
        off_duration = TIME_WINDOW - on_duration

        is_screen_on = on_duration > off_duration

        return on_duration, off_duration, is_screen_on

def run_server(host, port):
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/screen_status':
                on_duration, off_duration, is_screen_on = calculate_statistics()
                response = {
                    'is_screen_on': str(is_screen_on),
                    'on_duration': on_duration,
                    'off_duration': off_duration
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
    
    server = HTTPServer((host, port), RequestHandler)
    print(f"Starting server on http://{host}:{port}")
    server.serve_forever()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect if the monitor is on or off in a video.")
    parser.add_argument('--video_path', type=str, default='0', help='Path to the video file or camera index (0 for default camera)')
    parser.add_argument('--threshold', type=float, default=55, help='Brightness threshold to determine if the monitor is on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the HTTP server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the HTTP server on')
    parser.add_argument('--show', action='store_true', help='Show video frames')
    parser.add_argument('--no_http', action='store_true', help='Show video frames')
    return parser.parse_args()

def main():
    args = parse_arguments()
    video_path = args.video_path
    threshold = args.threshold
    host = args.host
    port = args.port
    show = args.show
    no_http = args.no_http

    # Start HTTP server in a separate thread
    if not no_http:
        server_thread = threading.Thread(target=run_server, args=(host, port))
        server_thread.start()

    # Start video processing
    video_processing(video_path, threshold, show)

if __name__ == "__main__":
    main()
