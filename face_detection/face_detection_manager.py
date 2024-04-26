import mediapipe as mp
import cv2
import os

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

class FaceDetectionManager:
    detector: 'FaceDetector'

    def __init__(self, on_detect_callback = None) -> None:
        self.on_detect_callback = on_detect_callback

        model_asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "blaze_face_short_range.tflite")

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.LIVE_STREAM if self.on_detect_callback else VisionRunningMode.IMAGE,
            result_callback=self.on_detect if self.on_detect_callback else None
        )
        
        self.detector = FaceDetector.create_from_options(options)

    def on_detect(self, result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        # print('face detector result: {}'.format(result))
        self.on_detect_callback(result, output_image, timestamp_ms)

    def detect(self, numpy_frame_from_opencv: cv2.typing.MatLike, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        self.detector.detect_async(mp_image, timestamp_ms)

    def detect_sync(self, numpy_frame_from_opencv: cv2.typing.MatLike):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        return self.detector.detect(mp_image)

    def __del__(self):
        self.detector.close()