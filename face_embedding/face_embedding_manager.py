import mediapipe as mp
import cv2
import os

BaseOptions = mp.tasks.BaseOptions
# ImageEmbedderResult = mp.tasks.vision.ImageEmbedder.ImageEmbedderResult
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class FaceEmbeddingManager:
    embedder: 'ImageEmbedder'

    def __init__(self, on_embed_callback=None) -> None:
        self.on_embed_callback = on_embed_callback

        model_asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "mobilenet_v3_large.tflite")

        options = options = ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.LIVE_STREAM if self.on_embed_callback else VisionRunningMode.IMAGE,
            quantize=True,
            result_callback=self.on_embed if self.on_embed_callback else None
        )
        
        self.embedder = ImageEmbedder.create_from_options(options)

    def on_embed(self, result, output_image: mp.Image, timestamp_ms: int):
        self.on_embed_callback(result, output_image, timestamp_ms)

    def embed(self, numpy_frame_from_opencv: cv2.typing.MatLike, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        self.embedder.embed_async(mp_image, timestamp_ms)

    def embed_sync(self, numpy_frame_from_opencv: cv2.typing.MatLike):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        return self.embedder.embed(mp_image)

    def __del__(self):
        self.embedder.close()