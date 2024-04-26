python -m pip install mediapipe

python -m pip install opencv-python

wget -q -O face_detection/model/blaze_face_short_range.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

wget -q -O face_embedding/model/mobilenet_v3_large.tflite -q https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite

wget -q -O face_embedding/model/mobilenet_v3_small.tflite -q https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite