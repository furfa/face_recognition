
import time
from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


class ResultVisualizer:

    # used to record the time when we processed last frame 
    prev_frame_time = 0

    def _normalized_to_pixel_coordinates(
            self,
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def add_fps(self, image):
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
    
        # Calculating the fps 
    
        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-self.prev_frame_time) 
        self.prev_frame_time = new_frame_time

        # converting the fps into integer 
        fps = int(fps) 
    
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = str(fps) 
    
        # putting the FPS count on the frame 
        cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 

    def visualize(
        self,
        image,
        detection_result,
        show_fps=False
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualize.
        Returns:
            Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            print(detection)
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = '' if category_name is None else category_name
                probability = round(category.score, 2)
                person_label = '' if not hasattr(detection, "person_label") else detection.person_label
                result_text = person_label + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


        if show_fps:
            self.add_fps(annotated_image)
        return annotated_image