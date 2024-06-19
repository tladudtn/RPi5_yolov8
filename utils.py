import cv2
import time
from ultralytics.utils.plotting import Annotator

class SimpleFPS:
    def __init__(self):
        self.start_time = time.time()
        self.display_time_sec = 1  # update fps display
        self.fps = 0
        self.frame_counter = 0
        self.is_fps_updated = False

    def get_fps(self):
        elapsed = time.time() - self.start_time
        self.frame_counter += 1
        is_fps_updated = False

        if elapsed > self.display_time_sec:
            self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.start_time = time.time()
            is_fps_updated = True

        return int(self.fps), is_fps_updated

def draw_fps(image, fps):
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def draw_annotation(image, label_names, results, conf_threshold=0.45, target_class=0):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > conf_threshold and cls == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{label_names[cls]} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def draw_center_lines(image):
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 0), 2)
    cv2.line(image, (0, center_y), (width, center_y), (0, 255, 0), 2)

def draw_ranges(image):
    height, width, _ = image.shape

    # Draw rectangles for the 25-75% range
    left_25 = int(width * 0.25)
    right_75 = int(width * 0.75)
    top_25 = int(height * 0.25)
    bottom_75 = int(height * 0.75)
    cv2.rectangle(image, (left_25, top_25), (right_75, bottom_75), (255, 255, 0), 2)

    # Draw rectangles for the 10-90% range
    left_10 = int(width * 0.10)
    right_90 = int(width * 0.90)
    top_10 = int(height * 0.10)
    bottom_90 = int(height * 0.90)
    cv2.rectangle(image, (left_10, top_10), (right_90, bottom_90), (255, 0, 255), 2)

    # Draw rectangles for the 40-60% range (servo motor stop range)
    left_40 = int(width * 0.40)
    right_60 = int(width * 0.60)
    top_40 = int(height * 0.40)
    bottom_60 = int(height * 0.60)
    cv2.rectangle(image, (left_40, top_40), (right_60, bottom_60), (0, 255, 255), 2)

def draw_person_label(image, obj_center_x, y1, y2, adjusted_center_y, label, x1, x2):
    cv2.line(image, (obj_center_x, y1), (obj_center_x, y2), (255, 0, 0), 2)
    cv2.line(image, (x1, adjusted_center_y), (x2, adjusted_center_y), (255, 0, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_temperature_grid(image, temperatures):
    height, width, _ = image.shape
    cell_width = width // 3
    cell_height = height // 3

    for i, row in enumerate(temperatures):
        for j, temp in enumerate(row):
            x = j * cell_width
            y = i * cell_height
            cv2.putText(image, f"{temp:.2f}", (x + cell_width // 2, y + cell_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # 주황색 텍스트
