from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from picamera2 import Picamera2
from yolo_manager import YoloDetectorWrapper
from utils import SimpleFPS, draw_fps, draw_annotation, draw_center_lines, draw_ranges, draw_person_label, draw_temperature_grid
import argparse
import time
import pyfirmata2
import smbus2

AMG8833_I2C_ADDR = 0x69
AMG8833_PIXEL_ARRAY_SIZE = 64

class ServoThread:
    def __init__(self, board, pin, initial_angle=90, min_angle=0, max_angle=180):
        self.servo = board.get_pin(f'd:{pin}:s')
        self.angle = initial_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.servo.write(self.angle)

    def set_angle(self, angle):
        self.angle = max(self.min_angle, min(self.max_angle, angle))  # Ensure the angle is within min and max bounds
        self.servo.write(self.angle)

class VideoThreadPiCam(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, frame_rate=15, width=800, height=800):
        super().__init__()
        self.frame_rate = frame_rate
        self.width = width
        self.height = height
        self.grab_frame = True

    def run(self):
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(main={"size": (self.width, self.height), "format": "RGB888"}, raw={"size": (self.width, self.height)})
        picam2.configure(camera_config)
        picam2.start()

        while True:
            if self.grab_frame:
                frame = picam2.capture_array()
                self.change_pixmap_signal.emit(frame)
                self.grab_frame = False
            time.sleep(1.0 / self.frame_rate)  # Frame rate control

class TemperatureThread(QThread):
    update_temperature_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.bus = smbus2.SMBus(1)
        self.running = True

    def run(self):
        while self.running:
            temperatures = self.read_temperature()
            averages = self.calculate_3x3_averages(temperatures)
            self.update_temperature_signal.emit(averages)
            time.sleep(1)

    def stop(self):
        self.running = False
        self.bus.close()

    def read_temperature(self):
        temp_data = []
        for i in range(AMG8833_PIXEL_ARRAY_SIZE):
            low_byte = self.bus.read_byte_data(AMG8833_I2C_ADDR, 0x80 + i * 2)
            high_byte = self.bus.read_byte_data(AMG8833_I2C_ADDR, 0x80 + i * 2 + 1)
            temperature = (high_byte << 8) | low_byte
            if temperature & 0x800:
                temperature -= 4096
            temperature *= 0.25
            temp_data.append(temperature)
        return temp_data

    def calculate_3x3_averages(self, temperatures):
        if len(temperatures) != AMG8833_PIXEL_ARRAY_SIZE:
            raise ValueError("Temperature data does not contain 64 elements.")
        
        averages = []
        for i in range(0, 8, 3):  # Rows: 0, 3, 6
            row = []
            for j in range(0, 8, 3):  # Columns: 0, 3, 6
                region = []
                for di in range(3):
                    for dj in range(3):
                        if i + di < 8 and j + dj < 8:
                            region.append(temperatures[(i + di) * 8 + (j + dj)])
                avg_temp = sum(region) / len(region)
                row.append(avg_temp)
            averages.append(row)
        
        return averages

class DetectionMonitorThread(QThread):
    update_timer_signal = pyqtSignal(float)

    def __init__(self, servo_thread_x, servo_thread_y, detection_timeout=5, y_initial_angle=90):
        super().__init__()
        self.servo_thread_x = servo_thread_x
        self.servo_thread_y = servo_thread_y
        self.detection_timeout = detection_timeout
        self.y_initial_angle = y_initial_angle
        self.last_detection_time = time.time()
        self.running = True
        self.sweeping = False
        self.x_increasing = True
        self.temperature_thread = None
        self.temperature_data = None

    def update_last_detection_time(self):
        self.last_detection_time = time.time()
        self.sweeping = False

    def run(self):
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.last_detection_time
            self.update_timer_signal.emit(elapsed_time)
            if elapsed_time > self.detection_timeout:
                self.sweeping = True
                print("No detection for 5 seconds. Sweeping to find target.")
                sweep_start_time = time.time()
                while self.sweeping:
                    current_sweep_time = time.time() - sweep_start_time
                    if self.servo_thread_x.angle >= 135:
                        self.x_increasing = False
                    elif self.servo_thread_x.angle <= 45:
                        self.x_increasing = True
                    
                    if self.x_increasing:
                        new_angle_x = self.servo_thread_x.angle + 1
                    else:
                        new_angle_x = self.servo_thread_x.angle - 1

                    self.servo_thread_x.set_angle(new_angle_x)

                    if new_angle_x == 45 or new_angle_x == 135:
                        self.x_increasing = not self.x_increasing

                        # Y축 조정
                        if self.servo_thread_y.angle >= 120:
                            self.servo_thread_y.set_angle(self.servo_thread_y.angle - 5)
                        elif self.servo_thread_y.angle <= 70:
                            self.servo_thread_y.set_angle(self.servo_thread_y.angle + 5)

                    time.sleep(0.1)  # Small delay to allow the servo to move

                    # 10초가 넘으면 온도 데이터 기반 서보 모터 제어 시작
                    if current_sweep_time > 10:
                        self.sweeping = False
                        self.temperature_based_sweeping()

            time.sleep(1)  # Check every second

    def temperature_based_sweeping(self):
        print("No detection for an additional 10 seconds. Using temperature data to find target.")
        temperature_sweep_start_time = time.time()
        while True:
            current_temp_sweep_time = time.time() - temperature_sweep_start_time

            highest_temp = -float('inf')
            highest_pos = (1, 1)
            for i, row in enumerate(self.temperature_data):
                for j, temp in enumerate(row):
                    if temp > highest_temp:
                        highest_temp = temp
                        highest_pos = (i, j)

            # 온도 값에 따른 X축 제어
            if highest_pos[1] == 0:  # 왼쪽
                new_angle_x = max(45, self.servo_thread_x.angle - 5)
            elif highest_pos[1] == 2:  # 오른쪽
                new_angle_x = min(135, self.servo_thread_x.angle + 5)
            else:  # 중앙
                new_angle_x = self.servo_thread_x.angle

            self.servo_thread_x.set_angle(new_angle_x)

            if current_temp_sweep_time > 10:
                break

            time.sleep(0.1)  # Small delay to allow the servo to move

        # After temperature-based sweeping, return to regular sweeping
        self.last_detection_time = time.time()  # Reset timer to switch back to regular sweeping

    def stop_sweeping(self):
        self.sweeping = False

    def set_temperature_data(self, temperature_data):
        self.temperature_data = temperature_data

class App(QWidget):
    def __init__(self, camera_test_only, use_coral_tpu, frame_rate, y_initial_angle):
        super().__init__()

        self.camera_test_only = camera_test_only

        if camera_test_only:
            self.yolo_detector = None
        else:
            self.yolo_detector = YoloDetectorWrapper(args.model, use_coral_tpu)

        self.setWindowTitle("Qt UI")
        self.disply_width = 800
        self.display_height = 800
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.fps_util = SimpleFPS()

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        self.thread = VideoThreadPiCam(frame_rate, self.disply_width, self.display_height)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # PyFirmata2 설정
        self.board = pyfirmata2.Arduino('/dev/ttyACM0')
        self.servo_thread_x = ServoThread(self.board, 9, min_angle=45, max_angle=135)  # X축 서보 모터 각도 제한
        self.servo_thread_y = ServoThread(self.board, 10, initial_angle=y_initial_angle)  # y축 초기 각도를 변수로 설정

        # Detection monitor thread 설정
        self.detection_monitor_thread = DetectionMonitorThread(self.servo_thread_x, self.servo_thread_y, y_initial_angle=y_initial_angle)
        self.detection_monitor_thread.update_timer_signal.connect(self.print_elapsed_time)
        self.detection_monitor_thread.start()

        # Temperature thread 설정
        self.temperature_thread = TemperatureThread()
        self.temperature_thread.update_temperature_signal.connect(self.update_temperature_display)
        self.temperature_thread.start()

        self.temperatures = None

    @pyqtSlot(float)
    def print_elapsed_time(self, elapsed_time):
        print(f"Time since last detection: {elapsed_time:.2f} seconds")

    @pyqtSlot(list)
    def update_temperature_display(self, temperatures):
        self.temperatures = temperatures
        self.detection_monitor_thread.set_temperature_data(temperatures)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        if self.yolo_detector is None:
            display_img = cv_img
        else:
            results = self.yolo_detector.predict(cv_img)
            display_img = draw_annotation(cv_img, self.yolo_detector.get_label_names(), results, conf_threshold=0.45, target_class=0)

        # Draw horizontal and vertical lines at the center of the screen
        draw_center_lines(display_img)

        # Draw rectangles for various ranges
        draw_ranges(display_img)

        person_detected = False
        person_count = 0  # Initialize person count
        largest_person_center_x = None
        largest_person_center_y = None
        largest_person_size = 0  # Initialize largest person size
        if results:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if conf > 0.45 and cls == 0:
                        person_detected = True
                        person_count += 1
                        label = f"Person {person_count}"
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        obj_center_x = (x1 + x2) // 2
                        obj_center_y = (y1 + y2) // 2
                        adjusted_center_y = obj_center_y - int((y2 - y1) * 0.1)  # Adjusted center_y to be 10% above the object's center
                        person_size = (x2 - x1) * (y2 - y1)  # Calculate the size of the person

                        if person_size > largest_person_size:
                            largest_person_size = person_size
                            largest_person_center_x = obj_center_x
                            largest_person_center_y = adjusted_center_y

                        draw_person_label(display_img, obj_center_x, y1, y2, adjusted_center_y, label, x1, x2)

        if largest_person_center_x is not None and largest_person_center_y is not None:
            diff_x = self.disply_width // 2 - largest_person_center_x
            diff_y = self.display_height // 2 - largest_person_center_y

            # 서보 모터 제어 (X축만)
            if not (self.disply_width * 0.40 < largest_person_center_x < self.disply_width * 0.60):
                new_angle_x = self.servo_thread_x.angle + (2 if diff_x > 0 else -2)
                self.servo_thread_x.set_angle(new_angle_x)
            else:
                new_angle_x = self.servo_thread_x.angle + (1 if diff_x > 0 else -1)
                self.servo_thread_x.set_angle(new_angle_x)

            # Y축 제어 조건 추가
            if largest_person_center_y > self.display_height * 0.75:
                new_angle_y = self.servo_thread_y.angle + 5
                self.servo_thread_y.set_angle(new_angle_y)
            elif largest_person_center_y < self.display_height * 0.25:
                new_angle_y = self.servo_thread_y.angle - 5
                self.servo_thread_y.set_angle(new_angle_y)

            if person_detected:
                self.detection_monitor_thread.update_last_detection_time()
                self.detection_monitor_thread.stop_sweeping()  # Stop the sweeping

        if self.temperatures:
            draw_temperature_grid(display_img, self.temperatures)

        fps, _ = self.fps_util.get_fps()
        draw_fps(display_img, fps)

        qt_img = self.convert_cv_qt(display_img)
        self.image_label.setPixmap(qt_img)

        # Signal to the thread to grab the next frame only after processing is done
        self.thread.grab_frame = True

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

def run_no_ui(use_coral_tpu, frame_rate, y_initial_angle):
    yolo_detector = YoloDetectorWrapper(args.model, use_coral_tpu)
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(main={"size": (800, 800), "format": "RGB888"}, raw={"size": (800, 800)})
    picam2.configure(camera_config)
    picam2.start()

    fps_util = SimpleFPS()

    # PyFirmata2 설정
    board = pyfirmata2.Arduino('/dev/ttyACM0')
    servo_thread_x = ServoThread(board, 9, min_angle=45, max_angle=135)  # X축 서보 모터 각도 제한
    servo_thread_y = ServoThread(board, 10, initial_angle=y_initial_angle)  # y축 초기 각도를 변수로 설정

    detection_monitor_thread = DetectionMonitorThread(servo_thread_x, servo_thread_y, y_initial_angle=y_initial_angle)
    detection_monitor_thread.update_timer_signal.connect(print_elapsed_time)
    detection_monitor_thread.start()

    temperature_thread = TemperatureThread()
    temperature_thread.update_temperature_signal.connect(update_temperature_display)
    temperature_thread.start()

    while True:
        frame = picam2.capture_array()
        results = yolo_detector.predict(frame)
        frame = draw_annotation(frame, yolo_detector.get_label_names(), results, conf_threshold=0.45, target_class=0)

        # Draw horizontal and vertical lines at the center of the screen
        draw_center_lines(frame)

        # Draw rectangles for various ranges
        draw_ranges(frame)

        person_detected = False
        person_count = 0  # Initialize person count
        largest_person_center_x = None
        largest_person_center_y = None
        largest_person_size = 0  # Initialize largest person size
        if results:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if conf > 0.45 and cls == 0:
                        person_detected = True
                        person_count += 1
                        label = f"Person {person_count}"
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        obj_center_x = (x1 + x2) // 2
                        obj_center_y = (y1 + y2) // 2
                        adjusted_center_y = obj_center_y - int((y2 - y1) * 0.1)  # Adjusted center_y to be 10% above the object's center
                        person_size = (x2 - x1) * (y2 - y1)  # Calculate the size of the person

                        if person_size > largest_person_size:
                            largest_person_size = person_size
                            largest_person_center_x = obj_center_x
                            largest_person_center_y = adjusted_center_y

                        draw_person_label(frame, obj_center_x, y1, y2, adjusted_center_y, label, x1, x2)

        if largest_person_center_x is not None and largest_person_center_y is not None:
            diff_x = 800 // 2 - largest_person_center_x
            diff_y = 800 // 2 - largest_person_center_y

            # 서보 모터 제어 (X축만)
            if not (800 * 0.40 < largest_person_center_x < 800 * 0.60):
                new_angle_x = servo_thread_x.angle + (2 if diff_x > 0 else -2)
                servo_thread_x.set_angle(new_angle_x)
            else:
                new_angle_x = servo_thread_x.angle + (1 if diff_x > 0 else -1)
                servo_thread_x.set_angle(new_angle_x)

            # Y축 제어 조건 추가
            if largest_person_center_y > 800 * 0.75:
                new_angle_y = servo_thread_y.angle - 5
                servo_thread_y.set_angle(new_angle_y)
            elif largest_person_center_y < 800 * 0.25:
                new_angle_y = servo_thread_y.angle + 5
                servo_thread_y.set_angle(new_angle_y)

            if person_detected:
                detection_monitor_thread.update_last_detection_time()
                detection_monitor_thread.stop_sweeping()  # Stop the sweeping

        if hasattr(App, 'temperatures'):
            draw_temperature_grid(frame, App.temperatures)

        fps, is_updated = fps_util.get_fps()
        if is_updated:
            print(fps)
        time.sleep(1.0 / frame_rate)  # Frame rate control

@pyqtSlot(float)
def print_elapsed_time(elapsed_time):
    print(f"Time since last detection: {elapsed_time:.2f} seconds")

@pyqtSlot(list)
def update_temperature_display(temperatures):
    App.temperatures = temperatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/yolov8s_full_integer_quant_edgetpu_192.tflite")  # Change to yolov8s model
    parser.add_argument('--camera_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--coral_tpu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--frame_rate', type=int, default=15, help='Frame rate for capturing video')  # Set default frame rate to 15
    parser.add_argument('--y_initial_angle', type=int, default=90, help='Initial angle for Y-axis servo motor')  # Initial angle for Y-axis

    args = parser.parse_args()

    if args.debug or args.camera_test:
        app = QApplication(sys.argv)
        a = App(camera_test_only=args.camera_test, use_coral_tpu=args.coral_tpu, frame_rate=args.frame_rate, y_initial_angle=args.y_initial_angle)
        a.show()
        sys.exit(app.exec_())
    else:
        run_no_ui(args.coral_tpu, args.frame_rate, args.y_initial_angle)
