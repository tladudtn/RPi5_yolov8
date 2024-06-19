import torch
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops  # for postprocess
from pathlib import Path
import cv2
import numpy as np
import logging

try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
except ModuleNotFoundError as m_err:
    pass

# .pt files contains names in there but exported onnx/tflite don't have them.
yolo_default_label_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                            7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                            12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
                            24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                            53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                            71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class YoloDetector:
    def __init__(self, model_path, task='detect', debug=False, debug_file='debug.log'):
        self.model = YOLO(model_path, task=task)
        self.debug = debug
        self.debug_file = debug_file
        self.imgsz = 640  # assume 640 at the moment since it is the default one

        if model_path.suffix == '.onnx':
            import onnx
            dummy_model = onnx.load(str(model_path))
            self.imgsz = dummy_model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
            del dummy_model

        if self.debug:
            logging.basicConfig(filename=self.debug_file, level=logging.DEBUG)

    def predict(self, frame, conf):
        if self.debug:
            logging.debug(f"Predicting with frame shape: {frame.shape} and confidence threshold: {conf}")
        predictions = self.model.predict(source=frame, save=False, conf=conf, save_txt=False, show=False, verbose=False, imgsz=self.imgsz)
        if self.debug:
            logging.debug(f"Predictions: {predictions}")
        return predictions

    def get_label_names(self):
        if self.model.names is None or len(self.model.names) == 0:
            return yolo_default_label_names
        return self.model.names

class YoloDetectorTFLite:
    def __init__(self, model_path, use_coral_tpu=False, debug=False, debug_file='debug.log'):
        self.name = model_path.name
        self.debug = debug
        self.debug_file = debug_file
        self.use_coral_tpu = use_coral_tpu

        if use_coral_tpu:
            self.interpreter = make_interpreter(str(model_path))
        else:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        # Get input details to determine the required input shape and type
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']  # [1, height, width, 3]
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']

        if self.debug:
            logging.basicConfig(filename=self.debug_file, level=logging.DEBUG)

    def preprocess(self, frame):
        # Resize frame to the model's expected input shape
        input_img = cv2.resize(frame, (self.input_width, self.input_height))
        input_img = input_img[np.newaxis, ...]  # Add batch dimension

        if self.use_coral_tpu:
            params = common.input_details(self.interpreter, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']
            input_mean = 128
            input_std = 128
            normalized_input = (input_img - input_mean) / input_std
            normalized_input = normalized_input / scale + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            return normalized_input.astype(self.input_dtype)
        else:
            input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            return input_img

    def predict(self, frame, conf):
        orig_imgs = [frame]

        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        if self.debug:
            logging.debug(f"Input shape: {self.input_shape}, Input image shape: {input_data.shape}")

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(self.output_details[0]['index'])
        if self.debug:
            logging.debug(f"Raw predictions shape: {preds.shape}")

        if self.use_coral_tpu:
            output_details = self.interpreter.get_output_details()[0]
            if np.issubdtype(preds.dtype, np.integer):
                scale, zero_point = output_details['quantization']
                preds = scale * (preds.astype(np.int64) - zero_point)
                preds = preds.astype(np.float32)

        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(preds, conf, 0.7, agnostic=False, max_det=300, classes=None)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] *= self.input_width
            pred[:, :4] = ops.scale_boxes(input_data.shape[1:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(Results(orig_img, path=img_path, names=yolo_default_label_names, boxes=pred))

        if self.debug:
            logging.debug(f"Processed results: {results}")

        return results

    def get_label_names(self):
        return yolo_default_label_names

class YoloDetectorWrapper:
    def __init__(self, model_path, use_coral_tpu=False, debug=False, debug_file='debug.log'):
        model_path = Path(model_path)
        self.debug = debug
        self.debug_file = debug_file

        if use_coral_tpu or model_path.suffix == '.tflite':
            self.detector = YoloDetectorTFLite(model_path, use_coral_tpu, debug, debug_file)
        else:
            self.detector = YoloDetector(model_path, debug=debug, debug_file=debug_file)

    def predict(self, frame, conf=0.5):
        if self.debug:
            logging.debug(f"Wrapper predict called with frame shape: {frame.shape} and confidence: {conf}")
        return self.detector.predict(frame, conf=conf)

    def get_label_names(self):
        return self.detector.get_label_names()
