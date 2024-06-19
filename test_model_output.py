from tflite_runtime.interpreter import Interpreter, load_delegate

# Edge TPU delegate 로드
interpreter = Interpreter(
    model_path='./models/yolov8n_full_integer_quant_edgetpu_192x192.tflite',
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# 입력 및 출력 세부 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
