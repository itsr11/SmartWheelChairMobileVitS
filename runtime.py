import cv2
#import serial
import numpy as np
import onnxruntime as ort
from collections import deque
import time

# Muat model ONNX dengan GPUExecutionProvider
onnx_model_path = "D:/datasetitsar/model/model11/model1.onnx"  # Ganti dengan path model ONNX Anda
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print("Execution providers:", session.get_providers())

# Konfigurasi serial Arduino
#arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)

# Daftar kelas sesuai urutan yang digunakan saat pelatihan
classes = ["bump", "lantai", "paving", "tangga"]  # Deteksi asli
#arduino_classes = [1, 3, 0, 2]  # Mapping ke kelas Arduino

# Buka file video atau kamera
cap = cv2.VideoCapture(0)  # Ganti dengan path video Anda jika menggunakan video

# Buffer untuk stabilisasi prediksi
buffer_size = 8  # jumlah frame yang akan digunakan untuk stabilisasi
pred_buffer = deque(maxlen=buffer_size)  # buffer prediksi
confidence_threshold = 0.3  # threshold confidence

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tentukan bounding box pada bagian bawah frame
    height, width, _ = frame.shape
    top_left_x = 0
    top_left_y = int(height * 0.50)
    bottom_right_x = width
    bottom_right_y = height

    # Crop frame pada area bounding box
    cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Preprocess cropped frame
    resized_frame = cv2.resize(cropped_frame, (256, 256))  # Resize ke ukuran input model
    input_tensor = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), axis=0).astype(np.float32) / 255.0  # NCHW dan normalisasi

    # Catat waktu sebelum inferensi
    start_time = time.time()

    # Inference dengan ONNX Runtime
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    probabilities = outputs[0]  # Ambil hasil output
    max_prob = np.max(probabilities)
    pred_class = np.argmax(probabilities)

    # Catat waktu setelah inferensi dan hitung waktu komputasi
    end_time = time.time()
    computation_time = end_time - start_time

    # Jika confidence di atas threshold, tambahkan ke buffer
    if max_prob >= confidence_threshold:
        pred_buffer.append(pred_class)

    # Tentukan label berdasarkan prediksi yang paling sering muncul dalam buffer
    if len(pred_buffer) > 7:
        stable_class = max(set(pred_buffer), key=pred_buffer.count)  # Prediksi paling sering
        label = classes[stable_class]
        #arduino_class = arduino_classes[stable_class]  # Mapping ke ID kelas Arduino
    else:
        label = "Unknown"
        arduino_class = None

    # Kirim data ke Arduino jika ada kelas yang terdeteksi
    #if arduino_class is not None:
    #    arduino.write(f"{arduino_class}\n".encode('utf-8'))  # Kirim ID kelas
     #   print(f"Kelas: {label}, ID Arduino: {arduino_class}, Waktu Komputasi: {computation_time:.4f} detik")

    # Tampilkan hasil prediksi pada bounding box
    cv2.putText(frame, f"{label} ({computation_time:.2f}s)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    color = (0, 0, 255)  # BGR format (merah)
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
    cv2.imshow('Deteksi Video', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
