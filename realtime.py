import cv2
import torch
from collections import deque
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification

# Muat model yang telah dilatih dan prosesor gambar
model = MobileViTForImageClassification.from_pretrained('D:/datasetitsar/model/model8')
processor = MobileViTFeatureExtractor(size=256)
model.eval()

# Daftar kelas sesuai urutan yang digunakan saat pelatihan
#classes = ["lantai", "paving", "bump", "tangga"]
classes = ["bump", "lantai", "paving", "tangga"]

# Buka file video
#video_path = "D:/gen5/Iwan/IPS_2024-11-14.14.51.42.5200.mp4"  # paving dan bump
#video_path = "C:/Users/ThinkCentre/Downloads/1117.mp4"
cap = cv2.VideoCapture(0)

# Buffer untuk menyimpan prediksi sebelumnya
buffer_size = 8  # jumlah frame yang akan digunakan untuk stabilisasi
pred_buffer = deque(maxlen=buffer_size)  # buffer prediksi
confidence_threshold = 0.4  # threshold confidence

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
    inputs = processor(images=cropped_frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        max_prob, pred_class = torch.max(probabilities, dim=-1)
        max_prob = max_prob.item()
        pred_class = pred_class.item()

    # Jika confidence di atas threshold, tambahkan ke buffer
    if max_prob >= confidence_threshold:
        pred_buffer.append(pred_class)

    # Tentukan label berdasarkan prediksi yang paling sering muncul dalam buffer
    if len(pred_buffer) > 0:
        stable_class = max(set(pred_buffer), key=pred_buffer.count)  # Prediksi paling sering
        label = classes[stable_class]
    else:
        label = "Unknown"

    print("Predicted label:", label)

    # Tampilkan hasil prediksi pada bounding box
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    color = (0, 0, 255)  # BGR format (merah)
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
    cv2.imshow('Deteksi Video', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
