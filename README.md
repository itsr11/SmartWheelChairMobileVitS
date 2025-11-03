
# 🦽 Smart Wheelchair: MobileViT-Based Obstacle & Surface Detection System

Sistem **Smart Wheelchair** ini dirancang untuk membantu kursi roda cerdas dalam mengenali kondisi permukaan jalan (seperti *paving*, *tangga*, *polisi tidur*, atau *rintangan lainnya*) secara **real-time**.
Model berbasis **MobileViT-S** dioptimalkan dan dijalankan pada perangkat **Jetson Nano** menggunakan **ONNX Runtime**, menjadikannya ringan, efisien, dan cocok untuk *edge deployment*.

---

## 🚀 Fitur Utama

### 🧠 **1. Object Detection MobileViT**

* Arsitektur model: **MobileViT-S**
* Framework: PyTorch → ekspor ke ONNX Runtime
* Optimizer: **AdamW**
* Input fleksibel dari kamera USB atau CSI Jetson
* Mendeteksi kondisi permukaan:
  🧱 *Paving*, 🪜 *Tangga*, 🚧 *Polisi Tidur*, 🔳 *Ubin*
* Output berupa label, bounding box, dan confidence score.

### ⚡ **2. Optimized for Jetson Nano**

* Model dikonversi ke **ONNX** dan dapat dipercepat dengan:

  * **ONNX Runtime GPU (TensorRT)**
  * **CUDA Provider** atau fallback ke CPU
* Ukuran model kecil (~30–50 MB) dengan latensi inferensi <100 ms/frame

### 🎥 **3. Realtime Detection**

* Deteksi langsung melalui kamera kursi roda
* Prediksi stabil dengan buffer frame
* Visualisasi bounding box langsung di video feed

---

## 🧩 Arsitektur Sistem

```plaintext
[Camera Input]
      │
      ▼
[Jetson Nano - ONNX Runtime]
  ├── Preprocessing (OpenCV)
  ├── Inference (MobileViT-S ONNX)
  ├── Postprocessing (Bounding Box + Label)
      │
      ▼
[Output Stream / Motor Controller / Arduino Signal]
```

---

## 📂 Struktur Direktori

```
SmartWheelchair/
│
├── models/
│   └── mobilevit_s.onnx            # Model konversi untuk Jetson Nano
│
├── dataset/
│   ├── paving.jpg
│   ├── tangga.jpg
│   ├── polisi_tidur.jpg
│
├── src/
│   ├── model.py                    # Arsitektur MobileViT-S
│   ├── train_model.py              # Training model
│   ├── evaluate.py                 # Evaluasi hasil pelatihan
│   ├── importONNX.py               # Konversi ke ONNX
│   ├── runtime.py                  # Inference di PC
│   └── realtime_jetson.py          # Deteksi realtime di Jetson Nano
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi di Jetson Nano

### 1️⃣ Update dan Install Dependensi

```bash
sudo apt-get update
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install torch torchvision onnxruntime-gpu opencv-python timm numpy
```

### 2️⃣ Clone Repository

```bash
git clone https://github.com/itsar-ai/smart-wheelchair-vision.git
cd smart-wheelchair-vision
```

### 3️⃣ Jalankan Model

```bash
python3 src/realtime_jetson.py
```

---

## 🧠 Pelatihan Model di PC (opsional)

### 1️⃣ Jalankan Training

```bash
python src/train_model.py
```

### 2️⃣ Konversi ke ONNX

```bash
python src/importONNX.py
```

### 3️⃣ Tes Model di PC

```bash
python src/runtime.py
```

---

## 🎥 Realtime Detection di Jetson Nano

Script `realtime_jetson.py` akan membuka kamera, melakukan inferensi, dan menampilkan hasil deteksi:

```python
python3 src/realtime_jetson.py
```

Keluaran (contoh di terminal):

```
[INFO] Detected: Tangga (0.89)
[INFO] Frame time: 0.072s
```

Output visual:
🖼️ Video feed dengan bounding box merah dan label deteksi.

---

## 🔧 Contoh Kode Inferensi (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np
import cv2

# Load model ONNX
session = ort.InferenceSession("mobilevit_s.onnx", providers=["CPUExecutionProvider"])

# Preprocess image
frame = cv2.imread("test.jpg")
img = cv2.resize(frame, (256, 256))
input_tensor = np.expand_dims(np.transpose(img, (2,0,1)), axis=0).astype(np.float32) / 255.0

# Run inference
outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
print("Predicted:", np.argmax(outputs[0]))
```

---

## ⚡ Performa di Jetson Nano

| Mode           | FPS        | Power | Precision |
| -------------- | ---------- | ----- | --------- |
| CPU            | ~8 FPS     | 5W    | FP32      |
| GPU (TensorRT) | ~20–25 FPS | 10W   | FP16      |
| INT8 Optimized | ~30 FPS    | 10W   | INT8      |

---

## 🔋 Integrasi ke Sistem Kursi Roda

Sistem ini dapat dihubungkan dengan **Arduino** atau **Raspberry Pi** melalui port serial:

* Jika terdeteksi *tangga* → kursi berhenti
* Jika *paving / datar* → lanjut jalan
* Jika *polisi tidur* → perlambat laju motor

Contoh:

```python
# arduino.write(f"{arduino_class}\n".encode('utf-8'))
```

---

## 📊 Teknologi yang Digunakan

| Komponen          | Teknologi             |
| ----------------- | --------------------- |
| Framework         | PyTorch, ONNX Runtime |
| Model             | MobileViT-S           |
| Device            | NVIDIA Jetson Nano    |
| Optimizer         | AdamW                 |
| Vision            | OpenCV                |
| Deployment        | TensorRT (opsional)   |
| Annotation Format | Pascal VOC (XML)      |

---

## 📚 Referensi

1. **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer**
   (Apple Machine Learning Research, 2021)
2. **Detection Transformer (DETR)**
   (Facebook AI Research, 2020)
3. **Jetson Nano Developer Guide** — NVIDIA
4. **ONNX Runtime for Edge AI** — Microsoft

---

