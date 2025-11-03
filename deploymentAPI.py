import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

# === Inisialisasi FastAPI ===
app = FastAPI(
    title="MobileViT Object Detection API (CPU)",
    description="Deteksi permukaan jalan menggunakan MobileViT (ONNX, CPU only)",
    version="1.0"
)

# === Load Model ONNX (CPU Only) ===
onnx_model_path = "D:/datasetitsar/model/model11/model1.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
print("✅ Model loaded on CPU. Execution Providers:", session.get_providers())

# === Daftar kelas ===
classes = ["bump", "lantai", "paving", "tangga"]
confidence_threshold = 0.3  # Minimum probabilitas agar deteksi diterima

# === Fungsi preprocessing ===
def preprocess_image(image: Image.Image):
    """
    Mengubah gambar menjadi tensor sesuai input model (NCHW format, float32, normalisasi 0–1).
    """
    image = image.convert("RGB")
    resized = image.resize((256, 256))
    img_array = np.array(resized).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return input_tensor

# === Fungsi prediksi ===
def predict(image_tensor):
    """
    Jalankan inferensi ONNX Runtime pada tensor input.
    """
    inputs = {session.get_inputs()[0].name: image_tensor}
    outputs = session.run(None, inputs)
    probabilities = outputs[0]  # Output model (probabilitas per kelas)
    pred_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    return pred_class, confidence

# === Endpoint untuk prediksi gambar ===
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Membaca file gambar dari request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess dan inferensi
        image_tensor = preprocess_image(image)
        pred_class, confidence = predict(image_tensor)

        # Hasil
        if confidence < confidence_threshold:
            result = {"class": "Unknown", "confidence": round(confidence, 4)}
        else:
            result = {"class": classes[pred_class], "confidence": round(confidence, 4)}

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Endpoint root ===
@app.get("/")
def root():
    return {"message": "MobileViT Object Detection API (CPU) is running!"}

# === Jalankan server ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
