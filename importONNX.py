import torch
from transformers import MobileViTForImageClassification
from safetensors.torch import load_file
import os

# Path ke folder model
model_folder = "D:/datasetitsar/model/model11"  # Path ke folder model
model_path = os.path.join(model_folder, "model.safetensors")  # Path ke SafeTensors
onnx_path = os.path.join(model_folder, "model1.onnx")         # Output path untuk ONNX

# Memuat model dari SafeTensors
print("Memuat model dari SafeTensors...")
state_dict = load_file(model_path)  # Memuat state_dict dari SafeTensors
model = MobileViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path=model_folder,  # Path ke folder konfigurasi model
    state_dict=state_dict                       # Memuat state_dict ke model
)

# Mengatur model ke mode evaluasi
model.eval()

# Dummy input tensor sesuai ukuran input model
dummy_input = torch.randn(1, 3, 256, 256)  # Batch Size, Channels, Height, Width

# Mengekspor model ke format ONNX
print("Mengekspor model ke format ONNX...")
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)  # Membuat folder jika belum ada

torch.onnx.export(
    model,                     # Model PyTorch
    dummy_input,               # Dummy input untuk tracing
    onnx_path,                 # Lokasi file ONNX
    export_params=True,        # Menyimpan parameter model di dalam file ONNX
    opset_version=11,          # Versi opset ONNX
    input_names=['input'],     # Nama tensor input
    output_names=['output'],   # Nama tensor output
    dynamic_axes={             # Dukungan untuk bentuk dinamis pada input/output
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model berhasil diekspor ke format ONNX di {onnx_path}")
