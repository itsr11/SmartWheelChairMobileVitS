import torch
from torch.utils.data import DataLoader
from transformers import MobileViTForImageClassification
from image_processing import get_transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score

# Muat model yang telah dilatih
model = MobileViTForImageClassification.from_pretrained('D:/datasetitsar/model/model7')
model.eval()

# Muat dataset pengujian
test_data = datasets.ImageFolder(root='dataset/test', transform=get_transforms())
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

def evaluate():
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs.logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()
