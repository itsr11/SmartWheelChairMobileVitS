import torch
from torch.utils.data import DataLoader, random_split
from transformers import MobileViTForImageClassification, MobileViTFeatureExtractor
from torchvision import datasets, transforms
from config_mobilevit import get_mobilevit_config
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl# Import matplotlib untuk visualisasi\
from sklearn.metrics import recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report  # Import recall dan f1-score
import csv  # Import pustaka CSV

# Tentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Konfigurasi dan inisialisasi model
config = get_mobilevit_config(num_classes=4)
print("Label to ID mapping in config:", config.label2id)
print("ID to Label mapping in config:", config.id2label)
model = MobileViTForImageClassification(config)
model.to(device)  # Pindahkan model ke device

# Hitung dan tampilkan jumlah parameter
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Optimizer dan loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Muat dataset dan terapkan transformasi
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal dengan probabilitas 50%
    transforms.RandomRotation(15),            # Rotasi random dengan derajat ±15
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Crop acak
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_data = datasets.ImageFolder(root='D:/datasetitsar/DatasetLagi-20241112T033305Z-001/DatasetLagi - Copy',
                                 transform=train_transforms)

# Cetak mapping class untuk memastikan urutan yang benar

print("Class to index mapping from dataset:", full_data.class_to_idx)

# Perbarui label2id dan id2label dalam konfigurasi berdasarkan class_to_idx dari dataset
config.label2id = full_data.class_to_idx
config.id2label = {v: k for k, v in config.label2id.items()}

# Pastikan kembali dengan mencetak hasil
print("Updated Label to ID mapping in config:", config.label2id)
print("Updated ID to Label mapping in config:", config.id2label)

# Bagi dataset menjadi train, val, dan test
train_size = int(0.7 * len(full_data))
val_size = int(0.15 * len(full_data))
test_size = len(full_data) - train_size - val_size
train_data, val_data, test_data = random_split(full_data, [train_size, val_size, test_size])

# Buat DataLoader untuk setiap bagian dataset
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


def train(num_epochs=50, patience=5):
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_recalls = []
    val_f1_scores = []

    # Variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Training Epoch {epoch + 1}")

                # Move data to the correct device
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs.logits, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.logits, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=loss.item())

        # Calculate and store average training loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(), tqdm(val_loader, unit="batch") as vepoch:
            for images, labels in vepoch:
                vepoch.set_description(f"Validation Epoch {epoch + 1}")

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.logits, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Collect predictions and labels for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                vepoch.set_postfix(val_loss=loss.item())

        # Calculate and store validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Calculate recall and F1-score
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)

        print(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Recall: {val_recall:.2f}, F1 Score: {val_f1:.2f}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            model.save_pretrained('D:/datasetitsar/model/model12')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Plot training and validation metrics
    plt.figure(figsize=(18, 5))

    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot recall and F1 score
    plt.subplot(1, 3, 3)
    plt.plot(val_recalls, label='Validation Recall')
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Validation Recall and F1 Score')
    plt.legend()

    plt.show()
    print("Training completed. The best model has been saved.")


def test():
    model.eval()  # Set model ke mode evaluasi
    model.to(device)
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []  # Untuk menyimpan semua prediksi
    all_labels = []  # Untuk menyimpan semua label asli

    with torch.no_grad(), tqdm(test_loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description("Testing")

            # Pindahkan data ke device
            images, labels = images.to(device), labels.to(device)

            # Prediksi model
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()

            # Hitung akurasi
            _, predicted = torch.max(outputs.logits, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Simpan prediksi dan label untuk penghitungan metrik
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            tepoch.set_postfix(test_loss=loss.item())

    # Hitung metrik evaluasi
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.id2label.values())
    disp.plot(cmap=mpl.colormaps['viridis'])
    plt.title("Confusion Matrix")
    plt.show()

    # Laporan klasifikasi
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config.id2label.values()))


if __name__ == "__main__":
    train(num_epochs=50, patience=5)
    test()
