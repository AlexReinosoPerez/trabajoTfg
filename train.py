import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import os
from sklearn.metrics import classification_report

# 🔧 Parámetros
num_epochs = 100
learning_rate = 0.00005
batch_size = 64
patience = 5

train_dir = 'C:/Users/User/Desktop/imagenes_descargadas/train'
val_dir = 'C:/Users/User/Desktop/imagenes_descargadas/val'

def get_class_distribution(dataset_path):
    return {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in os.listdir(dataset_path)}

if __name__ == '__main__':
    print("🔍 Calculando distribución de clases...")
    train_class_counts = get_class_distribution(train_dir)
    val_class_counts = get_class_distribution(val_dir)

    print(f"📊 Entrenamiento: {train_class_counts}")
    print(f"📊 Validación: {val_class_counts}")

    print("\n⚙️ Configurando transformaciones y DataLoader...")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("✅ Transformaciones listas.")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    print("📂 Calculando pesos balanceados para las clases...")

    class_sample_counts = np.array([train_class_counts[cls] for cls in train_dataset.classes])
    weights = 1.0 / (class_sample_counts + 1e-3)
    samples_weights = np.array([weights[label] for _, label in train_dataset.samples])

    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    print("🛠️ Cargando DataLoaders...")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("✅ DataLoaders listos.")

    # 🔥 Cargar modelo preentrenado
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 4)
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = True
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Entrenando en: {device}")
    model.to(device)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_acc = 0.0
    patience_counter = 0

    print("🏋️ Iniciando entrenamiento...")

    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss.mean()  # ✅ Aseguramos que la pérdida sea escalar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 📊 Validación
        model.eval()
        val_loss, correct = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss.mean()  # ✅ Igual aquí
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 🔍 Revisar distribución de `softmax`
                probs = torch.softmax(outputs, dim=1)
                print("📊 Probabilidades de predicción:", probs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset) * 100
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f"📊 Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        scheduler.step(val_loss)

        if accuracy > best_acc:
            best_acc = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early Stopping activado.")
                break

    print("\n📊 Generando reporte de clasificación...")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    torch.save(model.state_dict(), 'best_model.pth')
    print("✅ Modelo guardado correctamente.")
