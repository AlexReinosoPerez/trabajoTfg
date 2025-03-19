import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# 📌 Dispositivo (Usa GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Cargar modelo ResNet50
def load_model():
    modelo = models.resnet50(pretrained=False)  # Crear modelo base
    num_ftrs = modelo.fc.in_features
    modelo.fc = nn.Linear(num_ftrs, 5)  # Ajusta al número de clases (cambia 5 por el número real)
    
    # Cargar pesos entrenados en FastAI convertidos a PyTorch
    modelo.load_state_dict(torch.load("modelo_fastai_pytorch.pth", map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo

modelo = load_model()

# 📌 Transformaciones de imagen compatibles con PyTorch
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 📌 Cargar etiquetas de clases
with open("clases.json", "r") as f:
    clases = json.load(f)

# 📌 Función de predicción
def predecir_imagen(imagen_path):
    imagen = Image.open(imagen_path).convert("RGB")
    imagen = transformaciones(imagen)
    imagen = imagen.unsqueeze(0).to(device)  # Añadir batch dimension

    with torch.no_grad():
        salida = modelo(imagen)
        probabilidades = torch.nn.functional.softmax(salida[0], dim=0)
        clase_predicha = probabilidades.argmax().item()
    
    etiqueta_predicha = clases[str(clase_predicha)]
    confianza = probabilidades[clase_predicha].item()
    
    return etiqueta_predicha, confianza

# 📌 Prueba con una imagen
if __name__ == "__main__":
    imagen_prueba = "imagen_test.jpg"  # Cambia por una imagen válida
    clase, confianza = predecir_imagen(imagen_prueba)
    print(f"Predicción: {clase} ({confianza*100:.2f}%)")
