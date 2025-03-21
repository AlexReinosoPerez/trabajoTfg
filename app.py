import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Cargar etiquetas
with open("clases.json", "r") as f:
    class_names = json.load(f)

# Dispositivo de ejecución
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el modelo
model = models.resnet50(pretrained=False)  # No usamos pesos preentrenados
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))  # Adaptar salida al número de clases
model.load_state_dict(torch.load("modelo_fastai_pytorch.pth", map_location=device))
model.to(device)
model.eval()

# Función para hacer una predicción
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
    
    return label

# Ejemplo de uso
if __name__ == "__main__":
    img_path = "ejemplo.jpg"  # Ruta de la imagen de prueba
    resultado = predict(img_path)
    print(f"Predicción: {resultado}")
