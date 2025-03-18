import os
import requests
import torch
from torchvision import models

MODEL_URL = "https://huggingface.co/modelo_pytorch.pth/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

# Descargar el modelo si no existe localmente
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("Descargando el modelo desde Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Modelo descargado exitosamente.")

# Cargar el modelo con Torch
def load_model():
    download_model()

    # Asegurarse de que el archivo tiene un tamaño correcto antes de cargar
    if os.path.getsize(MODEL_PATH) < 1000:  # Ajusta este valor si es necesario
        raise RuntimeError("El archivo del modelo parece estar corrupto o vacío.")

    modelo = models.resnet34(pretrained=False)  # Usa la misma arquitectura
    modelo.fc = torch.nn.Linear(512, 10)  # Ajusta el número de clases

    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True))
    modelo.eval()
    return modelo

modelo = load_model()


# Transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Clasificación de Imágenes con PyTorch")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Aplicar transformaciones
    image_tensor = transform(image).unsqueeze(0)  # Agregar batch dimension

    # Hacer predicción
    with torch.no_grad():
        output = modelo(image_tensor)
        pred = output.argmax(dim=1).item()

    st.write(f"Predicción del modelo: {pred}")
