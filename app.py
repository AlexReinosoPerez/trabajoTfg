import torch
import torchvision.models as models
import streamlit as st
from torchvision import transforms
from PIL import Image
import requests

# URL del modelo en Hugging Face
MODEL_URL = "https://huggingface.co/tu_usuario/tu_repositorio/resolve/main/modelo_pytorch.pth"
MODEL_PATH = "modelo_pytorch.pth"

# Descargar el modelo si no existe localmente
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Descargando el modelo desde Hugging Face...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.write("Modelo descargado exitosamente.")

@st.cache_resource()
def load_model():
    # Descargar el modelo si no está en la carpeta
    download_model()

    # Crear el modelo con la misma arquitectura que usó FastAI
    modelo = models.resnet34(pretrained=False)
    modelo.fc = torch.nn.Linear(512, 10)  # Ajusta el número de salidas

    # Cargar los pesos guardados en modelo_pytorch.pth
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    modelo.eval()
    return modelo

modelo = load_model()

# Transformaciones de preprocesamiento (iguales a las usadas en el entrenamiento)
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
