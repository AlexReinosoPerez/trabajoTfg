import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import requests
from pathlib import Path

# 🚀 Configuración de la app
st.set_page_config(page_title="Clasificador de Pinturas", layout="centered")
st.title("🎨 Clasificador de estilos artísticos")
st.write("Sube una imagen para predecir su estilo artístico.")

# 📦 Descargar modelo desde Hugging Face si no está localmente
model_file = Path("modelo_fastai_pytorch.pth")
if not model_file.exists():
    with st.spinner("🔄 Descargando modelo desde Hugging Face..."):
        url = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/modelo_fastai_pytorch.pth"  # 🟡 REEMPLAZAR
        r = requests.get(url)
        model_file.write_bytes(r.content)
        st.success("Modelo descargado correctamente ✅")

# 📂 Cargar etiquetas de clase
with open("clases.json", "r") as f:
    class_names = json.load(f)

# 💻 Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🧠 Cargar modelo
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(model_file, map_location=device))
model.to(device)
model.eval()

# 🔍 Función de predicción
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

# 📷 Interfaz de usuario
uploaded_file = st.file_uploader("📁 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    with st.spinner("🔍 Clasificando..."):
        label = predict(image)
        st.success(f"🎯 Estilo artístico predicho: **{label}**")
