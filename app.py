import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Cargar el modelo de PyTorch
@st.cache_resource()
def load_model():
    modelo = models.resnet34(pretrained=False)  # Usa la misma arquitectura que en FastAI
    modelo.load_state_dict(torch.load("modelo_pytorch.pth", map_location=torch.device("cpu")))
    modelo.eval()
    return modelo

modelo = load_model()

# Transformaciones de preprocesamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("Clasificación de Obras de Arte")
st.write("Sube una imagen para clasificarla según su estilo artístico")

uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    imagen = Image.open(uploaded_file)
    st.image(imagen, caption="Imagen subida", use_column_width=True)
    
    # Preprocesar la imagen
    imagen = transform(imagen).unsqueeze(0)  # Añadir dimensión de batch
    
    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(imagen)
    
    clase_predicha = torch.argmax(salida, dim=1).item()
    st.write(f"Predicción: Clase {clase_predicha}")
