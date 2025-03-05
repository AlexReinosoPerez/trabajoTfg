import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# 📌 Clases del modelo (Asegúrate de que coincidan con las del entrenamiento)
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

# 📌 Función para cargar el modelo desde GitHub
@st.cache_resource
def load_model():
    model_path = "best_model.pth"

    # 📌 Si el modelo no está en local, descargarlo de GitHub
    if not os.path.exists(model_path):
        url = "https://raw.githubusercontent.com/TU-USUARIO/TU-REPO/main/best_model.pth"
        st.write("Descargando el modelo...")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    # 📌 Cargar modelo ResNet50 sin pesos preentrenados
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features

    # 📌 Redefinir la capa final para 4 clases
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(class_names))  # Número de clases
    )

    # 📌 Cargar pesos del modelo entrenado
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Poner en modo evaluación

    return model

model = load_model()

# 📌 Transformaciones para preprocesar la imagen antes de la predicción
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar
])

# 📌 Interfaz de Streamlit
st.title("Clasificación de Estilos Artísticos")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
image_url = st.text_input("O introduce una URL de imagen:")

# 📌 Procesar la imagen
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

elif image_url:
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Imagen cargada desde URL", use_column_width=True)
    except Exception as e:
        st.error("Error al cargar la imagen desde la URL.")

# 📌 Clasificación de imagen cuando el usuario presione el botón
if image and st.button("Clasificar Imagen"):
    img_tensor = transform(image).unsqueeze(0)  # Convertir imagen en tensor con batch=1

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    st.write(f"### 🎨 Predicción: {predicted_class}")
