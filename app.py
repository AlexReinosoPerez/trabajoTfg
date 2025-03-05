import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import requests
import os

# Definir las clases del modelo (deben coincidir con las del entrenamiento)
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

# 📌 Función para cargar el modelo de GitHub si no existe
@st.cache_resource
def load_model():
    model_path = "best_model.pth"

    # Descargar el modelo desde GitHub si no está en local
    if not os.path.exists(model_path):
        url = "https://github.com/AlexReinosoPerez/trabajoTfg/bob/main/app.py"
        st.write("Descargando modelo...")
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    # 📌 Definir la arquitectura exacta del modelo (debe coincidir con `train.py`)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(class_names))
    )

    # 📌 Cargar solo el `state_dict`
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

    model.eval()
    return model

# Cargar el modelo
model = load_model()
