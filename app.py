import streamlit as st
from fastai.vision.all import *
import json
import requests
from pathlib import Path

# URLs de Hugging Face
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model_fastai.pkl"
CLASSES_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/clases.json"

# Rutas locales temporales
MODEL_PATH = Path("best_model_fastai.pkl")
CLASSES_PATH = Path("clases.json")

# Descargar archivos si no existen
if not MODEL_PATH.exists():
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

if not CLASSES_PATH.exists():
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        f.write(requests.get(CLASSES_URL).text)

# Cargar etiquetas
with open(CLASSES_PATH) as f:
    class_names = json.load(f)

# Cargar el modelo entrenado en FastAI
learn = load_learner(MODEL_PATH)

# Interfaz Streamlit
st.title("🎨 Clasificador de Estilos Artísticos")
st.markdown("Sube una imagen de una obra de arte y el modelo predecirá su estilo.")

uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption="Imagen subida", use_column_width=True)

    with st.spinner("Clasificando..."):
        pred_class, pred_idx, outputs = learn.predict(img)
        st.success(f"Predicción: **{pred_class}**")
