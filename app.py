import streamlit as st
from fastai.vision.all import *
import requests
import json
from pathlib import Path

# URLs
CLASSES_URL = "https://raw.githubusercontent.com/AlexReinosoPerez/trabajoTfg/main/clases.json"
MODEL_URL = "https://huggingface.co/AlexReinoso/trabajoTFM/resolve/main/best_model_fastai.pkl"
MODEL_PATH = Path("best_model_fastai.pkl")
CLASSES_PATH = Path("clases.json")

# Descargar clases.json si no existe
if not CLASSES_PATH.exists():
    response = requests.get(CLASSES_URL)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)
    st.success("✅ clases.json descargado correctamente")

# Cargar clases
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Descargar modelo si no existe
if not MODEL_PATH.exists():
    response = requests.get(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("✅ Modelo descargado correctamente")

# Cargar modelo FastAI
learn = load_learner(MODEL_PATH)

# Interfaz Streamlit
st.title("🎨 Clasificador de Estilos Artísticos")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption="Imagen subida", use_column_width=True)

    if st.button("Clasificar"):
        pred, pred_idx, probs = learn.predict(img)
        st.markdown(f"### 🎯 Predicción: `{pred}`")
        st.write("Probabilidades:")
        for c, p in zip(class_names, probs):
            st.write(f"- {c}: {p:.4f}")
