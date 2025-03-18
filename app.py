from pathlib import Path, PosixPath
import streamlit as st
import os
import torch
import pickle
from fastai.vision.all import load_learner
from huggingface_hub import hf_hub_download

HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model_fastai.pkl"

@st.cache_resource
def load_model():
    os.system("pip install --upgrade fastcore==1.5.29 fastai==2.7.12")  # Asegura compatibilidad
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, force_download=True)

    # Convertimos la ruta a string explícitamente para evitar problemas con PosixPath/WindowsPath
    model_path = str(model_path)

    # Cargar el modelo sin `strict=False`
    learn = load_learner(model_path, pickle_module=pickle)
    learn.path = PosixPath("/")  # Corrige rutas en Linux
    return learn

learn = load_model()



st.title("🎨 Depuración Clasificación Artística (fastai)")

uploaded_file = st.file_uploader("Subir imagen", ["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    image_fastai = PILImage.create(uploaded_file)
    pred_class, pred_idx, probs = learn.predict(image_fastai)
    
    class_names = learn.dls.vocab
    
    with torch.no_grad():
        x = learn.dls.test_dl([image_fastai]).dataset[0][0].unsqueeze(0).to(learn.device)
        logits = learn.model.eval()(x)
        
    st.write("Outputs del modelo (logits):", logits.cpu().numpy())
    
    softmax_probs = torch.softmax(logits, dim=1)[0].cpu()
    st.write("Probabilidades (calculadas manualmente):",
             {class_names[i]: f"{p.item()*100:.2f}%" for i, p in enumerate(softmax_probs)})
    
    st.write("Probabilidades (vía learn.predict):",
             {class_names[i]: f"{p*100:.2f}%" for i, p in enumerate(probs)})
    
    st.write(f"Clase predicha: {pred_class} con {probs[pred_idx]*100:.2f}% confianza")
