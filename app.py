import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
from fastai.vision.all import load_learner, PILImage
import torch

HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model_fastai.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    learn = load_learner(model_path)
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
