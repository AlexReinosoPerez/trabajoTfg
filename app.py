import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

HF_REPO_ID = "AlexReinoso/trabajoTFM"
MODEL_FILENAME = "best_model.pth"
class_names = ["Impresionismo", "Post-Impresionismo", "Pop Art", "Renacentista"]

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, len(class_names)))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("🎨 Depuración Clasificación Artística")

uploaded_file = st.file_uploader("Subir imagen", ["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    st.write(f"Forma tensor entrada: {input_tensor.shape}")

    with torch.no_grad():
        outputs = model(input_tensor)
        st.write("Outputs del modelo (logits):", outputs.numpy())
        probs = torch.softmax(outputs, dim=1)
        st.write("Probabilidades:", {class_names[i]: f"{p*100:.2f}%" for i, p in enumerate(probs[0])})
        
        confidence, pred = torch.max(probs, 1)
        st.write(f"Clase predicha: {class_names[pred.item()]} con {confidence.item()*100:.2f}% confianza")
