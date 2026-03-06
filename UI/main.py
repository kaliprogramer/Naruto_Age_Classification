import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import base64
import numpy as np
import requests
from io import BytesIO  # <-- added for proper image handling

# -------------------- CONFIGURATION --------------------
st.set_page_config(
    page_title="Naruto Age Classifier",
    page_icon="🍥",
    layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ Your model was trained with 10 classes.
# Update this list with your actual 10 labels (in the correct order).
class_names = ["adult", "teen", "young"]  # placeholder

IMG_SIZE = 32

# -------------------- MODEL DEFINITION --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, img_size=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two pools on 32x32 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# -------------------- CACHED  LOADER --------------------
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=10, img_size=IMG_SIZE)  # force 10 classes
    try:
        state = torch.load("/Core/naruto_cnn_weights.pth", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model weights not found. Please ensure '../Core/naruto_cnn_weights.pth' exists.")
        return None
    except RuntimeError as e:
        st.error(f"Model architecture mismatch: {e}")
        st.info("Your saved model expects 10 output classes. Either:\n"
                "- Retrain the model with 3 classes, or\n"
                "- Replace the `class_names` list with your actual 10 labels.")
        return None

model = load_model()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .main-header {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        color: #FFD700;
        text-shadow: 0 0 10px #FF4500, 2px 2px 8px black;
        font-size: 3.5rem;
        margin-bottom: 0;
    }
    .sub-header {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        color: #FFA500;
        font-size: 1.8rem;
        background: rgba(0,0,0,0.5);
        border-radius: 50px;
        padding: 10px;
        width: 70%;
        margin: 20px auto;
        backdrop-filter: blur(5px);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 20px;
        background: rgba(30,30,30,0.8);
        backdrop-filter: blur(8px);
        border: 2px solid #FF8C00;
        box-shadow: 0 0 20px #FF4500;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #FFD700;
        font-weight: bold;
        margin-top: 50px;
        background: linear-gradient(90deg, transparent, #FF4500, transparent);
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF8C00, #FF4500);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 50px;
        padding: 10px 30px;
        box-shadow: 0 4px 15px rgba(255,69,0,0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255,69,0,0.6);
    }
    .stFileUploader {
        background: rgba(0,0,0,0.5);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(5px);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<h1 class="main-header">🍥 NARUTO AGE CLASSIFIER</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-header" >⚡ Upload a character image to predict age group ⚡</p>', unsafe_allow_html=True)


# -------------------- MAIN CONTENT --------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload or choose a sample")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png","webp"], label_visibility="collapsed")

    if "sample_used" not in st.session_state:
        st.session_state.sample_used = False

    # Direct image URL (replace if needed)
    SAMPLE_IMAGE_URL = "https://imgs.search.brave.com/Nk-lOQTwoeQ7tHysY89ZyDuFV1o3hPu3giNSes1_9Ro/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJhY2Nlc3Mu/Y29tL2Z1bGwvMzY5/NDguanBn"

    if st.button("✨ Load Sample Image", use_container_width=True):
        st.session_state.sample_used = True
        st.session_state.sample_url = SAMPLE_IMAGE_URL
        st.rerun()

    # Display image (uploaded or sample)
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Your Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error opening uploaded image: {e}")
            img = None
    elif st.session_state.get("sample_used"):
        try:
            response = requests.get(st.session_state.sample_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Sample Image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load sample image: {e}")
            img = None
    else:
        st.markdown("<div style='height:300px; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.3); border-radius:20px;'>👆 Upload or load a sample</div>", unsafe_allow_html=True)
        img = None

with col2:
    st.markdown("### 🔮 Prediction Result")
    if (uploaded_file is not None or st.session_state.get("sample_used")) and img is not None:
        if model is None:
            st.error("Model not loaded. Please check the error above.")
        else:
            with st.spinner("Analyzing chakra signature..."):
                transform = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    pred_idx = np.argmax(probs)
                    
                    # Ensure index is within class_names bounds (model outputs 10 classes)
                    if pred_idx >= len(class_names):
                        st.warning(f"Model predicted class index {pred_idx}, but you only defined {len(class_names)} labels. Update `class_names`.")
                        pred_class = f"Class {pred_idx}"
                    else:
                        pred_class = class_names[pred_idx]
                    
                    confidence = probs[pred_idx] * 100

                # Simple color mapping for first three classes (child, teen, adult)
                color_map = {"child": "#7CFC00", "teen": "#FFA500", "adult": "#FF4500"}
                card_color = color_map.get(pred_class, "#FF8C00")

                st.markdown(f"""
                <div class="prediction-card" style="border-color: {card_color};">
                    <h2 style="color: {card_color};">{pred_class.upper()}</h2>
                    <p style="font-size: 1.5rem;">Confidence: {confidence:.2f}%</p>
                    <div style="background: #444; border-radius: 10px; height: 20px; width: 100%;">
                        <div style="background: {card_color}; width: {confidence}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### Class probabilities")
                for i in range(3):
                    prob = probs[i] * 100
                    label = class_names[i] if i < len(class_names) else f"Class {i}"
                    st.markdown(f"**{i}: {label}:** {prob:.2f}%")
    else:
        st.markdown("<div style='height:300px; display:flex; align-items:center; justify-content:center; background:rgba(0,0,0,0.3); border-radius:20px;'>✨ Prediction will appear here</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown('<div class="footer">Made with 💛 for Naruto fans | Kiran K.C!</div>', unsafe_allow_html=True)
