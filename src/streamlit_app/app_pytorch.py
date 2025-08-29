import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# ----------------- Définir le modèle -----------------
class PriceRegressor(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=False,
                 num_types=2, num_categories=5, embed_dim=16):
        super().__init__()
        import torchvision
        backbone = getattr(torchvision.models, backbone_name)(
            weights=None  # Pas besoin des poids ImageNet pour l’inférence
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(in_features + 2 * embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x, type_idx=None, category_idx=None):
        feats = self.backbone(x)
        if type_idx is not None and category_idx is not None:
            type_emb = self.type_embed(type_idx)
            cat_emb = self.category_embed(category_idx)
            feats = torch.cat([feats, type_emb, cat_emb], dim=1)
        out = self.head(feats)
        return out


# ----------------- Charger le modèle -----------------
@st.cache_resource
def load_model(model_path="models/Pytorchruns/price_from_images/final_model.pt"):
    checkpoint = torch.load(model_path, map_location="cpu")
    type_map = checkpoint["type_map"]
    category_map = checkpoint["category_map"]

    model = PriceRegressor(
        backbone_name="resnet50",
        pretrained=False,
        num_types=len(type_map),
        num_categories=len(category_map)
    )
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, type_map, category_map


# ----------------- Pré-traitement image -----------------
def preprocess_image(image, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0)  # (1,3,H,W)


# ----------------- Streamlit UI -----------------
st.title("🏠 Real Estate Price Prediction")
st.write("Upload an image of a property, select its type and category, and get the predicted price.")

# Charger modèle
model, type_map, category_map = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload property image", type=["jpg", "jpeg", "png"])
property_type = st.selectbox("Select property type", list(type_map.keys()))
property_category = st.selectbox("Select property category", list(category_map.keys()))

if uploaded_file is not None:
    # Afficher image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Price"):
        x = preprocess_image(image)

        type_idx = torch.tensor([type_map[property_type]], dtype=torch.long)
        cat_idx = torch.tensor([category_map[property_category]], dtype=torch.long)

        with torch.no_grad():
            pred_log = model(x, type_idx, cat_idx)
            pred_price = torch.expm1(pred_log).item()  # inverse log1p

        st.success(f"💰 Predicted Price: {pred_price:,.0f} MAD")
