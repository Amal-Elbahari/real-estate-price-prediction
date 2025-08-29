import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from trainimage import PriceRegressor, safe_open_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location=device)
    type_map = checkpoint["type_map"]
    category_map = checkpoint["category_map"]
    num_types = len(type_map)
    num_categories = len(category_map)

    model = PriceRegressor(num_types=num_types, num_categories=num_categories)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, type_map, category_map

model_path = "models/Pytorch/runs/price_from_images/final_model.pt"
model, type_map, category_map = load_model(model_path)

# Transformation des images
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Fonction de prédiction
def predict_price(img_path: str, type_name: str, category_name: str):
    img = safe_open_image(img_path)
    if img is None:
        return None
    img = transform(img).unsqueeze(0).to(device)
    type_idx = torch.tensor([type_map.get(str(type_name), 0)], dtype=torch.long).to(device)
    category_idx = torch.tensor([category_map.get(str(category_name), 0)], dtype=torch.long).to(device)
    with torch.no_grad():
        log_pred = model(img, type_idx, category_idx)
        price_pred = math.expm1(log_pred.item())
    return price_pred
