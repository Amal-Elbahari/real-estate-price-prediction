import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import joblib
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import create_data_loaders, save_preprocessors
# --- Modèle CNN + Embeddings ---
class RealEstatePricePredictor(nn.Module):
    def __init__(self, num_features, num_locations=10, num_categories=5, emb_dim=4, dropout_rate=0.3):
        super().__init__()
        self.image_cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_cnn.fc = nn.Linear(512, 128)

        # Geler les premières couches
        for param in list(self.image_cnn.parameters())[:-4]:
            param.requires_grad = False

        # Embeddings
        self.location_emb = nn.Embedding(num_locations, emb_dim)
        self.category_emb = nn.Embedding(num_categories, emb_dim)

        # Traitement des features tabulaires
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features - 2 + emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Fusion CNN + tabulaire
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, images, features):
        image_feats = self.image_cnn(images)
        numeric_feats = features[:, :-2]
        category_idx = features[:, -2].long()
        location_idx = features[:, -1].long()
        cat_feats = self.category_emb(category_idx)
        loc_feats = self.location_emb(location_idx)

        tab_feats = torch.cat([numeric_feats, cat_feats, loc_feats], dim=1)
        tab_feats = self.feature_processor(tab_feats)
        combined = torch.cat([image_feats, tab_feats], dim=1)
        return self.fusion(combined).squeeze()
# --- Robust Loss ---
class RobustHybridLoss(nn.Module):
    def __init__(self, mae_weight=0.7, mse_weight=0.3, clip_value=3.0):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.clip_value = clip_value

    def forward(self, preds, targets):
        diff = preds - targets
        diff = torch.clamp(diff, -self.clip_value, self.clip_value)
        mae = torch.mean(torch.abs(diff))
        mse = torch.mean(diff ** 2)
        return self.mae_weight * mae + self.mse_weight * mse


# --- Évaluation ---
def evaluate_model(model, data_loader, device, price_scaler=None):
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for images, features, prices in data_loader:
            images, features, prices = images.to(device), features.to(device), prices.to(device)
            outputs = model(images, features)
            preds_list.append(outputs.view(-1).cpu().numpy())
            targets_list.append(prices.view(-1).cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    if price_scaler:
        preds = np.expm1(price_scaler.inverse_transform(preds.reshape(-1, 1))).flatten()
        targets = np.expm1(price_scaler.inverse_transform(targets.reshape(-1, 1))).flatten()

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = np.corrcoef(targets, preds)[0, 1] ** 2
    return mae, rmse, r2
# --- Entraînement ---
def train(num_epochs=20, batch_size=32, lr=1e-4, device='cuda', save_csv=True):
    # Création d'un répertoire de base pour cette exécution
    run_base_dir = f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = os.path.join(run_base_dir, "models")
    log_dir = os.path.join(run_base_dir, "logs")
    tensorboard_dir = os.path.join(run_base_dir, "tensorboard")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    train_loader, test_loader, scalers, encoders = create_data_loaders(batch_size=batch_size)
    num_numeric = 4
    num_bool = 7
    num_categorical = 2
    total_features = num_numeric + num_bool + num_categorical
    model = RealEstatePricePredictor(
        num_features=total_features,
        num_locations=len(encoders.get('location', {})),
        num_categories=len(encoders.get('property_category', {}))
    )
    model.to(device)
    criterion = RobustHybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir)
    # CSV pour metrics
    csv_file = None
    if save_csv:
        csv_file = os.path.join(log_dir, "metrics.csv")
        with open(csv_file, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'train_loss', 'MAE', 'RMSE', 'R2'])
    best_rmse = float('inf')
    best_r2 = float('-inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, features, prices in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, features, prices = images.to(device), features.to(device), prices.to(device)
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        # Évaluation
        mae, rmse, r2 = evaluate_model(model, test_loader, device, price_scaler=scalers['price'])
        writer.add_scalar('MAE/test', mae, epoch)
        writer.add_scalar('RMSE/test', rmse, epoch)
        writer.add_scalar('R2/test', r2, epoch)
        print(f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
        # Sauvegarde dans CSV
        if save_csv and csv_file:
            with open(csv_file, 'a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([epoch + 1, avg_loss, mae, rmse, r2])
        # Sauvegarde du meilleur modèle basé sur RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model_rmse.pth"))
            print(f"Nouveau meilleur modèle (RMSE) sauvegardé: {best_rmse:.2f}")

        # Sauvegarde du meilleur modèle basé sur R2
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model_r2.pth"))
            print(f"Nouveau meilleur modèle (R2) sauvegardé: {best_r2:.4f}")

    # Évaluation finale
    print("\nÉvaluation finale sur les données de test :")
    final_mae, final_rmse, final_r2 = evaluate_model(model, test_loader, device, price_scaler=scalers['price'])
    print(f"MAE final : {final_mae:.2f} DH")
    print(f"RMSE final : {final_rmse:.2f} DH")
    print(f"R2 final : {final_r2:.4f}")

    # Sauvegarde du modèle final et preprocessors
    torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))
    preprocessor_dir = os.path.join(model_dir, "preprocessors")
    save_preprocessors(scalers, encoders, save_dir=preprocessor_dir)
    
    print(f"\nModèles et préprocesseurs sauvegardés dans '{model_dir}'")
    if save_csv and csv_file:
        print(f"Metrics sauvegardés dans '{csv_file}'")
    
    print(f"Tensorboard logs dans '{tensorboard_dir}'")
# --- Point d'entrée ---
if __name__ == "__main__":
    print("Démarrage de l'entraînement du modèle de prédiction immobilière...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    train(num_epochs=50, batch_size=32, lr=1e-4, device=device)
    print("\nEntraînement terminé avec succès !")