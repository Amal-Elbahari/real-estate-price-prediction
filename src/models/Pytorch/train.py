# train_price_from_images.py
# ------------------------------------------------------------
# Predict property price from images using PyTorch (transfer learning)
# - Includes type and property_category as categorical features
# - Stores final model + mappings for later use
# ------------------------------------------------------------

import os
import math
import argparse
import random
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

# ----------------------------- Utils -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None

def price_bins_for_stratification(series: pd.Series, n_bins: int = 10) -> pd.Series:
    try:
        bins = pd.qcut(series, q=min(n_bins, series.nunique()), labels=False, duplicates="drop")
        return bins
    except Exception:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)

# ----------------------------- Config -----------------------------

@dataclass
class TrainConfig:
    csv_path: str = "data/image.csv"
    images_root: str = ""
    out_dir: str = "runs/price_from_images"
    type_filter: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    img_size: int = 224
    tiny_file_threshold: int = 4 * 1024
    val_split: float = 0.2
    num_workers: int = 4
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    freeze_backbone_epochs: int = 2
    amp: bool = True
    patience: int = 5
    resume: Optional[str] = None

# ----------------------------- Dataset -----------------------------

class PriceImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_root: str, transform=None, tiny_file_threshold: int = 4096,
                 type_map=None, category_map=None):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        self.tiny_file_threshold = tiny_file_threshold

        # Mapping categories to integers
        self.type_map = type_map or {t: i for i, t in enumerate(df['type'].astype(str).unique())}
        self.category_map = category_map or {c: i for i, c in enumerate(df['property_category'].astype(str).unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["local_path"] if not self.images_root else os.path.join(self.images_root, os.path.basename(row["local_path"]))
        price = float(row["price"])

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            return None
        try:
            if os.path.getsize(img_path) < self.tiny_file_threshold:
                return None
        except OSError:
            return None

        img = safe_open_image(img_path)
        if img is None:
            return None

        if self.transform:
            img = self.transform(img)

        target_log = torch.tensor([math.log1p(max(price, 0.0))], dtype=torch.float32)
        type_idx = torch.tensor(self.type_map[str(row["type"])], dtype=torch.long)
        category_idx = torch.tensor(self.category_map[str(row["property_category"])], dtype=torch.long)

        return img, target_log, float(price), type_idx, category_idx, img_path

def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, y_log, y_raw, types, categories, paths = zip(*batch)
    return (torch.stack(imgs, dim=0), torch.stack(y_log, dim=0),
            torch.tensor(y_raw, dtype=torch.float32),
            torch.stack(types, dim=0), torch.stack(categories, dim=0),
            list(paths))

# ----------------------------- Model -----------------------------

class PriceRegressor(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True,
                 num_types: int = 2, num_categories: int = 5, embed_dim: int = 16):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.type_embed = nn.Embedding(num_types, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(in_features + 2*embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
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

# ----------------------------- Metrics -----------------------------

def compute_metrics(y_true: np.ndarray, y_pred_log: np.ndarray):
    y_pred = np.expm1(y_pred_log)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ----------------------------- Training -----------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, cfg: TrainConfig, freeze_backbone: bool):
    model.train()
    for p in model.backbone.parameters():
        p.requires_grad = not freeze_backbone

    criterion = nn.MSELoss()
    running_loss = 0.0
    n = 0

    for batch in loader:
        if batch is None:
            continue
        imgs, y_log, _, type_idx, category_idx, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        y_log = y_log.to(device, non_blocking=True)
        type_idx = type_idx.to(device)
        category_idx = category_idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            with torch.cuda.amp.autocast():
                out = model(imgs, type_idx, category_idx)
                loss = criterion(out, y_log)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs, type_idx, category_idx)
            loss = criterion(out, y_log)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return running_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred_log = [], []

    for batch in loader:
        if batch is None:
            continue
        imgs, y_log, y_raw, type_idx, category_idx, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        type_idx = type_idx.to(device)
        category_idx = category_idx.to(device)
        out = model(imgs, type_idx, category_idx)
        y_true.append(y_raw.numpy())
        y_pred_log.append(out.cpu().numpy())

    if len(y_true) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "R2": float("nan")}

    y_true = np.concatenate(y_true, axis=0)
    y_pred_log = np.concatenate(y_pred_log, axis=0)
    return compute_metrics(y_true, y_pred_log)

def save_checkpoint(state: dict, is_best: bool, out_dir: str, filename: str = "last.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copy(path, os.path.join(out_dir, "best.pt"))

# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/image.csv")
    parser.add_argument("--images_root", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="runs/price_from_images")
    parser.add_argument("--type_filter", type=str, default=None)
    parser.add_argument("--min_price", type=float, default=None)
    parser.add_argument("--max_price", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--tiny_kb", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig(
        csv_path=args.csv_path,
        images_root=args.images_root,
        out_dir=args.out_dir,
        type_filter=args.type_filter,
        min_price=args.min_price,
        max_price=args.max_price,
        img_size=args.img_size,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        patience=args.patience,
        amp=args.amp,
        resume=args.resume,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        tiny_file_threshold=args.tiny_kb * 1024
    )

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(cfg.csv_path)
    for col in ["local_path", "price", "type", "property_category"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {cfg.csv_path}")

    if cfg.type_filter:
        df = df[df["type"].astype(str).str.lower() == cfg.type_filter.lower()]
    if cfg.min_price is not None:
        df = df[df["price"] >= cfg.min_price]
    if cfg.max_price is not None:
        df = df[df["price"] <= cfg.max_price]

    df = df.dropna(subset=["local_path", "price", "type", "property_category"]).reset_index(drop=True)
    assert len(df) > 0, "No samples after filtering."

    bins = price_bins_for_stratification(df["price"], n_bins=10)
    df["bin"] = bins
    val_ratio = float(cfg.val_split)
    val_idx, train_idx = [], []

    for b in sorted(df["bin"].unique()):
        sub = df[df["bin"] == b].index.tolist()
        random.shuffle(sub)
        n_val = max(1, int(len(sub) * val_ratio))
        val_idx.extend(sub[:n_val])
        train_idx.extend(sub[n_val:])

    df_train = df.loc[train_idx].drop(columns=["bin"]).reset_index(drop=True)
    df_val = df.loc[val_idx].drop(columns=["bin"]).reset_index(drop=True)

    print(f"Train: {len(df_train)} | Val: {len(df_val)} (Total: {len(df)})")

    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.15,0.15,0.15,0.03)], p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    type_map = {t: i for i, t in enumerate(df_train['type'].astype(str).unique())}
    category_map = {c: i for i, c in enumerate(df_train['property_category'].astype(str).unique())}

    train_ds = PriceImageDataset(df_train, cfg.images_root, transform=train_tf, tiny_file_threshold=cfg.tiny_file_threshold,
                                 type_map=type_map, category_map=category_map)
    val_ds = PriceImageDataset(df_val, cfg.images_root, transform=val_tf, tiny_file_threshold=cfg.tiny_file_threshold,
                               type_map=type_map, category_map=category_map)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              collate_fn=skip_none_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            collate_fn=skip_none_collate)

    model = PriceRegressor(
        backbone_name="resnet50",
        pretrained=True,
        num_types=len(type_map),
        num_categories=len(category_map)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - cfg.freeze_backbone_epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    start_epoch = 0
    best_rmse = float("inf")
    epochs_no_improve = 0

    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_rmse = ckpt.get("best_rmse", best_rmse)
        print(f"Resumed from {cfg.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, cfg.epochs):
        freeze_backbone = (epoch < cfg.freeze_backbone_epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, freeze_backbone)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train MSE: {train_loss:.5f} | "
              f"Val MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | "
              f"MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f}")

        is_best = metrics["RMSE"] < best_rmse
        if is_best:
            best_rmse = metrics["RMSE"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_rmse": best_rmse,
            "cfg": vars(cfg),
            "type_map": type_map,
            "category_map": category_map
        }, is_best=is_best, out_dir=cfg.out_dir, filename="last.pt")

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping triggered after {cfg.patience} epochs without improvement.")
            break

    # Save final model for inference
    final_model_path = os.path.join(cfg.out_dir, "final_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "type_map": type_map,
        "category_map": category_map
    }, final_model_path)
    print(f"Training done. Best Val RMSE: {best_rmse:.2f}. Model saved at: {final_model_path}")

if __name__ == "__main__":
    main()
