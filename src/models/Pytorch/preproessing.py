# preprocessing.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

DB_PARAMS = {
    "dbname": "real",
    "user": "postgres",
    "password": "2024",
    "host": "localhost",
    "port": 5432
}

class RealEstateDataset(Dataset):
    def __init__(self, dataframe, image_dir="data3/images", transform=None,
                 scalers=None, encoders=None, is_train=True):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.scalers = scalers or {}
        self.encoders = encoders or {}

        self.preprocess_features()

    def preprocess_features(self):
        df = self.dataframe.copy()

        # --- Price log-transform + StandardScaler ---
        if 'price' in df.columns:
            if self.is_train:
                df['price'] = np.log1p(df['price'])
                price_scaler = StandardScaler()
                df['price'] = price_scaler.fit_transform(df[['price']])
                self.scalers['price'] = price_scaler
            else:
                df['price'] = np.log1p(df['price'])
                if 'price' in self.scalers:
                    df['price'] = self.scalers['price'].transform(df[['price']])

        # --- Numeric features ---
        numeric_features = ['surface', 'rooms', 'bedrooms', 'bathrooms']
        for feat in numeric_features:
            if feat in df.columns:
                median_val = df[feat].median()
                df[feat] = df[feat].fillna(median_val)
                if self.is_train:
                    scaler = StandardScaler()
                    df[feat] = scaler.fit_transform(df[[feat]])
                    self.scalers[feat] = scaler
                else:
                    if feat in self.scalers:
                        df[feat] = self.scalers[feat].transform(df[[feat]])

        # --- Boolean features ---
        bool_features = ['terrace', 'garage', 'elevator', 'concierge', 'pool', 'security_system', 'garden']
        for feat in bool_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(False).astype(float)

        # --- Categorical features ---
        categorical_features = ['property_category', 'location']
        for feat in categorical_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna('Unknown')
                if self.is_train:
                    le = {v: i for i, v in enumerate(df[feat].unique())}
                    df[feat] = df[feat].map(le)
                    self.encoders[feat] = le
                else:
                    le = self.encoders.get(feat, {})
                    df[feat] = df[feat].apply(lambda x: le.get(x, le.get('Unknown', 0)))

        self.processed_data = df

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx]

        # --- Load and combine up to 3 images ---
        images = []
        for i in range(1, 4):
            img_path = row.get(f'local_path{i}', None)
            if pd.notna(img_path) and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
                except:
                    images.append(torch.zeros(3, 224, 224))
            else:
                images.append(torch.zeros(3, 224, 224))

        combined_image = torch.stack(images).mean(dim=0)

        # --- Features ---
        numeric_values = [row.get(f, 0) for f in ['surface', 'rooms', 'bedrooms', 'bathrooms']]
        bool_values = [row.get(f, 0) for f in ['terrace', 'garage', 'elevator', 'concierge', 'pool', 'security_system', 'garden']]
        categorical_values = [row.get(f, 0) for f in ['property_category', 'location']]

        features = np.concatenate([numeric_values, bool_values, categorical_values]).astype(np.float32)
        features_tensor = torch.tensor(features)

        price = torch.tensor(row['price'], dtype=torch.float32)

        return combined_image, features_tensor, price

# ---------------- Charger données depuis PostgreSQL ----------------
def load_data_from_db():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        query = """
        SELECT id, local_path1, local_path2, local_path3, price, surface, rooms, bedrooms, bathrooms,
               property_category, location, terrace, garage, elevator, concierge, pool, security_system, garden
        FROM images_predict_clean_sale
        WHERE type='For Sale'
        AND surface IS NOT NULL
        AND rooms IS NOT NULL
        AND bedrooms IS NOT NULL
        AND bathrooms IS NOT NULL;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return pd.DataFrame()

# ---------------- DataLoaders ----------------
def create_data_loaders(batch_size=32, test_size=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = load_data_from_db()

    if len(df) == 0:
        print("⚠️ Aucune donnée trouvée en base, jeu de test factice généré.")
        df = pd.DataFrame({
            'local_path1': ['test_image.jpg'],
            'local_path2': [None],
            'local_path3': [None],
            'price': [5000],
            'surface': [80],
            'rooms': [3],
            'bedrooms': [2],
            'bathrooms': [1],
            'property_category': ['Apartment'],
            'location': ['Casablanca'],
            'terrace': [True], 'garage': [False], 'elevator': [True], 'concierge': [False],
            'pool': [False], 'security_system': [True], 'garden': [False]
        })

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)

    train_dataset = RealEstateDataset(train_df, transform=transform, is_train=True)
    scalers, encoders = train_dataset.scalers, train_dataset.encoders

    test_dataset = RealEstateDataset(test_df, transform=transform, scalers=scalers, encoders=encoders, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scalers, encoders

# ---------------- Sauvegarde / chargement preprocessors ----------------
def save_preprocessors(scalers, encoders, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for feat, scaler in scalers.items():
        joblib.dump(scaler, f"{save_dir}/{feat}_scaler.pkl")
    for feat, enc in encoders.items():
        joblib.dump(enc, f"{save_dir}/{feat}_encoder.pkl")

def load_preprocessors(load_dir):
    scalers, encoders = {}, {}
    if os.path.exists(load_dir):
        for file in os.listdir(load_dir):
            if file.endswith("_scaler.pkl"):
                scalers[file.replace("_scaler.pkl", "")] = joblib.load(os.path.join(load_dir, file))
            elif file.endswith("_encoder.pkl"):
                encoders[file.replace("_encoder.pkl", "")] = joblib.load(os.path.join(load_dir, file))
    return scalers, encoders

# ---------------- Main ----------------
if __name__ == "__main__":
    print("📊 Chargement des données et création des DataLoaders...")
    train_loader, test_loader, scalers, encoders = create_data_loaders()

    print(f"✅ Nombre de lots d'entraînement : {len(train_loader)}")
    print(f"✅ Nombre de lots de test : {len(test_loader)}")

    # Créer un répertoire de sortie pour les préprocesseurs de test
    output_dir = "preprocessors_test"
    save_preprocessors(scalers, encoders, output_dir)
    print(f"💾 Préprocesseurs sauvegardés dans '{output_dir}'")

    # Charger les préprocesseurs pour vérifier
    loaded_scalers, loaded_encoders = load_preprocessors(output_dir)
    print(f"✅ Préprocesseurs chargés avec succès depuis '{output_dir}'")


    for images, features, prices in train_loader:
        print("📷 Taille des images :", images.shape)
        print("📊 Taille des features :", features.shape)
        print("💰 Taille des prix :", prices.shape)
        break
