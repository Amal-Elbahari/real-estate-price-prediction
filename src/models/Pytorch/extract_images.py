# db/extract_images.py
import os
import pandas as pd
import requests
import psycopg2
from urllib.parse import urlparse

# ---------------- PostgreSQL config ----------------
DB_PARAMS = {
    "dbname": "real",
    "user": "postgres",
    "password": "2024",
    "host": "localhost",
    "port": 5432
}

# ---------------- Create folders ----------------
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

# ---------------- Connect ----------------
conn = psycopg2.connect(**DB_PARAMS)
cursor = conn.cursor()

# ---------------- Fetch data ----------------
cursor.execute("""
    SELECT id, image_url, price, type, property_category 
    FROM combine
    WHERE image_url IS NOT NULL;
""")
rows = cursor.fetchall()

# ---------------- Headers for requests ----------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

data = []
failed = []

# ---------------- Download images ----------------
for row in rows:
    prop_id, image_url, price, type, property_category = row
    ext = os.path.splitext(urlparse(image_url).path)[1] or ".jpg"
    image_path = f"data/images/{prop_id}{ext}"

    # Skip already downloaded images
    if os.path.exists(image_path):
        print(f"Skipped {prop_id}, already exists")
        data.append((prop_id, image_url, image_path, price, type, property_category))
        continue

    # Attempt download
    try:
        response = requests.get(image_url, headers=headers, timeout=5)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {prop_id}")
            data.append((prop_id, image_url, image_path, price, type, property_category))
        else:
            print(f"Failed {prop_id}: Status code {response.status_code}")
            failed.append((prop_id, image_url, price, type, property_category))
    except Exception as e:
        print(f"Erreur pour {prop_id} : {e}")
        failed.append((prop_id, image_url, price, type, property_category))

# ---------------- Save CSVs ----------------
df = pd.DataFrame(data, columns=["id", "image_url", "local_path", "price", "type", "property_category"])
df.to_csv("data/image.csv", index=False)
print("CSV enregistré → data/image.csv")

if failed:
    df_failed = pd.DataFrame(failed, columns=["id", "image_url", "price", "type", "property_category"])
    df_failed.to_csv("data/logs/failed_downloads.csv", index=False)
    print(f"Failed downloads saved → data/logs/failed_downloads.csv")

# ---------------- Create / Insert table ----------------
cursor.execute("""
    DROP TABLE IF EXISTS images_predict;
    CREATE TABLE IF NOT EXISTS images_predict (
        id INT PRIMARY KEY,
        image_url TEXT,
        local_path TEXT,
        price FLOAT,
        type TEXT,
        property_category TEXT
    );
""")

for row in data:
    try:
        cursor.execute(
            "INSERT INTO images_predict (id, image_url, local_path, price, type, property_category) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING;",
            row
        )
    except Exception as e:
        print(f"Erreur insertion {row[0]} : {e}")

conn.commit()
cursor.close()
conn.close()
print("Données insérées dans table 'images_predict'")
