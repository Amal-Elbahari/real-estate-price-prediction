import psycopg2
import re
from datetime import datetime

# 🧱 Features à détecter
target_features = ["terrasse", "garage", "ascenseur", "concierge", "piscine", "sécurité", "jardin"]

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname="real",
    user="postgres",
    password="2024",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# 🔄 Créer table nettoyée
cursor.execute("""
DROP TABLE IF EXISTS clean_data_avito;
CREATE TABLE clean_data_avito (
    id SERIAL PRIMARY KEY,
    title TEXT,
    price FLOAT,
    location TEXT,
    description TEXT,
    surface FLOAT,
    rooms INT,
    bedrooms INT,
    bathrooms INT,
    floor TEXT,
    transport TEXT,
    type TEXT,
    link TEXT,
    image_url TEXT,
    published_time TEXT,
    property_category TEXT,
    terrace BOOLEAN,
    garage BOOLEAN,
    elevator BOOLEAN,
    concierge BOOLEAN,
    pool BOOLEAN,
    security BOOLEAN,
    garden BOOLEAN,
    insert_date TIMESTAMP
);
""")
conn.commit()

# 📥 Charger données brutes
cursor.execute("""
SELECT title, price, location, description,
       surface, bedrooms, bathrooms,
       published_time, type, property_category,
       features, rooms,
       link, image_url
FROM avito
""")
rows = cursor.fetchall()
cleaned_rows = []

# 🔧 Fonctions utilitaires
def clean_price_val(val):
    if val is None: return None
    try:
        val_str = str(val)
        num = re.sub(r"[^\d.]", "", val_str)
        return float(num) if num else None
    except:
        return None

def clean_surface_val(val):
    if val is None: return None
    try:
        return float(re.search(r"\d+", str(val)).group())
    except:
        return None

def to_int(val):
    if val is None: return None
    try:
        return int(re.search(r"\d+", str(val)).group())
    except:
        return None

def clean_text(val):
    if val is None: return None
    val = val.strip().replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", val)

def extract_floor(description):
    if not description: return None
    match = re.search(r"(\d+)[èe]?[e]?\s*(étage|floor)", description.lower())
    return match.group(1) if match else None

def extract_transport(description):
    if not description: return None
    keywords = ["tram", "bus", "métro", "station", "train", "taxis", "transport"]
    desc_lower = description.lower()
    transports = [kw for kw in keywords if kw in desc_lower]
    return ", ".join(transports) if transports else None

# 🧹 Nettoyage et extraction
for row in rows:
    (title, price, location, description, surface, bedrooms, bathrooms,
     published_time, prop_type, prop_category, features, rooms,
     link, image_url) = row

    clean_price = clean_price_val(price)
    clean_surface = clean_surface_val(surface)
    clean_rooms = to_int(rooms)
    clean_bedrooms = to_int(bedrooms)
    clean_bathrooms = to_int(bathrooms)

    title = clean_text(title)
    location = clean_text(location)
    description = clean_text(description)
    published_time = clean_text(published_time)
    prop_type = clean_text(prop_type)
    prop_category = clean_text(prop_category)
    link = clean_text(link)
    image_url = clean_text(image_url)

    floor = extract_floor(description)
    transport = extract_transport(description)

    # Features binaires
    feature_list = [f.strip().lower() for f in features.split(",")] if features else []
    binary_values = [(feature.lower() in feature_list) for feature in target_features]

    # Vérification obligatoire
    if clean_price is None or clean_surface is None or not title or not location:
        continue  # on ignore les annonces sans prix ou surface

    insert_date = datetime.now()
    cleaned_rows.append((
        title, clean_price, location, description,
        clean_surface, clean_rooms, clean_bedrooms, clean_bathrooms,
        floor, transport,
        prop_type, link, image_url, published_time, prop_category,
        *binary_values,
        insert_date
    ))

# 🟢 Insertion
insert_query = """
INSERT INTO clean_data_avito(
    title, price, location, description,
    surface, rooms, bedrooms, bathrooms,
    floor, transport,
    type, link, image_url, published_time, property_category,
    terrace, garage, elevator, concierge, pool, security, garden,
    insert_date
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
cursor.executemany(insert_query, cleaned_rows)
conn.commit()

print(f"✅ {len(cleaned_rows)} lignes insérées dans clean_data_avito.")

cursor.close()
conn.close()
