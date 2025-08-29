import psycopg2
import re
from datetime import datetime

# Liste des features à transformer en colonnes binaires
target_features = ["Terrace", "Garage", "Elevator", "Concierge", "Pool", "Security system", "Garden"]

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="real",
    user="postgres",
    password="2024",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Supprimer et créer une nouvelle table clean_data avec les champs supplémentaires
cursor.execute("""
    DROP TABLE IF EXISTS clean_rent;
    CREATE TABLE clean_rent(
        id SERIAL PRIMARY KEY,
        title TEXT,
        price FLOAT,
        location TEXT,
        description TEXT,
        surface FLOAT,
        rooms INT,
        bedrooms INT,
        bathrooms INT,
        type TEXT,
        property_category TEXT,
        link TEXT,
        image_url TEXT,
        published_time TEXT,
        terrace BOOLEAN,
        garage BOOLEAN,
        elevator BOOLEAN,
        concierge BOOLEAN,
        pool BOOLEAN,
        security_system BOOLEAN,
        garden BOOLEAN,
        insert_date TIMESTAMP
    );
""")
conn.commit()

# Charger les données brutes avec les nouveaux champs
cursor.execute("""
    SELECT title, price, location, description, surface,
           rooms, bedrooms, bathrooms, type,
           property_category, link, image_url, published_time, features
    FROM mubawab_rent
""")
rows = cursor.fetchall()

cleaned_rows = []

for row in rows:
    (title, price, location, description, surface,
     rooms, bedrooms, bathrooms, prop_type,
     prop_category, link, image_url, published_time, features) = row

    # Nettoyage du prix
    try:
        clean_price = float(re.sub(r"[^\d.]", "", price)) if price else None
    except:
        clean_price = None

    # Nettoyage surface
    try:
        clean_surface = float(re.search(r"\d+", surface).group()) if surface else None
    except:
        clean_surface = None

    # Fonction de nettoyage int
    def to_int(val):
        try:
            return int(re.search(r"\d+", val).group()) if val else None
        except:
            return None

    clean_rooms = to_int(rooms)
    clean_bedrooms = to_int(bedrooms)
    clean_bathrooms = to_int(bathrooms)

    # Nettoyage texte
    def clean_text(val):
        return val.strip().replace("\n", " ").replace("\t", " ") if val else None

    title = clean_text(title)
    location = clean_text(location)
    description = clean_text(description)
    prop_type = clean_text(prop_type)
    prop_category = clean_text(prop_category)
    link = clean_text(link)
    image_url = clean_text(image_url)
    published_time = clean_text(published_time)

    # Binarisation des features
    feature_list = [f.strip().lower() for f in features.split(",")] if features else []
    binary_values = [(feature.lower() in feature_list) for feature in target_features]

    # Vérification des champs essentiels
    if title and clean_price is not None and location and clean_surface is not None:
        insert_date = datetime.now()
        cleaned_rows.append((
            title, clean_price, location, description, clean_surface,
            clean_rooms, clean_bedrooms, clean_bathrooms, prop_type,
            prop_category, link, image_url, published_time,
            *binary_values,
            insert_date
        ))

# Insertion dans la table clean_data
insert_query = f"""
    INSERT INTO clean_rent (
        title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type,
        property_category, link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security_system, garden,
        insert_date
    ) VALUES ({', '.join(['%s'] * 21)})
"""

cursor.executemany(insert_query, cleaned_rows)
conn.commit()

print(f"✅ {len(cleaned_rows)} lignes nettoyées et insérées dans clean_data.")

cursor.close()
conn.close()
