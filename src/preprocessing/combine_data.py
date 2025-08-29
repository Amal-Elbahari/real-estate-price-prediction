import psycopg2

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname="real",
    user="postgres",
    password="2024",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# 🔄 1. Créer la table fusionnée
cursor.execute("""
    
    Drop table if exists combine;
    CREATE TABLE combine (
        id SERIAL PRIMARY KEY,
        source TEXT,
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
        security BOOLEAN,
        garden BOOLEAN,
        insert_date TIMESTAMP
    );
""")
conn.commit()

# 🔄 2. Insérer les données depuis clean_data_avito
cursor.execute("""
    INSERT INTO combine (
        source, title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security, garden,
        insert_date
    )
     SELECT
        'mubawab', title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool,security_system AS security, garden,
        insert_date
    FROM clean_rent;
""")
conn.commit()

# 🔄 3. Insérer les données depuis clean_data (mubawab)
cursor.execute("""
    INSERT INTO combine (
        source, title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security, garden,
        insert_date
    )
    SELECT
        'mubawab', title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security_system AS security, garden,
        insert_date
    FROM clean_sall;
""")
conn.commit()

# 🔄 4. Insérer les données depuis clean_data_avito
cursor.execute("""
    INSERT INTO combine (
        source, title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security, garden,
        insert_date
    )
    SELECT
        'avito', title, price, location, description, surface,
        rooms, bedrooms, bathrooms, type, property_category,
        link, image_url, published_time,
        terrace, garage, elevator, concierge, pool, security, garden,
        insert_date
    FROM clean_data_avito;
""")
conn.commit()

print("✅ Fusion terminée : données combinées dans 'combined_clean_data'.")

cursor.close()
conn.close()   