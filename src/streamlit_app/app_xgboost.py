import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration de la page ---
st.set_page_config(page_title="Prédiction de prix immobilier", page_icon="🏠", layout="centered")
st.title("🏠 Estimation de prix immobilier")

# --- Formulaire de saisie ---
with st.form("prediction_form"):
    surface = st.number_input("Surface (m²)", min_value=10.0, step=1.0)
    rooms = st.number_input("Nombre de pièces", min_value=1, step=1)
    bedrooms = st.number_input("Chambres", min_value=0, step=1)
    bathrooms = st.number_input("Salles de bain", min_value=0, step=1)
    location = st.text_input("Localisation (ex: Casablanca, Ain Sebaa)")
    property_category = st.selectbox("Catégorie du bien", ["Apartement", "Maison", "Villa", "Bureau", "Terrain"])
    type_bien = st.selectbox("Type de transaction", ["For Sale", "For Rent"])

    terrace = st.checkbox("Terrasse")
    garage = st.checkbox("Garage")
    elevator = st.checkbox("Ascenseur")
    concierge = st.checkbox("Concierge")
    pool = st.checkbox("Piscine")
    security = st.checkbox("Sécurité")
    garden = st.checkbox("Jardin")

    submit = st.form_submit_button("Prédire le prix")

if submit:
    # --- Définir le chemin du modèle selon le type ---
    model_dir = r"C:\Users\elbah\Desktop\real-estate-price-prediction\venv\models\xgboost"
    model_file = os.path.join(model_dir, f"real_estate_model_{type_bien.replace(' ', '_')}.joblib")

    # --- Charger le modèle ---
    try:
        saved_obj = joblib.load(model_file)
        preprocessor = saved_obj['preprocessor']
        model = saved_obj['model']
    except FileNotFoundError:
        st.error(f"Modèle pour '{type_bien}' non trouvé : {model_file}")
        st.stop()

    # --- Feature engineering ---
    location_split = [x.strip() for x in location.split(",")] + ["Other"]
    location_city = location_split[0]
    location_district = location_split[1]

    rooms_per_surface = rooms / surface if surface > 0 else 0
    bathrooms_per_surface = bathrooms / surface if surface > 0 else 0
    bedrooms_per_surface = bedrooms / surface if surface > 0 else 0

    surface_x_rooms = surface * rooms
    surface_x_bathrooms = surface * bathrooms
    surface_x_bedrooms = surface * bedrooms

    luxury_features_count = int(concierge) + int(pool) + int(security) + int(garden)

    data = pd.DataFrame([{
        "location_city": location_city,
        "location_district": location_district,
        "surface": surface,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "rooms_per_surface": rooms_per_surface,
        "bathrooms_per_surface": bathrooms_per_surface,
        "bedrooms_per_surface": bedrooms_per_surface,
        "surface_x_rooms": surface_x_rooms,
        "surface_x_bathrooms": surface_x_bathrooms,
        "surface_x_bedrooms": surface_x_bedrooms,
        "luxury_features_count": luxury_features_count,
        "terrace": int(terrace),
        "garage": int(garage),
        "elevator": int(elevator),
        "concierge": int(concierge),
        "pool": int(pool),
        "security": int(security),
        "garden": int(garden),
        "property_category": property_category
    }])

    # --- Transformation et prédiction ---
    X_proc = preprocessor.transform(data)
    pred_log = model.predict(X_proc)
    pred_price = np.expm1(pred_log)[0]

    st.success(f"💰 Prix estimé : {pred_price:,.0f} MAD")
