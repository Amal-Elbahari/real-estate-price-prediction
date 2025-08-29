# Architecture du projet Real Estate Price Prediction

## Aperçu
Le projet suit une architecture modulaire inspirée de l’architecture hexagonale :

- **Adaptateurs d’entrée** : Scraping avec Selenium (Mubawab) et Scrapy (Avito)
- **Domaine métier** : Nettoyage, feature engineering, modèles ML
- **Adaptateurs de sortie** : Sauvegarde des modèles, API Flask, application Streamlit

## Schéma
[Scraping] --> [Preprocessing] --> [Models] --> [API / Streamlit]

## Modules
- `scraping/` : collecte des données
- `preprocessing/` : nettoyage et préparation des données
- `models/` : entraînement XGBoost et PyTorch
- `api/` : Flask pour prédiction
- `streamlit_app/` : interface utilisateur
