# Pipelines

## Pipeline complet
Scraping -> Prétraitement -> Modélisation -> API / Streamlit

## Étapes détaillées
1. **Scraping** : collecter les données via Selenium/ Scrapy
2. **Prétraitement** : nettoyer, normaliser, générer les features
3. **Entraînement** : XGBoost (tabulaire) et PyTorch (images)
4. **Déploiement** : API Flask et Streamlit app
