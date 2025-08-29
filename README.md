# Real Estate Price Prediction (Morocco)

**Projet de prédiction des prix immobiliers au Maroc**  
Pipeline complet : scraping, nettoyage, modélisation (XGBoost + PyTorch pour images), API Flask et application Streamlit.

---

## 📌 À propos du projet
Ce projet fournit une solution complète pour :

- Collecter des données immobilières à partir de **Mubawab** (Selenium) et **Avito** (Scrapy)  
- Nettoyer et préparer un dataset exploitable  
- Entraîner un modèle **XGBoost** sur les features tabulaires  
- Entraîner un modèle **PyTorch (ResNet50)** sur les images des biens  
- Exposer les modèles via une **API Flask** pour prédiction en temps réel  
- Fournir une **interface Streamlit** pour tester facilement les prédictions  

---

## 🗂 Structure du projet
```
real-estate-price-prediction/
├── data/ # Données brutes et exemples
├── visualisation/ # Notebooks Jupyter pour exploration ou visualisation
├── src/
│ ├── scrap/ # Scraping avec Selenium (Mubawab) et Scrapy (Avito)
│ │ ├── mubawab_scraper_Rent.py
│ │ └── mubawab_scraper_Sale.py
│ │ └── scrapping (avito)
│
│ ├── preprocessing/ # Nettoyage, feature engineering et normalisation
│ │ └── clean_avito.py
│ │ └── cleanrent.py
│ │ └── cleansall.py
│ │ └── combine_data.py
│ ├── models/ # Entraînement XGBoost et PyTorch
│ │ ├── Pytorch
│ │ └── Xgboost
│ ├── api/ # API Flask pour prédiction
│ │ ├── api_pytorch.py
│ │ └── api_xgboost.py
│ │ └── model_utils.py
│ ├── streamlit_app/ # Interface Streamlit
│ │  └── app_pytorch.py
│ │  └── app_xgboost.py
├── requirements.txt # Dépendances Python
├── Dockerfile # Pour déploiement Docker
├── docker-compose.yml # Déploiement multi-container (optionnel)
├── .gitignore
├── README.md
├── LICENSE
└── docs/ # Documentation détaillée
├── architecture.md
├── data_model.md
├── guide_utilisation.md
├── model_card.md
├── scraping.md
└── pipelines.md

```
---

## 🛠 Technologies et bibliothèques utilisées

- **Python 3.10+**  
- **Scraping** : Selenium (Mubawab), Scrapy (Avito)  
- **Data cleaning** : Pandas, NumPy  
- **Modélisation tabulaire** : XGBoost  
- **Modélisation image** : PyTorch, torchvision (ResNet50)  
- **API** : Flask  
- **Interface utilisateur** : Streamlit  
- **Visualisation** : Matplotlib, Seaborn  
- **Déploiement** : Docker / Docker Compose  

---

## ⚡ Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/real-estate-price-prediction.git
cd real-estate-price-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

🚀 Utilisation
1️⃣ Scraping
# Scraping Mubawab
python src/scraping/mubawab_scraper_Sale.py --out data/raw/mubawab.csv

# Scraping Avito
scrapy crawl avito --out data/raw/avito.csv

2️⃣ Prétraitement
python src/preprocessing/clean_avito.py 

3️⃣ Entraînement des modèles

XGBoost (tabulaire)

python src/models/train_xgboost.py --config config/xgb.yaml


PyTorch (images)

python src/models/train_pytorch.py --config config/pt.yaml

4️⃣ Lancer l’API Flask
cd src/api
flask run --host=0.0.0.0 --port=5000


Endpoints disponibles :

POST /predict → Prédiction à partir des features tabulaires

POST /predict-image → Prédiction à partir d’une image

5️⃣ Lancer l’application Streamlit
streamlit run src/streamlit_app/app_pytorch.py
streamlit run src/streamlit_app/app_xgboost.py

📦 Déploiement avec Docker
# Construire l'image
docker build -t real-estate-app .

# Lancer le conteneur
docker run -p 5000:5000 real-estate-app

# Avec Docker Compose
docker-compose up -d

📊 Modèles et métriques

XGBoost : MAE, RMSE sur features tabulaires

PyTorch : MAE sur prix log-transformés via images

📚 Documentation

Pour plus d’informations, consultez le dossier docs/ :

architecture.md : architecture et modules du projet

data_model.md : description du dataset et colonnes principales

guide_utilisation.md : guide complet d’installation et d’utilisation

model_card.md : résumé des modèles utilisés + métriques

scraping.md : détails sur Selenium et Scrapy, liens officiels

pipelines.md : explication du pipeline complet

🔗 Liens utiles

XGBoost : https://xgboost.readthedocs.io

PyTorch : https://pytorch.org/docs/stable/index.html

Scrapy : https://docs.scrapy.org/en/latest/

Selenium : https://www.selenium.dev/documentation/

📝 Contribution

Forker le projet

Créer une branche pour votre fonctionnalité :

git checkout -b feature/nom-feature


Committer vos changements :

git commit -m "Ajout de la feature"


Pousser vers la branche :

git push origin feature/nom-feature


Ouvrir une Pull Request

⚖️ Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.


---

