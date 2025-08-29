# Guide d’utilisation

## Installation
```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
Scraping
python src/scraping/mubawab_scraper.py --out data/raw/mubawab.csv
python src/scraping/avito_scraper.py --out data/raw/avito.csv
Prétraitement
python src/preprocessing/clean_data.py --in data/raw --out data/processed
Entraînement
python src/models/train_xgboost.py
python src/models/train_pytorch.py

API Flask
cd src/api
flask run --host=0.0.0.0 --port=5000
Streamlit
streamlit run src/streamlit_app/app.py
