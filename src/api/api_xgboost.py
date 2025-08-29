from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load("C:/Users/elbah/Desktop/real_estate/venv/model_prediction/results/best_model_RandomForest_target_encoded.pkl")

# Route d'accueil
@app.route("/")
def home():
    return "Bienvenue sur l’API de prédiction de prix !"

# Route de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Données reçues depuis la requête JSON
        data = request.get_json()

        # Création DataFrame à partir des données envoyées
        input_df = pd.DataFrame([data])

        # Prédiction avec le modèle
        prediction = model.predict(input_df)[0]

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# Lancer le serveur
if __name__ == "__main__":
    app.run(debug=True)
