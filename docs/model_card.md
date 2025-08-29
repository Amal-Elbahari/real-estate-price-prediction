
---

### 4️⃣ `model_card.md`
```markdown
# Model Card - Real Estate Price Prediction

## XGBoost (Tabulaire)
- Objectif : prédire le prix à partir des features tabulaires
- Entrées : surface, chambres, localisation, type de bien
- Sortie : prix prédit
- Métriques : MAE, RMSE
- Documentation officielle : [XGBoost](https://xgboost.readthedocs.io/)

## PyTorch (Images)
- Objectif : prédire le prix à partir des images des biens
- Backbone : ResNet50 pré-entraîné + tête de régression
- Entrées : images 224x224 RGB
- Sortie : prix prédit
- Métriques : MAE sur prix log-transformés
- Documentation officielle : [PyTorch](https://pytorch.org/docs/stable/index.html)
