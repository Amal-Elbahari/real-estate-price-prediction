# Modèle des données

## Colonnes principales du dataset
| Nom          | Type     | Description                   |
|--------------|----------|-------------------------------|
| price        | float    | Prix du bien                  |
| surface      | float    | Surface en m²                 |
| bedrooms     | integer  | Nombre de chambres            |
| bathrooms    | integer  |   Nombre de salles de bains   |
| location     | text     |    Ville / quartier           |
| property_type| text     | Type de bien                  |
| type         | text     | Type (Sale/Rent)              |
| image_url    | text     | Chemin vers l’image           |
| title        | text     | titre de l'annonce            |
| description  | text     | description de l'annonce      |
| link         | text     | Chemin vers l'annonce         |
## Notes
- Les données sont nettoyées dans `preprocessing/`
- Les images sont redimensionnées pour PyTorch (224x224)
