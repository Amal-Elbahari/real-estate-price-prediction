import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model_utils import predict_price

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files["image"]
    type_name = request.form.get("type", "default")
    category_name = request.form.get("property_category", "default")

    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    price = predict_price(filepath, type_name, category_name)
    if price is None:
        return jsonify({"error": "Invalid image"}), 400

    return jsonify({"predicted_price": round(price, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
