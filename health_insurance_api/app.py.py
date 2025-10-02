from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from model_utils import train_model_from_csv, test_model_from_csv, predict_from_dict

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = train_model_from_csv(filepath, model_path=os.path.join(MODEL_FOLDER, "model.joblib"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = test_model_from_csv(filepath, model_path=os.path.join(MODEL_FOLDER, "model.joblib"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    try:
        result = predict_from_dict(data, model_path=os.path.join(MODEL_FOLDER, "model.joblib"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
