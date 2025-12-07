# app.py
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------- Configuration ----------
UPLOAD_FOLDER = "static/uploads"
MODEL_PATHS = ["models/best_model.keras", "models/best_model.h5", "animal_skin_model.h5"]
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

CLASS_NAMES = [
    "allergy",
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "mange",
    "parasitic_infestation",
    "ringworm"
]

# ---------- Flask init ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------- Load model ----------
model = None
for p in MODEL_PATHS:
    try:
        if os.path.exists(p):
            model = load_model(p)
            print(f"Loaded model from: {p}")
            break
    except Exception as e:
        print(f"Failed loading {p}: {e}")

if model is None:
    raise FileNotFoundError(
        "No model found. Put your model at one of: " + ", ".join(MODEL_PATHS)
    )


# ---------- Helper functions ----------
def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXT


# ************ UPDATED predict_disease() WITH FIXED INVALID IMAGE LOGIC ************
def predict_disease(img_path: str):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)[0]
    preds = np.clip(preds, 0.0, 1.0)

    max_prob = float(np.max(preds))
    class_idx = int(np.argmax(preds))
    predicted_label = CLASS_NAMES[class_idx]
    confidence = max_prob * 100.0
    all_confidences = [float(p * 100.0) for p in preds]

    # ==============================
    # 🚨 UPDATED INVALID IMAGE CHECK
    # ==============================

    # 1️⃣ Model extremely unsure (< 40% confidence)
    if max_prob < 0.40:
        return "invalid_image", 0.0, all_confidences

    # 2️⃣ If top two probabilities are too close (< 10% difference)
    sorted_probs = sorted(preds, reverse=True)
    if sorted_probs[0] - sorted_probs[1] < 0.10:
        return "invalid_image", 0.0, all_confidences

    # ==============================

    # NORMAL CONFIDENCE RULES
    if predicted_label == "healthy":
        confidence = round(confidence, 2)
    else:
        if confidence > 50:
            confidence = 50.0

    return predicted_label, round(confidence, 2), all_confidences


# Load treatments
treatment_map = {}
if os.path.exists("treatment_map.json"):
    try:
        with open("treatment_map.json", "r", encoding="utf-8") as f:
            treatment_map = json.load(f)
    except Exception as e:
        print("Failed to load treatment_map.json:", e)


def get_treatment_text(label: str):
    return treatment_map.get(label, {}).get("treatment", "No treatment info available.")


def get_vet_advice(label: str):
    return treatment_map.get(label, {}).get("vet_advice", "No vet advice available.")


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        i = 1
        save_name = filename

        while os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], save_name)):
            save_name = f"{base}_{i}{ext}"
            i += 1

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(filepath)
        return jsonify({"filename": save_name})

    filename = request.args.get("file")
    if not filename:
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return redirect(url_for("index"))

    try:
        label, confidence, all_confidences = predict_disease(filepath)
    except Exception as e:
        return render_template(
            "result.html",
            uploaded_image="/" + filepath.replace(os.path.sep, "/"),
            prediction="Error",
            confidence=0.0,
            conf_color="#ff4d4d",
            treatment=f"Prediction error: {e}",
            vet_advice="",
            all_confidences=[0] * len(CLASS_NAMES),
            all_labels=CLASS_NAMES,
        )

    # ---------- FIXED INVALID IMAGE HANDLING ----------
    if label == "invalid_image":
        return render_template(
            "result.html",
            uploaded_image="/" + filepath.replace(os.path.sep, "/"),
            prediction="Error: Upload an ANIMAL SKIN image only",
            confidence=0.0,
            conf_color="#ff4d4d",
            treatment="This is not an animal skin image. Please upload a clear image of an animal skin disease.",
            vet_advice="",
            all_confidences=[0] * len(CLASS_NAMES),
            all_labels=CLASS_NAMES,
        )
    # --------------------------------------------------

    # Confidence color
    if confidence < 50:
        conf_color = "#ff4d4d"
    elif confidence < 80:
        conf_color = "#ffcc00"
    else:
        conf_color = "#00ff88"

    treatment_text = get_treatment_text(label)
    vet_text = get_vet_advice(label)

    web_path = "/" + filepath.replace(os.path.sep, "/")

    return render_template(
        "result.html",
        uploaded_image=web_path,
        prediction=label,
        confidence=confidence,
        conf_color=conf_color,
        treatment=treatment_text,
        vet_advice=vet_text,
        all_confidences=all_confidences,
        all_labels=CLASS_NAMES,
    )


# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
