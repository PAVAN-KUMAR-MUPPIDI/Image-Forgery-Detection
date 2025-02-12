import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Ensure 'static' folder exists for storing uploaded images
os.makedirs("static", exist_ok=True)

# Load Pre-trained Model (Use compile=False to avoid optimizer issues)
MODEL_PATH = "image_forgery_detection_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found! Train the model first.")

model = load_model(MODEL_PATH, compile=False)

# Recompile model before using it
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Define image size
IMG_SIZE = (128, 128)

def predict_image(img_path):
    """Process image and make prediction."""
    img_path = os.path.abspath(img_path)  # Convert relative to absolute path
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) * 100
    
    return ("Authentic", confidence) if prediction >= 0.5 else ("Forged", 100 - confidence)

# Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    # Get prediction
    result, confidence = predict_image(file_path)

    return render_template("result.html", result=result, confidence=confidence, image_url=file_path)

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
