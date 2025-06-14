import sys, os
sys.path.append(os.path.join(os.getcwd(), "src"))

import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnclassifier.utils.common import decodeImage
from cnnclassifier import logging
import subprocess
from threading import Thread
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set locale environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
app.config['APP_NAME'] = 'PoultryGuard AI'
CORS(app)

# --------- Prediction Pipeline ---------
class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, filename):
        img = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        preds = self.model.predict(img_array)
        result = np.argmax(preds)
        return result

# --------- Client App ---------
class ClientsApp:
    def __init__(self):
        self.filename = Path("InputImage.jpg")
        self.model_path = os.path.join("artifacts", "training", "model.keras")
        self.classifier = None  # Lazy initialization

    def get_classifier(self):
        if self.classifier is None:
            try:
                logging.info(f"Loading model from: {self.model_path}")
                self.classifier = PredictionPipeline(self.model_path)
                logging.info("Model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise
        return self.classifier

c1App = ClientsApp()

# --------- DVC-Based Training ---------
def run_training():
    subprocess.run(["dvc", "repro"])
    subprocess.run(["dvc", "push"])

# --------- Routes ---------
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html', app_name=app.config['APP_NAME'])

@app.route("/train", methods=['GET', 'POST'])
def trainRoute():
    Thread(target=run_training).start()
    logging.info("Training started via DVC asynchronously.")
    return f"Training started using DVC in {app.config['APP_NAME']}."

@app.route("/predict", methods=['POST'])
def predictionRoute():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        logging.debug("Decoding image...")
        decodeImage(data['image'], c1App.filename)
        logging.debug("Image decoded successfully.")

        logging.debug("Loading model and running prediction...")
        classifier = c1App.get_classifier()
        result = classifier.predict(str(c1App.filename))
        logging.debug(f"Prediction result: {result}")

        prediction = "Healthy" if result == 1 else "Coccidiosis"

        return jsonify({
            "success": True,
            "app": app.config['APP_NAME'],
            "result": [{"image": prediction}]
        })

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Exception during prediction: {str(e)}\n{tb}")
        return jsonify({"success": False, "error": "An internal error occurred"}), 500

# --------- Start Server ---------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting {app.config['APP_NAME']} server on port {port}...")
    app.run(host='0.0.0.0', port=port)
