import os
import sys
import traceback
from threading import Thread
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- ✅ Ensure 'cnnclassifier' is accessible ---
src_path = os.path.join(os.getcwd(), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from cnnclassifier.utils.common import decodeImage
from cnnclassifier import logging

# ✅ Locale environment settings
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# ✅ Flask App Initialization
app = Flask(__name__)
app.config['APP_NAME'] = 'PoultryGuard AI'
CORS(app)

# ✅ Prediction Pipeline
class PredictionPipeline:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, filename):
        img = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        preds = self.model.predict(img_array)
        result = int(np.argmax(preds))
        return result

# ✅ App Logic Wrapper
class ClientsApp:
    def __init__(self):
        self.filename = Path("InputImage.jpg")
        self.model_path = os.path.join("artifacts", "training", "model.keras")
        self.classifier = None

    def get_classifier(self):
        if self.classifier is None:
            logging.info(f"Loading model from: {self.model_path}")
            self.classifier = PredictionPipeline(self.model_path)
            logging.info("Model loaded successfully.")
        return self.classifier

c1App = ClientsApp()

# ✅ Use main.py training pipeline (not DVC)
def run_training_pipeline():
    try:
        # Import your custom pipeline stages
        from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
        from cnnclassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
        from cnnclassifier.pipeline.stage_03_training import ModelTrainingPipeline
        from cnnclassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

        # Run each stage sequentially
        logging.info(">>> Running training pipeline...")

        DataIngestionTrainingPipeline().main()
        logging.info("✅ Data ingestion completed.")

        PrepareBaseModelTrainingPipeline().main()
        logging.info("✅ Base model prepared.")

        ModelTrainingPipeline().main()
        logging.info("✅ Model training completed.")

        ModelEvaluationPipeline().main()
        logging.info("✅ Model evaluation completed.")

        logging.info("✅ Training pipeline finished successfully.")

    except Exception as e:
        logging.error(f"❌ Error during training pipeline: {e}")
        raise

# ✅ ROUTES

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", app_name=app.config['APP_NAME'])

@app.route("/train", methods=["GET", "POST"])
def trainRoute():
    Thread(target=run_training_pipeline).start()
    logging.info("Training triggered asynchronously.")
    return f"Training started for {app.config['APP_NAME']}."

@app.route("/predict", methods=["POST"])
def predictionRoute():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        logging.debug("Decoding image...")
        decodeImage(data['image'], c1App.filename)

        logging.debug("Running prediction...")
        classifier = c1App.get_classifier()
        result = classifier.predict(str(c1App.filename))

        prediction = "Healthy" if result == 1 else "Coccidiosis"
        return jsonify({
            "success": True,
            "app": app.config['APP_NAME'],
            "result": [{"image": prediction}]
        })

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Prediction error: {e}\n{tb}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# ✅ App Entry Point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting {app.config['APP_NAME']} server on port {port}...")
    app.run(host="0.0.0.0", port=port)
