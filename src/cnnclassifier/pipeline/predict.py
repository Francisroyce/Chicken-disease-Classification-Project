import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import json

class PredictionPipeline:
    def __init__(self, model_path, class_indices_path):
        """
        Args:
            model_path (str): Path to the trained keras model (.keras or .h5).
            class_indices_path (str): Path to the JSON file containing class indices.
        """
        self.model = load_model(model_path)
        
        # Load class indices from JSON file
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        # Reverse dict to map prediction index back to class name
        self.idx_to_class = {v: k for k, v in class_indices.items()}

    def predict(self, image_path):
        """
        Args:
            image_path (str): Path to the image to classify.
        
        Returns:
            str: Predicted class label.
        """
        # Load image with target size your model expects
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
        img_array = img_array / 255.0  # Normalize to [0, 1]

        preds = self.model.predict(img_array)
        pred_index = np.argmax(preds)
        predicted_class = self.idx_to_class.get(pred_index, "Unknown")

        return predicted_class


# Example usage:
if __name__ == "__main__":
    model_path = os.path.join("artifacts", "training", "model.keras")
    class_indices_path = os.path.join("artifacts", "training", "class_indices.json")

    pipeline = PredictionPipeline(model_path, class_indices_path)
    
    image_path = "InputImage.jpg"  # Replace with your test image path
    prediction = pipeline.predict(image_path)
    print(f"Predicted class: {prediction}")
