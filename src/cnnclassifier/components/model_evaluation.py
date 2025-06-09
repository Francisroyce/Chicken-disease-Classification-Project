from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from cnnclassifier.utils.common import save_json
from cnnclassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.config.trained_model_path)

    def _load_validation_data(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2  # same as training
        )

        self.val_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False  # important for label alignment
        )

    def _evaluate_model(self):
        self._load_model()
        self._load_validation_data()

        self.score = self.model.evaluate(self.val_generator)
        self.y_pred = self.model.predict(self.val_generator)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)
        self.y_true = self.val_generator.classes

        self.classification_report = classification_report(
            self.y_true, self.y_pred_classes, output_dict=True
        )
        self.conf_matrix = confusion_matrix(self.y_true, self.y_pred_classes)

    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "classification_report": self.classification_report,
            "confusion_matrix": self.conf_matrix.tolist()
        }

        save_path = Path("artifacts/evaluation/scores.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if missing

        save_json(path=save_path, data=scores)

    def evaluate(self):
        self._evaluate_model()
        self.save_score()
