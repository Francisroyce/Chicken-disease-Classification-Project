# stage_model_training.py

from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.prepare_callbacks import PrepCallbacks
from cnnclassifier.components.training import Training
from cnnclassifier import logger

import matplotlib.pyplot as plt

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Starting model training pipeline...")
        try:
            config = ConfigurationManager()

            logger.info("Fetching PrepareCallbacks configuration...")
            prepare_callbacks_config = config.get_prepare_callback_config()
            prepare_callbacks = PrepCallbacks(config=prepare_callbacks_config)
            callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
            logger.info(f"Callback list ready with {len(callback_list)} callbacks.")

            logger.info("Fetching training configuration...")
            training_config = config.get_training_config()
            training = Training(config=training_config)

            logger.info("Starting model training...")
            history = training.train(callback_list=callback_list)

            logger.info("Plotting training history...")
            plot_history(history)

            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.exception(f"Pipeline failed due to: {e}")
            raise

def plot_history(history):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
