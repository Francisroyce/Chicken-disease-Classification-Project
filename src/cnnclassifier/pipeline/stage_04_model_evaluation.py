# stage_model_evaluation.py

from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.model_evaluation import Evaluation
from cnnclassifier import logger

STAGE_NAME = "Model Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Initializing ConfigurationManager...")
        config = ConfigurationManager()
        logger.info("ConfigurationManager initialized.")

        logger.info("Fetching evaluation configuration...")
        val_config = config.get_validation_config()
        logger.info("Evaluation configuration fetched.")

        logger.info("Initializing Evaluation class...")
        evaluation = Evaluation(val_config)
        logger.info("Evaluation class initialized.")

        logger.info("Starting model evaluation...")
        evaluation.evaluate()
        logger.info("Model evaluation completed successfully.")

        logger.info("Saving evaluation scores...")
        evaluation.save_score()
        logger.info("Evaluation scores saved successfully.")

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception("An error occurred during the evaluation pipeline execution.")
        raise e
