from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.prepare_base_model import PrepareBaseModel
from cnnclassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        logger.info("Starting prepare base model pipeline...")
        try:
            config = ConfigurationManager()
            prepare_base_model_config = config.get_prepare_base_model_config()
            prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

            logger.info("Getting base model...")
            prepare_base_model.get_base_model()

            logger.info("Updating base model...")
            prepare_base_model.update_base_model()

            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.exception(f"Pipeline failed due to: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
