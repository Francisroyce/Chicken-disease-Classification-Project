from cnnclassifier import logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnclassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnclassifier.pipeline.stage_03_training import ModelTrainingPipeline

#dataingestion
STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


#base model
STAGE_NAME = "Prepare base model"

try:
    logger.info("**************")
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Exception occurred in stage {STAGE_NAME}: {e}")
    raise


#training
STAGE_NAME = "Training"

try:
    logger.info("***********")
    logger.info(f">>>>> Stage: {STAGE_NAME} started <<<<<")
    
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    
    logger.info(f">>>>> Stage: {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(f"Stage {STAGE_NAME} failed due to: {e}")
    raise