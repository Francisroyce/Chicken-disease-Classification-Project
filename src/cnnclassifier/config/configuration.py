from cnnclassifier.constants import *
import os
from pathlib import Path
from cnnclassifier.utils.common import read_yaml, create_directories
from cnnclassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig
)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAM_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        create_directories([config['root_dir']])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config['root_dir'],
            source_URL=config['source_URL'],
            local_data_file=config['local_data_file'],
            unzip_dir=config['unzip_dir'],
            expected_hash=config.get('expected_hash')  # optional
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config['prepare_base_model']

        create_directories([config['root_dir']])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config['root_dir']),
            base_model_dir=Path(config['base_model_dir']),
            updated_base_model_dir=Path(config['updated_base_model_dir']),
            params_image_size=self.params['MODEL_PARAMS']['IMAGE_SIZE'],
            params_learning_rate=self.params['MODEL_PARAMS']['LEARNING_RATE'],
            params_include_top=self.params['MODEL_PARAMS']['INCLUDE_TOP'],
            params_weights=self.params['MODEL_PARAMS']['WEIGHTS'],
            params_classes=self.params['MODEL_PARAMS']['CLASSES']
        )

        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config['prepare_callbacks']

        model_ckpt_dir = os.path.dirname(config['checkpoint_model_filepath'])

        create_directories([
            Path(model_ckpt_dir),
            Path(config['tensorboard_root_log_dir'])
        ])

        prepare_callbacks_config = PrepareCallbacksConfig(
            root_dir=Path(config['root_dir']),
            tensorboard_root_log_dir=Path(config['tensorboard_root_log_dir']),
            checkpoint_model_filepath=Path(config['checkpoint_model_filepath'])
        )

        return prepare_callbacks_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config['training']
        prepare_base_model = self.config['prepare_base_model']
        params = self.params

        training_data = os.path.join("artifacts/data_ingestion", "Chicken-fecal-images")

        create_directories([Path(training['root_dir'])])
        training_config = TrainingConfig(
            root_dir=Path(training['root_dir']),
            trained_model_path=Path(training['trained_model_path']),
            updated_base_model_dir=Path(prepare_base_model['updated_base_model_dir']),
            training_data=Path(training_data),
            params_epochs=params['TRAINING']['EPOCHS'],
            params_batch_size=params['MODEL_PARAMS']['BATCH_SIZE'],
            params_is_augmentation=params['AUGMENTATION'],
            params_image_size=params['MODEL_PARAMS']['IMAGE_SIZE'],
            params_learning_rate=params['MODEL_PARAMS']['LEARNING_RATE']
        )

        return training_config
    


    def get_validation_config(self) -> EvaluationConfig:
        model_params = self.params.get("MODEL_PARAMS", {})
    
        image_size = model_params.get("IMAGE_SIZE")
        batch_size = model_params.get("BATCH_SIZE")
    
        if image_size is None or batch_size is None:
            raise KeyError("IMAGE_SIZE or BATCH_SIZE not found in MODEL_PARAMS section of params.yaml")
    
        eval_config = EvaluationConfig(
            trained_model_path=os.path.join("artifacts", "training", "model.keras"),
            training_data=os.path.join("artifacts", "data_ingestion", "Chicken-fecal-images"),
            all_params=self.params,
            params_image_size=image_size,
            params_batch_size=batch_size
        )
        return eval_config

    


 

