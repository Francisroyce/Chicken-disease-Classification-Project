from cnnclassifier.constants import *
import os
from cnnclassifier.utils.common import  read_yaml, create_directories # Assumed utility file
from cnnclassifier.entity.config_entity import (DataIngestionConfig, 
                                                PrepareBaseModelConfig, PrepareCallbacksConfig)

# config manager for data ingestion
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAM_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['artifacts_root']])  # fix here

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
    

#configuration manager for model
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config['prepare_base_model']

        create_directories([config['root_dir']])  # ✅ Treat as dictionary

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
    

#configuration for callbacks
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAM_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['artifacts_root']])

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config['prepare_callbacks']  # dictionary-style access

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
