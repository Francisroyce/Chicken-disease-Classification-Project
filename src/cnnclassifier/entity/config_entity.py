from dataclasses import dataclass
from pathlib import Path
from typing import List

#entity for daat ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    expected_hash: str = None

#entity for Basemodel
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path                          # Not in YAML – likely set in code
    base_model_dir: Path                    # Not in YAML – likely set in code
    updated_base_model_dir: Path            # Not in YAML – likely set in code
    params_image_size: List[int]            # From YAML → MODEL_PARAMS.IMAGE_SIZE
    params_learning_rate: float             # From YAML → MODEL_PARAMS.LEARNING_RATE
    params_include_top: bool                # From YAML → MODEL_PARAMS.INCLUDE_TOP
    params_weights: str                     # From YAML → MODEL_PARAMS.WEIGHTS
    params_classes: int 

#entity for callbacks

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path