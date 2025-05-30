#components
import os
from datetime import datetime
import tensorflow as tf
from pathlib import Path
import time
from cnnclassifier.entity.config_entity import PrepareCallbacksConfig

class PrepCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self) -> tf.keras.callbacks.TensorBoard:
        """Creates a TensorBoard callback with timestamped log directory."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_log_dir = os.path.join(self.config.tensorboard_root_log_dir, f"tb_logs_{timestamp}")
        return tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)

    @property
    def _create_ckpt_callbacks(self) -> tf.keras.callbacks.ModelCheckpoint:
        """Creates a ModelCheckpoint callback."""
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )

    def get_tb_ckpt_callbacks(self) -> list:
        """Returns a list of TensorBoard and ModelCheckpoint callbacks."""
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]

       