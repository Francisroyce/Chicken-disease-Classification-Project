#components
import os
from datetime import datetime
import tensorflow as tf
from pathlib import Path
import time
from cnnclassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = tf.keras.models.load_model(self.config.updated_base_model_dir)

    def train(self, callback_list: list):
        if self.config.params_is_augmentation:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=0.2
            )

        train_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="training"
        )

        val_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            subset="validation"
        )

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            metrics=["accuracy"]
        )

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator),
            callbacks=callback_list
        )

        self.save_model()  # ✅ Now this will work
        return history

    def save_model(self):
        self.model.save(self.config.trained_model_path)
