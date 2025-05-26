# component
import os
import urllib.request as request
from zipfile import ZipFile, BadZipFile
import tensorflow as tf
from pathlib import Path
from cnnclassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            classes=self.config.params_classes
        )

        self.save_model(
            path=self.config.base_model_dir,
            model=self.model
        )

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(classes, activation='softmax')(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = PrepareBaseModel._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(
            path=self.config.updated_base_model_dir,
            model=self.full_model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
