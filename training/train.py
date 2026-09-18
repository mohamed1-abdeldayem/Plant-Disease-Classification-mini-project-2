from tensorflow.keras.callbacks import (
    EarlyStopping,
)
from plant_disease_mlops import Preprocessor
from pathlib import Path
from plant_disease_mlops import settings
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from .wandb_utils import setup_wandb
from .sweep_config import SWEEP_CONFIG

import mlflow


class Trainer:
    def __init__(
        self,
    ):
        self.preprocessor = Preprocessor()
        self.model_export_path = Path(settings.base_dir_path, settings.MODEL_KERAS_PATH)
        self.wandb_project = settings.WANDB_PROJECT_NAME
        self.sweep_config = SWEEP_CONFIG
        self.wandb_entity = settings.WANDB_ENTITY

    def get_train_generator(
        self,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        return self.preprocessor.get_train_generator(
            image_size, batch_size, validation_split
        )

    def get_validation_generator(
        self,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        return self.preprocessor.get_validation_generator(
            image_size, batch_size, validation_split
        )

    def build_model(self, hp, image_size: tuple[int, int] = (224, 224)):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights="imagenet",
        )

        base_model.trainable = False

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    hp["dense_units"],
                    activation="relu",
                ),
                tf.keras.layers.Dropout(hp["dropout"]),
                tf.keras.layers.Dense(
                    4,
                    activation="softmax",
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, config=None):
        with wandb.init(
            project=self.wandb_project,
            config=config,
        ):
            config = wandb.config

            train_generator = self.get_train_generator(batch_size=config.batch_size)

            validation_generator = self.get_validation_generator(
                batch_size=config.batch_size,
            )

            model = self.build_model(config)

            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=config.epochs,
                callbacks=[
                    WandbMetricsLogger(),
                ],
            )

            return model, history

    def run_sweep(self):
        setup_wandb()

        sweep_id = wandb.sweep(
            sweep=self.sweep_config,
            project=self.wandb_project,
        )

        wandb.agent(
            sweep_id,
            function=self.train,
            count=10,
        )

        return sweep_id

    def get_best_hyperparameters(self, sweep_id):
        api = wandb.Api()

        sweep = api.sweep(f"{self.wandb_entity}/{self.wandb_project}/{sweep_id}")

        best_run = max(
            sweep.runs,
            key=lambda run: run.summary.get("val_accuracy", 0),
        )

        return dict(best_run.config)

    def train_best_model(self, best_hyper):
        best_hyperparameters = best_hyper
        train_generator = self.get_train_generator(
            batch_size=best_hyperparameters["batch_size"],
        )

        validation_generator = self.get_validation_generator(
            batch_size=best_hyperparameters["batch_size"],
        )

        model = self.build_model(best_hyperparameters)

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=best_hyperparameters["epochs"],
            callbacks=[
                EarlyStopping(
                    monitor="val_accuracy",
                    patience=3,
                    mode="max",
                    restore_best_weights=True,
                ),
            ],
        )

        return model, history

    def register_model(self, model, history, params):
        mlflow.set_experiment("Plant Disease Classification")

        metrics = {
            "best_val_accuracy": max(history.history["val_accuracy"]),
            "best_val_loss": min(history.history["val_loss"]),
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            model_info = mlflow.keras.log_model(
                model,
                name="model",
                registered_model_name="PlantDiseaseClassifier",
            )

            client = mlflow.MlflowClient()

            client.set_model_version_tag(
                name="PlantDiseaseClassifier",
                version=model_info.registered_model_version,
                key="model_status",
                value="approved",
            )

        return model_info

    def export_model_from_mlflow(self):
        client = mlflow.MlflowClient()

        model_name = "PlantDiseaseClassifier"
        tag_key = "model_status"
        tag_value = "approved"

        versions = client.search_model_versions(f"name='{model_name}'")

        approved_versions = [
            version for version in versions if version.tags.get(tag_key) == tag_value
        ]

        if not approved_versions:
            raise ValueError(
                f"No model version found with " f"tag '{tag_key}={tag_value}'"
            )

        approved_version = max(
            approved_versions,
            key=lambda version: int(version.version),
        )

        model_uri = f"models:/{model_name}/{approved_version.version}"

        model = mlflow.keras.load_model(model_uri)

        model.save(self.model_export_path)

        return self.model_export_path

    def run_pipeline(self):
        sweep_id = self.run_sweep()

        best_hyperparameters = self.get_best_hyperparameters(sweep_id)

        model, history = self.train_best_model(best_hyperparameters)

        self.register_model(
            model=model,
            history=history,
            params=best_hyperparameters,
        )

        self.export_model_from_mlflow()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_pipeline()
