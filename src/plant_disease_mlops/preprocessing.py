from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class PlantDiseasePreprocessor:
    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> None:
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split

    def get_train_generator(
        self,
        data_dir: Path,
    ):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=self.validation_split,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        return train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

    def get_validation_generator(
        self,
        data_dir: Path,
    ):
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=self.validation_split,
        )

        return val_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    def preprocess_image(
        self,
        image_path: str | Path,
    ) -> tf.Tensor:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=self.image_size,
        )

        image = tf.keras.utils.img_to_array(image)

        image = image / 255.0

        image = tf.expand_dims(image, axis=0)

        return image
