import pytest
from src.plant_disease_mlops import Preprocessor
import tensorflow as tf


@pytest.fixture
def preprocessor(image_size=(224, 224), batch_size=32, validation_split=0.2):
    return Preprocessor(
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
    )


@pytest.fixture
def sample_dataset(tmp_path):
    classes = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
    ]

    for class_name in classes:
        class_dir = tmp_path / class_name
        class_dir.mkdir()

        for i in range(5):
            image = tf.random.uniform(
                (300, 300, 3),
                maxval=255,
                dtype=tf.float32,
            )

            tf.keras.utils.save_img(
                class_dir / f"image_{i}.jpg",
                image,
            )

    return tmp_path


def test_preprocessor_initialization(preprocessor):
    assert preprocessor.image_size == (224, 224)
    assert preprocessor.batch_size == 32
    assert preprocessor.validation_split == 0.2


def test_preprocess_image_shape(tmp_path, preprocessor):
    image = tf.random.uniform(
        shape=(300, 300, 3),
        maxval=255,
        dtype=tf.float32,
    )

    image_path = tmp_path / "test.jpg"

    tf.keras.utils.save_img(
        image_path,
        image,
    )

    preprocessor = preprocessor

    result = preprocessor.preprocess_image(image_path)

    assert result.shape == (1, 224, 224, 3)


def test_train_generator(sample_dataset):
    preprocessor = Preprocessor(
        batch_size=4,
    )

    generator = preprocessor.get_train_generator(sample_dataset)

    assert generator.batch_size == 4
    assert generator.target_size == (224, 224)
    assert generator.class_mode == "categorical"


def test_validation_generator(sample_dataset):
    preprocessor = Preprocessor(
        batch_size=4,
    )

    generator = preprocessor.get_validation_generator(sample_dataset)

    assert generator.batch_size == 4
    assert generator.target_size == (224, 224)
    assert generator.class_mode == "categorical"
    assert generator.shuffle is False
