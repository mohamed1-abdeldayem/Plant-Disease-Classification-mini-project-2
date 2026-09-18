from pathlib import Path

import pytest

from src.plant_disease_mlops import settings


@pytest.fixture
def data_path():
    return Path(settings.base_dir_path, settings.RAW_DATA_DIR_PATH)


@pytest.fixture
def data_classes() -> set:
    return {
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
    }


@pytest.fixture
def image_extensions():
    return {".jpeg", ".png", ".jpg"}


def test_expected_classes_exist(data_classes, data_path):
    actual_classes = {
        directory.name for directory in data_path.iterdir() if directory.is_dir()
    }

    assert actual_classes == data_classes


def test_each_class_contains_images(data_path, image_extensions):
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            images = [
                file
                for file in class_dir.iterdir()
                if file.suffix.lower() in image_extensions
            ]

            assert images, f"No images found in {class_dir}"


def test_only_image_files_exist(data_path, image_extensions):
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            for file in class_dir.iterdir():
                assert file.suffix.lower() in image_extensions, f"Invalid file: {file}"
