from plant_disease_mlops import Inference
import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def inference():
    return Inference()


def test_inference_initialization(inference):
    assert inference.session is not None
    assert inference.model_path.exists()


def test_preprocess_image(inference, tmp_path):
    image_path = tmp_path / "image.jpg"

    image = Image.new("RGB", (300, 300))
    image.save(image_path)

    result = inference.preprocess_image(image_path)

    assert result.shape == (1, 224, 224, 3)


def test_predict(mocker, inference):
    mock_session = mocker.Mock()

    mock_session.get_inputs.return_value = [mocker.Mock(name="input")]

    mock_session.get_inputs.return_value[0].name = "input"

    mock_session.run.return_value = [
        np.array([[0.05, 0.10, 0.80, 0.05]], dtype=np.float32)
    ]

    inference.session = mock_session

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)

    predicted_class, confidence = inference.predict(image)

    assert predicted_class == "Tomato___Leaf_Mold"
    assert confidence == pytest.approx(0.80)
