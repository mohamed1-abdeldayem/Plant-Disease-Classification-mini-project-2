from fastapi.testclient import TestClient

from plant_disease_mlops import app

import pytest


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Plant-disease API is running"}


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict(mocker, client):
    mock_inference = mocker.Mock()

    mock_inference.preprocess_image.return_value = "processed_image"

    mock_inference.predict.return_value = (
        "Tomato___Leaf_Mold",
        0.80,
    )

    app.state.session = mock_inference

    response = client.post(
        "/predict",
        files={
            "image": (
                "test.jpg",
                b"fake image content",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "predicted_class": "Tomato___Leaf_Mold",
        "confidence": pytest.approx(0.80),
    }
