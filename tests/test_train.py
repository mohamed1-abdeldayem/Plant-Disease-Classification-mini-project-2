from training import Trainer
from unittest.mock import Mock
import pytest


@pytest.fixture
def trainer():
    return Trainer()


def test_build_model(trainer):
    hp = {
        "dense_units": 128,
        "dropout": 0.3,
        "learning_rate": 0.001,
    }

    model = trainer.build_model(hp)

    assert model.layers[-1].units == 4
    assert model.layers[-1].activation.__name__ == "softmax"
    assert model.layers[-2].rate == 0.3
    assert model.optimizer.learning_rate.numpy() == 0.001
    assert model.loss == "categorical_crossentropy"


def test_get_train_generator(trainer):
    fake_generator = Mock()

    trainer.preprocessor.get_train_generator = Mock(return_value=fake_generator)

    result = trainer.get_train_generator(
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
    )

    assert result is fake_generator

    trainer.preprocessor.get_train_generator.assert_called_once_with(
        (224, 224),
        32,
        0.2,
    )


def test_get_best_hyperparameters(mocker):
    trainer = Trainer()

    run1 = mocker.Mock()
    run1.summary = {"val_accuracy": 0.90}
    run1.config = {
        "batch_size": 32,
        "dense_units": 128,
    }

    run2 = mocker.Mock()
    run2.summary = {"val_accuracy": 0.96}
    run2.config = {
        "batch_size": 64,
        "dense_units": 256,
    }

    fake_sweep = mocker.Mock()
    fake_sweep.runs = [run1, run2]

    mock_api = mocker.Mock()
    mock_api.sweep.return_value = fake_sweep

    mocker.patch(
        "wandb.Api",
        return_value=mock_api,
    )

    result = trainer.get_best_hyperparameters("sweep123")

    assert result == {
        "batch_size": 64,
        "dense_units": 256,
    }


def test_register_model(mocker):
    trainer = Trainer()

    history = mocker.Mock()
    history.history = {
        "val_accuracy": [0.80, 0.90, 0.95],
        "val_loss": [0.5, 0.3, 0.2],
    }

    model = mocker.Mock()

    mock_model_info = mocker.Mock()
    mock_model_info.registered_model_version = 1

    mocker.patch(
        "mlflow.keras.log_model",
        return_value=mock_model_info,
    )

    mocker.patch("mlflow.MlflowClient")

    params = {
        "batch_size": 32,
        "epochs": 10,
    }

    result = trainer.register_model(
        model,
        history,
        params,
    )

    assert result is mock_model_info


def test_run_pipeline(mocker):
    trainer = Trainer()

    trainer.run_sweep = mocker.Mock(return_value="sweep123")

    trainer.get_best_hyperparameters = mocker.Mock(return_value={"batch_size": 32})

    trainer.train_best_model = mocker.Mock(return_value=("model", "history"))

    trainer.register_model = mocker.Mock()

    trainer.export_model_from_mlflow = mocker.Mock()

    trainer.run_pipeline()

    trainer.run_sweep.assert_called_once()

    trainer.get_best_hyperparameters.assert_called_once_with("sweep123")

    trainer.train_best_model.assert_called_once_with({"batch_size": 32})

    trainer.register_model.assert_called_once_with(
        model="model",
        history="history",
        params={"batch_size": 32},
    )

    trainer.export_model_from_mlflow.assert_called_once()
