from pathlib import Path
import subprocess

import tensorflow as tf

from plant_disease_mlops import settings


def export_to_onnx() -> Path:
    keras_model_path = Path(
        settings.base_dir_path,
        settings.MODEL_KERAS_PATH,
    )

    onnx_model_path = Path(
        settings.base_dir_path,
        settings.MODEL_ONNX_PATH,
    )

    saved_model_path = Path(settings.base_dir_path) / "models" / "saved_model"

    model = tf.keras.models.load_model(keras_model_path)

    model.export(saved_model_path)

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(saved_model_path),
            "--output",
            str(onnx_model_path),
        ],
        check=True,
    )

    return onnx_model_path


if __name__ == "__main__":
    export_to_onnx()
