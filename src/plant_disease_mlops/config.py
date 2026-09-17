from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    RAW_DATA_DIR_PATH: Path
    MODEL_KERAS_PATH: Path
    MODEL_ONNX_PATH: Path
    WANDB_KEY: str
    WANDB_PROJECT_NAME: str
    WANDB_ENTITY: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8"
    )

    @property
    def base_dir_path(self):
        return BASE_DIR


settings = Settings()
