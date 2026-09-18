<div align="center">

<br/>

# Plant Disease Classification — MLOps Service

**An end-to-end computer-vision MLOps pipeline that detects tomato leaf disease from a photo, from versioned raw images to a containerized, ONNX-served API.**

DVC-versioned data · W&B hyperparameter sweeps · MLflow model registry · ONNX-optimized inference · FastAPI backend · Static frontend · Docker Compose orchestration · Terraform-provisioned infra · GitHub Actions CI/CD

<br/>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.14x-009688?logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?logo=mlflow&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/W%26B-Sweeps-FFBE00?logo=weightsandbiases&logoColor=black)
![Terraform](https://img.shields.io/badge/Terraform-Infra-7B42BC?logo=terraform&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9)

<br/>

[Overview](#overview) · [Architecture](#architecture) · [Pipeline Stages](#pipeline-stages) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Testing](#testing) · [Configuration](#configuration) · [CI/CD](#cicd) · [Infrastructure](#infrastructure-terraform) · [Tech Stack](#tech-stack)

<br/>

</div>

---

> This README documents the `feature/terraform` branch, which adds Terraform-provisioned infrastructure and a Terraform-plan job to the CD workflow on top of the existing training/serving pipeline.

## Overview

This project is a **complete, containerized MLOps pipeline** for classifying tomato leaf diseases from an uploaded image, covering four classes: `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`, and `Tomato___Leaf_Mold`.

It takes the model lifecycle from a versioned image dataset to a served API: raw images are pulled through **DVC** (backed by Google Drive), a **MobileNetV2** transfer-learning model is tuned via **Weights & Biases** Bayesian sweeps, the best configuration is retrained with early stopping and registered in **MLflow**'s model registry, the approved version is exported to **ONNX** for fast, TensorFlow-free inference, and the whole system — API and frontend — is packaged into two Docker images orchestrated with Docker Compose (with the supporting network already provisioned via Terraform).

| Capability | Details |
|---|---|
| **Versioned data** | The raw image dataset is tracked with `DVC` and stored on a Google Drive remote, not committed to Git |
| **Hyperparameter search** | `W&B Sweeps` (Bayesian, optimizing `val_accuracy`) searches learning rate, batch size, dropout, and dense-layer width |
| **Experiment tracking & registry** | `MLflow` logs params/metrics for the best run and registers the model; only versions tagged `model_status=approved` are promoted |
| **Optimized serving** | The registered Keras model is exported to **ONNX** via `tf2onnx` and served with `onnxruntime` instead of TensorFlow |
| **Typed API** | FastAPI accepts an uploaded image (`multipart/form-data`) and returns the predicted class + confidence |
| **Automated tests** | Pytest suite covering data integrity, preprocessing, training, inference, and the HTTP API (with mocked W&B/MLflow/ONNX calls) |
| **Full containerization** | Separate Docker images for the FastAPI backend and the static frontend, wired together with `docker-compose.yml` |
| **CI/CD** | GitHub Actions workflows for linting/tests/compose smoke-tests, on-demand model (re)training, and building/pushing Docker images |
| **Infrastructure as code** | `Terraform` provisions the Docker network the services run on, with `terraform plan` running in CI |
| **Environment-driven config** | Data/model paths and tracking credentials are injected via `.env` / `pydantic-settings`, not hardcoded |

> **Who is this for?** Anyone learning how to take a computer-vision model from a training notebook to a served, containerized API — including dataset versioning, hyperparameter sweeps, a model registry with an approval gate, ONNX export, and CI/CD around all of it.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                Plant Disease Classification — Architecture                    │
└───────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────┐     ┌────────────────────────┐     ┌───────────────────────┐
  │  DATA LAYER   │     │   TRAINING (offline)    │     │   TRACKING & REGISTRY │
  │               │     │                         │     │                       │
  │ data/raw/     │     │ training/train.py        │     │  W&B  ─ sweeps,        │
  │  4 tomato     │────►│  ├─ W&B sweep (bayes)    │────►│        best-run pick  │
  │  disease      │     │  ├─ retrain best config  │     │  MLflow ─ params,      │
  │  classes      │     │  │   + EarlyStopping     │     │          metrics,      │
  │ versioned via │     │  ├─ register_model()     │────►│          registry,     │
  │ DVC (gdrive)  │     │  └─ export_from_mlflow() │     │          approval tag  │
  └───────────────┘     │                         │     └───────────┬───────────┘
                         │ training/export_to_onnx │                 │
                         │  Keras ──► ONNX (tf2onnx)┼─────────────────┘
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │     MODEL ARTIFACTS     │
                         │  models/*.keras          │
                         │  models/*.onnx           │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────────────────────────────┐
                    │     BACKEND  — FastAPI (:8000)           │
                    │     src/plant_disease_mlops/             │
                    │                                          │
                    │  lifespan → load ONNX model once          │
                    │  POST /predict   image → class + confidence│
                    │  GET  /health    model_loaded status       │
                    │  GET  /          liveness message           │
                    └──────────────────┬───────────────────────┘
                                       │ REST (CORS-enabled)
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │     FRONTEND — Static site (nginx :80)   │
                    │     frontend/index.html + script.js       │
                    │                                          │
                    │  Image upload → POST /predict → result    │
                    └─────────────────────────────────────────┘

     docker-compose.yml orchestrates backend + frontend, on the Docker
     network provisioned by terraform/main.tf. GitHub Actions runs CI
     (lint/tests/compose), an on-demand DVC + training workflow, and a
     CD workflow that builds/pushes images and runs `terraform plan`.
```

---

## Pipeline Stages

### 1 — Data Versioning

The raw image dataset (`data/raw/`, 4 class folders, ~5.4k images) is **not committed to Git**. It's tracked with `DVC` (`data/raw.dvc`) against a Google Drive remote (`.dvc/config`, `pydrive2.yaml`). `uv run dvc pull` fetches it locally; `dvc.yaml` defines the pipeline stages (`train`, `export_onnx`) with their dependencies/outputs, so `dvc repro` re-runs only the stages whose inputs changed.

### 2 — Preprocessing & Augmentation

`src/plant_disease_mlops/preprocessing.py → PlantDiseasePreprocessor` builds Keras `ImageDataGenerator`s for training (rescaling, rotation, shift, shear, zoom, horizontal flip — an 80/20 train/validation split) and validation (rescaling only), both targeting `224×224` RGB images. The same class also exposes `preprocess_image()`, used at inference time to turn a raw uploaded image into a `(1, 224, 224, 3)` float32 tensor.

### 3 — Hyperparameter Sweep

`training/train.py → Trainer.run_sweep()` launches a **W&B Bayesian sweep** (`training/sweep_config.py`) over learning rate, batch size, dropout, and dense-layer width (10 epochs, 10 runs), optimizing `val_accuracy`. Each run fine-tunes a `MobileNetV2` (frozen ImageNet backbone) + `GlobalAveragePooling2D` + `Dense`/`Dropout` head, logging metrics via `WandbMetricsLogger`.

### 4 — Best-Model Training & Registration

`Trainer.get_best_hyperparameters()` pulls the top sweep run by `val_accuracy` from the W&B API. `Trainer.train_best_model()` retrains that configuration with an `EarlyStopping` callback (`patience=3`, restoring best weights). `Trainer.register_model()` then logs params/metrics and the model itself to **MLflow**, registers it as `PlantDiseaseClassifier`, and tags the new version `model_status=approved`.

### 5 — ONNX Export

`Trainer.export_model_from_mlflow()` fetches the latest `approved` model version from the MLflow registry and saves it as a Keras model. `training/export_to_onnx.py` then loads that Keras model, exports it to TensorFlow's `SavedModel` format, and converts it to **ONNX** via `tf2onnx`, so the API never needs TensorFlow at inference time — only the lightweight `onnxruntime`.

### 6 — Model Serving

`src/plant_disease_mlops/inference.py → Inference` loads the ONNX model once at API startup. Each request is:

1. Read as raw image bytes from the uploaded file
2. Preprocessed to a `(1, 224, 224, 3)` float32 tensor (resize, rescale to `[0, 1]`)
3. Run through the ONNX session; the class with the highest softmax probability and its confidence score are returned

### 7 — API & Frontend

A FastAPI app (`src/plant_disease_mlops/main.py`) exposes the model behind three endpoints (see [API Reference](#api-reference)), with CORS opened for a static HTML/CSS/JS frontend (`frontend/`) served separately by nginx — a drag-and-drop image uploader with a live API status indicator and a confidence bar.

### 8 — Testing

Five pytest modules cover the pipeline end to end — see [Testing](#testing).

### 9 — CI/CD & Infrastructure

Three GitHub Actions workflows (lint/test/compose-smoke-test, on-demand DVC training, and Docker build/push + `terraform plan`) automate the pipeline above — see [CI/CD](#cicd) and [Infrastructure](#infrastructure-terraform).

---

## Project Structure

```
Plant-Disease-Classification-mini-project-2/    # branch: feature/terraform
├── .github/workflows/
│   ├── ci.yaml                       # Lint (ruff) + tests + docker-compose smoke test
│   ├── model-training.yaml           # Manual: dvc pull → dvc repro → dvc push
│   └── cd.yaml                       # Build/push Docker images + terraform plan
├── data/
│   └── raw.dvc                       # DVC pointer to the image dataset (gdrive remote)
├── src/
│   └── plant_disease_mlops/
│       ├── main.py                   # FastAPI app: /, /health, /predict
│       ├── inference.py              # ONNX model loading + prediction
│       ├── preprocessing.py          # PlantDiseasePreprocessor: generators + image preprocessing
│       ├── config.py                 # pydantic-settings, .env driven
│       ├── schemas.py                # (reserved for request/response models)
│       ├── logging.py                # (reserved for logging configuration)
│       └── __init__.py
├── training/
│   ├── train.py                      # Trainer: sweep → best model → register → export
│   ├── sweep_config.py               # W&B Bayesian sweep search space
│   ├── wandb_utils.py                # W&B login helper
│   ├── export_to_onnx.py             # Keras → SavedModel → ONNX (tf2onnx)
│   └── __init__.py
├── models/
│   ├── plant_disease_model.keras     # Trained Keras model (committed)
│   └── plant_disease_model.onnx      # ONNX-exported model (committed)
├── notebooks/
│   └── model.ipynb                   # Exploratory model development
├── frontend/
│   ├── index.html                    # Image upload UI
│   ├── script.js                     # Calls the /predict endpoint
│   ├── style.css
│   └── Dockerfile                    # nginx:alpine static image
├── terraform/
│   ├── main.tf                       # Docker provider + docker_network resource
│   ├── variables.tf                  # (currently empty — reserved for future variables)
│   └── outputs.tf                    # (currently empty — reserved for future outputs)
├── tests/
│   ├── test_data_integrity.py        # Dataset class/file-type sanity checks
│   ├── test_preprocessing.py         # Preprocessor unit tests
│   ├── test_train.py                 # Trainer unit tests (W&B/MLflow mocked)
│   ├── test_inference.py             # Inference/preprocessing tests
│   └── test_api.py                   # FastAPI endpoint tests
├── reports/                          # Reserved for generated evaluation reports
├── docker-compose.yml                # Orchestrates backend + frontend
├── Dockerfile                        # Backend image (python:3.12-slim + uv)
├── dvc.yaml / dvc.lock                # DVC pipeline stages (train, export_onnx)
├── .dvc/config                       # DVC remote configuration (gdrive)
├── pydrive2.yaml                     # Google Drive OAuth client config for DVC
├── pyproject.toml                    # uv-managed dependencies
├── uv.lock
├── .pre-commit-config.yaml           # ruff check --fix + ruff format on commit
└── .env.example                      # Environment variable template (fill in locally)
```

---

## Quick Start

### Prerequisites

```
Python 3.12+   git   uv   docker   docker-compose   dvc (via uv)
```

You'll also want accounts/credentials for **Weights & Biases** and **MLflow** if you plan to (re)run training, and access to the project's DVC Google Drive remote if you plan to pull the raw dataset.

### 1. Clone & install

```bash
git clone -b feature/terraform https://github.com/mohamed1-abdeldayem/Plant-Disease-Classification-mini-project-2.git
cd Plant-Disease-Classification-mini-project-2
uv sync --group dev
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in the required variables (relative paths resolve against the project root):

```
RAW_DATA_DIR_PATH=data/raw
MODEL_KERAS_PATH=models/plant_disease_model.keras
MODEL_ONNX_PATH=models/plant_disease_model.onnx
WANDB_KEY=your-wandb-api-key
WANDB_PROJECT_NAME=your-wandb-project
WANDB_ENTITY=your-wandb-entity
```

> A trained `models/plant_disease_model.onnx` is already committed to the repo, so the API works out of the box **without** pulling the dataset or retraining — the `RAW_DATA_DIR_PATH`/`WANDB_*` variables are only needed if you plan to retrain.

### 3. (Optional) Pull the versioned dataset

```bash
uv run dvc pull
```

### 4. (Optional) Retrain the model

```bash
uv run python training/train.py
```

This runs a W&B sweep, retrains the best configuration, registers it in MLflow, and exports `models/plant_disease_model.keras`. Alternatively, `uv run dvc repro` re-runs the `train`/`export_onnx` stages defined in `dvc.yaml`.

### 5. (Optional) Export to ONNX

```bash
uv run python training/export_to_onnx.py
```

### 6. Run the API locally

```bash
uv run uvicorn plant_disease_mlops.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/leaf.jpg"
```

### 7. Run the test suite

```bash
uv run pytest tests/ -v
```

### 8. Full Docker stack

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| Frontend | http://localhost:5500 |

---

## API Reference

Interactive docs auto-generated by FastAPI at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### `POST /predict` — Classify a leaf image

**Request:** `multipart/form-data` with a single field:

| Field | Type | Description |
|---|---|---|
| `image` | file (`.jpg` / `.jpeg` / `.png`) | Photo of a tomato leaf |

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@leaf.jpg"
```

**Response**

```json
{
  "predicted_class": "Tomato___Leaf_Mold",
  "confidence": 0.80
}
```

`predicted_class` is one of `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`. `confidence` is the model's softmax probability for that class.

---

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true}
```

---

### `GET /` — Liveness message

```bash
curl http://localhost:8000/
# {"message": "Plant-disease API is running"}
```

---

## Testing

```
tests/
├── test_data_integrity.py  — Dataset sanity: expected class folders exist,
│                              each class has images, no unexpected file types
├── test_preprocessing.py   — PlantDiseasePreprocessor: image shape/scale,
│                              train/validation generator configuration
├── test_train.py           — Trainer: model architecture, generator wiring,
│                              best-hyperparameter selection, MLflow registration,
│                              full pipeline orchestration (W&B/MLflow mocked)
├── test_inference.py       — Inference: model loads, image preprocessing shape,
│                              prediction logic (ONNX session mocked)
└── test_api.py             — FastAPI TestClient: /, /health, /predict (inference mocked)
```

```bash
uv run pytest tests/ -v
```

> `test_data_integrity.py` and `test_train.py`'s data-dependent parts require the raw dataset (`uv run dvc pull` first); CI runs only the dataset-independent suite (`test_preprocessing.py`, `test_inference.py`, `test_api.py`), since those mock out W&B, MLflow, and the ONNX session.

---

## Configuration

All paths and tracking credentials are supplied through environment variables loaded via `pydantic-settings` (`src/plant_disease_mlops/config.py`), resolved relative to the project root:

```bash
# .env
RAW_DATA_DIR_PATH=data/raw
MODEL_KERAS_PATH=models/plant_disease_model.keras
MODEL_ONNX_PATH=models/plant_disease_model.onnx
WANDB_KEY=your-wandb-api-key
WANDB_PROJECT_NAME=your-wandb-project
WANDB_ENTITY=your-wandb-entity
```

The training pipeline reads `RAW_DATA_DIR_PATH` and writes `MODEL_KERAS_PATH`, using the `WANDB_*` variables for sweep logging; the API reads `MODEL_ONNX_PATH` at startup to load the ONNX inference session. DVC's Google Drive remote is configured separately in `.dvc/config` (and, in CI, via the `DVC_GDRIVE_CLIENT_ID` / `DVC_GDRIVE_CLIENT_SECRET` secrets).

---

## Docker

The backend uses a slim, `uv`-driven build; the frontend is a plain static nginx image.

**Backend (`Dockerfile`)**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir uv
COPY src ./src
COPY models ./models
RUN uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "plant_disease_mlops.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend (`frontend/Dockerfile`)**
```dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
```

**Orchestration (`docker-compose.yml`)** builds and wires both services, with the frontend depending on the backend and reaching it over the network on port `8000`.

```bash
docker-compose up --build
```

---

## CI/CD

Three GitHub Actions workflows automate the pipeline:

| Workflow | Trigger | What it does |
|---|---|---|
| **`ci.yaml`** | Push to `main`/`feature/**`, PRs into `main` | Runs `ruff check` + `ruff format --check`, runs the dataset-independent pytest suite, then builds and boots the full `docker-compose` stack as a smoke test |
| **`model-training.yaml`** | Manual (`workflow_dispatch`) | Authenticates DVC against Google Drive, `dvc pull`s the dataset, `dvc repro`s the `train`/`export_onnx` stages, then `dvc push`s the resulting artifacts |
| **`cd.yaml`** | Push to `feature/cd`, `feature/terraform`, `main` | Builds and pushes the backend and frontend images to Docker Hub, then runs `terraform init` / `validate` / `plan` against the `terraform/` configuration |

---

## Infrastructure (Terraform)

`terraform/main.tf` currently provisions a single resource: a dedicated Docker network (`plant-disease-network`) via the `kreuzwerker/docker` provider, intended as the network the Compose services (or future Terraform-managed containers) run on. `variables.tf` and `outputs.tf` are scaffolded but currently empty, reserved for expanding this into a fuller infrastructure definition.

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Dataset | Tomato leaf images (4 classes) | Multi-class disease classification |
| Data versioning | `DVC` + Google Drive remote | Version and sync the raw image dataset without storing it in Git |
| ML | TensorFlow/Keras — `MobileNetV2` (frozen) + dense head | Transfer-learning image classifier |
| Hyperparameter search | `W&B Sweeps` (Bayesian, optimizing `val_accuracy`) | Tunes learning rate, batch size, dropout, dense units |
| Experiment tracking & registry | `MLflow` | Logs params/metrics, registers and version-tags approved models |
| Model export | `tf2onnx` | Converts the trained Keras model (via SavedModel) to ONNX |
| Serving runtime | `onnxruntime` | Fast, TensorFlow-free inference |
| API | FastAPI + `pydantic-settings` | Multipart image upload endpoint, env-driven config |
| Frontend | Static HTML / CSS / JS | Drag-and-drop image uploader, calls the API directly |
| Testing | Pytest + `pytest-mock` + FastAPI `TestClient` | Data integrity, preprocessing, training, inference, and API test coverage |
| Code quality | `pre-commit` + `ruff` (`ruff --fix`, `ruff-format`) | Lint/format enforced automatically on every commit |
| CI/CD | GitHub Actions | Lint/test/compose smoke test, on-demand training pipeline, image build/push + Terraform plan |
| Infrastructure | Terraform (`kreuzwerker/docker` provider) | Provisions the Docker network the services run on |
| Package management | `uv` | Dependency resolution and locking, with a separate `dev` dependency group |
| Containerization | Docker (backend + frontend) + Docker Compose | Reproducible local orchestration |

---

## Contributing

Pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

This repo uses [`pre-commit`](https://pre-commit.com/) with `ruff --fix` and `ruff-format`, so install the hooks once after cloning:

```bash
uv run pre-commit install
```

```bash
git checkout -b feature/your-feature
# make changes
uv run pytest tests/ -v
git commit -m "feat: add your feature"   # pre-commit lints/formats automatically
git push origin feature/your-feature
# open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [mohamed1-abdeldayem](https://github.com/mohamed1-abdeldayem) · [Repository](https://github.com/mohamed1-abdeldayem/Plant-Disease-Classification-mini-project-2/tree/feature/terraform)

</div>