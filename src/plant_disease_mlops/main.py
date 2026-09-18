from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import Inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""
    app.state.session = Inference()
    yield


app = FastAPI(
    title="Plant_disease_API",
    description="API for predicting plant disease",
    version="0.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
):
    image_bytes = await image.read()

    image_processed = request.app.state.session.preprocess_image(image_bytes)

    predicted_class, confidence = request.app.state.session.predict(image_processed)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
    }


@app.get("/health")
async def health_check(request: Request):
    return {"status": "healthy", "model_loaded": request.app.state.session is not None}


@app.get("/")
def root():
    return {"message": "Plant-disease API is running"}
