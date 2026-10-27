"""
Ham vs Spam - Backend API
FastAPI application for SMS / Email spam classification using a pre-trained Keras 3 model.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import warnings

# Suppress harmless warnings and ensure writable Keras cache
warnings.filterwarnings("ignore")
os.environ.setdefault("KERAS_HOME", "/tmp/keras")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import tensorflow as tf

# Use Keras 3 standalone if available, otherwise tf.keras
try:
    import keras
    load_model = keras.models.load_model
except ImportError:
    from tensorflow.keras.models import load_model


# ─────────────────────────────────────────────
# Constants & Model Path
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "model.keras"

_model = None


# ─────────────────────────────────────────────
# Lifespan – Load model at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

    print(f"Loading Keras model from {MODEL_PATH} ...", flush=True)
    try:
        _model = load_model(str(MODEL_PATH))
        # Warm up the model with a dummy prediction
        _ = _model(tf.constant(["Warmup test message"]), training=False)
        print("Model loaded and warmed up successfully ✓", flush=True)
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr, flush=True)
        raise exc

    yield

    _model = None
    print("Application shutdown complete.", flush=True)


# ─────────────────────────────────────────────
# App Setup & CORS
# ─────────────────────────────────────────────
app = FastAPI(
    title="KhushiDL - Ham V/s Spam API",
    description="Deep learning SMS/email classification API for Ham vs Spam.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"]
    if allowed_origins_env == "*"
    else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="Text message to classify")


class PredictResponse(BaseModel):
    label: str          # "Ham" or "Spam"
    confidence: float   # 0.0 to 1.0 (certainty of the ruling)
    message: str        # original message evaluated
    probability: float  # raw model sigmoid probability (0.0 to 1.0)


# ─────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────
def _predict(text: str) -> tuple[str, float, float]:
    """
    Run inference on the pre-trained model.
    The model includes a TextVectorization layer that expects string tensors directly.
    """
    text_cleaned = text.strip()
    tensor_input = tf.constant([text_cleaned])

    # Direct tensor call for low latency inference
    out = _model(tensor_input, training=False)
    prob = float(out[0][0])

    # In binary classification: >= 0.5 is Spam, < 0.5 is Ham
    label = "Spam" if prob >= 0.5 else "Ham"
    confidence = prob if label == "Spam" else (1.0 - prob)

    return label, round(confidence, 4), round(prob, 4)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", summary="Root health check")
def root():
    return {
        "status": "ok",
        "service": "KhushiDL - Ham V/s Spam API",
        "version": "1.0.0",
        "model_loaded": _model is not None,
    }


@app.get("/health", summary="Detailed health status")
def health():
    return {
        "status": "healthy" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "tensorflow_version": tf.__version__,
    }


@app.post("/predict", response_model=PredictResponse, summary="Classify text message")
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Classification model is not initialized.")

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be blank.")

    try:
        label, confidence, prob = _predict(body.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}") from exc

    return PredictResponse(
        label=label,
        confidence=confidence,
        message=body.message,
        probability=prob,
    )
