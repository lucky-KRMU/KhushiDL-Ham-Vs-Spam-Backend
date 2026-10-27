"""
Ham vs Spam - Backend API
FastAPI application for SMS spam classification using a pre-trained Keras model.
"""

import os
import re
import pickle
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "Model" / "model.keras"
TOKENIZER_PATH = Path(__file__).parent / "Model" / "tokenizer.pkl"
MAX_LEN = 150  # must match training configuration


# ─────────────────────────────────────────────
# Global model / tokenizer holders
# ─────────────────────────────────────────────
_model = None
_tokenizer = None


def preprocess_text(text: str) -> str:
    """Basic text cleaning matching training pipeline."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # strip URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Lifespan – load model once at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    print("Loading model …", flush=True)
    _model = load_model(str(MODEL_PATH))
    print("Model loaded ✓", flush=True)

    if TOKENIZER_PATH.exists():
        print("Loading tokenizer …", flush=True)
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
        print("Tokenizer loaded ✓", flush=True)
    else:
        print("⚠  tokenizer.pkl not found – using character-level fallback", flush=True)

    yield  # app runs here

    print("Shutting down …", flush=True)


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Ham vs Spam API",
    description="Classify SMS / email messages as Ham (legitimate) or Spam.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow all origins in dev; tighten in production via env var
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"]
    if allowed_origins_env == "*"
    else [o.strip() for o in allowed_origins_env.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    message: str

    @validator("message")
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message must not be empty.")
        if len(v) > 5000:
            raise ValueError("Message is too long (max 5 000 characters).")
        return v.strip()


class PredictResponse(BaseModel):
    label: str          # "Ham" | "Spam"
    confidence: float   # 0.0 – 1.0
    message: str        # original (cleaned) message


# ─────────────────────────────────────────────
# Helper: run inference
# ─────────────────────────────────────────────
def _predict(text: str) -> tuple[str, float]:
    """Return (label, confidence) for the given text."""
    cleaned = preprocess_text(text)

    if _tokenizer is not None:
        seq = _tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        x = np.array(padded, dtype=np.float32)
    else:
        # Character-level fallback: encode each char as its ASCII value / 127
        encoded = [min(ord(c) / 127.0, 1.0) for c in cleaned[:MAX_LEN]]
        encoded += [0.0] * (MAX_LEN - len(encoded))
        x = np.array([encoded], dtype=np.float32)

    prob = float(_model.predict(x, verbose=0)[0][0])

    # Convention: model output close to 1 → Spam
    label = "Spam" if prob >= 0.5 else "Ham"
    confidence = prob if label == "Spam" else 1.0 - prob
    return label, round(confidence, 4)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "Ham vs Spam API", "version": "1.0.0"}


@app.get("/health", summary="Detailed health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "tokenizer_loaded": _tokenizer is not None,
    }


@app.post("/predict", response_model=PredictResponse, summary="Classify a message")
def predict(body: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        label, confidence = _predict(body.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictResponse(
        label=label,
        confidence=confidence,
        message=body.message,
    )
