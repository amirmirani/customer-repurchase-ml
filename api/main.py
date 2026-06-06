from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "features.json"
PREDICTION_THRESHOLD = 0.5


def load_feature_config() -> dict[str, Any]:
    with FEATURES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model() -> Any:
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


FEATURE_CONFIG = load_feature_config()
FEATURE_COLUMNS = FEATURE_CONFIG["feature_columns"]
FEATURE_DEFAULTS = FEATURE_CONFIG["defaults"]
MODEL = load_model()


class PredictionRequest(BaseModel):
    orders_last_30d: int | None = Field(default=None, ge=0)
    orders_last_90d: int | None = Field(default=None, ge=0)
    total_spend_90d: float | None = Field(default=None, ge=0)
    avg_order_value_90d: float | None = Field(default=None, ge=0)
    days_since_last_order: int | None = Field(default=None, ge=0)
    customer_tenure_days: int | None = Field(default=None, ge=0)
    category_diversity_90d: int | None = Field(default=None, ge=0)
    avg_quantity_per_order: float | None = Field(default=None, ge=0)
    favorite_category: str | None = Field(default=None, min_length=1)


class PredictionResponse(BaseModel):
    repurchase_probability: float
    prediction: int


app = FastAPI(
    title="Customer Repurchase Prediction API",
    description="FastAPI inference service for customer repurchase prediction.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


def request_to_features(request: PredictionRequest) -> pd.DataFrame:
    request_values = request.model_dump()
    row = {
        feature: request_values.get(feature)
        if request_values.get(feature) is not None
        else FEATURE_DEFAULTS[feature]
        for feature in FEATURE_COLUMNS
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    features = request_to_features(request)
    probability = float(MODEL.predict_proba(features)[0, 1])
    prediction = int(probability >= PREDICTION_THRESHOLD)

    return PredictionResponse(
        repurchase_probability=round(probability, 4),
        prediction=prediction,
    )
