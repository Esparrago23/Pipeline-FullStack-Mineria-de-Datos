from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.pipeline.config import (
    ANALYSIS_SUMMARY_PATH,
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_MODEL_PATH,
    DUCKDB_PATH,
    MODEL_METADATA_PATH,
    MODEL_METRICS_PATH,
    PROJECT_ROOT,
    REGRESSION_FEATURES,
    REGRESSION_MODEL_PATH,
    WAREHOUSE_SUMMARY_PATH,
    json_safe,
)
from backend.pipeline.run_pipeline import run_pipeline
from backend.pipeline.warehouse import OLAP_QUERIES, run_olap_query


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Pipeline Full Stack de Mineria de Datos",
    version="1.0.0",
    description="API para EDA, warehouse DuckDB, OLAP e inferencia de exoplanetas Kepler.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, float | None] = Field(default_factory=dict)


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require_metadata() -> dict[str, Any]:
    metadata = read_json(MODEL_METADATA_PATH)
    if metadata is None:
        raise HTTPException(status_code=404, detail="No hay modelos entrenados. Ejecuta /api/pipeline/run.")
    return metadata


def build_feature_frame(
    incoming: dict[str, float | None],
    features: list[str],
    defaults: dict[str, float],
) -> pd.DataFrame:
    row = {}
    for feature in features:
        value = incoming.get(feature, defaults.get(feature))
        row[feature] = value
    return pd.DataFrame([row], columns=features)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "warehouse_exists": DUCKDB_PATH.exists(),
        "classification_model_exists": CLASSIFICATION_MODEL_PATH.exists(),
        "regression_model_exists": REGRESSION_MODEL_PATH.exists(),
    }


@app.post("/api/pipeline/run")
def execute_pipeline() -> dict[str, Any]:
    try:
        return json_safe(run_pipeline())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    return {
        "analysis": read_json(ANALYSIS_SUMMARY_PATH),
        "warehouse": read_json(WAREHOUSE_SUMMARY_PATH),
        "modeling": read_json(MODEL_METRICS_PATH),
    }


@app.get("/api/olap")
def list_olap_queries() -> dict[str, Any]:
    return {"queries": sorted(OLAP_QUERIES)}


@app.get("/api/olap/{query_name}")
def olap(query_name: str, limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    try:
        df = run_olap_query(query_name, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"query": query_name, "rows": json_safe(df), "columns": list(df.columns)}


@app.get("/api/prediction-sample")
def prediction_sample() -> dict[str, Any]:
    metadata = require_metadata()
    return {
        "classification_features": CLASSIFICATION_FEATURES,
        "regression_features": REGRESSION_FEATURES,
        "classification_defaults": metadata["classification_defaults"],
        "regression_defaults": metadata["regression_defaults"],
    }


@app.post("/api/predict/classification")
def predict_classification(request: PredictionRequest) -> dict[str, Any]:
    metadata = require_metadata()
    if not CLASSIFICATION_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No existe el modelo de clasificacion entrenado.")

    model = joblib.load(CLASSIFICATION_MODEL_PATH)
    frame = build_feature_frame(
        request.features,
        CLASSIFICATION_FEATURES,
        metadata["classification_defaults"],
    )
    prediction = str(model.predict(frame)[0])
    result: dict[str, Any] = {"prediction": prediction, "features_used": json_safe(frame.iloc[0].to_dict())}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = [str(value) for value in model.classes_]
        result["probabilities"] = {label: float(prob) for label, prob in zip(classes, probabilities)}

    return result


@app.post("/api/predict/regression")
def predict_regression(request: PredictionRequest) -> dict[str, Any]:
    metadata = require_metadata()
    if not REGRESSION_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No existe el modelo de regresion entrenado.")

    model = joblib.load(REGRESSION_MODEL_PATH)
    frame = build_feature_frame(
        request.features,
        REGRESSION_FEATURES,
        metadata["regression_defaults"],
    )
    prediction_log = float(model.predict(frame)[0])
    prediction_radius = float(np.expm1(prediction_log))
    return {
        "prediction_log1p_koi_prad": prediction_log,
        "prediction_radius_earth": prediction_radius,
        "features_used": json_safe(frame.iloc[0].to_dict()),
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
