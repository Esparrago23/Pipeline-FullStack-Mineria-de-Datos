from __future__ import annotations

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
DB_PATH = ROOT_DIR / "mineria" / "data" / "warehouse" / "exoplanets.duckdb"

FEATURES_CLASIFICACION = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "ra",
    "dec",
    "koi_kepmag",
]

FEATURES_REGRESION = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "ra",
    "dec",
    "koi_kepmag",
]


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "koi_period": 9.49,
                "koi_impact": 0.15,
                "koi_duration": 3.4,
                "koi_depth": 615.8,
                "koi_teq": 900.0,
                "koi_insol": 180.0,
                "koi_model_snr": 35.0,
                "koi_steff": 5600.0,
                "koi_slogg": 4.4,
                "koi_srad": 1.0,
                "ra": 290.0,
                "dec": 44.5,
                "koi_kepmag": 14.2,
            }
        }
    )

    koi_period: float | None = Field(None, description="Periodo orbital en dias.")
    koi_impact: float | None = Field(None, description="Parametro de impacto del transito.")
    koi_duration: float | None = Field(None, description="Duracion del transito en horas.")
    koi_depth: float | None = Field(None, description="Profundidad del transito en ppm.")
    koi_teq: float | None = Field(None, description="Temperatura de equilibrio en K.")
    koi_insol: float | None = Field(None, description="Flujo de insolacion.")
    koi_model_snr: float | None = Field(None, description="Relacion senal/ruido del modelo.")
    koi_steff: float | None = Field(None, description="Temperatura efectiva de la estrella.")
    koi_slogg: float | None = Field(None, description="Gravedad superficial estelar.")
    koi_srad: float | None = Field(None, description="Radio estelar.")
    ra: float | None = Field(None, description="Ascension recta.")
    dec: float | None = Field(None, description="Declinacion.")
    koi_kepmag: float | None = Field(None, description="Magnitud Kepler.")


class ModelState:
    classifier: Pipeline | None = None
    regressor: Pipeline | None = None
    feature_defaults: dict[str, float | None] = {}
    class_counts: dict[str, int] = {}
    rows_loaded: int = 0
    error: str | None = None


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DB_PATH.exists():
        try:
            train_models()
        except Exception as exc:  # pragma: no cover - queda visible en /api/health
            model_state.error = str(exc)
    else:
        model_state.error = (
            "No se encontro mineria/data/warehouse/exoplanets.duckdb. "
            "Ejecuta primero el notebook 02_capa_datos_warehouse.ipynb."
        )
    yield


app = FastAPI(
    title="Pipeline Full Stack de Mineria de Datos",
    description="API para consultas OLAP e inferencia de exoplanetas Kepler.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_database() -> Path:
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No se encontro mineria/data/warehouse/exoplanets.duckdb. "
                "Ejecuta primero mineria/02_capa_datos_warehouse.ipynb."
            ),
        )
    return DB_PATH


def open_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(require_database()), read_only=True)


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def records_from_query(query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    con = open_connection()
    try:
        df = con.execute(query, params or []).df()
    finally:
        con.close()
    return [
        {column: clean_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def load_modeling_data() -> pd.DataFrame:
    con = open_connection()
    try:
        return con.execute(
            """
            SELECT
                star_id AS kepid,
                candidate_id AS kepoi_name,
                kepler_name,
                koi_disposition,
                planet_radius_earth AS koi_prad,
                orbital_period_days AS koi_period,
                transit_duration_hours AS koi_duration,
                transit_depth_ppm AS koi_depth,
                equilibrium_temp_k AS koi_teq,
                insolation_flux AS koi_insol,
                model_snr AS koi_model_snr,
                impact_parameter AS koi_impact,
                koi_steff,
                koi_slogg,
                koi_srad,
                ra,
                dec,
                koi_kepmag
            FROM v_koi_observations
            """
        ).df()
    finally:
        con.close()


def make_numeric_preprocessor(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline(
                    steps=[
                        ("imputar_mediana", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )


def train_models() -> None:
    kepler = load_modeling_data()
    model_state.rows_loaded = int(len(kepler))

    model_state.feature_defaults = {
        feature: clean_value(float(pd.to_numeric(kepler[feature], errors="coerce").median()))
        for feature in FEATURES_CLASIFICACION
    }

    clasificacion_df = kepler[FEATURES_CLASIFICACION + ["koi_disposition"]].dropna(
        subset=["koi_disposition"]
    )
    y_clf = np.where(
        clasificacion_df["koi_disposition"] == "CONFIRMED",
        "CONFIRMED",
        "NO_CONFIRMED",
    )
    clf_pipeline = Pipeline(
        steps=[
            ("preprocesador", make_numeric_preprocessor(FEATURES_CLASIFICACION)),
            ("modelo", LogisticRegression(max_iter=3000)),
        ]
    )
    clf_pipeline.fit(clasificacion_df[FEATURES_CLASIFICACION], y_clf)

    regresion_df = kepler[FEATURES_REGRESION + ["koi_prad"]].dropna(subset=["koi_prad"])
    y_reg = np.log1p(regresion_df["koi_prad"])
    reg_pipeline = Pipeline(
        steps=[
            ("preprocesador", make_numeric_preprocessor(FEATURES_REGRESION)),
            ("modelo", Ridge(alpha=1.0)),
        ]
    )
    reg_pipeline.fit(regresion_df[FEATURES_REGRESION], y_reg)

    class_counts = pd.Series(y_clf).value_counts().to_dict()
    model_state.class_counts = {str(key): int(value) for key, value in class_counts.items()}
    model_state.classifier = clf_pipeline
    model_state.regressor = reg_pipeline
    model_state.error = None


def ensure_models() -> None:
    if model_state.classifier is None or model_state.regressor is None:
        if model_state.error:
            raise HTTPException(status_code=503, detail=model_state.error)
        train_models()


def prediction_row(payload: PredictionInput, features: list[str]) -> pd.DataFrame:
    data = payload.model_dump()
    row = {
        feature: np.nan if data.get(feature) is None else data[feature]
        for feature in features
    }
    return pd.DataFrame([row], columns=features)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if DB_PATH.exists() and model_state.error is None else "warning",
        "database_exists": DB_PATH.exists(),
        "database_path": str(DB_PATH),
        "models_ready": model_state.classifier is not None and model_state.regressor is not None,
        "rows_loaded": model_state.rows_loaded,
        "class_counts": model_state.class_counts,
        "error": model_state.error,
    }


@app.get("/api/model/default-input")
def default_input() -> dict[str, Any]:
    ensure_models()
    return {
        "features": FEATURES_CLASIFICACION,
        "defaults": model_state.feature_defaults,
    }


@app.post("/api/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    ensure_models()
    assert model_state.classifier is not None
    assert model_state.regressor is not None

    clf_input = prediction_row(payload, FEATURES_CLASIFICACION)
    reg_input = prediction_row(payload, FEATURES_REGRESION)

    predicted_class = str(model_state.classifier.predict(clf_input)[0])
    class_probabilities: dict[str, float] = {}
    if hasattr(model_state.classifier, "predict_proba"):
        classes = model_state.classifier.named_steps["modelo"].classes_
        probabilities = model_state.classifier.predict_proba(clf_input)[0]
        class_probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities, strict=False)
        }

    predicted_log_radius = float(model_state.regressor.predict(reg_input)[0])
    predicted_radius = max(0.0, float(np.expm1(predicted_log_radius)))

    return {
        "classification": {
            "predicted_class": predicted_class,
            "probabilities": class_probabilities,
        },
        "regression": {
            "predicted_log_radius": round(predicted_log_radius, 4),
            "predicted_radius_earth": round(predicted_radius, 4),
        },
        "models": {
            "classification": "LogisticRegression",
            "regression": "Ridge",
        },
    }


@app.get("/api/olap/disposition")
def olap_disposition() -> dict[str, Any]:
    rows = records_from_query(
        """
        SELECT
            koi_disposition,
            COUNT(*) AS n_observations,
            ROUND(AVG(planet_radius_earth), 3) AS avg_radius_earth,
            ROUND(MEDIAN(planet_radius_earth), 3) AS median_radius_earth,
            ROUND(AVG(model_snr), 3) AS avg_snr
        FROM v_koi_observations
        GROUP BY koi_disposition
        ORDER BY n_observations DESC
        """
    )
    return {"rows": rows}


@app.get("/api/olap/drilldown")
def olap_drilldown(
    limit: int = Query(30, ge=1, le=200),
) -> dict[str, Any]:
    rows = records_from_query(
        """
        SELECT
            koi_disposition,
            star_temp_band,
            planet_radius_band,
            COUNT(*) AS n_observations,
            ROUND(MEDIAN(planet_radius_earth), 3) AS median_radius_earth
        FROM v_koi_observations
        GROUP BY koi_disposition, star_temp_band, planet_radius_band
        ORDER BY n_observations DESC
        LIMIT ?
        """,
        [limit],
    )
    return {"rows": rows}


@app.get("/api/olap/habitable-slice")
def olap_habitable_slice() -> dict[str, Any]:
    rows = records_from_query(
        """
        SELECT
            koi_disposition,
            star_temp_band,
            planet_radius_band,
            COUNT(*) AS n_observations,
            ROUND(MEDIAN(equilibrium_temp_k), 2) AS median_teq
        FROM v_koi_observations
        WHERE equilibrium_temp_k BETWEEN 180 AND 320
          AND planet_radius_band IN ('Tipo Tierra', 'Super Tierra')
        GROUP BY koi_disposition, star_temp_band, planet_radius_band
        ORDER BY n_observations DESC
        """
    )
    return {"rows": rows}


@app.get("/api/olap/pivot")
def olap_pivot() -> dict[str, Any]:
    rows = records_from_query(
        """
        PIVOT v_koi_observations
        ON koi_disposition
        USING COUNT(*)
        GROUP BY star_temp_band
        ORDER BY star_temp_band
        """
    )
    return {"rows": rows}


@app.get("/api/olap/cube")
def olap_cube(
    min_count: int = Query(50, ge=1, le=1000),
    limit: int = Query(40, ge=1, le=200),
) -> dict[str, Any]:
    rows = records_from_query(
        """
        SELECT
            star_temp_band,
            planet_radius_band,
            koi_disposition,
            COUNT(*) AS n_observations,
            ROUND(AVG(model_snr), 3) AS avg_snr
        FROM v_koi_observations
        GROUP BY star_temp_band, planet_radius_band, koi_disposition
        HAVING COUNT(*) >= ?
        ORDER BY n_observations DESC
        LIMIT ?
        """,
        [min_count, limit],
    )
    return {"rows": rows}


@app.get("/api/records/sample")
def sample_records(limit: int = Query(8, ge=1, le=50)) -> dict[str, Any]:
    rows = records_from_query(
        """
        SELECT
            candidate_id,
            kepler_name,
            koi_disposition,
            star_temp_band,
            planet_radius_band,
            ROUND(planet_radius_earth, 3) AS planet_radius_earth,
            ROUND(orbital_period_days, 3) AS orbital_period_days,
            ROUND(model_snr, 3) AS model_snr
        FROM v_koi_observations
        ORDER BY model_snr DESC NULLS LAST
        LIMIT ?
        """,
        [limit],
    )
    return {"rows": rows}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
