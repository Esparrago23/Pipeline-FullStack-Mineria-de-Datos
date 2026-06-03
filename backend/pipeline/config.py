from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"

RAW_KEPLER_PATH = RAW_DIR / "cumulative_2026.06.01_20.09.17.csv"
RAW_PSCOMP_PATH = RAW_DIR / "PSCompPars_2026.06.01_20.09.10.csv"
KEPLER_PROCESSED_PATH = PROCESSED_DIR / "kepler_koi_processed.csv"
PSCOMP_PROCESSED_PATH = PROCESSED_DIR / "pscomppars_processed.csv"
DUCKDB_PATH = WAREHOUSE_DIR / "exoplanets.duckdb"

ANALYSIS_SUMMARY_PATH = REPORTS_DIR / "analysis_summary.json"
WAREHOUSE_SUMMARY_PATH = REPORTS_DIR / "warehouse_summary.json"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.json"

CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "regression_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

IDENTIFIER_COLUMNS = ["kepid", "kepoi_name", "kepler_name"]
LEAKAGE_COLUMNS = [
    "koi_score",
    "koi_pdisposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

CLASSIFICATION_FEATURES = [
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

REGRESSION_FEATURES = [
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


def ensure_project_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, WAREHOUSE_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, pd.DataFrame):
        return dataframe_records(value)
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if pd.isna(value):
        return None
    return value
