from __future__ import annotations

import json
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_MODEL_PATH,
    DUCKDB_PATH,
    MODEL_METADATA_PATH,
    MODEL_METRICS_PATH,
    REGRESSION_FEATURES,
    REGRESSION_MODEL_PATH,
    ensure_project_dirs,
    json_safe,
)


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


def load_modeling_data() -> pd.DataFrame:
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError("No se encontro el warehouse. Ejecuta primero la capa de datos.")
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
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
                koi_kepmag,
                star_temp_band,
                planet_radius_band,
                ra_bin,
                dec_bin
            FROM v_koi_observations
            """
        ).df()


def train_classification(kepler: pd.DataFrame) -> tuple[dict[str, Any], Pipeline, dict[str, float]]:
    classification_df = kepler[CLASSIFICATION_FEATURES + ["koi_disposition"]].dropna(
        subset=["koi_disposition"]
    ).copy()
    classification_df["target_confirmed"] = np.where(
        classification_df["koi_disposition"] == "CONFIRMED",
        "CONFIRMED",
        "NO_CONFIRMED",
    )

    x_clf = classification_df[CLASSIFICATION_FEATURES]
    y_clf = classification_df["target_confirmed"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_clf,
        y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )

    models = {
        "regresion_logistica": LogisticRegression(max_iter=3000),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "arbol_decision": DecisionTreeClassifier(max_depth=6, random_state=42),
        "naive_bayes": GaussianNB(),
    }

    f1_confirmed = make_scorer(f1_score, pos_label="CONFIRMED", zero_division=0)
    rows = []
    trained_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocesador", make_numeric_preprocessor(CLASSIFICATION_FEATURES)),
                ("modelo", model),
            ]
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        cv_f1 = cross_val_score(pipe, x_train, y_train, cv=5, scoring=f1_confirmed)
        rows.append(
            {
                "modelo": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, pos_label="CONFIRMED", zero_division=0),
                "Recall": recall_score(y_test, pred, pos_label="CONFIRMED", zero_division=0),
                "F1": f1_score(y_test, pred, pos_label="CONFIRMED", zero_division=0),
                "CV_F1_promedio": cv_f1.mean(),
                "CV_F1_desv": cv_f1.std(),
            }
        )
        trained_models[name] = pipe

    results = pd.DataFrame(rows).sort_values(["F1", "Recall"], ascending=False)
    best_name = str(results.iloc[0]["modelo"])
    best_model = trained_models[best_name]
    best_pred = best_model.predict(x_test)
    labels = ["CONFIRMED", "NO_CONFIRMED"]
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Real_{label}" for label in labels], columns=[f"Predicho_{label}" for label in labels])

    defaults = x_clf.median(numeric_only=True).to_dict()
    metrics = json_safe(
        {
            "target": "CONFIRMED vs NO_CONFIRMED",
            "positive_class": "CONFIRMED",
            "features": CLASSIFICATION_FEATURES,
            "best_model": best_name,
            "rows_total": len(classification_df),
            "split": {"train": len(x_train), "test": len(x_test), "test_size": 0.20, "random_state": 42},
            "class_distribution": y_clf.value_counts().rename_axis("clase").reset_index(name="n"),
            "model_results": results,
            "confusion_matrix": cm_df,
            "decision": (
                "Se elige el modelo con mejor F1 para CONFIRMED porque combina precision y recall; "
                "accuracy sola puede ocultar errores si las clases no estan balanceadas."
            ),
        }
    )
    return metrics, best_model, json_safe(defaults)


def train_regression(kepler: pd.DataFrame) -> tuple[dict[str, Any], Pipeline, dict[str, float]]:
    regression_df = kepler[REGRESSION_FEATURES + ["koi_prad"]].dropna(subset=["koi_prad"]).copy()
    x_reg = regression_df[REGRESSION_FEATURES]
    y_reg = np.log1p(regression_df["koi_prad"])

    x_train, x_test, y_train, y_test = train_test_split(
        x_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "regresion_lineal": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=10000),
    }

    rows = []
    trained_models: dict[str, Pipeline] = {}
    y_test_original = np.expm1(y_test)

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocesador", make_numeric_preprocessor(REGRESSION_FEATURES)),
                ("modelo", model),
            ]
        )
        pipe.fit(x_train, y_train)
        pred_log = pipe.predict(x_test)
        pred_original = np.expm1(pred_log)
        cv_r2 = cross_val_score(pipe, x_train, y_train, cv=5, scoring="r2")
        rows.append(
            {
                "modelo": name,
                "MSE_log": mean_squared_error(y_test, pred_log),
                "R2_log": r2_score(y_test, pred_log),
                "MSE_radio_tierra": mean_squared_error(y_test_original, pred_original),
                "RMSE_radio_tierra": np.sqrt(mean_squared_error(y_test_original, pred_original)),
                "CV_R2_promedio": cv_r2.mean(),
                "CV_R2_desv": cv_r2.std(),
            }
        )
        trained_models[name] = pipe

    results = pd.DataFrame(rows).sort_values("R2_log", ascending=False)
    best_name = str(results.iloc[0]["modelo"])
    best_model = trained_models[best_name]
    pred_log = best_model.predict(x_test)
    comparison = pd.DataFrame(
        {
            "y_real_radio_tierra": np.expm1(y_test),
            "y_pred_radio_tierra": np.expm1(pred_log),
            "y_real_log": y_test,
            "y_pred_log": pred_log,
        }
    ).head(15)

    defaults = x_reg.median(numeric_only=True).to_dict()
    metrics = json_safe(
        {
            "target": "log1p(koi_prad)",
            "reported_original_unit": "radio terrestre",
            "features": REGRESSION_FEATURES,
            "best_model": best_name,
            "rows_total": len(regression_df),
            "split": {"train": len(x_train), "test": len(x_test), "test_size": 0.20, "random_state": 42},
            "model_results": results,
            "sample_predictions": comparison,
            "decision": (
                "Se compara regresion lineal, Ridge y Lasso sobre log1p(koi_prad); "
                "R2 y MSE en escala log miden ajuste. La RMSE en radios terrestres puede ser alta por outliers "
                "muy grandes, por eso el modelo se interpreta principalmente en escala log."
            ),
        }
    )
    return metrics, best_model, json_safe(defaults)


def run_modeling() -> dict[str, Any]:
    ensure_project_dirs()
    kepler = load_modeling_data()
    classification_metrics, classification_model, classification_defaults = train_classification(kepler)
    regression_metrics, regression_model, regression_defaults = train_regression(kepler)

    joblib.dump(classification_model, CLASSIFICATION_MODEL_PATH)
    joblib.dump(regression_model, REGRESSION_MODEL_PATH)

    metadata = json_safe(
        {
            "classification_model_path": str(CLASSIFICATION_MODEL_PATH),
            "regression_model_path": str(REGRESSION_MODEL_PATH),
            "classification_features": CLASSIFICATION_FEATURES,
            "regression_features": REGRESSION_FEATURES,
            "classification_defaults": classification_defaults,
            "regression_defaults": regression_defaults,
            "preprocessing": "SimpleImputer + StandardScaler dentro de Pipeline ajustado solo con train",
        }
    )
    metrics = {"classification": classification_metrics, "regression": regression_metrics, "metadata": metadata}

    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_modeling(), indent=2))
