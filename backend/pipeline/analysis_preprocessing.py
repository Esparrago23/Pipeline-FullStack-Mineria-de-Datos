from __future__ import annotations

import json
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from .config import (
    ANALYSIS_SUMMARY_PATH,
    CLASSIFICATION_FEATURES,
    DATA_DIR,
    IDENTIFIER_COLUMNS,
    KEPLER_PROCESSED_PATH,
    LEAKAGE_COLUMNS,
    PSCOMP_PROCESSED_PATH,
    RAW_KEPLER_PATH,
    RAW_PSCOMP_PATH,
    REGRESSION_FEATURES,
    ensure_project_dirs,
    json_safe,
    project_relative,
)


KEPLER_HINTS = {
    "kepid": "nominal",
    "kepoi_name": "nominal",
    "kepler_name": "nominal",
    "koi_disposition": "nominal",
    "koi_pdisposition": "nominal",
    "koi_score": "numerico",
    "koi_fpflag_nt": "binario",
    "koi_fpflag_ss": "binario",
    "koi_fpflag_co": "binario",
    "koi_fpflag_ec": "binario",
    "koi_tce_plnt_num": "ordinal",
}

KEPLER_USAGE = {
    "kepid": "identificador",
    "kepoi_name": "identificador",
    "kepler_name": "identificador",
    "koi_disposition": "objetivo_clasificacion",
    "koi_prad": "objetivo_regresion",
    "koi_pdisposition": "excluir_fuga",
    "koi_score": "excluir_fuga",
    "koi_fpflag_nt": "excluir_fuga",
    "koi_fpflag_ss": "excluir_fuga",
    "koi_fpflag_co": "excluir_fuga",
    "koi_fpflag_ec": "excluir_fuga",
}

PSCOMP_HINTS = {
    "pl_name": "nominal",
    "hostname": "nominal",
    "discoverymethod": "nominal",
    "disc_facility": "nominal",
    "disc_year": "ordinal",
    "pl_controv_flag": "binario",
    "ttv_flag": "binario",
    "sy_snum": "ordinal",
    "sy_pnum": "ordinal",
    "rastr": "nominal",
    "decstr": "nominal",
    "st_spectype": "nominal",
    "pl_bmassprov": "nominal",
    "st_metratio": "nominal",
}

PSCOMP_USAGE = {
    "pl_name": "identificador",
    "hostname": "identificador",
    "discoverymethod": "analisis",
    "disc_facility": "analisis",
    "disc_year": "analisis",
    "pl_rade": "analisis",
    "pl_orbper": "analisis",
    "st_teff": "analisis",
    "st_rad": "analisis",
    "st_mass": "analisis",
}

NUMERIC_EDA_COLUMNS = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
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


def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace({"": pd.NA})
    return cleaned


def dataframe_info_text(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def summarize_dataframe(name: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": name,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "filas_duplicadas": int(df.duplicated().sum()),
        "celdas_nulas": int(df.isna().sum().sum()),
        "pct_nulos_total": round(float(df.isna().sum().sum() / df.size * 100), 2),
        "columnas_con_nulos": int((df.isna().sum() > 0).sum()),
    }


def classify_attributes(
    df: pd.DataFrame,
    hints: dict[str, str] | None = None,
    usage: dict[str, str] | None = None,
) -> pd.DataFrame:
    valid_types = {"nominal", "binario", "ordinal", "numerico"}
    hints = hints or {}
    usage = usage or {}
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        n_unique = int(series.nunique(dropna=True))

        if column in hints:
            data_type = hints[column]
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            data_type = "nominal"
        elif n_unique == 2:
            data_type = "binario"
        elif pd.api.types.is_integer_dtype(series) and n_unique <= 12:
            data_type = "ordinal"
        elif pd.api.types.is_numeric_dtype(series):
            data_type = "numerico"
        else:
            data_type = "nominal"

        if data_type not in valid_types:
            raise ValueError(f"Tipo no permitido para {column}: {data_type}")

        examples = series.dropna().astype(str).unique()[:4]
        rows.append(
            {
                "columna": column,
                "dtype_pandas": str(series.dtype),
                "tipo_dato": data_type,
                "uso": usage.get(column, "analisis"),
                "n_unique": n_unique,
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(float(series.isna().mean() * 100), 2),
                "ejemplos": list(examples),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["uso", "tipo_dato", "pct_missing", "columna"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def missing_summary(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    table = df.isna().agg(["sum", "mean"]).T
    table.columns = ["n_missing", "pct_missing"]
    table["pct_missing"] = (table["pct_missing"] * 100).round(2)
    return table.sort_values("pct_missing", ascending=False).head(n)


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    valid_columns = [column for column in columns if column in df.columns]
    return df[valid_columns].describe().T.round(3)


def detect_iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"La serie debe ser numerica, recibio {series.dtype}")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return ((series < low) | (series > high)).fillna(False)


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        mask = detect_iqr_outliers(df[column])
        rows.append(
            {
                "columna": column,
                "outliers": int(mask.sum()),
                "pct_outliers": round(float(mask.mean() * 100), 2),
                "min": df[column].min(),
                "mediana": df[column].median(),
                "max": df[column].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


def chi_square(df: pd.DataFrame, col_a: str, col_b: str) -> dict[str, Any]:
    subset = df[[col_a, col_b]].dropna()
    observed = pd.crosstab(subset[col_a], subset[col_b])
    chi2, p, dof, expected = stats.chi2_contingency(observed.values)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "observed": observed,
        "expected": expected_df,
    }


def distance_matrix(df: pd.DataFrame, metric: str = "euclidean") -> np.ndarray:
    non_numeric = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Solo se aceptan columnas numericas. No numericas: {non_numeric}")
    return pairwise_distances(df.values, metric=metric)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_KEPLER_PATH.exists() or not RAW_PSCOMP_PATH.exists():
        raise FileNotFoundError(
            "No se encontraron los CSV crudos en backend/data/raw. Revisa README.md para descargarlos."
        )
    kepler = clean_whitespace(pd.read_csv(RAW_KEPLER_PATH, comment="#"))
    pscomppars = clean_whitespace(pd.read_csv(RAW_PSCOMP_PATH, comment="#"))
    return kepler, pscomppars


def build_processed_tables(kepler: pd.DataFrame, pscomppars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_columns = IDENTIFIER_COLUMNS + ["koi_disposition", "koi_prad"] + CLASSIFICATION_FEATURES
    kepler_processed = kepler[classification_columns].dropna(subset=["koi_disposition"]).copy()
    kepler_processed["koi_prad_log"] = np.log1p(kepler_processed["koi_prad"])

    pscomp_columns = [
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "disc_facility",
        "pl_orbper",
        "pl_rade",
        "pl_bmasse",
        "pl_eqt",
        "st_teff",
        "st_rad",
        "st_mass",
        "sy_dist",
        "ra",
        "dec",
    ]
    pscomp_processed = pscomppars[pscomp_columns].copy()
    for column in ["pl_name", "hostname", "discoverymethod", "disc_facility"]:
        pscomp_processed[column] = pscomp_processed[column].fillna("Desconocido")

    return kepler_processed, pscomp_processed


def build_analysis_summary(
    kepler: pd.DataFrame,
    pscomppars: pd.DataFrame,
    kepler_processed: pd.DataFrame,
    pscomp_processed: pd.DataFrame,
) -> dict[str, Any]:
    class_distribution = (
        kepler["koi_disposition"].value_counts(dropna=False).rename_axis("clase").reset_index(name="n")
    )
    class_distribution["pct"] = (
        class_distribution["n"] / class_distribution["n"].sum() * 100
    ).round(2)

    correlations = (
        kepler[NUMERIC_EDA_COLUMNS]
        .corr(numeric_only=True)["koi_prad"]
        .drop("koi_prad")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )
    correlations.columns = ["variable", "correlacion_con_koi_prad"]

    clf_ready = kepler_processed[CLASSIFICATION_FEATURES + ["koi_disposition"]].dropna(subset=["koi_disposition"])
    x_clf = clf_ready[CLASSIFICATION_FEATURES]
    y_clf = clf_ready["koi_disposition"]
    x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
        x_clf,
        y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )

    reg_ready = kepler_processed[REGRESSION_FEATURES + ["koi_prad"]].dropna(subset=["koi_prad"])
    x_reg = reg_ready[REGRESSION_FEATURES]
    y_reg = np.log1p(reg_ready["koi_prad"])
    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
        x_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    return json_safe(
        {
            "data_dir": project_relative(DATA_DIR),
            "raw": [
                summarize_dataframe("Kepler KOI cumulative", kepler),
                summarize_dataframe("PSCompPars", pscomppars),
            ],
            "processed": [
                summarize_dataframe("kepler_koi_processed", kepler_processed),
                summarize_dataframe("pscomppars_processed", pscomp_processed),
            ],
            "attribute_types_kepler": classify_attributes(kepler, KEPLER_HINTS, KEPLER_USAGE).head(30),
            "attribute_types_pscomppars": classify_attributes(pscomppars, PSCOMP_HINTS, PSCOMP_USAGE).head(30),
            "missing_kepler_top": missing_summary(kepler, n=12),
            "missing_pscomppars_top": missing_summary(pscomppars, n=12),
            "numeric_summary_kepler": numeric_summary(kepler, NUMERIC_EDA_COLUMNS),
            "outliers_iqr_kepler": outlier_summary(kepler, NUMERIC_EDA_COLUMNS),
            "class_distribution": class_distribution,
            "radius_correlations": correlations,
            "preprocessing_decisions": {
                "excluded_identifiers": IDENTIFIER_COLUMNS,
                "excluded_leakage": LEAKAGE_COLUMNS,
                "classification_features": CLASSIFICATION_FEATURES,
                "regression_features": REGRESSION_FEATURES,
                "split": "train_test_split antes de ajustar imputacion/escalado en los pipelines de modelado",
                "scaling": "SimpleImputer(strategy='median') + StandardScaler dentro de Pipeline",
            },
            "split_shapes": {
                "classification": {
                    "X_train": list(x_train_clf.shape),
                    "X_test": list(x_test_clf.shape),
                    "y_train": list(y_train_clf.shape),
                    "y_test": list(y_test_clf.shape),
                },
                "regression": {
                    "X_train": list(x_train_reg.shape),
                    "X_test": list(x_test_reg.shape),
                    "y_train": list(y_train_reg.shape),
                    "y_test": list(y_test_reg.shape),
                },
            },
        }
    )


def run_analysis_preprocessing() -> dict[str, Any]:
    ensure_project_dirs()
    kepler, pscomppars = load_raw_data()
    kepler_processed, pscomp_processed = build_processed_tables(kepler, pscomppars)

    kepler_processed.to_csv(KEPLER_PROCESSED_PATH, index=False)
    pscomp_processed.to_csv(PSCOMP_PROCESSED_PATH, index=False)

    summary = build_analysis_summary(kepler, pscomppars, kepler_processed, pscomp_processed)
    ANALYSIS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_analysis_preprocessing(), indent=2))
