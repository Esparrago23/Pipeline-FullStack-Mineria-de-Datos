from __future__ import annotations

import json
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .config import (
    DUCKDB_PATH,
    KEPLER_PROCESSED_PATH,
    PSCOMP_PROCESSED_PATH,
    WAREHOUSE_SUMMARY_PATH,
    dataframe_records,
    ensure_project_dirs,
    json_safe,
    project_relative,
)


def band_temperature(value: float) -> str:
    if pd.isna(value):
        return "Desconocida"
    if value < 3700:
        return "Fria"
    if value < 5200:
        return "Templada"
    if value < 6500:
        return "Solar"
    return "Caliente"


def band_radius(value: float) -> str:
    if pd.isna(value):
        return "Desconocido"
    if value < 1.25:
        return "Tipo Tierra"
    if value < 2.0:
        return "Super Tierra"
    if value < 6.0:
        return "Sub Neptuno"
    if value < 15.0:
        return "Gigante"
    return "Muy grande"


def sky_bin(value: float, size: int) -> str:
    if pd.isna(value):
        return "Desconocido"
    low = int(np.floor(value / size) * size)
    high = low + size
    return f"{low}-{high}"


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not KEPLER_PROCESSED_PATH.exists() or not PSCOMP_PROCESSED_PATH.exists():
        raise FileNotFoundError("Ejecuta primero la capa de analisis/preprocesamiento.")
    kepler = pd.read_csv(KEPLER_PROCESSED_PATH)
    pscomppars = pd.read_csv(PSCOMP_PROCESSED_PATH)
    for df in (kepler, pscomppars):
        for column in df.select_dtypes(include=["object", "string"]).columns:
            df[column] = df[column].astype("string").str.strip().replace({"": pd.NA})
    return kepler, pscomppars


def build_dimension_tables(kepler: pd.DataFrame, pscomppars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    koi_base = kepler.copy()
    koi_base["star_temp_band"] = koi_base["koi_steff"].apply(band_temperature)
    koi_base["planet_radius_band"] = koi_base["koi_prad"].apply(band_radius)
    koi_base["ra_bin"] = koi_base["ra"].apply(lambda value: sky_bin(value, 30))
    koi_base["dec_bin"] = koi_base["dec"].apply(lambda value: sky_bin(value + 90, 30))

    planet_base = pscomppars.copy()
    planet_base["star_temp_band"] = planet_base["st_teff"].apply(band_temperature)
    planet_base["planet_radius_band"] = planet_base["pl_rade"].apply(band_radius)
    planet_base["ra_bin"] = planet_base["ra"].apply(lambda value: sky_bin(value, 30))
    planet_base["dec_bin"] = planet_base["dec"].apply(lambda value: sky_bin(value + 90, 30))

    dim_star = (
        koi_base[["kepid", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag", "star_temp_band"]]
        .drop_duplicates("kepid")
        .copy()
        .rename(columns={"kepid": "star_id"})
    )
    dim_star.insert(0, "star_key", range(1, len(dim_star) + 1))

    dim_candidate = (
        koi_base[["kepoi_name", "kepler_name", "planet_radius_band"]]
        .drop_duplicates("kepoi_name")
        .copy()
        .rename(columns={"kepoi_name": "candidate_id"})
    )
    dim_candidate.insert(0, "candidate_key", range(1, len(dim_candidate) + 1))

    dim_disposition = pd.DataFrame(
        {"koi_disposition": sorted(koi_base["koi_disposition"].dropna().unique())}
    )
    dim_disposition.insert(0, "disposition_key", range(1, len(dim_disposition) + 1))

    dim_sky_position = koi_base[["ra", "dec", "ra_bin", "dec_bin"]].drop_duplicates().copy()
    dim_sky_position.insert(0, "sky_key", range(1, len(dim_sky_position) + 1))

    fact_koi = koi_base.merge(dim_star[["star_key", "star_id"]], left_on="kepid", right_on="star_id", how="left")
    fact_koi = fact_koi.merge(
        dim_candidate[["candidate_key", "candidate_id"]],
        left_on="kepoi_name",
        right_on="candidate_id",
        how="left",
    )
    fact_koi = fact_koi.merge(dim_disposition, on="koi_disposition", how="left")
    fact_koi = fact_koi.merge(dim_sky_position, on=["ra", "dec", "ra_bin", "dec_bin"], how="left")
    fact_koi = fact_koi[
        [
            "candidate_key",
            "star_key",
            "disposition_key",
            "sky_key",
            "koi_period",
            "koi_impact",
            "koi_duration",
            "koi_depth",
            "koi_prad",
            "koi_teq",
            "koi_insol",
            "koi_model_snr",
        ]
    ].rename(
        columns={
            "koi_period": "orbital_period_days",
            "koi_impact": "impact_parameter",
            "koi_duration": "transit_duration_hours",
            "koi_depth": "transit_depth_ppm",
            "koi_prad": "planet_radius_earth",
            "koi_teq": "equilibrium_temp_k",
            "koi_insol": "insolation_flux",
            "koi_model_snr": "model_snr",
        }
    )
    fact_koi.insert(0, "observation_key", range(1, len(fact_koi) + 1))

    dim_discovery_method = (
        planet_base[["discoverymethod", "disc_facility"]]
        .fillna("Desconocido")
        .drop_duplicates()
        .copy()
    )
    dim_discovery_method.insert(0, "method_key", range(1, len(dim_discovery_method) + 1))

    dim_discovery_year = planet_base[["disc_year"]].dropna().drop_duplicates().sort_values("disc_year").copy()
    dim_discovery_year["disc_year"] = dim_discovery_year["disc_year"].astype(int)
    dim_discovery_year.insert(0, "year_key", range(1, len(dim_discovery_year) + 1))

    fact_planets = planet_base.merge(dim_discovery_method, on=["discoverymethod", "disc_facility"], how="left")
    fact_planets["disc_year_int"] = fact_planets["disc_year"].astype("Int64")
    fact_planets = fact_planets.merge(dim_discovery_year, left_on="disc_year_int", right_on="disc_year", how="left")
    fact_planets = fact_planets[
        [
            "pl_name",
            "hostname",
            "method_key",
            "year_key",
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
    ].copy()
    fact_planets.insert(0, "planet_fact_key", range(1, len(fact_planets) + 1))

    return {
        "dim_star": dim_star,
        "dim_candidate": dim_candidate,
        "dim_disposition": dim_disposition,
        "dim_sky_position": dim_sky_position,
        "fact_koi_observations": fact_koi,
        "dim_discovery_method": dim_discovery_method,
        "dim_discovery_year": dim_discovery_year,
        "fact_confirmed_planets": fact_planets,
    }


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE VIEW v_koi_observations AS
        SELECT
            f.observation_key,
            c.candidate_id,
            c.kepler_name,
            c.planet_radius_band,
            s.star_id,
            s.star_temp_band,
            s.koi_steff,
            s.koi_slogg,
            s.koi_srad,
            s.koi_kepmag,
            d.koi_disposition,
            sky.ra,
            sky.dec,
            sky.ra_bin,
            sky.dec_bin,
            f.orbital_period_days,
            f.impact_parameter,
            f.transit_duration_hours,
            f.transit_depth_ppm,
            f.planet_radius_earth,
            f.equilibrium_temp_k,
            f.insolation_flux,
            f.model_snr
        FROM fact_koi_observations f
        JOIN dim_candidate c USING (candidate_key)
        JOIN dim_star s USING (star_key)
        JOIN dim_disposition d USING (disposition_key)
        JOIN dim_sky_position sky USING (sky_key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW v_confirmed_planets AS
        SELECT
            f.planet_fact_key,
            f.pl_name,
            f.hostname,
            m.discoverymethod,
            m.disc_facility,
            y.disc_year,
            f.pl_orbper,
            f.pl_rade,
            f.pl_bmasse,
            f.pl_eqt,
            f.st_teff,
            f.st_rad,
            f.st_mass,
            f.sy_dist,
            f.ra,
            f.dec
        FROM fact_confirmed_planets f
        LEFT JOIN dim_discovery_method m USING (method_key)
        LEFT JOIN dim_discovery_year y USING (year_key)
        """
    )


OLAP_QUERIES: dict[str, str] = {
    "rollup_disposition": """
        SELECT
            koi_disposition,
            COUNT(*) AS n_observations,
            AVG(planet_radius_earth) AS avg_radius_earth,
            MEDIAN(planet_radius_earth) AS median_radius_earth,
            AVG(model_snr) AS avg_snr
        FROM v_koi_observations
        GROUP BY koi_disposition
        ORDER BY n_observations DESC
    """,
    "drilldown_temp": """
        SELECT
            koi_disposition,
            star_temp_band,
            planet_radius_band,
            COUNT(*) AS n_observations,
            MEDIAN(planet_radius_earth) AS median_radius_earth
        FROM v_koi_observations
        GROUP BY koi_disposition, star_temp_band, planet_radius_band
        ORDER BY koi_disposition, star_temp_band, n_observations DESC
    """,
    "slice_dice_habitable": """
        SELECT
            koi_disposition,
            star_temp_band,
            planet_radius_band,
            COUNT(*) AS n_observations,
            MEDIAN(equilibrium_temp_k) AS median_teq
        FROM v_koi_observations
        WHERE equilibrium_temp_k BETWEEN 180 AND 320
          AND planet_radius_band IN ('Tipo Tierra', 'Super Tierra')
        GROUP BY koi_disposition, star_temp_band, planet_radius_band
        ORDER BY n_observations DESC
    """,
    "pivot_disposition": """
        PIVOT v_koi_observations
        ON koi_disposition
        USING COUNT(*)
        GROUP BY star_temp_band
        ORDER BY star_temp_band
    """,
    "cube": """
        SELECT
            star_temp_band,
            planet_radius_band,
            koi_disposition,
            COUNT(*) AS n_observations,
            AVG(model_snr) AS avg_snr
        FROM v_koi_observations
        GROUP BY CUBE (star_temp_band, planet_radius_band, koi_disposition)
        ORDER BY n_observations DESC
    """,
    "rollup_radius": """
        SELECT
            star_temp_band,
            planet_radius_band,
            COUNT(*) AS n_observations,
            AVG(planet_radius_earth) AS avg_radius_earth
        FROM v_koi_observations
        GROUP BY ROLLUP (star_temp_band, planet_radius_band)
        ORDER BY star_temp_band, planet_radius_band
    """,
    "grouping_sets": """
        SELECT
            star_temp_band,
            koi_disposition,
            COUNT(*) AS n_observations,
            MEDIAN(planet_radius_earth) AS median_radius_earth
        FROM v_koi_observations
        GROUP BY GROUPING SETS ((star_temp_band), (koi_disposition), (star_temp_band, koi_disposition), ())
        ORDER BY star_temp_band, koi_disposition
    """,
    "iceberg_cube": """
        SELECT
            star_temp_band,
            planet_radius_band,
            koi_disposition,
            COUNT(*) AS n_observations,
            AVG(model_snr) AS avg_snr
        FROM v_koi_observations
        GROUP BY star_temp_band, planet_radius_band, koi_disposition
        HAVING COUNT(*) >= 50
        ORDER BY n_observations DESC
    """,
    "confirmed_by_year": """
        SELECT
            disc_year,
            discoverymethod,
            COUNT(*) AS n_planets,
            MEDIAN(pl_rade) AS median_radius_earth
        FROM v_confirmed_planets
        WHERE disc_year IS NOT NULL
        GROUP BY disc_year, discoverymethod
        ORDER BY disc_year DESC, n_planets DESC
    """,
}


def run_olap_query(name: str, limit: int | None = 100) -> pd.DataFrame:
    if name not in OLAP_QUERIES:
        valid = ", ".join(sorted(OLAP_QUERIES))
        raise ValueError(f"Consulta OLAP desconocida: {name}. Opciones: {valid}")
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError("No existe el warehouse DuckDB. Ejecuta el pipeline primero.")
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute(OLAP_QUERIES[name]).df()
    return df.head(limit) if limit is not None else df


def timed_query(con: duckdb.DuckDBPyConnection, query: str, repetitions: int = 10) -> float:
    start = time.perf_counter()
    for _ in range(repetitions):
        con.execute(query).fetchall()
    return (time.perf_counter() - start) / repetitions


def run_warehouse() -> dict[str, Any]:
    ensure_project_dirs()
    kepler, pscomppars = load_processed_data()
    tables = build_dimension_tables(kepler, pscomppars)

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        for name, df in tables.items():
            con.register(f"tmp_{name}", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_{name}")
            con.unregister(f"tmp_{name}")

        create_views(con)

        level_counts = con.execute(
            """
            SELECT
                COUNT(DISTINCT star_temp_band) AS l_star_temp,
                COUNT(DISTINCT planet_radius_band) AS l_radius,
                COUNT(DISTINCT koi_disposition) AS l_disposition
            FROM v_koi_observations
            """
        ).df().iloc[0]
        expected_cube_rows = int(
            (level_counts["l_star_temp"] + 1)
            * (level_counts["l_radius"] + 1)
            * (level_counts["l_disposition"] + 1)
        )
        observed_cube_rows = int(
            con.execute(
                """
                SELECT SUM(n_groups) AS n_rows
                FROM (
                    SELECT COUNT(*) AS n_groups FROM (SELECT DISTINCT star_temp_band FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT planet_radius_band FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT koi_disposition FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT star_temp_band, planet_radius_band FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT star_temp_band, koi_disposition FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT planet_radius_band, koi_disposition FROM v_koi_observations)
                    UNION ALL
                    SELECT COUNT(*) FROM (SELECT DISTINCT star_temp_band, planet_radius_band, koi_disposition FROM v_koi_observations)
                    UNION ALL
                    SELECT 1
                )
                """
            ).fetchone()[0]
        )

        count_query = """
            SELECT star_temp_band, planet_radius_band, COUNT(*) AS n_observations
            FROM v_koi_observations
            GROUP BY GROUPING SETS ((star_temp_band), (planet_radius_band), (star_temp_band, planet_radius_band), ())
        """
        median_query = """
            SELECT star_temp_band, planet_radius_band, MEDIAN(planet_radius_earth) AS median_radius
            FROM v_koi_observations
            GROUP BY GROUPING SETS ((star_temp_band), (planet_radius_band), (star_temp_band, planet_radius_band), ())
        """

        summary = json_safe(
            {
                "duckdb_path": project_relative(DUCKDB_PATH),
                "tables": {name: {"filas": len(df), "columnas": df.shape[1]} for name, df in tables.items()},
                "olap_queries": sorted(OLAP_QUERIES),
                "sample_olap": {
                    "rollup_disposition": dataframe_records(con.execute(OLAP_QUERIES["rollup_disposition"]).df()),
                    "slice_dice_habitable": dataframe_records(con.execute(OLAP_QUERIES["slice_dice_habitable"]).df()),
                },
                "cube_validation": {
                    "filas_cube": len(con.execute(OLAP_QUERIES["cube"]).df()),
                    "filas_esperadas_observadas": observed_cube_rows,
                    "maximo_teorico_cubo_denso": expected_cube_rows,
                    "nota": "DuckDB devuelve grupos observados; el maximo teorico incluye combinaciones vacias.",
                },
                "measure_timing_seconds": {
                    "COUNT_distributiva": timed_query(con, count_query),
                    "MEDIAN_holistica": timed_query(con, median_query),
                },
            }
        )

    WAREHOUSE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_warehouse(), indent=2))
