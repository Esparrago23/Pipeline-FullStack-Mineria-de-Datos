from __future__ import annotations

import argparse
import shutil
from urllib.parse import urlencode
from urllib.request import urlopen

from .config import RAW_KEPLER_PATH, RAW_PSCOMP_PATH, ensure_project_dirs


NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

KEPLER_COLUMNS = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_period_err1",
    "koi_period_err2",
    "koi_time0bk",
    "koi_time0bk_err1",
    "koi_time0bk_err2",
    "koi_impact",
    "koi_impact_err1",
    "koi_impact_err2",
    "koi_duration",
    "koi_duration_err1",
    "koi_duration_err2",
    "koi_depth",
    "koi_depth_err1",
    "koi_depth_err2",
    "koi_prad",
    "koi_prad_err1",
    "koi_prad_err2",
    "koi_teq",
    "koi_teq_err1",
    "koi_teq_err2",
    "koi_insol",
    "koi_insol_err1",
    "koi_insol_err2",
    "koi_model_snr",
    "koi_tce_plnt_num",
    "koi_tce_delivname",
    "koi_steff",
    "koi_steff_err1",
    "koi_steff_err2",
    "koi_slogg",
    "koi_slogg_err1",
    "koi_slogg_err2",
    "koi_srad",
    "koi_srad_err1",
    "koi_srad_err2",
    "ra",
    "dec",
    "koi_kepmag",
]

PSCOMP_COLUMNS = [
    "pl_name",
    "hostname",
    "sy_snum",
    "sy_pnum",
    "discoverymethod",
    "disc_year",
    "disc_facility",
    "pl_controv_flag",
    "pl_orbper",
    "pl_orbpererr1",
    "pl_orbpererr2",
    "pl_orbperlim",
    "pl_orbsmax",
    "pl_orbsmaxerr1",
    "pl_orbsmaxerr2",
    "pl_orbsmaxlim",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_radelim",
    "pl_radj",
    "pl_radjerr1",
    "pl_radjerr2",
    "pl_radjlim",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_bmasselim",
    "pl_bmassj",
    "pl_bmassjerr1",
    "pl_bmassjerr2",
    "pl_bmassjlim",
    "pl_bmassprov",
    "pl_orbeccen",
    "pl_orbeccenerr1",
    "pl_orbeccenerr2",
    "pl_orbeccenlim",
    "pl_insol",
    "pl_insolerr1",
    "pl_insolerr2",
    "pl_insollim",
    "pl_eqt",
    "pl_eqterr1",
    "pl_eqterr2",
    "pl_eqtlim",
    "ttv_flag",
    "st_spectype",
    "st_teff",
    "st_tefferr1",
    "st_tefferr2",
    "st_tefflim",
    "st_rad",
    "st_raderr1",
    "st_raderr2",
    "st_radlim",
    "st_mass",
    "st_masserr1",
    "st_masserr2",
    "st_masslim",
    "st_met",
    "st_meterr1",
    "st_meterr2",
    "st_metlim",
    "st_metratio",
    "st_logg",
    "st_loggerr1",
    "st_loggerr2",
    "st_logglim",
    "rastr",
    "ra",
    "decstr",
    "dec",
    "sy_dist",
    "sy_disterr1",
    "sy_disterr2",
    "sy_vmag",
    "sy_vmagerr1",
    "sy_vmagerr2",
    "sy_kmag",
    "sy_kmagerr1",
    "sy_kmagerr2",
    "sy_gaiamag",
    "sy_gaiamagerr1",
    "sy_gaiamagerr2",
]


def tap_url(table: str, columns: list[str]) -> str:
    query = f"select {', '.join(columns)} from {table}"
    return f"{NASA_TAP_URL}?{urlencode({'query': query, 'format': 'csv'})}"


def download_csv(table: str, columns: list[str], output_path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Existe, no se descarga de nuevo: {output_path}")
        return

    url = tap_url(table, columns)
    print(f"Descargando {table} -> {output_path}")
    with urlopen(url, timeout=120) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga los CSV crudos desde NASA Exoplanet Archive.")
    parser.add_argument("--force", action="store_true", help="Sobrescribe los CSV si ya existen.")
    args = parser.parse_args()

    ensure_project_dirs()
    download_csv("cumulative", KEPLER_COLUMNS, RAW_KEPLER_PATH, force=args.force)
    download_csv("pscomppars", PSCOMP_COLUMNS, RAW_PSCOMP_PATH, force=args.force)


if __name__ == "__main__":
    main()
