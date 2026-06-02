# Proyecto de mineria - Exoplanetas Kepler

Proyecto de mineria de datos usando NASA Exoplanet Archive con dos fuentes:

- `cumulative` / Kepler Objects of Interest (KOI): senales candidatas observadas por Kepler.
- `pscomppars`: planetas confirmados de NASA Exoplanet Archive para referencia y capa analitica.

El proyecto esta dividido por capas:

1. `mineria/01_analisis_eda_preprocesamiento.ipynb` - Capa de analisis.
2. `mineria/02_capa_datos_warehouse.ipynb` - Capa de datos / DuckDB / OLAP.
3. `mineria/03_capa_modelado.ipynb` - Capa de modelado.

## 1. Crear entorno virtual

Ejecutar desde la carpeta raiz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn plotly duckdb jupyter nbformat ipykernel
python -m ipykernel install --user --name mineria-exoplanetas --display-name "Python (mineria-exoplanetas)"
```

Si VS Code pregunta por kernel, elegir `Python (mineria-exoplanetas)`.

## 2. Descargar los CSV reducidos

No descargar con `select *`, porque trae demasiadas columnas y vuelve dificil explicar el proyecto. Estos comandos descargan solo las columnas que se usan en el trabajo.

```powershell
New-Item -ItemType Directory -Force mineria\data | Out-Null

$keplerCols = @(
  "kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
  "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
  "koi_period", "koi_period_err1", "koi_period_err2",
  "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2",
  "koi_impact", "koi_impact_err1", "koi_impact_err2",
  "koi_duration", "koi_duration_err1", "koi_duration_err2",
  "koi_depth", "koi_depth_err1", "koi_depth_err2",
  "koi_prad", "koi_prad_err1", "koi_prad_err2",
  "koi_teq", "koi_teq_err1", "koi_teq_err2",
  "koi_insol", "koi_insol_err1", "koi_insol_err2",
  "koi_model_snr", "koi_tce_plnt_num", "koi_tce_delivname",
  "koi_steff", "koi_steff_err1", "koi_steff_err2",
  "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
  "koi_srad", "koi_srad_err1", "koi_srad_err2",
  "ra", "dec", "koi_kepmag"
) -join ","

$keplerQuery = [uri]::EscapeDataString("select $keplerCols from cumulative")
Invoke-WebRequest `
  -Uri "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=$keplerQuery&format=csv" `
  -OutFile "mineria\data\cumulative_2026.06.01_20.09.17.csv"

$psCols = @(
  "pl_name", "hostname", "sy_snum", "sy_pnum", "discoverymethod", "disc_year", "disc_facility",
  "pl_controv_flag", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbperlim",
  "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbsmaxlim",
  "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_radelim",
  "pl_radj", "pl_radjerr1", "pl_radjerr2", "pl_radjlim",
  "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmasselim",
  "pl_bmassj", "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim",
  "pl_bmassprov", "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim",
  "pl_insol", "pl_insolerr1", "pl_insolerr2", "pl_insollim",
  "pl_eqt", "pl_eqterr1", "pl_eqterr2", "pl_eqtlim", "ttv_flag",
  "st_spectype", "st_teff", "st_tefferr1", "st_tefferr2", "st_tefflim",
  "st_rad", "st_raderr1", "st_raderr2", "st_radlim",
  "st_mass", "st_masserr1", "st_masserr2", "st_masslim",
  "st_met", "st_meterr1", "st_meterr2", "st_metlim", "st_metratio",
  "st_logg", "st_loggerr1", "st_loggerr2", "st_logglim",
  "rastr", "ra", "decstr", "dec",
  "sy_dist", "sy_disterr1", "sy_disterr2",
  "sy_vmag", "sy_vmagerr1", "sy_vmagerr2",
  "sy_kmag", "sy_kmagerr1", "sy_kmagerr2",
  "sy_gaiamag", "sy_gaiamagerr1", "sy_gaiamagerr2"
) -join ","

$psQuery = [uri]::EscapeDataString("select $psCols from pscomppars")
Invoke-WebRequest `
  -Uri "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=$psQuery&format=csv" `
  -OutFile "mineria\data\PSCompPars_2026.06.01_20.09.10.csv"
```

## 3. Ejecutar notebooks

Abrir la carpeta `mineria` en Jupyter o VS Code y ejecutar en este orden:

```powershell
jupyter lab mineria
```

1. `01_analisis_eda_preprocesamiento.ipynb`
   - Carga los CSV reducidos.
   - Hace EDA: `df.info()`, `df.describe()`, nulos, distribuciones, correlacion y outliers.
   - Limpia espacios en blanco y prepara variables.
   - Guarda:
     - `mineria/data/processed/kepler_koi_processed.csv`
     - `mineria/data/processed/pscomppars_processed.csv`

2. `02_capa_datos_warehouse.ipynb`
   - Usa los CSV procesados.
   - Construye el modelo dimensional en DuckDB.
   - Crea:
     - `mineria/data/warehouse/exoplanets.duckdb`
   - Ejecuta consultas OLAP: roll-up, drill-down, slice/dice, pivot, CUBE, ROLLUP y GROUPING SETS.

3. `03_capa_modelado.ipynb`
   - Usa el DuckDB creado por la capa de datos.
   - Clasifica `koi_disposition` como `CONFIRMED` vs `NO_CONFIRMED`.
   - Predice `log1p(koi_prad)` para la tarea de regresion.
   - Evalua con metricas vistas en clase y evita fuga de datos usando `Pipeline`.

## 4. Guia del dataset y del codigo

La explicacion detallada esta en:

- `docs/guia_dataset_y_codigo.md`

Ese documento explica que significa cada grupo de columnas, por que se eligieron, como se diseno el warehouse y como defender el codigo de cada capa.
