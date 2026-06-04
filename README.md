# Pipeline Full Stack de Mineria de Datos - Exoplanetas Kepler

Proyecto de corte para Mineria de Datos. El sistema integra EDA, preprocesamiento, warehouse en DuckDB, modelos de clasificacion/regresion, API con FastAPI y frontend web basico en HTML, CSS y JavaScript.

El proyecto ejecutable no depende de notebooks. Todo lo necesario para reconstruir los resultados esta en scripts dentro de `backend/pipeline` y en la aplicacion dentro de `backend/app` y `frontend`.

## Objetivo del sistema

El proyecto usa datos reales de NASA Exoplanet Archive para responder dos preguntas:

- Clasificacion: decidir si una senal de Kepler se parece a un exoplaneta confirmado (`CONFIRMED`) o a una senal no confirmada (`NO_CONFIRMED`).
- Regresion: estimar el radio planetario en radios terrestres usando `log1p(koi_prad)` para reducir el efecto de valores extremos.

Importante: la app no confirma exoplanetas cientificamente. La app calcula una prediccion basada en patrones historicos del dataset; la confirmacion real pertenece al proceso cientifico de NASA.

## Estructura del repositorio

```text
backend/
  app/
    main.py                         API FastAPI y servidor del frontend
  pipeline/
    analysis_preprocessing.py       EDA y preprocesamiento reproducible
    warehouse.py                    Warehouse DuckDB y consultas OLAP
    modeling.py                     Entrenamiento de clasificacion y regresion
    run_pipeline.py                 Ejecuta todo el pipeline en orden
    download_data.py                Descarga opcional de CSV crudos desde NASA
    config.py                       Rutas, columnas y configuracion compartida
  data/
    raw/                            CSV crudos incluidos para reproducibilidad
    processed/                      CSV generados por preprocesamiento, no versionados
    warehouse/                      Base DuckDB generada por el pipeline, no versionada
  models/                           Modelos generados por el pipeline, no versionados
  reports/                          JSON con resumenes y metricas, no versionados
frontend/
  index.html                        Interfaz web
  styles.css                        Estilos
  app.js                            Consumo de la API real
AI_USAGE.md                         Declaracion de uso de IA
requirements.txt                    Dependencias Python
README.md                           Este archivo
```

## Requisitos

- Python 3.10 o superior.
- PowerShell en Windows.
- Conexion a internet solo si se quieren volver a descargar los CSV desde NASA. Los CSV crudos ya estan incluidos en `backend/data/raw`.

## Reproduccion exacta desde cero

Ejecutar todos los comandos desde la raiz del repositorio.

El punto de partida reproducible son los CSV crudos en `backend/data/raw`. Los archivos de
`backend/data/processed`, `backend/data/warehouse`, `backend/models` y `backend/reports` son
salidas generadas: se pueden borrar y reconstruir con `python -m backend.pipeline.run_pipeline`.

### 1. Crear y activar entorno virtual

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Verificar datos crudos

Los dos archivos deben existir:

```powershell
Test-Path backend\data\raw\cumulative_2026.06.01_20.09.17.csv
Test-Path backend\data\raw\PSCompPars_2026.06.01_20.09.10.csv
```

Resultado esperado:

```text
True
True
```

Si alguno no existe, descargarlos con:

```powershell
python -m backend.pipeline.download_data
```

Si se quiere forzar una descarga nueva y sobrescribir los CSV:

```powershell
python -m backend.pipeline.download_data --force
```

### 3. Ejecutar el pipeline completo

```powershell
python -m backend.pipeline.run_pipeline
```

Si se quiere comprobar que todo se reconstruye desde cero antes de ejecutar el pipeline:

```powershell
Remove-Item backend\data\processed, backend\data\warehouse, backend\models, backend\reports -Recurse -Force -ErrorAction SilentlyContinue
python -m backend.pipeline.run_pipeline
```

Este comando ejecuta, en orden:

1. `analysis_preprocessing.py`: carga CSV crudos, limpia datos, genera CSV procesados y resumen EDA.
2. `warehouse.py`: construye `backend/data/warehouse/exoplanets.duckdb` con tablas de hechos, dimensiones, vistas y consultas OLAP.
3. `modeling.py`: entrena modelos de clasificacion y regresion con `Pipeline` de scikit-learn, sin fuga de datos.

Salida esperada al final:

```json
{
  "status": "ok",
  "analysis": { "...": "..." },
  "warehouse": { "...": "..." },
  "modeling": {
    "classification_best_model": "arbol_decision",
    "regression_best_model": "regresion_lineal"
  }
}
```

El pipeline genera o actualiza:

```text
backend/data/processed/kepler_koi_processed.csv
backend/data/processed/pscomppars_processed.csv
backend/data/warehouse/exoplanets.duckdb
backend/models/classification_model.joblib
backend/models/regression_model.joblib
backend/models/model_metadata.json
backend/reports/analysis_summary.json
backend/reports/warehouse_summary.json
backend/reports/model_metrics.json
```

### 4. Revisar metricas principales

```powershell
python -c "import json; m=json.load(open('backend/reports/model_metrics.json')); print(m['classification']['best_model']); print(m['regression']['best_model'])"
```

Resultado esperado:

```text
arbol_decision
regresion_lineal
```

Metricas actuales despues de ejecutar el pipeline:

- Clasificacion: mejor modelo `arbol_decision`; Accuracy aproximado `0.842`; Precision `0.707`; Recall `0.765`; F1 `0.735`.
- Regresion: mejor modelo `regresion_lineal`; `R2_log` aproximado `0.443`.

La clasificacion se decide por F1 para la clase `CONFIRMED`, no por accuracy sola. La regresion se evalua principalmente en escala log porque existen radios planetarios extremos.

### 5. Arrancar backend y frontend

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Abrir en el navegador:

```text
http://127.0.0.1:8000
```

Documentacion interactiva de la API:

```text
http://127.0.0.1:8000/docs
```

### 6. Verificar que la API esta lista

Con el servidor corriendo, abrir otra terminal PowerShell desde la raiz del proyecto y ejecutar:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Resultado esperado despues de correr el pipeline:

```text
status                      : ok
warehouse_exists            : True
classification_model_exists : True
regression_model_exists     : True
```

## Uso de la aplicacion

La interfaz web consume el backend real. No usa imagenes estaticas ni resultados copiados a mano.

Apartados principales:

- Ejecucion: permite correr el pipeline desde la API.
- Metricas generales: resume tamano de datasets y mejores modelos.
- Clasificacion: compara los algoritmos vistos en clase y muestra matriz de confusion.
- Regresion: muestra el ajuste del modelo para radio planetario.
- Consultas OLAP: consulta el warehouse DuckDB desde el frontend.
- Evaluar candidato: envia datos al backend y muestra al mismo tiempo clase estimada y radio estimado.

## Endpoints principales

```text
GET  /api/health
POST /api/pipeline/run
GET  /api/summary
GET  /api/olap
GET  /api/olap/{query_name}
GET  /api/prediction-sample
POST /api/predict/classification
POST /api/predict/regression
```

Ejemplo rapido de prediccion desde PowerShell:

```powershell
$body = @{
  features = @{
    koi_period = 6.002548
    koi_impact = 0.914
    koi_duration = 4.1628
    koi_depth = 202.0
    koi_teq = 1209
    koi_insol = 504.79
    koi_model_snr = 71.6
    koi_steff = 6204
    koi_slogg = 4.269
    koi_srad = 1.360
    ra = 290.49512
    dec = 38.795479
    koi_kepmag = 12.730
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/predict/classification `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Decisiones metodologicas principales

- Dataset principal: Kepler Objects of Interest (`cumulative`).
- Dataset de referencia analitica: planetas confirmados (`pscomppars`).
- Objetivo de clasificacion: `koi_disposition`, convertido a `CONFIRMED` contra `NO_CONFIRMED`.
- Objetivo de regresion: `log1p(koi_prad)`.
- Columnas excluidas por fuga: `koi_score`, `koi_pdisposition`, `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`.
- Separacion train/test antes de entrenar.
- Imputacion y escalado dentro de `Pipeline`, ajustados solo con train.
- Clasificacion comparada con regresion logistica, K-NN, arbol de decision y Naive Bayes.
- Regresion comparada con regresion lineal, Ridge y Lasso.
