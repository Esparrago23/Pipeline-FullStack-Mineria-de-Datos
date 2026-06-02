# Declaracion de uso de IA

## Herramientas usadas

- ChatGPT / Codex.

## Que genere con IA

- Organizacion del proyecto en tres notebooks por capa.
- Codigo base para EDA, preprocesamiento, warehouse DuckDB y modelado.
- README con pasos de entorno virtual, descarga de CSV reducidos y orden de ejecucion.
- Guia del dataset y del codigo en `docs/guia_dataset_y_codigo.md`.

## Que entendi y modifique yo

- Se eligio NASA Exoplanet Archive porque el proyecto tiene una narrativa cientifica clara.
- Se decidio usar Kepler KOI como dataset principal y PSCompPars como referencia de planetas confirmados.
- Se corrigio la clasificacion para usar `koi_disposition` como `CONFIRMED` vs `NO_CONFIRMED`, asi las metricas coinciden con la diapositiva.
- Se pidio quitar metricas que no estaban en la diapositiva, como `balanced_accuracy`, `f1_macro` y `f1_weighted`.
- Se cambio regresion para quedarse con modelos mas faciles de defender en clase: regresion lineal, Ridge y Lasso.
- Se excluyeron columnas de fuga como `koi_score`, `koi_pdisposition` y `koi_fpflag_*`.

## Que NO se explicar de lo que genero la IA

- Nada queda como caja negra intencionalmente. La guia del proyecto explica las columnas, el flujo por capas, las metricas y las decisiones principales.
