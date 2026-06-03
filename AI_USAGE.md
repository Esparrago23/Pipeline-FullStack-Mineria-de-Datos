# Declaracion de uso de IA

## Herramientas usadas

- ChatGPT / Codex.
- Tambien use la IA como apoyo para revisar ideas, entender documentacion y organizar el codigo.

## Que genere o mejore con IA

- Comparacion inicial de posibles datasets reales. Pedi recomendaciones de datasets de fisica o del espacio y, despues de revisar varias opciones, seleccione NASA Exoplanet Archive / Kepler Objects of Interest porque permite plantear clasificacion y regresion en el mismo proyecto.
- Explicacion del diccionario de datos de NASA Exoplanet Archive, porque varias columnas de la pagina original no eran claras al inicio.
- Apoyo para decidir que columnas usar y cuales dejar fuera. En especial se revisaron variables de identificacion y posibles fugas de datos como `koi_score`, `koi_pdisposition` y `koi_fpflag_*`.
- Apoyo para organizar la separacion train/test antes de ajustar transformaciones, dejando imputacion y escalado dentro de un `Pipeline` de scikit-learn.
- Mejora del diseno dimensional del warehouse. Mi primer diseno estaba mas limitado a un solo hecho, y con apoyo de IA lo reorganice para incluir dimensiones, hechos de observaciones KOI y hechos de planetas confirmados.
- Apoyo para entender y aplicar operaciones OLAP como `CUBE`, `ROLLUP`, `GROUPING SETS`, iceberg cube y la diferencia entre medidas distributivas y holisticas.
- Apoyo para convertir el flujo de pruebas que eran los notebooks a scripts reproducibles dentro de `backend/pipeline`, porque el proyecto final no debia depender de notebooks.
- Apoyo para crear el backend con FastAPI y exponer endpoints para ejecutar pipeline, consultar OLAP y hacer inferencia.
- Apoyo para crear un frontend basico con HTML, CSS y JavaScript que consuma la API real.
- Apoyo para redactar y ordenar el `README.md` con pasos de reproduccion y arranque de la aplicacion.

## Que entendi y modifique yo

- Elegi quedarme con el dataset de exoplanetas porque me intereso la idea, aunque es muy complicado de entender: usar senales de Kepler para estimar si un objeto se parece a un exoplaneta confirmado y estimar su radio.
- Decidi usar Kepler KOI como dataset principal y PSCompPars como referencia para planetas confirmados para poder validar los datos.
- Decidi excluir columnas con fuga de datos: `koi_score`, `koi_pdisposition` y `koi_fpflag_*`, porque pueden adelantar informacion muy cercana a la clasificacion oficial.
- Para clasificacion, yo ya tenia claro que debia reportar metricas como precision, recall y F1. Con apoyo de IA ajuste el problema para que fuera binario: `CONFIRMED` contra `NO_CONFIRMED`, agrupando `CANDIDATE` y `FALSE POSITIVE`.
- Decidi usar F1 para la clase `CONFIRMED` como criterio principal porque accuracy sola puede verse bien aunque el modelo falle en la clase que mas interesa.
- Para regresion, entendi que `koi_prad` tenia outliers muy grandes y por eso no convenia modelarlo directamente en su escala original.
- Decidi comparar modelos de regresion lineal, Ridge y Lasso porque son mas defendibles dentro de lo visto en clase.


## Que NO supe explicar de lo que genero la IA

- No supe explicar al inicio como defender tecnicamente el uso de `log1p(koi_prad)`. Yo ya habia visto que `koi_prad` tenia outliers demasiado grandes, pero pedi ayuda para entender por que transformar el radio ayuda a estabilizar la escala y como regresar la prediccion a radios terrestres con `expm1`.
- No supe explicar por mi cuenta completa como pasar de una clasificacion con varias etiquetas (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`) a una clasificacion binaria. Pedi ayuda para justificar que `CANDIDATE` y `FALSE POSITIVE` se agruparan como `NO_CONFIRMED`, dejando el problema como `CONFIRMED` contra `NO_CONFIRMED`.
- No supe disenar desde cero un warehouse con mas de una tabla de hechos. Mi primer diseno estaba mas enfocado en un solo hecho ya que solo tome en cuenta un dataset, y pedi ayuda para mejorarlo con hechos de observaciones KOI y hechos de planetas confirmados.
- No supe aterrizar completamente la sintaxis de `CUBE`, `ROLLUP`, `GROUPING SETS` e iceberg cube sin ejemplos. Puedo explicar para que se usan en el proyecto, pero la sintaxis exacta fue una parte donde use apoyo de IA mas que nada debido a la gran cantidad de columnas, con las cuales no estaba muy acostumbrado a trabajar o no entendia muy bien debido a que la pagina oficial no explica bien sus datos.

