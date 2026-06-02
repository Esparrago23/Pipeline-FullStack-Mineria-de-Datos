# Declaración de uso de IA
## Herramientas usadas
- Claude / ChatGPT / Copilot / etc.
## Qué generé con IA
- Pipeline inicial con ColumnTransformer (prompt: "...")
- Sintaxis de cross_val_score con scoring custom
- Gráfica de matriz de confusión
## Qué entendí y modifiqué yo
- Cambié RandomForestRegressor por Ridge porque el rubro pedía un modelo lineal
regularizado
- Reescribí el manejo de NaN en TotalCharges porque la sugerencia de IA imputaba con la
media y yo decidí eliminar esas filas porque son solo 11 (decisión defendible en la
oral)
## Qué NO sé explicar de lo que generó la IA
- (Sé honesto. Esto es exactamente lo que te voy a preguntar.)