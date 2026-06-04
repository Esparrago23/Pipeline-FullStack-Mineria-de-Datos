const state = {
  sample: null,
  summary: null,
};

const apiStatus = document.querySelector("#apiStatus");
const pipelineSummary = document.querySelector("#pipelineSummary");
const metricGrid = document.querySelector("#metricGrid");
const classificationDecision = document.querySelector("#classificationDecision");
const regressionDecision = document.querySelector("#regressionDecision");
const classificationMetricsTable = document.querySelector("#classificationMetricsTable");
const regressionMetricsTable = document.querySelector("#regressionMetricsTable");
const confusionMatrixTable = document.querySelector("#confusionMatrixTable");
const olapSelect = document.querySelector("#olapSelect");
const olapDescription = document.querySelector("#olapDescription");
const olapTable = document.querySelector("#olapTable");
const predictionForm = document.querySelector("#predictionForm");
const predictionResult = document.querySelector("#predictionResult");

const queryDescriptions = {
  rollup_disposition: "Resumen general por dictamen KOI.",
  drilldown_temp: "Detalle por dictamen, temperatura estelar y banda de radio.",
  slice_dice_habitable: "Filtro de candidatos templados con radio tipo Tierra o super Tierra.",
  pivot_disposition: "Comparacion de clases como columnas por banda de temperatura.",
  cube: "Cubo OLAP por temperatura, radio y disposicion.",
  rollup_radius: "Jerarquia de subtotales por temperatura y radio.",
  grouping_sets: "Agrupaciones especificas para comparar disposicion y temperatura.",
  iceberg_cube: "Grupos del cubo con al menos 50 observaciones.",
  confirmed_by_year: "Planetas confirmados por anio y metodo de descubrimiento.",
};

const featureLabels = {
  koi_period: "Periodo orbital",
  koi_impact: "Impacto",
  koi_duration: "Duracion transito",
  koi_depth: "Profundidad",
  koi_teq: "Temp. equilibrio",
  koi_insol: "Insolacion",
  koi_model_snr: "Senal/ruido",
  koi_steff: "Temp. estrella",
  koi_slogg: "Gravedad estrella",
  koi_srad: "Radio estrella",
  ra: "Ascension recta",
  dec: "Declinacion",
  koi_kepmag: "Magnitud Kepler",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return String(value);
  }
  if (Math.abs(number) >= 1000) {
    return number.toLocaleString("es-MX", { maximumFractionDigits: 0 });
  }
  return number.toLocaleString("es-MX", { maximumFractionDigits: digits });
}

function formatValue(value) {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  return escapeHtml(value ?? "-");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Error HTTP ${response.status}`);
  }
  return data;
}

function setStatus(text, className) {
  apiStatus.textContent = text;
  apiStatus.className = `status ${className || ""}`;
}

function metric(label, value, note) {
  return `
    <article class="metric">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(value)}</div>
      <div class="note">${escapeHtml(note || "")}</div>
    </article>
  `;
}

function pipelineItem(label, status, note, isReady) {
  return `
    <article class="pipeline-item ${isReady ? "ready" : ""}">
      <div class="pipeline-status">${isReady ? "Listo" : "Pendiente"}</div>
      <strong>${escapeHtml(label)}</strong>
      <span>${escapeHtml(note)}</span>
    </article>
  `;
}

function renderPipelineSummary(message) {
  const analysis = state.summary?.analysis;
  const warehouse = state.summary?.warehouse;
  const modeling = state.summary?.modeling;
  const kepler = analysis?.processed?.find((item) => item.dataset === "kepler_koi_processed");
  const cube = warehouse?.cube_validation;

  if (message) {
    pipelineSummary.innerHTML = `<div class="notice">${escapeHtml(message)}</div>`;
    return;
  }

  pipelineSummary.innerHTML = [
    pipelineItem("Analisis", Boolean(analysis), `${formatNumber(kepler?.filas)} filas KOI procesadas`, Boolean(analysis)),
    pipelineItem("Warehouse", Boolean(warehouse), cube ? `CUBE validado: ${cube.filas_cube} grupos` : "DuckDB no generado", Boolean(warehouse)),
    pipelineItem("Clasificacion", Boolean(modeling?.classification), modeling?.classification?.best_model || "sin modelo", Boolean(modeling?.classification)),
    pipelineItem("Regresion", Boolean(modeling?.regression), modeling?.regression?.best_model || "sin modelo", Boolean(modeling?.regression)),
  ].join("");
}

function renderMetrics() {
  const analysis = state.summary?.analysis;
  const warehouse = state.summary?.warehouse;
  const modeling = state.summary?.modeling;
  const processed = analysis?.processed || [];
  const kepler = processed.find((item) => item.dataset === "kepler_koi_processed");
  const pscomp = processed.find((item) => item.dataset === "pscomppars_processed");
  const clfBest = modeling?.classification?.model_results?.find(
    (item) => item.modelo === modeling?.classification?.best_model
  );
  const regBest = modeling?.regression?.model_results?.find(
    (item) => item.modelo === modeling?.regression?.best_model
  );

  metricGrid.innerHTML = [
    metric("Kepler KOI", formatNumber(kepler?.filas), "filas procesadas"),
    metric("PSCompPars", formatNumber(pscomp?.filas), "planetas confirmados de referencia"),
    metric("Mejor F1", formatNumber(clfBest?.F1), modeling?.classification?.best_model || "clasificacion"),
    metric("Mejor R2", formatNumber(regBest?.R2_log), modeling?.regression?.best_model || "regresion"),
  ].join("");

  renderPipelineSummary();
  renderModelResults();
}

function renderTable(table, rows, columns, labels = {}) {
  if (!rows || !rows.length) {
    table.innerHTML = "<tbody><tr><td>Sin resultados.</td></tr></tbody>";
    return;
  }
  const cols = columns && columns.length ? columns : Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map((col) => `<th>${escapeHtml(labels[col] || col)}</th>`).join("")}</tr></thead>`;
  const tbody = rows
    .map((row) => `<tr>${cols.map((col) => `<td>${formatValue(row[col])}</td>`).join("")}</tr>`)
    .join("");
  table.innerHTML = `${thead}<tbody>${tbody}</tbody>`;
}

function renderModelResults() {
  const classification = state.summary?.modeling?.classification;
  const regression = state.summary?.modeling?.regression;

  classificationDecision.textContent = classification?.decision || "Ejecuta el pipeline para ver la comparacion.";
  regressionDecision.textContent = regression?.decision || "Ejecuta el pipeline para ver la comparacion.";

  renderTable(
    classificationMetricsTable,
    classification?.model_results || [],
    ["modelo", "Accuracy", "Precision", "Recall", "F1", "CV_F1_promedio"],
    { modelo: "Modelo", CV_F1_promedio: "CV F1" }
  );

  renderTable(
    confusionMatrixTable,
    classification?.confusion_matrix || [],
    classification?.confusion_matrix?.[0] ? Object.keys(classification.confusion_matrix[0]) : [],
    {
      Predicho_CONFIRMED: "Pred. CONFIRMED",
      Predicho_NO_CONFIRMED: "Pred. NO_CONFIRMED",
    }
  );

  renderTable(
    regressionMetricsTable,
    regression?.model_results || [],
    ["modelo", "MSE_log", "R2_log", "RMSE_radio_tierra", "CV_R2_promedio"],
    {
      modelo: "Modelo",
      MSE_log: "MSE log",
      R2_log: "R2 log",
      RMSE_radio_tierra: "RMSE radio tierra",
      CV_R2_promedio: "CV R2",
    }
  );
}

function activeFeatures() {
  if (!state.sample) {
    return [];
  }
  return [
    ...state.sample.classification_features,
    ...state.sample.regression_features.filter(
      (feature) => !state.sample.classification_features.includes(feature)
    ),
  ];
}

function activeDefaults() {
  if (!state.sample) {
    return {};
  }
  return {
    ...state.sample.regression_defaults,
    ...state.sample.classification_defaults,
  };
}

function renderPredictionForm() {
  const defaults = activeDefaults();
  predictionForm.innerHTML = activeFeatures()
    .map(
      (feature) => `
        <div class="field">
          <label for="${feature}">${escapeHtml(featureLabels[feature] || feature)}</label>
          <input id="${feature}" name="${feature}" type="number" step="any" value="${defaults[feature] ?? ""}" />
        </div>
      `
    )
    .join("");
  predictionResult.innerHTML = "<p>Presiona Evaluar para obtener la clase estimada y el radio del candidato.</p>";
}

function renderPrediction(classificationResult, regressionResult) {
  const prediction = classificationResult.prediction;
  const probabilities = classificationResult.probabilities || {};
  const probabilityRows = Object.entries(probabilities)
    .map(([label, value]) => `<span>${escapeHtml(label)}: <strong>${formatNumber(value * 100, 1)}%</strong></span>`)
    .join("");
  const interpretation =
    prediction === "CONFIRMED"
      ? "El radio se interpreta como estimacion del candidato clasificado como posible exoplaneta."
      : "Como la senal no queda confirmada, el radio se muestra solo como estimacion de la senal, no como radio real de un planeta confirmado.";

  predictionResult.innerHTML = `
    <div class="combined-result">
      <article class="result-card">
        <span class="result-label">Clase estimada</span>
        <strong>${escapeHtml(prediction)}</strong>
        <p>Primero se evalua si la senal se parece a un exoplaneta confirmado.</p>
        <div class="probabilities">${probabilityRows}</div>
      </article>
      <article class="result-card">
        <span class="result-label">Radio estimado</span>
        <strong>${formatNumber(regressionResult.prediction_radius_earth)} radios terrestres</strong>
        <p>Escala log1p(koi_prad): ${formatNumber(regressionResult.prediction_log1p_koi_prad)}.</p>
      </article>
    </div>
    <p class="result-note">${escapeHtml(interpretation)}</p>
  `;
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    const ready = health.warehouse_exists && health.classification_model_exists && health.regression_model_exists;
    setStatus(ready ? "Backend listo" : "Backend sin modelos", ready ? "ok" : "");
  } catch (error) {
    setStatus("Backend no disponible", "bad");
  }
}

async function loadSummary() {
  state.summary = await api("/api/summary");
  renderMetrics();
}

async function loadOlapList() {
  const data = await api("/api/olap");
  const queries = data.queries || [];
  olapSelect.innerHTML = queries.map((name) => `<option value="${name}">${name}</option>`).join("");
  if (queries.includes("rollup_disposition")) {
    olapSelect.value = "rollup_disposition";
  }
}

async function loadOlap() {
  const name = olapSelect.value || "rollup_disposition";
  olapDescription.textContent = queryDescriptions[name] || "Consulta OLAP sobre el warehouse DuckDB.";
  const data = await api(`/api/olap/${name}?limit=100`);
  renderTable(olapTable, data.rows, data.columns);
}

async function loadSample() {
  state.sample = await api("/api/prediction-sample");
  renderPredictionForm();
}

async function runPipeline() {
  const button = document.querySelector("#runPipelineBtn");
  button.disabled = true;
  renderPipelineSummary("Ejecutando pipeline completo...");
  try {
    await api("/api/pipeline/run", { method: "POST" });
    await Promise.all([loadHealth(), loadSummary(), loadOlapList(), loadSample()]);
    await loadOlap();
    renderPipelineSummary("Pipeline ejecutado correctamente. Resultados actualizados.");
    setTimeout(() => renderPipelineSummary(), 1800);
  } catch (error) {
    renderPipelineSummary(error.message);
  } finally {
    button.disabled = false;
  }
}

async function predict(event) {
  event.preventDefault();
  const features = {};
  new FormData(predictionForm).forEach((value, key) => {
    features[key] = value === "" ? null : Number(value);
  });
  try {
    const [classificationResult, regressionResult] = await Promise.all([
      api("/api/predict/classification", {
        method: "POST",
        body: JSON.stringify({ features }),
      }),
      api("/api/predict/regression", {
        method: "POST",
        body: JSON.stringify({ features }),
      }),
    ]);
    renderPrediction(classificationResult, regressionResult);
  } catch (error) {
    predictionResult.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
  }
}

function wireEvents() {
  document.querySelector("#runPipelineBtn").addEventListener("click", runPipeline);
  document.querySelector("#loadOlapBtn").addEventListener("click", loadOlap);
  document.querySelector("#olapSelect").addEventListener("change", loadOlap);
  document.querySelector("#predictBtn").addEventListener("click", predict);
  document.querySelector("#resetBtn").addEventListener("click", (event) => {
    event.preventDefault();
    renderPredictionForm();
  });
}

async function init() {
  wireEvents();
  renderPipelineSummary("Cargando estado del backend...");
  await loadHealth();
  try {
    await Promise.all([loadSummary(), loadOlapList()]);
    await loadOlap();
    await loadSample();
  } catch (error) {
    renderPipelineSummary(error.message);
  }
}

init();
