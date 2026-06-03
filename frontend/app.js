const API_BASE = window.location.protocol.startsWith("http") ? "" : "http://127.0.0.1:8000";

const featureLabels = {
  koi_period: "Periodo orbital",
  koi_impact: "Impacto",
  koi_duration: "Duración",
  koi_depth: "Profundidad",
  koi_teq: "Temp. equilibrio",
  koi_insol: "Insolación",
  koi_model_snr: "SNR",
  koi_steff: "Temp. estrella",
  koi_slogg: "Log g",
  koi_srad: "Radio estrella",
  ra: "RA",
  dec: "DEC",
  koi_kepmag: "Magnitud Kepler",
};

const olapEndpoints = {
  disposition: "/api/olap/disposition",
  drilldown: "/api/olap/drilldown?limit=30",
  habitable: "/api/olap/habitable-slice",
  pivot: "/api/olap/pivot",
  cube: "/api/olap/cube?min_count=50&limit=40",
};

const state = {
  currentTable: "disposition",
  defaults: {},
};

const elements = {
  apiStatus: document.querySelector("#api-status"),
  refreshButton: document.querySelector("#refresh-button"),
  rowsLoaded: document.querySelector("#rows-loaded"),
  modelsReady: document.querySelector("#models-ready"),
  confirmedCount: document.querySelector("#confirmed-count"),
  noConfirmedCount: document.querySelector("#no-confirmed-count"),
  predictionForm: document.querySelector("#prediction-form"),
  predictButton: document.querySelector("#predict-button"),
  predictedClass: document.querySelector("#predicted-class"),
  confirmedProbability: document.querySelector("#confirmed-probability"),
  predictedRadius: document.querySelector("#predicted-radius"),
  olapHead: document.querySelector("#olap-head"),
  olapBody: document.querySelector("#olap-body"),
  recordsHead: document.querySelector("#records-head"),
  recordsBody: document.querySelector("#records-body"),
  toast: document.querySelector("#toast"),
};

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString("es-MX", {
    maximumFractionDigits: 4,
  });
}

function formatCell(value) {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 4200);
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Error HTTP ${response.status}`);
  }
  return response.json();
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Error HTTP ${response.status}`);
  }
  return response.json();
}

function renderTable(headElement, bodyElement, rows) {
  headElement.innerHTML = "";
  bodyElement.innerHTML = "";

  if (!rows.length) {
    bodyElement.innerHTML = `<tr><td>Sin datos</td></tr>`;
    return;
  }

  const columns = Object.keys(rows[0]);
  headElement.innerHTML = `<tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr>`;
  bodyElement.innerHTML = rows
    .map(
      (row) =>
        `<tr>${columns.map((column) => `<td>${formatCell(row[column])}</td>`).join("")}</tr>`,
    )
    .join("");
}

function renderPredictionForm(defaults) {
  elements.predictionForm.innerHTML = Object.entries(defaults)
    .map(([feature, value]) => {
      const label = featureLabels[feature] || feature;
      return `
        <div class="field">
          <label for="${feature}">${label}</label>
          <input id="${feature}" name="${feature}" type="number" step="any" value="${value ?? ""}" />
        </div>
      `;
    })
    .join("");
}

function readPredictionForm() {
  const formData = new FormData(elements.predictionForm);
  return Object.fromEntries(
    Array.from(formData.entries()).map(([key, value]) => [
      key,
      value === "" ? null : Number(value),
    ]),
  );
}

async function loadHealth() {
  const health = await apiGet("/api/health");
  elements.rowsLoaded.textContent = formatNumber(health.rows_loaded);
  elements.modelsReady.textContent = health.models_ready ? "Listos" : "No";
  elements.confirmedCount.textContent = formatNumber(health.class_counts?.CONFIRMED || 0);
  elements.noConfirmedCount.textContent = formatNumber(health.class_counts?.NO_CONFIRMED || 0);

  elements.apiStatus.textContent = health.status === "ok" ? "API activa" : "Revisar API";
  elements.apiStatus.classList.toggle("is-ok", health.status === "ok");
  elements.apiStatus.classList.toggle("is-error", health.status !== "ok");

  if (health.error) {
    showToast(health.error);
  }
}

async function loadDefaults() {
  const data = await apiGet("/api/model/default-input");
  state.defaults = data.defaults;
  renderPredictionForm(state.defaults);
}

async function loadOlapTable(name = state.currentTable) {
  state.currentTable = name;
  const data = await apiGet(olapEndpoints[name]);
  renderTable(elements.olapHead, elements.olapBody, data.rows);
}

async function loadRecords() {
  const data = await apiGet("/api/records/sample?limit=8");
  renderTable(elements.recordsHead, elements.recordsBody, data.rows);
}

async function runPrediction() {
  try {
    elements.predictButton.disabled = true;
    const result = await apiPost("/api/predict", readPredictionForm());
    elements.predictedClass.textContent = result.classification.predicted_class;
    elements.confirmedProbability.textContent = `${formatNumber(
      (result.classification.probabilities.CONFIRMED || 0) * 100,
    )}%`;
    elements.predictedRadius.textContent = `${formatNumber(
      result.regression.predicted_radius_earth,
    )} R⊕`;
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.predictButton.disabled = false;
  }
}

async function refreshDashboard() {
  try {
    await loadHealth();
    await loadDefaults();
    await loadOlapTable();
    await loadRecords();
  } catch (error) {
    elements.apiStatus.textContent = "Sin conexión";
    elements.apiStatus.classList.remove("is-ok");
    elements.apiStatus.classList.add("is-error");
    showToast(error.message);
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", async () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
    button.classList.add("is-active");
    try {
      await loadOlapTable(button.dataset.table);
    } catch (error) {
      showToast(error.message);
    }
  });
});

elements.refreshButton.addEventListener("click", refreshDashboard);
elements.predictButton.addEventListener("click", runPrediction);

refreshDashboard();
