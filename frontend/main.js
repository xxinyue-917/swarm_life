const canvas = document.getElementById("simulation-canvas");
const ctx = canvas.getContext("2d");
const statusElement = document.getElementById("connection-status");
const matrixTable = document.getElementById("matrix-table");
const presetSelect = document.getElementById("preset-select");
const applyPresetButton = document.getElementById("apply-preset");
const resetButton = document.getElementById("reset-config");
const applySettingsButton = document.getElementById("apply-settings");
const speciesInput = document.getElementById("species-input");
const particleInput = document.getElementById("particle-input");
const statsList = document.getElementById("stats-list");

const baseColors = ["#ff5a5f", "#00a699", "#f7b733", "#3c91e6", "#9b59b6", "#2ecc71"];

let socket;
let state = {
  config: null,
  matrix: [],
  particles: [],
  presets: [],
};
let matrixInputs = [];
let pendingConfig = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfig();
  await loadPresets();
  setupControls();
  connect();
});

function setupControls() {
  applySettingsButton.addEventListener("click", () => {
    void submitConfigForm(true);
  });

  applyPresetButton.addEventListener("click", () => {
    const selected = presetSelect.value;
    if (selected) {
      sendMessage({ type: "use_preset", name: selected });
    }
  });

  resetButton.addEventListener("click", () => {
    sendMessage({ type: "update_config", config: {}, reset_matrix: true });
  });

  [speciesInput, particleInput].forEach((input) => {
    input.addEventListener("change", () => {
      void submitConfigForm(true);
    });
  });
}

async function loadConfig() {
  const response = await fetch("/config");
  if (!response.ok) {
    setStatus("Failed to load config");
    return;
  }
  const data = await response.json();
  state.config = data.config;
  state.matrix = data.matrix;
  resizeCanvas();
  populateConfigControls();
  buildMatrixEditor();
  updateStats();
}

async function loadPresets() {
  const response = await fetch("/presets");
  if (!response.ok) {
    return;
  }
  state.presets = await response.json();
  presetSelect.innerHTML =
    '<option value="">Select a preset</option>' +
    state.presets
      .map((preset) => `<option value="${preset.name}">${preset.name} — ${preset.description}</option>`)
      .join("");
}

function resizeCanvas() {
  if (!state.config) {
    return;
  }
  canvas.width = state.config.width;
  canvas.height = state.config.height;
}

function populateConfigControls() {
  if (!state.config) {
    return;
  }
  if (
    pendingConfig &&
    (state.config.species_count !== pendingConfig.species ||
      state.config.particle_count !== pendingConfig.particles)
  ) {
    return;
  }
  if (document.activeElement !== speciesInput) {
    speciesInput.value = state.config.species_count;
  }
  if (document.activeElement !== particleInput) {
    particleInput.value = state.config.particle_count;
  }
}

function buildMatrixEditor() {
  const size = state.matrix.length;
  matrixTable.innerHTML = "";
  matrixInputs = new Array(size).fill(null).map(() => new Array(size));

  if (size === 0) {
    return;
  }

  const headerRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.className = "matrix-corner";
  headerRow.appendChild(corner);
  for (let column = 0; column < size; column += 1) {
    headerRow.appendChild(createHeaderCell(column));
  }
  matrixTable.appendChild(headerRow);

  for (let i = 0; i < size; i += 1) {
    const row = document.createElement("tr");
    row.appendChild(createHeaderCell(i));
    for (let j = 0; j < size; j += 1) {
      const cell = document.createElement("td");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "0.1";
      input.value = state.matrix[i][j];
      input.dataset.row = String(i);
      input.dataset.col = String(j);
      input.addEventListener("input", (event) => {
        const { rowIndex, colIndex, value } = extractMatrixInput(event);
        if (value !== null) {
          state.matrix[rowIndex][colIndex] = value;
        }
      });
      input.addEventListener("change", (event) => {
        const { rowIndex, colIndex, value } = extractMatrixInput(event);
        if (value !== null) {
          state.matrix[rowIndex][colIndex] = value;
          sendMatrixUpdate();
        } else {
          event.target.value = state.matrix[rowIndex][colIndex];
        }
      });
      cell.appendChild(input);
      row.appendChild(cell);
      matrixInputs[i][j] = input;
    }
    matrixTable.appendChild(row);
  }
}

function createHeaderCell(index) {
  const cell = document.createElement("th");
  const legend = document.createElement("div");
  legend.className = "legend";
  const chip = document.createElement("span");
  chip.className = "color-chip";
  chip.style.backgroundColor = getSpeciesColor(index);
  legend.appendChild(chip);

  const label = document.createElement("span");
  label.textContent = `S${index + 1}`;
  legend.appendChild(label);

  cell.appendChild(legend);
  return cell;
}

function extractMatrixInput(event) {
  const target = event.target;
  const rowIndex = Number.parseInt(target.dataset.row, 10);
  const colIndex = Number.parseInt(target.dataset.col, 10);
  const value = Number.parseFloat(target.value);
  if (!Number.isFinite(value)) {
    return { rowIndex, colIndex, value: null };
  }
  return { rowIndex, colIndex, value };
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  socket.addEventListener("open", () => setStatus("Connected"));
  socket.addEventListener("close", () => {
    setStatus("Disconnected – retrying…");
    setTimeout(connect, 2000);
  });
  socket.addEventListener("error", () => setStatus("Connection error"));
  socket.addEventListener("message", onMessage);
}

function onMessage(event) {
  const message = JSON.parse(event.data);
  if (message.type === "state") {
    const payload = message.payload;
    state.particles = payload.particles;
    state.matrix = payload.matrix;
    state.config = payload.config;
    if (
      pendingConfig &&
      state.config.species_count === pendingConfig.species &&
      state.config.particle_count === pendingConfig.particles
    ) {
      pendingConfig = null;
    }
    resizeCanvas();
    populateConfigControls();
    refreshMatrixEditor();
    render();
    updateStats();
    return;
  }
  if (message.type === "error") {
    setStatus(message.detail || "Server error");
  }
}

function applyMatrixToInputs() {
  const size = state.matrix.length;
  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < size; j += 1) {
      const input = matrixInputs[i]?.[j];
      if (!input) {
        continue;
      }
      if (document.activeElement === input) {
        continue;
      }
      const value = state.matrix[i][j];
      if (Number.parseFloat(input.value) !== value) {
        input.value = value;
      }
    }
  }
}

function refreshMatrixEditor() {
  const size = state.matrix.length;
  const needsRebuild =
    matrixInputs.length !== size ||
    matrixInputs.some((row) => !row || row.length !== size);
  if (needsRebuild) {
    buildMatrixEditor();
  } else {
    applyMatrixToInputs();
  }
}

function render() {
  if (!state.config) {
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const radius = Math.max(1.5, Math.min(3.5, state.config.interaction_radius * 0.08));
  for (const particle of state.particles) {
    const color = getSpeciesColor(particle.species);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateStats() {
  if (!state.config) {
    return;
  }
  const items = [
    `Particles: ${state.particles.length}`,
    `Species: ${state.config.species_count}`,
    `Frame interval: ${state.config.frame_interval.toFixed(2)}s`,
  ];
  statsList.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function setStatus(text) {
  statusElement.textContent = text;
}

function sendMessage(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  }
}

function getSpeciesColor(index) {
  if (index < baseColors.length) {
    return baseColors[index];
  }
  const hue = (index * 137.508) % 360;
  return `hsl(${Math.round(hue)}, 70%, 55%)`;
}

async function submitConfigForm(resetMatrix) {
  if (!state.config) {
    return;
  }
  const values = readConfigInputs();
  if (!values) {
    return;
  }
  const { species, particles } = values;
  const unchanged =
    species === state.config.species_count &&
    particles === state.config.particle_count;
  if (unchanged && !resetMatrix) {
    return;
  }
  pendingConfig = { species, particles };
  try {
    const response = await fetch("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config: { species_count: species, particle_count: particles },
        reset_matrix: resetMatrix,
      }),
    });
    if (!response.ok) {
      throw new Error(`Failed to update config (${response.status})`);
    }
    const payload = await response.json();
    if (payload.config) {
      state.config = payload.config;
    }
    if (payload.matrix) {
      state.matrix = payload.matrix;
    }
    if (
      state.config &&
      pendingConfig &&
      state.config.species_count === pendingConfig.species &&
      state.config.particle_count === pendingConfig.particles
    ) {
      pendingConfig = null;
    }
    populateConfigControls();
    refreshMatrixEditor();
    setStatus("");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Failed to update config");
    pendingConfig = null;
  }
}

function readConfigInputs() {
  const species = Number.parseInt(speciesInput.value, 10);
  const particles = Number.parseInt(particleInput.value, 10);
  if (!Number.isInteger(species) || species < 1) {
    setStatus("Species must be a positive integer");
    return null;
  }
  if (!Number.isInteger(particles) || particles < 1) {
    setStatus("Particles must be a positive integer");
    return null;
  }
  setStatus("");
  return { species, particles };
}

function sendMatrixUpdate() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  sendMessage({ type: "update_matrix", matrix: state.matrix });
}
