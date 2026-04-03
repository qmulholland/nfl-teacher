const state = {
  config: null,
  players: [],
  losX: 0.5,
  firstDownX: 0.7,
  maxPlayersPerSide: 11,
  defenseCall: "Unknown Coverage",
  defenseFront: "",
  defenseCoverage: "",
  defenseTemplateFront: "",
  defenseMatchScore: null,
  selectedId: null,
  draggingId: null,
  inferInFlight: false,
  inferQueued: false,
  inferTimer: null,
};

const fieldEl = document.getElementById("field");
const playersLayerEl = document.getElementById("players-layer");
const losLineEl = document.getElementById("los-line");
const firstDownLineEl = document.getElementById("first-down-line");
const ballMarkerEl = document.getElementById("ball-marker");
const playersTableEl = document.getElementById("players-table");
const defenseCallEl = document.getElementById("defense-call");
const defenseCallDetailEl = document.getElementById("defense-call-detail");
const qbWarningEl = document.getElementById("qb-warning");

const selectedIdInputEl = document.getElementById("selected-id");
const sideSelectEl = document.getElementById("side-select");
const lockToggleEl = document.getElementById("lock-toggle");
const labelSelectEl = document.getElementById("label-select");
const selectedSummaryEl = document.getElementById("selected-summary");

const playIdInputEl = document.getElementById("play-id-input");
const splitSelectEl = document.getElementById("split-select");
const saveStatusEl = document.getElementById("save-status");
const layoutEl = document.querySelector(".layout");
const sidePanelEl = document.getElementById("side-panel");
const detailsToggleEl = document.getElementById("details-toggle");
const IMMOBILE_OL_LABELS = new Set(["LT", "LG", "C", "RG", "RT"]);

document.getElementById("reset-btn").addEventListener("click", handleReset);
document.getElementById("save-btn").addEventListener("click", handleSave);

sideSelectEl.addEventListener("change", handleSideChange);
lockToggleEl.addEventListener("change", handleLockChange);
labelSelectEl.addEventListener("change", handleLockedLabelChange);

fieldEl.addEventListener("pointermove", handlePointerMove);
fieldEl.addEventListener("pointerup", endDrag);
fieldEl.addEventListener("pointerleave", endDrag);
fieldEl.addEventListener("pointercancel", endDrag);
detailsToggleEl.addEventListener("click", toggleDetailsPanel);

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function byId(id) {
  return state.players.find((player) => player.id === id) || null;
}

function sideLabels(side) {
  if (!state.config) return [];
  return side === "offense" ? state.config.offense : state.config.defense;
}

function countBySide(side) {
  return state.players.filter((player) => player.side === side).length;
}

function setSelected(id) {
  state.selectedId = id;
  render();
}

function setDetailsPanelOpen(open) {
  const shouldOpen = Boolean(open);
  if (layoutEl) {
    layoutEl.classList.toggle("details-open", shouldOpen);
  }
  sidePanelEl.classList.toggle("collapsed", !shouldOpen);
  detailsToggleEl.textContent = shouldOpen ? "Hide Details" : "Show Details";
  detailsToggleEl.setAttribute("aria-expanded", String(shouldOpen));
}

function toggleDetailsPanel() {
  const isOpen = !sidePanelEl.classList.contains("collapsed");
  setDetailsPanelOpen(!isOpen);
}

function uniquePlayerId(prefix) {
  const existing = new Set(state.players.map((player) => player.id));
  let n = 1;
  while (existing.has(`${prefix}${n}`)) n += 1;
  return `${prefix}${n}`;
}

function isImmovablePlayer(player) {
  return Boolean(player && player.locked_label && IMMOBILE_OL_LABELS.has(player.locked_label));
}

function handleReset() {
  if (!state.config) return;
  state.players = (state.starterPlayers || []).map((p) => ({ ...p }));
  state.selectedId = state.players[0]?.id || null;
  queueInfer(0);
}

function handlePointerDown(event, id) {
  const player = byId(id);
  if (!player) return;
  setSelected(id);
  if (isImmovablePlayer(player)) {
    saveStatusEl.textContent = `${player.locked_label} is fixed and cannot be moved.`;
    state.draggingId = null;
    return;
  }
  event.preventDefault();
  state.draggingId = id;
}

function handlePointerMove(event) {
  if (!state.draggingId) return;
  const rect = fieldEl.getBoundingClientRect();
  const x = clamp01((event.clientX - rect.left) / rect.width);
  const y = clamp01((event.clientY - rect.top) / rect.height);
  const player = byId(state.draggingId);
  if (!player) return;
  player.x = x;
  player.y = y;
  render();
  queueInfer(30);
}

function endDrag() {
  state.draggingId = null;
}

function handleSideChange() {
  const player = byId(state.selectedId);
  if (!player) return;
  if (isImmovablePlayer(player)) {
    sideSelectEl.value = player.side;
    saveStatusEl.textContent = `${player.locked_label} is fixed and cannot change side.`;
    return;
  }
  const nextSide = sideSelectEl.value;
  if (nextSide !== player.side && countBySide(nextSide) >= state.maxPlayersPerSide) {
    saveStatusEl.textContent = `You can only have ${state.maxPlayersPerSide} ${nextSide} players.`;
    sideSelectEl.value = player.side;
    return;
  }

  player.side = nextSide;
  const labels = sideLabels(player.side);
  if (!labels.includes(player.locked_label)) {
    player.locked_label = null;
    lockToggleEl.checked = false;
  }
  queueInfer(0);
  render();
}

function handleLockChange() {
  const player = byId(state.selectedId);
  if (!player) return;
  if (isImmovablePlayer(player)) {
    lockToggleEl.checked = true;
    saveStatusEl.textContent = `${player.locked_label} is fixed and cannot be unlocked.`;
    return;
  }
  if (!lockToggleEl.checked) {
    player.locked_label = null;
  } else {
    const labels = sideLabels(player.side);
    player.locked_label = labels[0] || null;
  }
  queueInfer(0);
  render();
}

function handleLockedLabelChange() {
  const player = byId(state.selectedId);
  if (!player) return;
  if (isImmovablePlayer(player)) {
    labelSelectEl.value = player.locked_label;
    saveStatusEl.textContent = `${player.locked_label} is fixed and cannot be changed.`;
    return;
  }
  player.locked_label = labelSelectEl.value || null;
  queueInfer(0);
  render();
}

function queueInfer(delayMs) {
  if (state.inferTimer) {
    window.clearTimeout(state.inferTimer);
  }
  state.inferTimer = window.setTimeout(() => {
    state.inferTimer = null;
    runInfer();
  }, delayMs);
}

async function runInfer() {
  if (state.inferInFlight) {
    state.inferQueued = true;
    return;
  }
  state.inferInFlight = true;

  try {
    const payload = {
      los_x: state.losX,
      players: state.players.map((player) => ({
        id: player.id,
        side: player.side,
        x: player.x,
        y: player.y,
        locked_label: player.locked_label || null,
      })),
    };

    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("inference request failed");
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    const map = new Map(data.players.map((player) => [player.id, player]));
    state.players = state.players.map((player) => ({ ...player, ...(map.get(player.id) || {}) }));
    state.defenseCall = data.defense_call || "Unknown Coverage";
    state.defenseFront = data.defense_front || "";
    state.defenseCoverage = data.defense_coverage || "";
    state.defenseTemplateFront = data.defense_template_front || "";
    state.defenseMatchScore = data.defense_match_score ?? null;
    render();
  } catch (error) {
    console.error(error);
    saveStatusEl.textContent = error.message || "Inference failed.";
  } finally {
    state.inferInFlight = false;
    if (state.inferQueued) {
      state.inferQueued = false;
      runInfer();
    }
  }
}

function renderPlayerEditor() {
  const player = byId(state.selectedId);
  if (!player) {
    selectedIdInputEl.value = "";
    sideSelectEl.value = "offense";
    lockToggleEl.checked = false;
    labelSelectEl.innerHTML = "";
    labelSelectEl.disabled = true;
    selectedSummaryEl.textContent = "Click a player to edit side/label lock.";
    return;
  }

  selectedIdInputEl.value = player.id;
  sideSelectEl.value = player.side;
  lockToggleEl.checked = Boolean(player.locked_label);

  const labels = sideLabels(player.side);
  labelSelectEl.innerHTML = labels
    .map((label) => `<option value="${label}">${label}</option>`)
    .join("");
  if (player.locked_label && labels.includes(player.locked_label)) {
    labelSelectEl.value = player.locked_label;
  } else if (labels.length > 0) {
    labelSelectEl.value = labels[0];
  }
  const immovable = isImmovablePlayer(player);
  sideSelectEl.disabled = immovable;
  lockToggleEl.disabled = immovable;
  labelSelectEl.disabled = !lockToggleEl.checked || immovable;

  const confidence = player.predicted_confidence
    ? `${(player.predicted_confidence * 100).toFixed(1)}%`
    : "0.0%";
  selectedSummaryEl.textContent = `Predicted: ${player.predicted_label || "-"} (${confidence})`;
}

function renderPlayersTable() {
  const rows = [...state.players].sort((a, b) => {
    if (a.side !== b.side) return a.side < b.side ? -1 : 1;
    return a.y - b.y;
  });
  playersTableEl.innerHTML = rows
    .map((player) => {
      const selectedClass = player.id === state.selectedId ? "selected" : "";
      const confidence = player.predicted_confidence
        ? `${(player.predicted_confidence * 100).toFixed(0)}%`
        : "0%";
      return `
        <button class="players-row ${selectedClass}" data-row-id="${player.id}" type="button">
          <span>${player.id}</span>
          <span>${player.side.slice(0, 3).toUpperCase()}</span>
          <span>${player.predicted_label || "-"}</span>
          <span>${confidence}</span>
        </button>
      `;
    })
    .join("");

  playersTableEl.querySelectorAll("[data-row-id]").forEach((el) => {
    el.addEventListener("click", () => setSelected(el.getAttribute("data-row-id")));
  });
}

function renderDefenseCall() {
  defenseCallEl.textContent = state.defenseCall || "Unknown Coverage";
  if (state.defenseMatchScore == null) {
    defenseCallDetailEl.textContent = "No defense template match score available.";
  } else {
    const scoreText = Number(state.defenseMatchScore).toFixed(3);
    const templateText = state.defenseTemplateFront
      ? ` | Closest template front: ${state.defenseTemplateFront}`
      : "";
    defenseCallDetailEl.textContent = `Front: ${state.defenseFront || "Unknown"} | Coverage: ${
      state.defenseCoverage || "Unknown"
    }${templateText} | Match score: ${scoreText} (lower is better)`;
  }

  qbWarningEl.textContent = computeQbWarning();
}

function computeQbWarning() {
  const offense = state.players.filter((player) => player.side === "offense");
  const center =
    offense.find((player) => player.locked_label === "C") ||
    offense.find((player) => player.predicted_label === "C") ||
    null;
  const qb = offense.find((player) => player.predicted_label === "QB") || null;
  if (!center || !qb) return "";

  const olLabels = new Set(["LT", "LG", "C", "RG", "RT"]);
  let pool = offense.filter((player) => !olLabels.has(player.predicted_label));
  if (pool.length === 0) return "";

  let behind = pool.filter((player) => player.x <= state.losX + 0.02);
  if (behind.length === 0) behind = pool;
  const immediate = [...behind].sort((a, b) => {
    if (b.x !== a.x) return b.x - a.x;
    return Math.abs(a.y - center.y) - Math.abs(b.y - center.y);
  })[0];

  const laneHalfWidth = 0.02;
  const lateralGap = Math.abs(qb.y - center.y);
  const overlap = Math.max(0, 1 - lateralGap / laneHalfWidth);

  if (qb.id !== immediate.id) {
    return "QB warning: QB should be the first player directly behind the ball.";
  }
  if (overlap < 0.8) {
    return `QB warning: QB is only ${Math.round(overlap * 100)}% centered behind the ball.`;
  }
  return "";
}

function renderField() {
  losLineEl.style.left = `${state.losX * 100}%`;
  firstDownLineEl.style.left = `${state.firstDownX * 100}%`;

  const offensePlayers = state.players.filter((player) => player.side === "offense");
  let centerPlayer =
    offensePlayers.find((player) => player.locked_label === "C") ||
    offensePlayers.find((player) => player.predicted_label === "C") ||
    offensePlayers.find((player) => player.id === "o5") ||
    offensePlayers[0] ||
    null;
  if (centerPlayer) {
    ballMarkerEl.style.left = `${state.losX * 100}%`;
    ballMarkerEl.style.top = `${centerPlayer.y * 100}%`;
    ballMarkerEl.style.display = "flex";
  } else {
    ballMarkerEl.style.display = "none";
  }

  playersLayerEl.innerHTML = "";
  for (const player of state.players) {
    const el = document.createElement("button");
    const classes = ["player", player.side];
    if (player.id === state.selectedId) classes.push("selected");
    if (player.locked_label) classes.push("locked");
    if (isImmovablePlayer(player)) classes.push("immovable");
    el.className = classes.join(" ");
    el.type = "button";
    el.style.left = `${player.x * 100}%`;
    el.style.top = `${player.y * 100}%`;
    const label = player.predicted_label || "?";
    el.innerHTML = `<span>${label}</span>`;
    el.addEventListener("pointerdown", (event) => handlePointerDown(event, player.id));
    el.addEventListener("click", () => setSelected(player.id));
    playersLayerEl.appendChild(el);
  }
}

function render() {
  renderField();
  renderDefenseCall();
  renderPlayersTable();
  renderPlayerEditor();
}

async function handleSave() {
  saveStatusEl.textContent = "Saving...";
  try {
    const playId = playIdInputEl.value.trim();
    const response = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        play_id: playId || null,
        split: splitSelectEl.value,
        los_x: state.losX,
        players: state.players,
      }),
    });
    if (!response.ok) throw new Error("save failed");
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    playIdInputEl.value = data.play_id;
    const callSuffix = data.defense_call ? ` | ${data.defense_call}` : "";
    saveStatusEl.textContent = `Saved ${data.play_id} to ${data.path} (${data.split})${callSuffix}`;
  } catch (error) {
    console.error(error);
    saveStatusEl.textContent = "Save failed. Check terminal logs.";
  }
}

async function boot() {
  const response = await fetch("/api/config");
  const data = await response.json();
  state.config = data.config;
  state.losX = data.los_x ?? 0.5;
  state.firstDownX = clamp01(state.losX + 0.2);
  state.maxPlayersPerSide = data.max_players_per_side ?? 11;
  state.starterPlayers = data.starter_players || [];
  state.players = state.starterPlayers.map((player) => ({ ...player }));
  state.selectedId = state.players[0]?.id || null;
  const templateCount = data.template_count ?? 0;
  state.defenseCall = templateCount > 0 ? "Move players to predict call" : "No defense templates found";
  setDetailsPanelOpen(false);
  render();
  queueInfer(0);
}

boot().catch((error) => {
  console.error(error);
});
