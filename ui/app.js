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

const selectedSummaryEl = document.getElementById("selected-summary");
const selectedDescriptionEl = document.getElementById("selected-description");
const selectedConfidenceEl = document.getElementById("selected-confidence");
const selectedAlternativesEl = document.getElementById("selected-alternatives");

const playIdInputEl = document.getElementById("play-id-input");
const splitSelectEl = document.getElementById("split-select");
const saveStatusEl = document.getElementById("save-status");
const layoutEl = document.querySelector(".layout");
const sidePanelEl = document.getElementById("side-panel");
const detailsToggleEl = document.getElementById("details-toggle");
const IMMOBILE_OL_LABELS = new Set(["LT", "LG", "C", "RG", "RT"]);
const FIRST_DOWN_OFFSET_X = 0.3;
const FIELD_CONTENT_SHIFT_X = FIRST_DOWN_OFFSET_X / 10;
const YARD_LINE_STEP_UI = 0.15;
const LINE_RIGHT_SHIFT_UI = 0.019;
const YARD_TO_UI_X = 0.02;
const OL_SETBACK_YARDS_FROM_ORANGE = 1.0;
const RIGHTMOST_YARD_LINE_UI = 1 - (4 * (YARD_LINE_STEP_UI / 5));
const LEFTMOST_YARD_LINE_UI =
  RIGHTMOST_YARD_LINE_UI -
  Math.floor(RIGHTMOST_YARD_LINE_UI / YARD_LINE_STEP_UI) * YARD_LINE_STEP_UI;

document.getElementById("reset-btn").addEventListener("click", handleReset);
document.getElementById("save-btn").addEventListener("click", handleSave);

fieldEl.addEventListener("pointermove", handlePointerMove);
fieldEl.addEventListener("pointerup", endDrag);
fieldEl.addEventListener("pointerleave", endDrag);
fieldEl.addEventListener("pointercancel", endDrag);
detailsToggleEl.addEventListener("click", toggleDetailsPanel);

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function modelXToUiX(modelX) {
  return clamp01(Number(modelX) - FIELD_CONTENT_SHIFT_X);
}

function uiXToModelX(uiX) {
  return clamp01(Number(uiX) + FIELD_CONTENT_SHIFT_X);
}

function snapUiXToLeftYardLine(uiX) {
  const clamped = clamp01(Number(uiX));
  const stepsFromRight = Math.ceil((RIGHTMOST_YARD_LINE_UI - clamped) / YARD_LINE_STEP_UI);
  const snapped = RIGHTMOST_YARD_LINE_UI - stepsFromRight * YARD_LINE_STEP_UI;
  const bounded = Math.min(RIGHTMOST_YARD_LINE_UI, Math.max(LEFTMOST_YARD_LINE_UI, snapped));
  return clamp01(bounded);
}

function snapModelXToLeftYardLine(modelX) {
  return uiXToModelX(snapUiXToLeftYardLine(modelXToUiX(modelX)));
}

function modelXToLineUiX(modelX) {
  return clamp01(snapUiXToLeftYardLine(modelXToUiX(modelX)) + LINE_RIGHT_SHIFT_UI);
}

function nextWhiteLineRightUiX(modelX) {
  return clamp01(modelXToLineUiX(modelX) + YARD_LINE_STEP_UI);
}

function alignStarterOffensiveLineToOrangeLine(players) {
  const olUiX = clamp01(modelXToLineUiX(state.losX) - (OL_SETBACK_YARDS_FROM_ORANGE * YARD_TO_UI_X));
  const olModelX = uiXToModelX(olUiX);
  return players.map((player) => {
    if (
      player.side === "offense" &&
      IMMOBILE_OL_LABELS.has(String(player.locked_label || ""))
    ) {
      return { ...player, x: olModelX };
    }
    return player;
  });
}

function byId(id) {
  return state.players.find((player) => player.id === id) || null;
}

function displayPositionLabel(label) {
  if (!label) return "-";
  const positionNames = {
    DE: "Defensive End",
    LB: "Linebacker",
    DT: "Defensive Tackle",
    CB: "Cornerback",
    LT: "Left Tackle",
    LG: "Left Guard",
    C: "Center",
    RT: "Right Tackle",
    RG: "Right Guard",
    QB: "Quarterback",
    RB: "Running Back",
    TE: "Tight End",
    WR: "Wide Receiver",
    S: "Safety",
  };
  return positionNames[label] || label;
}

const POSITION_DESCRIPTIONS = {
  QB: "Quarterback (QB): The quarterback starts each play by receiving the snap and decides where the ball goes. They can throw the ball, hand it to a running back, or run it themselves.",
  RB: "Running Back (RB): The running back usually takes handoffs from the quarterback and runs the ball forward. They can also catch short passes and help block defenders.",
  WR: "Wide Receiver (WR): Wide receivers run routes to get open and catch passes from the quarterback. After catching the ball, they try to gain as many yards as possible.",
  TE: "Tight End (TE): The tight end lines up near the offensive line and can either block defenders or run routes to catch passes. Their role changes depending on the play.",
  C:  "Center (C): The center snaps the ball to the quarterback to begin the play. After that, they block defenders in the middle of the line.",
  LG: "Left Guard (LG): The left guard blocks defenders on the left side of the center to protect the quarterback and help create running lanes.",
  RG: "Right Guard (RG): The right guard blocks defenders on the right side of the center. They help protect the quarterback and support running plays.",
  LT: "Left Tackle (LT): The left tackle blocks defenders on the left edge of the line, helping protect the quarterback from outside pressure.",
  RT: "Right Tackle (RT): The right tackle blocks defenders on the right edge. They help stop defenders from getting around the line.",
  DE: "Defensive End (DE): Defensive ends line up on the outside of the defensive line and try to get past blockers to tackle the quarterback or stop runs to the outside.",
  DT: "Defensive Tackle (DT): Defensive tackles line up in the middle and try to stop runs up the center or push through to pressure the quarterback.",
  LB: "Linebacker (LB): Linebackers watch the play and move toward the ball, helping stop runs, covering short passes, or rushing the quarterback.",
  CB: "Cornerback (CB): Cornerbacks cover wide receivers and try to stop them from catching passes. They stay close and try to knock the ball away or intercept it.",
  S:  "Safety (S): Safeties play deeper in the field and help stop long passes or runs that get past other defenders. They are the last line of defense.",
};

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
  const uiX = clamp01((event.clientX - rect.left) / rect.width);
  const y = clamp01((event.clientY - rect.top) / rect.height);
  const player = byId(state.draggingId);
  if (!player) return;
  player.x = uiXToModelX(uiX);
  player.y = y;
  render();
  queueInfer(30);
}

function endDrag() {
  state.draggingId = null;
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
    selectedSummaryEl.textContent = "Predicted Position: -";
    selectedDescriptionEl.textContent = "Position Description: -";
    selectedConfidenceEl.textContent = "Position Confidence: -";
    selectedAlternativesEl.textContent = "Other Likely Positions: -";
    return;
  }

  const rawLabel = player.predicted_label || "";
  const displayLabel = displayPositionLabel(player.predicted_label);
  const confidenceText = `${displayConfidencePercent(player)}%`;
  selectedSummaryEl.textContent = `Predicted Position: ${displayLabel}`;
  selectedDescriptionEl.textContent = `Position Description: ${
    POSITION_DESCRIPTIONS[rawLabel] || "-"
  }`;
  selectedConfidenceEl.textContent = `Position Confidence: ${confidenceText}`;
  selectedAlternativesEl.textContent = `Other Likely Positions: ${formatAlternativePositions(player)}`;
}

function formatAlternativePositions(player) {
  const probs = player && typeof player.label_probabilities === "object"
    ? player.label_probabilities
    : null;
  if (!probs) return "-";

  const side = String(player.side || "");
  const offenseAllowed = new Set(["QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT"]);
  const defenseAllowed = new Set(["DE", "DT", "CB", "S", "LB"]);
  const allowed = side === "defense" ? defenseAllowed : offenseAllowed;
  const predicted = String(player.predicted_label || "");
  const alternatives = Object.entries(probs)
    .map(([label, prob]) => [String(label), Number(prob)])
    .filter(([label, prob]) => label && Number.isFinite(prob))
    .filter(([label]) => allowed.has(label))
    .filter(([label]) => label !== predicted)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  if (alternatives.length === 0) return "-";

  return alternatives
    .map(([label, prob]) => `${displayPositionLabel(label)} (${Math.round(clamp01(prob) * 100)}%)`)
    .join(", ");
}

function displayConfidencePercent(player) {
  if (!player) return 0;
  const label = String(player.predicted_label || "");
  const locked = String(player.locked_label || "");
  const side = String(player.side || "");

  // Locked OL labels are immovable in this UI and should display as certain.
  if (side === "offense" && IMMOBILE_OL_LABELS.has(locked)) {
    return 100;
  }

  let basePercent = Math.round(clamp01(Number(player.predicted_confidence) || 0) * 100);
  const xRaw = Number(player.x);
  const yRaw = Number(player.y);
  const x = Number.isFinite(xRaw) ? clamp01(xRaw) : Number(state.losX || 0.5);
  const y = Number.isFinite(yRaw) ? clamp01(yRaw) : 0.5;
  const sidelineGap = Math.min(y, 1.0 - y);
  const depthFromLos = Math.abs(x - Number(state.losX || 0.5));

  if (side !== "offense") {
    if (label === "CB") {
      // Boundary + off-ball depth strongly indicate corner alignment.
      const boundaryFactor = 1 - clamp01(sidelineGap / 0.30);
      const depthFactor = clamp01((depthFromLos - 0.06) / 0.20);
      const cbFloor = Math.round(55 + (boundaryFactor * 30) + (depthFactor * 10));
      basePercent = Math.max(basePercent, cbFloor);
    } else if (label === "S") {
      // Safeties are typically deeper and somewhat interior.
      const deepFactor = clamp01((depthFromLos - 0.10) / 0.20);
      const interiorFactor = clamp01((sidelineGap - 0.12) / 0.25);
      const sFloor = Math.round(58 + (deepFactor * 28) + (interiorFactor * 8));
      basePercent = Math.max(basePercent, sFloor);
    } else if (label === "LB") {
      const interiorFactor = clamp01((sidelineGap - 0.14) / 0.22);
      const lbFloor = Math.round(52 + (interiorFactor * 20));
      basePercent = Math.max(basePercent, lbFloor);
    } else if (label === "DE" || label === "DT" || label === "NT") {
      const lineFactor = 1 - clamp01(depthFromLos / 0.12);
      const dlFloor = Math.round(68 + (lineFactor * 27));
      basePercent = Math.max(basePercent, dlFloor);
    } else {
      basePercent = Math.max(basePercent, 45);
    }

    return Math.round(clamp01(basePercent / 100) * 100);
  }

  const offensePlayers = state.players.filter((p) => String(p.side || "") === "offense");
  const offenseLabel = (p) => String(p.locked_label || p.predicted_label || "");
  const lt = offensePlayers.find((p) => offenseLabel(p) === "LT");
  const rt = offensePlayers.find((p) => offenseLabel(p) === "RT");
  const center =
    offensePlayers.find((p) => offenseLabel(p) === "C") ||
    offensePlayers.find((p) => String(p.predicted_label || "") === "C") ||
    null;

  // Keep skill-position displays from reading as near-zero confidence.
  if (label === "QB") {
    basePercent = Math.max(basePercent, 60);
  } else if (label === "RB") {
    basePercent = Math.max(basePercent, 50);
  } else if (label === "WR" || label === "TE") {
    basePercent = Math.max(basePercent, 45);
  }

  // WR confidence floor from geometry:
  // outside tackle box and/or wide toward a sideline.
  if (label === "WR") {
    const maxWideGap = 0.30;
    if (sidelineGap <= maxWideGap) {
      const wideFactor = clamp01((maxWideGap - sidelineGap) / maxWideGap);
      const wrWideFloor = Math.round(60 + (wideFactor * 35));
      basePercent = Math.max(basePercent, wrWideFloor);
    }

    const ltY = lt ? Number(lt.y) : NaN;
    const rtY = rt ? Number(rt.y) : NaN;
    if (Number.isFinite(ltY) && Number.isFinite(rtY)) {
      const lowTackleY = Math.min(ltY, rtY);
      const highTackleY = Math.max(ltY, rtY);
      const outsideTackle = y < lowTackleY || y > highTackleY;
      if (outsideTackle) {
        const outsideGap = Math.min(Math.abs(y - lowTackleY), Math.abs(y - highTackleY));
        const outsideFactor = clamp01(outsideGap / 0.20);
        const wrOutsideFloor = Math.round(65 + (outsideFactor * 30));
        basePercent = Math.max(basePercent, wrOutsideFloor);
      }
    }
  }

  // TE confidence floor when geometry strongly matches a tight-end alignment:
  // close to LT/RT lane and not detached deep from LOS.
  if (label === "TE") {
    if (lt || rt) {
      const ltY = lt ? Number(lt.y) : NaN;
      const rtY = rt ? Number(rt.y) : NaN;
      const tackleGap = Math.min(
        Number.isFinite(ltY) ? Math.abs(y - ltY) : 1.0,
        Number.isFinite(rtY) ? Math.abs(y - rtY) : 1.0
      );
      const nearTackleLane = tackleGap <= 0.16;
      const edgeBand = sidelineGap >= 0.08 && sidelineGap <= 0.50;
      const nearLine = depthFromLos <= 0.22;
      if (nearTackleLane && edgeBand && nearLine) {
        const closeness = 1 - clamp01(tackleGap / 0.16);
        const teFloor = Math.round(62 + (closeness * 33));
        basePercent = Math.max(basePercent, teFloor);
      }
    }
  }

  // QB confidence floor from center-relative alignment:
  // QB should be first player directly behind center and reasonably centered.
  if (label === "QB" && center) {
    const centerX = Number(center.x);
    const centerY = Number(center.y);
    if (Number.isFinite(centerX) && Number.isFinite(centerY)) {
      const lateralGapBase = Math.abs(y - centerY);
      const centeredFactor = 1 - clamp01(lateralGapBase / 0.08);
      const qbBackfieldFactor = clamp01((depthFromLos - 0.01) / 0.18);
      const qbGeometryFloor = Math.round(62 + (centeredFactor * 18) + (qbBackfieldFactor * 10));
      basePercent = Math.max(basePercent, qbGeometryFloor);

      const olLabels = new Set(["LT", "LG", "C", "RG", "RT"]);
      let pool = offensePlayers.filter((p) => !olLabels.has(String(p.predicted_label || "")));
      if (pool.length === 0) pool = offensePlayers;
      let behind = pool.filter((p) => Number(p.x) <= Number(state.losX) + 0.02);
      if (behind.length === 0) behind = pool;
      const immediate = [...behind].sort((a, b) => {
        const ax = Number(a.x);
        const bx = Number(b.x);
        if (bx !== ax) return bx - ax;
        return Math.abs(Number(a.y) - centerY) - Math.abs(Number(b.y) - centerY);
      })[0];

      if (immediate && String(immediate.id) === String(player.id)) {
        const lateralGap = Math.abs(y - centerY);
        const overlap = Math.max(0, 1 - (lateralGap / 0.03));
        const qbDepth = Math.abs(x - centerX);
        const depthFactor = clamp01((0.20 - qbDepth) / 0.20);
        const qbFloor = Math.round(78 + (overlap * 16) + (depthFactor * 6));
        basePercent = Math.max(basePercent, qbFloor);
      }
    }
  }

  // RB confidence floor from backfield geometry:
  // aligned in the backfield with reasonable center proximity.
  if (label === "RB" && center) {
    const centerX = Number(center.x);
    const centerY = Number(center.y);
    if (Number.isFinite(centerX) && Number.isFinite(centerY)) {
      const qb = offensePlayers.find((p) => String(p.predicted_label || "") === "QB") || null;
      const lateralGap = Math.abs(y - centerY);
      const lateralFactor = 1 - clamp01(lateralGap / 0.22);
      const backfieldFactor = clamp01((depthFromLos - 0.04) / 0.20);
      let rbFloor = Math.round(58 + (backfieldFactor * 24) + (lateralFactor * 10));

      if (qb) {
        const qbX = Number(qb.x);
        const qbDepth = Number.isFinite(qbX) ? Math.abs(clamp01(qbX) - Number(state.losX || 0.5)) : 0;
        const deeperThanQb = depthFromLos >= (qbDepth + 0.012);
        if (deeperThanQb) rbFloor = Math.max(rbFloor, 74);
      }

      basePercent = Math.max(basePercent, rbFloor);
    }
  }

  return Math.round(clamp01(basePercent / 100) * 100);
}

function renderPlayersTable() {
  const rows = [...state.players].sort((a, b) => {
    if (a.side !== b.side) return a.side < b.side ? -1 : 1;
    return a.y - b.y;
  });
  playersTableEl.innerHTML = rows
    .map((player) => {
      const selectedClass = player.id === state.selectedId ? "selected" : "";
      const confidence = `${displayConfidencePercent(player)}%`;
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

function computedDefensiveFrontFromLabels() {
  const defensePlayers = state.players.filter((player) => player.side === "defense");
  const deCount = defensePlayers.filter((player) => player.predicted_label === "DE").length;
  const dtCount = defensePlayers.filter((player) => player.predicted_label === "DT").length;
  const lbCount = defensePlayers.filter((player) => player.predicted_label === "LB").length;
  return `${deCount + dtCount}-${lbCount}`;
}

function defenseScoreToPercent(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return null;
  // Display mapping only: convert distance-like score (lower is better) to a 40-100% band.
  // This keeps confidence readable while preserving rank ordering.
  const maxReasonableScore = 3.0;
  const normalized = 1 - clamp01(value / maxReasonableScore);
  return Math.round(40 + normalized * 60);
}

function renderDefenseCall() {
  const computedFront = computedDefensiveFrontFromLabels();
  const computedCoverage = state.defenseCoverage || "Unknown";
  defenseCallEl.textContent = `${computedFront} ${computedCoverage}`.trim();
  if (state.defenseMatchScore == null) {
    defenseCallDetailEl.textContent = "No defense template confidence available.";
  } else {
    const confidencePct = defenseScoreToPercent(state.defenseMatchScore);
    defenseCallDetailEl.textContent = `Front (DE+DT-LB): ${computedFront} | Coverage: ${
      computedCoverage
    } | Match confidence: ${confidencePct ?? 0}%`;
  }

  // Keep QB placement logic active without surfacing the warning text in details UI.
  computeQbWarning();
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
  losLineEl.style.left = `${modelXToLineUiX(state.losX) * 100}%`;
  firstDownLineEl.style.left = `${nextWhiteLineRightUiX(state.firstDownX) * 100}%`;

  const offensePlayers = state.players.filter((player) => player.side === "offense");
  let centerPlayer =
    offensePlayers.find((player) => player.locked_label === "C") ||
    offensePlayers.find((player) => player.predicted_label === "C") ||
    offensePlayers.find((player) => player.id === "o5") ||
    offensePlayers[0] ||
    null;
  if (centerPlayer) {
    ballMarkerEl.style.left = `${modelXToLineUiX(state.losX) * 100}%`;
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
    el.style.left = `${modelXToUiX(player.x) * 100}%`;
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
  state.losX = snapModelXToLeftYardLine(data.los_x ?? 0.5);
  state.firstDownX = snapModelXToLeftYardLine(clamp01(state.losX + FIRST_DOWN_OFFSET_X));
  state.maxPlayersPerSide = data.max_players_per_side ?? 11;
  state.starterPlayers = alignStarterOffensiveLineToOrangeLine(data.starter_players || []);
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
