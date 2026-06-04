import { MAZES, regenerateMaze } from "./maze.js";
import { ACTIONS } from "./maze.js";
import { MazeRenderer } from "./render-maze.js";
import { compileReward } from "./reward.js";
import { createModel, predict, predictBatch, trainStep, setupBackend, warmupModel } from "./az-network.js";
import { AZMCTS } from "./az-mcts.js";
import { ReplayBuffer, playOneGame } from "./az-selfplay.js";
import { CHANNELS } from "./az-encode.js";
import { LineChart, ProbePanel } from "./az-charts.js";
import { TreeRenderer } from "./render-tree.js";

const Z_PRESETS = {
  "Pure win/loss": "isGoal ? 1 : -1",
  "Distance-shaped": "isGoal ? 1 : (1 - dist / maxDist)",
  "Step penalty": "isGoal ? Math.max(0, 1 - steps / (maxDist * 4)) : -1",
};

let state = null; // initialized inside initAzTab to defer TF.js access
let activeBackend = null;

export async function initAzTab() {
  activeBackend = await setupBackend();
  const sizeSel = document.getElementById("az-ctl-size");
  const zInput = document.getElementById("az-ctl-z-formula");
  const zPreset = document.getElementById("az-ctl-z-preset");
  const cPuctSlider = document.getElementById("az-ctl-cpuct");
  const cPuctVal = document.getElementById("az-ctl-cpuct-val");
  const simsSlider = document.getElementById("az-ctl-sims");
  const simsVal = document.getElementById("az-ctl-sims-val");
  const lrSlider = document.getElementById("az-ctl-lr");
  const lrVal = document.getElementById("az-ctl-lr-val");
  const stepBtn = document.getElementById("az-btn-step");
  const run10Btn = document.getElementById("az-btn-run10");
  const runContBtn = document.getElementById("az-btn-runcont");
  const resetMctsBtn = document.getElementById("az-btn-reset-mcts");
  const resetNetBtn = document.getElementById("az-btn-reset-net");
  const canvas = document.getElementById("az-maze-canvas");
  const mazeRenderer = new MazeRenderer(canvas);
  let treeRenderer = new TreeRenderer(document.getElementById("az-tree-svg"));

  zInput.value = Z_PRESETS["Pure win/loss"];

  const zChart = new LineChart(document.getElementById("az-chart-z"),
    { title: "z per game (raw + moving avg)", yMin: -1.1, yMax: 1.1 });
  zChart.addSeries("raw", "#888");
  zChart.addSeries("ma", "#6ab0ff");
  const lossChart = new LineChart(document.getElementById("az-chart-loss"),
    { title: "training loss" });
  lossChart.addSeries("total", "#6ab0ff");
  lossChart.addSeries("policy", "#ffb86b");
  lossChart.addSeries("value", "#f06b6b");
  const probe = new ProbePanel(document.getElementById("az-probe"));

  state = freshState(parseInt(sizeSel.value, 10), parseFloat(lrSlider.value));
  state.zChart = zChart;
  state.lossChart = lossChart;
  state.probe = probe;
  state.treeRenderer = treeRenderer;
  state.zFn = compileReward(zInput.value);
  warmupModel(state.model, state.size);
  renderMaze(mazeRenderer);
  renderCharts();

  // Slider labels
  cPuctSlider.addEventListener("input", () => {
    state.cPuct = parseFloat(cPuctSlider.value);
    cPuctVal.textContent = state.cPuct.toFixed(2);
  });
  simsSlider.addEventListener("input", () => {
    state.simsPerMove = parseInt(simsSlider.value, 10);
    simsVal.textContent = simsSlider.value;
  });
  lrSlider.addEventListener("input", () => {
    const lr = Math.pow(10, parseFloat(lrSlider.value));
    state.optimizer = tf.train.adam(lr);
    lrVal.textContent = lr.toExponential(1);
  });
  zPreset.addEventListener("change", () => {
    const key = zPreset.value;
    if (!key) return;
    zInput.value = Z_PRESETS[key];
    state.zFn = compileReward(zInput.value);
    zPreset.value = "";
  });
  zInput.addEventListener("change", () => {
    try {
      state.zFn = compileReward(zInput.value);
      zInput.classList.remove("invalid");
    } catch (err) {
      zInput.classList.add("invalid");
      zInput.title = err.message;
    }
  });

  sizeSel.addEventListener("change", () => {
    const size = parseInt(sizeSel.value, 10);
    if (!confirm("Changing maze size will reset the AlphaZero network and replay buffer. Continue?")) {
      sizeSel.value = String(state.size);
      return;
    }
    state = freshState(size, Math.pow(10, parseFloat(lrSlider.value)));
    state.zChart = zChart;
    state.lossChart = lossChart;
    state.probe = probe;
    treeRenderer = new TreeRenderer(document.getElementById("az-tree-svg"));
    state.treeRenderer = treeRenderer;
    zChart.clear();
    lossChart.clear();
    state.zFn = compileReward(zInput.value);
    warmupModel(state.model, state.size);
    renderMaze(mazeRenderer);
    updateStatusBar();
    renderCharts();
  });

  stepBtn.addEventListener("click", async () => {
    await runSelfPlayStep(mazeRenderer);
    setIdleProgress();
  });
  run10Btn.addEventListener("click", () => runMany(10, mazeRenderer));
  runContBtn.addEventListener("click", () => toggleContinuous(runContBtn, mazeRenderer));
  resetMctsBtn.addEventListener("click", () => {
    treeRenderer = new TreeRenderer(document.getElementById("az-tree-svg"));
    state.treeRenderer = treeRenderer;
  });
  resetNetBtn.addEventListener("click", () => {
    if (!confirm("Reset network: erase all weights, replay buffer, and charts?")) return;
    const lr2 = Math.pow(10, parseFloat(lrSlider.value));
    state = freshState(state.size, lr2);
    state.zChart = zChart;
    state.lossChart = lossChart;
    state.probe = probe;
    treeRenderer = new TreeRenderer(document.getElementById("az-tree-svg"));
    state.treeRenderer = treeRenderer;
    zChart.clear();
    lossChart.clear();
    state.zFn = compileReward(zInput.value);
    warmupModel(state.model, state.size);
    renderMaze(mazeRenderer);
    updateStatusBar();
    renderCharts();
  });

  for (const r of document.querySelectorAll('input[name="az-overlay"]')) {
    r.addEventListener("change", () => renderMaze(mazeRenderer));
  }
  // classic-tab owns the theme toggle and flips data-theme; we just re-render.
  document.getElementById("theme-toggle").addEventListener("click", () => {
    renderMaze(mazeRenderer);
  });

  updateStatusBar();
}

function currentMaze() { return MAZES[state.size]; }

function freshState(size, lr) {
  return {
    size,
    cPuct: 1.0,
    simsPerMove: 50,
    optimizer: tf.train.adam(lr),
    model: createModel(size),
    buffer: new ReplayBuffer(10000),
    gamesPlayed: 0,
    latestZ: NaN,
    zHistory: [],
    lastLosses: null,
    zFn: null,
    running: false,
  };
}

async function runSelfPlayStep(mazeRenderer) {
  if (!state.zFn) return;
  const maze = currentMaze();
  const network = { predict: (m, pos) => predict(state.model, m, pos) };
  const maxSteps = state.size * 4;
  const result = await playOneGame({
    maze, network, zFn: state.zFn,
    simsPerMove: state.simsPerMove, cPuct: state.cPuct,
    maxSteps, temperatureMoves: 5,
    onProgress: (ev) => updateProgress(ev),
  });
  for (const t of result.trajectory) {
    state.buffer.append({ state: t.state, pi: t.pi, z: result.z });
  }
  if (state.buffer.size() >= 32) {
    const inputDim = state.size * state.size * CHANNELS;
    const batch = state.buffer.sampleBatch(32, inputDim);
    state.lastLosses = trainStep(state.model, state.optimizer, batch, inputDim);
  }
  state.gamesPlayed += 1;
  state.latestZ = result.z;
  state.zHistory.push(result.z);
  if (state.zHistory.length > 1000) state.zHistory.shift();
  updateStatusBar();
  state.zChart.addPoint("raw", result.z);
  const w = state.zHistory.slice(-20);
  state.zChart.addPoint("ma", w.reduce((a, b) => a + b, 0) / w.length);
  if (state.lastLosses) {
    state.lossChart.addPoint("total", state.lastLosses.total);
    state.lossChart.addPoint("policy", state.lastLosses.policy);
    state.lossChart.addPoint("value", state.lastLosses.value);
  }
  state.lastMcts = result.lastMcts ?? null;
  state.cumulativeVisits = result.cumulativeVisits ?? new Map();
  renderCharts();
  updateProbe();
  renderMaze(mazeRenderer);
  if (result.lastMcts) state.treeRenderer.render(result.lastMcts);
}

async function runMany(N, mazeRenderer) {
  state.loopActive = N;
  state.loopDone = 0;
  for (let i = 0; i < N; i++) {
    state.loopDone = i;
    await runSelfPlayStep(mazeRenderer);
    await new Promise((r) => setTimeout(r, 0));
  }
  state.loopActive = null;
  setIdleProgress();
}

async function toggleContinuous(btn, mazeRenderer) {
  if (state.running) { state.running = false; btn.textContent = "Run continuously"; return; }
  state.running = true;
  btn.textContent = "Stop";
  state.loopActive = Infinity;
  state.loopDone = 0;
  while (state.running) {
    await runSelfPlayStep(mazeRenderer);
    state.loopDone += 1;
    await new Promise((r) => setTimeout(r, 0));
  }
  state.loopActive = null;
  setIdleProgress();
  btn.textContent = "Run continuously";
}

function updateProgress({ move, maxSteps }) {
  const fill = document.getElementById("az-progress-fill");
  const text = document.getElementById("az-progress-text");
  const root = document.getElementById("az-progress");
  root.classList.remove("idle");
  const pct = Math.min(100, ((move + 1) / maxSteps) * 100);
  fill.style.width = `${pct.toFixed(1)}%`;
  let prefix = "";
  if (state.loopActive === Infinity) {
    prefix = `game ${state.loopDone + 1} (running) · `;
  } else if (state.loopActive && state.loopActive > 1) {
    prefix = `game ${state.loopDone + 1} / ${state.loopActive} · `;
  }
  // The "/maxSteps" is the agent's step budget per game (= size × 4), NOT
  // the MCTS sims/move slider. Surface simsPerMove here too so users can
  // see the slider value is taking effect inside each move.
  text.textContent =
    `${prefix}agent step ${move + 1} / ${maxSteps} · ${state.simsPerMove} MCTS sims this move`;
}

function setIdleProgress() {
  const root = document.getElementById("az-progress");
  const fill = document.getElementById("az-progress-fill");
  const text = document.getElementById("az-progress-text");
  if (!root || !fill || !text) return;
  root.classList.add("idle");
  fill.style.width = "0%";
  text.textContent = "idle";
}

function updateStatusBar() {
  document.getElementById("az-games").textContent = state.gamesPlayed;
  document.getElementById("az-latest-z").textContent =
    isNaN(state.latestZ) ? "—" : state.latestZ.toFixed(3);
  if (state.zHistory.length === 0) {
    document.getElementById("az-avg-z").textContent = "—";
  } else {
    const w = state.zHistory.slice(-20);
    const avg = w.reduce((a, b) => a + b, 0) / w.length;
    document.getElementById("az-avg-z").textContent = avg.toFixed(3);
  }
  document.getElementById("az-loss").textContent =
    state.lastLosses ? state.lastLosses.total.toFixed(4) : "—";
  document.getElementById("az-buf").textContent = state.buffer.size();
  document.getElementById("az-backend").textContent = activeBackend ?? "—";
}

function renderCharts() {
  state.zChart.render();
  state.lossChart.render();
}

function updateProbe() {
  const maze = currentMaze();
  const { p, v } = predict(state.model, maze, maze.start);
  state.probe.update({ p, v });
}

function computeVHeatmap() {
  const maze = currentMaze();
  const size = maze.size;
  const positions = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) positions.push([r, c]);
  }
  const { vs } = predictBatch(state.model, maze, positions);
  return vs;
}

function computePArrows() {
  const maze = currentMaze();
  const size = maze.size;
  const positions = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) positions.push([r, c]);
  }
  const { ps } = predictBatch(state.model, maze, positions);
  return ps; // length size*size*4
}

// "What the agent would do right now from the start cell" — greedy walk of
// the network's policy. At each cell, pick the legal non-revisiting action
// with the highest p(a|s). Stops at the goal, when stuck, or after maxSteps.
// Returns an array of [r,c] positions; render-maze.js draws it as the green
// principal-variation polyline.
function computeBestRoute() {
  const maze = currentMaze();
  const maxSteps = maze.size * 4;
  const visited = new Set();
  const route = [[...maze.start]];
  visited.add(maze.start.join(","));
  let pos = [...maze.start];
  for (let t = 0; t < maxSteps; t++) {
    if (maze.isGoal(pos)) break;
    const { p } = predict(state.model, maze, pos);
    // Rank legal, non-revisiting actions by network prior; pick the best.
    let bestIdx = -1;
    let bestProb = -Infinity;
    for (let i = 0; i < ACTIONS.length; i++) {
      const next = maze.step(pos, ACTIONS[i]);
      if (!maze.isOpen(next)) continue;
      if (visited.has(next.join(","))) continue;
      if (p[i] > bestProb) {
        bestProb = p[i];
        bestIdx = i;
      }
    }
    if (bestIdx === -1) break; // dead-end under no-revisit
    pos = maze.step(pos, ACTIONS[bestIdx]);
    visited.add(pos.join(","));
    route.push([...pos]);
  }
  return route;
}

function renderMaze(mazeRenderer) {
  const maze = currentMaze();
  const mode = document.querySelector('input[name="az-overlay"]:checked').value;
  const opts = {};
  try {
    if (mode === "v") opts.vHeatmap = computeVHeatmap();
    else if (mode === "p") opts.pArrows = computePArrows();
    else if (mode === "visits")
      opts.visits = state.cumulativeVisits ?? new Map();
    // Best route is always drawn (regardless of overlay mode), so users can
    // see what the trained policy currently thinks the path looks like.
    // Skip until the first self-play step so the route doesn't show a
    // meaningless random walk from an untrained network.
    if (state.gamesPlayed > 0) {
      opts.principalVariation = computeBestRoute();
    }
  } catch (err) {
    // Surface silent overlay-compute failures (TF.js issues, NaN, etc.) so a
    // user reporting "the overlay shows nothing" can paste an actual error.
    console.error(`[az-tab] overlay '${mode}' failed:`, err);
  }
  mazeRenderer.render(maze, opts);
}
