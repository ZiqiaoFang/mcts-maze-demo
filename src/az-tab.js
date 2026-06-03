import { MAZES, regenerateMaze } from "./maze.js";
import { MazeRenderer } from "./render-maze.js";
import { compileReward } from "./reward.js";
import { createModel, predict, predictBatch, trainStep } from "./az-network.js";
import { AZMCTS } from "./az-mcts.js";
import { ReplayBuffer, playOneGame } from "./az-selfplay.js";
import { CHANNELS } from "./az-encode.js";

const Z_PRESETS = {
  "Pure win/loss": "isGoal ? 1 : -1",
  "Distance-shaped": "isGoal ? 1 : (1 - dist / maxDist)",
  "Step penalty": "isGoal ? Math.max(0, 1 - steps / (maxDist * 4)) : -1",
};

let state = null; // initialized inside initAzTab to defer TF.js access

export function initAzTab() {
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

  zInput.value = Z_PRESETS["Pure win/loss"];

  state = freshState(parseInt(sizeSel.value, 10), parseFloat(lrSlider.value));
  state.zFn = compileReward(zInput.value);
  mazeRenderer.render(currentMaze(), {});

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
    state.zFn = compileReward(zInput.value);
    mazeRenderer.render(currentMaze(), {});
    updateStatusBar();
  });

  stepBtn.addEventListener("click", () => runSelfPlayStep(mazeRenderer));
  run10Btn.addEventListener("click", () => runMany(10, mazeRenderer));
  runContBtn.addEventListener("click", () => toggleContinuous(runContBtn, mazeRenderer));
  resetMctsBtn.addEventListener("click", () => { /* no-op for now; tree pane added later */ });
  resetNetBtn.addEventListener("click", () => {
    if (!confirm("Reset network: erase all weights, replay buffer, and charts?")) return;
    const lr = Math.pow(10, parseFloat(lrSlider.value));
    state = freshState(state.size, lr);
    state.zFn = compileReward(zInput.value);
    mazeRenderer.render(currentMaze(), {});
    updateStatusBar();
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
  const result = playOneGame({
    maze, network, zFn: state.zFn,
    simsPerMove: state.simsPerMove, cPuct: state.cPuct,
    maxSteps, temperatureMoves: 5,
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
}

async function runMany(N, mazeRenderer) {
  for (let i = 0; i < N; i++) {
    await runSelfPlayStep(mazeRenderer);
    await new Promise((r) => requestAnimationFrame(r));
  }
}

async function toggleContinuous(btn, mazeRenderer) {
  if (state.running) { state.running = false; btn.textContent = "Run continuously"; return; }
  state.running = true;
  btn.textContent = "Stop";
  while (state.running) {
    await runSelfPlayStep(mazeRenderer);
    await new Promise((r) => requestAnimationFrame(r));
  }
  btn.textContent = "Run continuously";
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
}
