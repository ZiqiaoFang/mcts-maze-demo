# AlphaZero MCTS Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second top-level tab to the MCTS Maze Demo that runs AlphaZero-style MCTS with a TensorFlow.js–backed network learned online via self-play, with visualizations of the learned value/policy fields and the training progress.

**Architecture:** A tab switcher at the top of `index.html` toggles between the existing Classic MCTS view and a new AZ view. The AZ view is wired by a new `src/az-tab.js`, which depends on: `src/az-encode.js` (maze state → tensor), `src/az-network.js` (TF.js model + training step), `src/az-mcts.js` (PUCT search), `src/az-selfplay.js` (replay buffer + game loop), and `src/az-charts.js` (lightweight canvas line charts and a probe panel). `src/render-maze.js` is extended with two new overlay modes (`v`-heatmap, `p`-arrows). Existing files (`maze.js`, `node.js`, `mcts.js`, `rng.js`, `reward.js`, `render-tree.js`, `animate.js`) are untouched; the classic-tab logic moves intact from `app.js` into `src/classic-tab.js`, leaving `app.js` as a tiny tab router.

**Tech Stack:** Vanilla JavaScript (ES modules), `node:test`, Node 26+ for tests, TensorFlow.js 4.x via CDN, `python3 -m http.server` for browser smoke testing.

**Spec:** `docs/superpowers/specs/2026-06-03-alphazero-tab-design.md`

**Note on testing strategy:** Node tests cover pure logic (encoding, PUCT formula, replay buffer FIFO, MCTS tree shape with a stub network). TensorFlow.js itself is only available in the browser in this project (we do not add `@tensorflow/tfjs-node` to keep the install lightweight), so all training-loop and rendering behaviour is validated by a manual browser smoke test at the end of each relevant task. Tasks make this explicit.

---

## File Map

- **Modify**: `index.html` — add tab switcher above the header, AZ tab markup, TF.js CDN script
- **Modify**: `styles.css` — tab button styles, AZ pane layout, charts panel grid, overlay-toggle radio
- **Modify**: `src/app.js` — strip down to a top-level tab router; defer to classic-tab / az-tab init
- **Modify**: `src/render-maze.js` — add `vHeatmap` and `pArrows` overlay modes
- **Create**: `src/classic-tab.js` — receives the entire body of the old `app.js`
- **Create**: `src/az-encode.js` — maze + agent position → Float32Array (size·size·3)
- **Create**: `src/az-network.js` — TF.js model construction, forward pass, training step
- **Create**: `src/az-mcts.js` — `AZNode`, PUCT formula, `AZMCTS` class with iterate/expand/backprop
- **Create**: `src/az-selfplay.js` — `ReplayBuffer` class, `playOneGame(mcts, network, maze, zFn)`
- **Create**: `src/az-charts.js` — `LineChart` class, `ProbePanel` class
- **Create**: `src/az-tab.js` — wires the AZ tab: DOM bindings, button handlers, render loop, run-N orchestration
- **Create**: `tests/az-encode.test.js`
- **Create**: `tests/az-mcts.test.js`
- **Create**: `tests/az-selfplay.test.js`

---

## Task 1: Add TensorFlow.js to the page (manual smoke)

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add CDN script tag for TF.js**

In `/Users/brandon/mcts-maze-demo/index.html`, find the existing `<script type="module" src="src/app.js" defer></script>` line in `<head>` and add a TF.js CDN script immediately above it:

```html
  <script src="https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js" defer></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js" defer></script>
  <script type="module" src="src/app.js" defer></script>
```

- [ ] **Step 2: Manual smoke that TF.js loads**

Run: `cd /Users/brandon/mcts-maze-demo && python3 -m http.server 8000`
Open: `http://localhost:8000` and check the JS console.
Type: `tf.tensor([1, 2, 3]).shape` and press Enter.
Expected output: `[3]`

Stop the server (Ctrl-C).

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat(az): load TensorFlow.js from CDN"
```

---

## Task 2: Refactor — extract classic-tab.js (mechanical, no behavior change)

**Files:**
- Create: `src/classic-tab.js`
- Modify: `src/app.js`

- [ ] **Step 1: Create classic-tab.js — the entire body of the old app.js wrapped in initClassicTab()**

Create `/Users/brandon/mcts-maze-demo/src/classic-tab.js` with these exact contents:

```js
// classic-tab.js — wraps the original Classic MCTS app body. The theme toggle
// stays here because it needs to call renderAll(), which closes over mcts.
import { animateIteration } from "./animate.js";
import { MAZES, regenerateMaze } from "./maze.js";
import { MCTS } from "./mcts.js";
import { MazeRenderer } from "./render-maze.js";
import { TreeRenderer } from "./render-tree.js";
import { compileReward, PRESETS } from "./reward.js";

export function initClassicTab() {
  const DEBUG = new URLSearchParams(location.search).get("debug") === "1";
  if (DEBUG) document.getElementById("debug-panel").hidden = false;

  const canvas = document.getElementById("maze-canvas");
  const svg = document.getElementById("tree-svg");
  const sizeSel = document.getElementById("ctl-size");
  const formulaInput = document.getElementById("ctl-reward-formula");
  const presetSel = document.getElementById("ctl-reward-preset");
  formulaInput.value = PRESETS["Distance-shaped"];
  let currentRewardFn = null;
  const cSlider = document.getElementById("ctl-c");
  const hSlider = document.getElementById("ctl-horizon");
  const cVal = document.getElementById("ctl-c-val");
  const hVal = document.getElementById("ctl-horizon-val");
  const statIter = document.getElementById("stat-iter");
  const statNodes = document.getElementById("stat-nodes");
  const statPV = document.getElementById("stat-pv");
  const statPhase = document.getElementById("stat-phase");

  const mazeRenderer = new MazeRenderer(canvas);
  const treeRenderer = new TreeRenderer(svg);

  let mcts = null;

  function currentMaze() {
    return MAZES[parseInt(sizeSel.value, 10)];
  }

  function newMcts() {
    const maze = currentMaze();
    mcts = new MCTS(maze, {
      C: parseFloat(cSlider.value),
      rolloutHorizon: parseInt(hSlider.value, 10),
      rewardFn: currentRewardFn,
      seed: 42,
    });
    renderAll();
  }

  function renderAll() {
    const maze = currentMaze();
    const pv = mcts.principalVariation().map((n) => n.position);
    mazeRenderer.render(maze, {
      visits: mcts.cellVisits(),
      principalVariation: pv,
    });
    treeRenderer.render(mcts);
    const s = mcts.treeStats();
    statIter.textContent = mcts.iterationCount;
    statNodes.textContent = s.totalNodes;
    statPV.textContent = pv.length - 1;
    if (maze.isGoal(pv[pv.length - 1])) statPV.style.color = "#4ade80";
    else statPV.style.color = "";
    updateDebug();
  }

  let lastRollout = null;
  function updateDebug() {
    if (!DEBUG || !mcts) return;
    const s = mcts.treeStats();
    document.getElementById("debug-stats").textContent =
      `iterations: ${mcts.iterationCount}\nnodes: ${s.totalNodes}\nmax depth: ${s.maxDepth}\nmax visits: ${s.maxVisits}`;
    if (lastRollout) {
      const start = lastRollout.positions[0];
      const end = lastRollout.positions[lastRollout.positions.length - 1];
      document.getElementById("debug-last-rollout").textContent =
        `rollout:\n  start: (${start})\n  end:   (${end})\n  steps: ${lastRollout.positions.length - 1}\n  reward: ${lastRollout.reward.toFixed(3)}`;
    }
  }

  let busy = false;
  const setButtonsEnabled = (enabled) => {
    for (const id of ["btn-step", "btn-run10", "btn-runend", "btn-reset", "btn-regenerate"]) {
      document.getElementById(id).disabled = !enabled;
    }
  };

  document.getElementById("btn-step").addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    setButtonsEnabled(false);
    await animateIteration(mcts, {
      mazeRenderer,
      treeRenderer,
      getMaze: currentMaze,
      setPhase: (p) => (statPhase.textContent = p),
      setStats: renderAll,
      onRollout: (ev) => {
        lastRollout = ev;
        updateDebug();
      },
    });
    updateDebug();
    setButtonsEnabled(true);
    busy = false;
  });
  document.getElementById("btn-run10").addEventListener("click", () => {
    if (busy) return;
    busy = true;
    setButtonsEnabled(false);
    statPhase.textContent = "running 10";
    for (let i = 0; i < 10; i++) for (const _ev of mcts.iterate()) {}
    renderAll();
    updateDebug();
    statPhase.textContent = "idle";
    setButtonsEnabled(true);
    busy = false;
  });
  document.getElementById("btn-runend").addEventListener("click", () => {
    if (busy) return;
    busy = true;
    setButtonsEnabled(false);
    statPhase.textContent = "running to end";
    const maze = currentMaze();
    let stableCount = 0;
    let lastPVKey = "";
    for (let i = 0; i < 500; i++) {
      for (const _ev of mcts.iterate()) {}
      const pv = mcts.principalVariation();
      const last = pv[pv.length - 1].position;
      const pvKey = pv.map((n) => n.position.join(",")).join("|");
      if (maze.isGoal(last) && pvKey === lastPVKey) stableCount++;
      else stableCount = 0;
      lastPVKey = pvKey;
      if (stableCount >= 5) break;
    }
    renderAll();
    updateDebug();
    statPhase.textContent = "idle";
    setButtonsEnabled(true);
    busy = false;
  });
  document.getElementById("btn-reset").addEventListener("click", () => {
    if (busy) return;
    newMcts();
  });
  document.getElementById("btn-regenerate").addEventListener("click", () => {
    if (busy) return;
    regenerateMaze(parseInt(sizeSel.value, 10));
    newMcts();
  });

  sizeSel.addEventListener("change", () => {
    const size = parseInt(sizeSel.value, 10);
    hSlider.max = size * 8;
    hSlider.value = size * 4;
    hVal.textContent = hSlider.value;
    newMcts();
  });
  function applyFormula(formulaString) {
    if (busy) return;
    try {
      const fn = compileReward(formulaString);
      formulaInput.classList.remove("invalid");
      formulaInput.title = "";
      if (statPhase.textContent.startsWith("reward error")) {
        statPhase.textContent = "idle";
      }
      currentRewardFn = fn;
      newMcts();
    } catch (err) {
      formulaInput.classList.add("invalid");
      formulaInput.title = err.message;
      statPhase.textContent = `reward error: ${err.message}`;
    }
  }

  formulaInput.addEventListener("change", () => applyFormula(formulaInput.value));
  presetSel.addEventListener("change", () => {
    const key = presetSel.value;
    if (!key) return;
    formulaInput.value = PRESETS[key];
    presetSel.value = "";
    applyFormula(formulaInput.value);
  });
  cSlider.addEventListener("input", () => {
    cVal.textContent = parseFloat(cSlider.value).toFixed(2);
    if (mcts) mcts.setConfig({ C: parseFloat(cSlider.value) });
  });
  hSlider.addEventListener("input", () => {
    hVal.textContent = hSlider.value;
    if (mcts) mcts.setConfig({ rolloutHorizon: parseInt(hSlider.value, 10) });
  });

  applyFormula(formulaInput.value);

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const html = document.documentElement;
    html.dataset.theme = html.dataset.theme === "light" ? "dark" : "light";
    renderAll();
  });

  if (DEBUG) {
    document.getElementById("debug-bench").addEventListener("click", () => {
      const out = document.getElementById("debug-bench-out");
      out.textContent = "running...";
      setTimeout(() => {
        const t0 = performance.now();
        for (let i = 0; i < 1000; i++) for (const _ev of mcts.iterate()) {}
        const t1 = performance.now();
        out.textContent = `1000 iterations in ${(t1 - t0).toFixed(1)}ms (${((t1 - t0) / 1000).toFixed(3)}ms/iter)`;
        renderAll();
        updateDebug();
      }, 10);
    });
  }
}
```

- [ ] **Step 2: Replace app.js with a minimal bootstrap**

Fully replace `/Users/brandon/mcts-maze-demo/src/app.js` with:

```js
import { initClassicTab } from "./classic-tab.js";

initClassicTab();
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/brandon/mcts-maze-demo && npm test`
Expected: all 47 existing tests pass.

- [ ] **Step 4: Manual smoke**

Run: `python3 -m http.server 8000` and open `http://localhost:8000`.
Verify: maze renders, controls work, Step / Run 10 / Run to end / Reset / New maze all behave as before.

Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/app.js src/classic-tab.js
git commit -m "refactor(app): extract Classic MCTS body into classic-tab.js"
```

---

## Task 3: Top-level tab switcher (HTML + CSS + app.js routing)

**Files:**
- Modify: `index.html`
- Modify: `styles.css`
- Modify: `src/app.js`

- [ ] **Step 1: Add tab switcher and two containers in index.html**

In `/Users/brandon/mcts-maze-demo/index.html`, replace the `<body>` open with:

```html
<body>
  <div id="tab-bar">
    <button class="tab-btn active" data-tab="classic">Classic MCTS</button>
    <button class="tab-btn" data-tab="az">AlphaZero MCTS</button>
  </div>

  <div id="tab-classic" class="tab-pane active">
```

Insert `</div>` immediately before the `</body>` closing tag, and right after that closing `</div>` add an empty AZ tab placeholder div:

```html
  </div>
  <div id="tab-az" class="tab-pane">
    <p style="padding: 2rem; text-align: center;">AlphaZero tab content — wired in later tasks.</p>
  </div>
</body>
```

The existing `<header>`, `<main>`, `<footer>`, and `<aside>` should all be inside `<div id="tab-classic" class="tab-pane active">` now.

- [ ] **Step 2: Add tab styles to styles.css**

Append to `/Users/brandon/mcts-maze-demo/styles.css`:

```css
#tab-bar {
  display: flex;
  gap: 0.25rem;
  padding: 0.5rem 1rem 0;
  background: var(--bg);
  border-bottom: 1px solid var(--border, #444);
}
.tab-btn {
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px solid var(--border, #444);
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  color: inherit;
  cursor: pointer;
  font-size: 0.95rem;
}
.tab-btn.active {
  background: var(--bg-elev, #2a2a2a);
  font-weight: 600;
}
.tab-pane { display: none; }
.tab-pane.active { display: block; }
```

- [ ] **Step 3: Wire tab switching in app.js**

Replace `/Users/brandon/mcts-maze-demo/src/app.js`:

```js
import { initClassicTab } from "./classic-tab.js";

initClassicTab();

let azInitialized = false;
function activate(tabName) {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  }
  for (const pane of document.querySelectorAll(".tab-pane")) {
    pane.classList.toggle("active", pane.id === `tab-${tabName}`);
  }
  if (tabName === "az" && !azInitialized) {
    azInitialized = true;
    import("./az-tab.js").then((m) => m.initAzTab());
  }
}
for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => activate(btn.dataset.tab));
}
```

The dynamic `import("./az-tab.js")` deferral keeps the page snappy when only Classic is being used. Theme-toggle handling stays inside `initClassicTab` (already registered there); the AZ tab will add its own listener in Task 10.

- [ ] **Step 4: Manual smoke**

Run: `python3 -m http.server 8000`. Open `http://localhost:8000`. Click `AlphaZero MCTS` tab — placeholder text shows. Click `Classic MCTS` — maze re-appears. Console shows a 404 for `az-tab.js` after clicking the AZ tab (we have not created it yet); ignore. Verify all Classic controls still work.

Stop the server.

- [ ] **Step 5: Commit**

```bash
git add index.html styles.css src/app.js
git commit -m "feat(az): tab switcher with lazy AZ tab init"
```

---

## Task 4: az-encode.js — maze state → tensor (TDD)

**Files:**
- Create: `tests/az-encode.test.js`
- Create: `src/az-encode.js`

- [ ] **Step 1: Write the failing test file**

Create `/Users/brandon/mcts-maze-demo/tests/az-encode.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { encodeState, CHANNELS } from "../src/az-encode.js";
import { Maze } from "../src/maze.js";

// 3x3 grid:
//  0 0 1
//  0 0 1
//  0 0 0   start [0,0], goal [2,2]
const GRID = [
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("encodeState returns Float32Array of length size*size*CHANNELS", () => {
  const v = encodeState(maze, [0, 0]);
  assert.ok(v instanceof Float32Array);
  assert.equal(v.length, 3 * 3 * CHANNELS);
});

test("encodeState channel layout: [walls, agent, goal] interleaved per cell", () => {
  // We use channel-last layout: index = (r * size + c) * CHANNELS + ch
  const v = encodeState(maze, [1, 0]);
  const at = (r, c, ch) => v[(r * 3 + c) * CHANNELS + ch];
  // walls channel: 1 where grid==1
  assert.equal(at(0, 0, 0), 0);
  assert.equal(at(0, 2, 0), 1);
  assert.equal(at(1, 2, 0), 1);
  assert.equal(at(2, 2, 0), 0);
  // agent channel: 1 only at agent_pos = [1,0]
  assert.equal(at(1, 0, 1), 1);
  assert.equal(at(0, 0, 1), 0);
  assert.equal(at(2, 2, 1), 0);
  // goal channel: 1 only at goal = [2,2]
  assert.equal(at(2, 2, 2), 1);
  assert.equal(at(0, 0, 2), 0);
});

test("encodeState: changing agent position only flips the agent channel", () => {
  const a = encodeState(maze, [0, 0]);
  const b = encodeState(maze, [2, 1]);
  // Walls and goal channels equal in both
  for (let i = 0; i < a.length; i += CHANNELS) {
    assert.equal(a[i], b[i]);       // walls
    assert.equal(a[i + 2], b[i + 2]); // goal
  }
  // Exactly two indices differ in agent channel
  let diffs = 0;
  for (let i = 1; i < a.length; i += CHANNELS) if (a[i] !== b[i]) diffs++;
  assert.equal(diffs, 2);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/az-encode.test.js 2>&1 | head -20`
Expected: FAIL with "Cannot find module 'src/az-encode.js'"

- [ ] **Step 3: Write az-encode.js**

Create `/Users/brandon/mcts-maze-demo/src/az-encode.js`:

```js
// Number of input channels: walls, agent one-hot, goal one-hot.
export const CHANNELS = 3;

// Encode a maze + agent position into a flat Float32Array suitable for
// reshaping to [size, size, CHANNELS] on the TF.js side.
// Layout: index = (r * size + c) * CHANNELS + ch
//   ch 0: 1.0 where the cell is a wall (grid[r][c] === 1), else 0.
//   ch 1: 1.0 at the agent position, else 0.
//   ch 2: 1.0 at the goal position, else 0.
export function encodeState(maze, agentPos) {
  const size = maze.size;
  const out = new Float32Array(size * size * CHANNELS);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const base = (r * size + c) * CHANNELS;
      if (maze.grid[r][c] === 1) out[base] = 1;
    }
  }
  const [ar, ac] = agentPos;
  out[(ar * size + ac) * CHANNELS + 1] = 1;
  const [gr, gc] = maze.goal;
  out[(gr * size + gc) * CHANNELS + 2] = 1;
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/az-encode.test.js`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/az-encode.js tests/az-encode.test.js
git commit -m "feat(az): state encoder (maze + agent → tensor)"
```

---

## Task 5: az-network.js — model construction + forward pass (browser-only smoke)

**Files:**
- Create: `src/az-network.js`

- [ ] **Step 1: Write az-network.js**

Create `/Users/brandon/mcts-maze-demo/src/az-network.js`. This file uses the global `tf` from the TF.js CDN (loaded in `index.html`). It is intentionally not unit-tested in Node — see plan header.

```js
// AZ network: small two-headed MLP. Lives in the browser only.
// Assumes the `tf` global is available (loaded via CDN in index.html).
import { CHANNELS, encodeState } from "./az-encode.js";

// Build a 2-hidden-layer MLP with two output heads.
// Input: flat [size*size*CHANNELS]
// Hidden: 64 → 64 (relu)
// Heads: p (4, softmax) and v (1, tanh)
export function createModel(size) {
  const inputDim = size * size * CHANNELS;
  const input = tf.input({ shape: [inputDim] });
  const h1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(input);
  const h2 = tf.layers.dense({ units: 64, activation: "relu" }).apply(h1);
  const pHead = tf.layers
    .dense({ units: 4, activation: "softmax", name: "policy" })
    .apply(h2);
  const vHead = tf.layers
    .dense({ units: 1, activation: "tanh", name: "value" })
    .apply(h2);
  return tf.model({ inputs: input, outputs: [pHead, vHead] });
}

// Single-state prediction. Returns { p: Float32Array(4), v: number }.
// Allocates and immediately disposes intermediate tensors.
export function predict(model, maze, agentPos) {
  const enc = encodeState(maze, agentPos);
  return tf.tidy(() => {
    const x = tf.tensor2d(enc, [1, enc.length]);
    const [pTensor, vTensor] = model.predict(x);
    const p = pTensor.dataSync().slice(); // copy out of WebGL
    const v = vTensor.dataSync()[0];
    return { p: Float32Array.from(p), v };
  });
}

// Batched prediction over many (maze, pos) states sharing the same maze.
// Returns { ps: Float32Array(N*4), vs: Float32Array(N) } in input order.
export function predictBatch(model, maze, positions) {
  const size = maze.size;
  const inputDim = size * size * CHANNELS;
  const N = positions.length;
  const flat = new Float32Array(N * inputDim);
  for (let i = 0; i < N; i++) {
    const row = encodeState(maze, positions[i]);
    flat.set(row, i * inputDim);
  }
  return tf.tidy(() => {
    const x = tf.tensor2d(flat, [N, inputDim]);
    const [pTensor, vTensor] = model.predict(x);
    const ps = Float32Array.from(pTensor.dataSync());
    const vs = Float32Array.from(vTensor.dataSync());
    return { ps, vs };
  });
}

// One gradient step on a single minibatch.
// batch: { states: Float32Array(B*inputDim), pis: Float32Array(B*4), zs: Float32Array(B) }
// Returns { total, policy, value } as plain numbers.
export function trainStep(model, optimizer, batch, inputDim) {
  const B = batch.zs.length;
  const x = tf.tensor2d(batch.states, [B, inputDim]);
  const piTarget = tf.tensor2d(batch.pis, [B, 4]);
  const zTarget = tf.tensor2d(batch.zs, [B, 1]);
  let losses;
  optimizer.minimize(() => {
    const [pPred, vPred] = model.apply(x);
    // value loss: MSE
    const vLoss = tf.losses.meanSquaredError(zTarget, vPred);
    // policy loss: -sum(pi * log(p)) averaged over batch
    const eps = tf.scalar(1e-8);
    const pLoss = tf
      .neg(tf.sum(tf.mul(piTarget, tf.log(tf.add(pPred, eps)))))
      .div(tf.scalar(B));
    const total = tf.add(vLoss, pLoss);
    losses = {
      total: total.dataSync()[0],
      policy: pLoss.dataSync()[0],
      value: vLoss.dataSync()[0],
    };
    eps.dispose();
    return total;
  }, /* returnCost */ false);
  x.dispose();
  piTarget.dispose();
  zTarget.dispose();
  return losses;
}
```

- [ ] **Step 2: Manual smoke that model builds and predicts**

This is a temporary smoke; remove after verification.

In the browser console (with the page open and `tab-az` active enough to lazy-load — but the AZ tab still has placeholder), open the console and run:

```js
const { createModel, predict } = await import('./src/az-network.js');
const { MAZES } = await import('./src/maze.js');
const m = createModel(10);
m.summary();
const out = predict(m, MAZES[10], [0,0]);
console.log('p:', out.p, 'v:', out.v);
```

Expected: `m.summary()` prints layer table; `out.p` is a Float32Array length 4 summing to ~1; `out.v` is a number in `[-1, 1]`.

- [ ] **Step 3: Commit**

```bash
git add src/az-network.js
git commit -m "feat(az): TF.js network — createModel, predict, trainStep"
```

---

## Task 6: az-mcts.js — AZNode + PUCT formula (TDD)

**Files:**
- Create: `tests/az-mcts.test.js`
- Create: `src/az-mcts.js`

- [ ] **Step 1: Write the failing test file**

Create `/Users/brandon/mcts-maze-demo/tests/az-mcts.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { AZNode, puctScore, AZMCTS } from "../src/az-mcts.js";
import { Maze } from "../src/maze.js";

test("AZNode initializes with prior, zero visits, zero W", () => {
  const n = new AZNode([0, 0], null, null, 0.25);
  assert.deepEqual(n.position, [0, 0]);
  assert.equal(n.parent, null);
  assert.equal(n.action, null);
  assert.equal(n.prior, 0.25);
  assert.equal(n.visits, 0);
  assert.equal(n.totalReward, 0);
  assert.equal(n.children.size, 0);
});

test("AZNode.meanReward returns 0 when unvisited", () => {
  const n = new AZNode([0, 0], null, null, 1.0);
  assert.equal(n.meanReward(), 0);
});

test("AZNode.meanReward returns W/N when visited", () => {
  const n = new AZNode([0, 0], null, null, 1.0);
  n.visits = 4;
  n.totalReward = 2.0;
  assert.equal(n.meanReward(), 0.5);
});

test("puctScore: unvisited child uses Q=0 and prior-scaled exploration", () => {
  const n = new AZNode([0, 0], null, null, 0.5);
  // c_puct=1, parentVisits=4, prior=0.5, N_child=0 → 0 + 1*0.5*sqrt(4)/(1+0) = 1
  assert.equal(puctScore(n, 4, 1), 1);
});

test("puctScore: visited child mixes Q with prior bonus", () => {
  const n = new AZNode([0, 0], null, null, 0.5);
  n.visits = 1;
  n.totalReward = 0.4;
  // Q=0.4, c_puct=1, parentVisits=4, prior=0.5, N_child=1
  //   → 0.4 + 1*0.5*sqrt(4)/(1+1) = 0.4 + 0.5 = 0.9
  assert.equal(Math.abs(puctScore(n, 4, 1) - 0.9) < 1e-9, true);
});

// AZMCTS integration with a stub network
const STUB = {
  // Always returns uniform prior and v=0.
  predict: () => ({ p: new Float32Array([0.25, 0.25, 0.25, 0.25]), v: 0 }),
};

const GRID = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("AZMCTS.iterate populates root children with priors on first call", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  assert.equal(m.root.children.size, 0);
  m.iterate();
  // From [0,0] legal actions are [right, down] (up/left out of bounds).
  // Uniform prior over those two = 0.5 each after re-normalization.
  assert.equal(m.root.children.size, 2);
  for (const child of m.root.children.values()) {
    assert.ok(Math.abs(child.prior - 0.5) < 1e-9);
  }
});

test("AZMCTS.iterate increments root visits by 1", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  m.iterate();
  assert.equal(m.root.visits, 1);
  m.iterate();
  assert.equal(m.root.visits, 2);
});

test("AZMCTS.rootPolicy returns visit-count distribution", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  for (let i = 0; i < 10; i++) m.iterate();
  const pi = m.rootPolicy();
  // pi has length 4 (one per ACTION); illegal actions get 0; sum = 1.
  assert.equal(pi.length, 4);
  const sum = pi.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `sum was ${sum}`);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/az-mcts.test.js 2>&1 | head -20`
Expected: FAIL with "Cannot find module 'src/az-mcts.js'"

- [ ] **Step 3: Write src/az-mcts.js**

Create `/Users/brandon/mcts-maze-demo/src/az-mcts.js`:

```js
import { ACTIONS } from "./maze.js";

const actionKey = ([dr, dc]) => `${dr},${dc}`;

export class AZNode {
  constructor(position, parent, action, prior) {
    this.position = position;
    this.parent = parent;
    this.action = action;          // [dr, dc] taken from parent
    this.prior = prior;            // p(a|s) from the network at parent state
    this.children = new Map();     // "dr,dc" → AZNode
    this.visits = 0;
    this.totalReward = 0;
  }
  meanReward() {
    return this.visits === 0 ? 0 : this.totalReward / this.visits;
  }
}

// PUCT score with the conventional formulation:
//   Q + c_puct * prior * sqrt(parentVisits) / (1 + N_child)
// For unvisited children, Q is taken to be 0 (per spec §5.1).
export function puctScore(child, parentVisits, cPuct) {
  const Q = child.visits === 0 ? 0 : child.totalReward / child.visits;
  const U = cPuct * child.prior * Math.sqrt(parentVisits) / (1 + child.visits);
  return Q + U;
}

// Collect positions on the path from `node` up to root (inclusive).
function ancestorPositionKeys(node) {
  const out = new Set();
  let n = node;
  while (n !== null) {
    out.add(n.position.join(","));
    n = n.parent;
  }
  return out;
}

export class AZMCTS {
  constructor(maze, network, config = {}) {
    this.maze = maze;
    this.network = network;        // must expose predict(maze, pos) → {p, v}
    this.cPuct = config.cPuct ?? 1.0;
    // The MCTS root represents the agent's *current* position. Default to
    // maze.start; callers running self-play pass a startPos for the current
    // game move. Root has no prior (it isn't selected via PUCT); 1 is a
    // placeholder.
    const startPos = config.startPos ?? maze.start;
    this.root = new AZNode([...startPos], null, null, 1);
    this.iterationCount = 0;
  }

  // One PUCT-MCTS simulation.
  iterate() {
    // 1. Selection: walk down by PUCT until we reach a leaf
    //    (a node with no children) or a terminal node.
    const path = [this.root];
    let node = this.root;
    while (!this.maze.isGoal(node.position) && node.children.size > 0) {
      let best = null;
      let bestScore = -Infinity;
      for (const child of node.children.values()) {
        const s = puctScore(child, node.visits || 1, this.cPuct);
        if (s > bestScore) {
          bestScore = s;
          best = child;
        }
      }
      node = best;
      path.push(node);
    }

    // 2. Expand + evaluate: if non-terminal, query the network at the leaf,
    //    create children for every cycle-free legal action, with priors.
    let leafValue;
    if (this.maze.isGoal(node.position)) {
      leafValue = 1; // terminal: definite success
    } else {
      const { p, v } = this.network.predict(this.maze, node.position);
      leafValue = v;
      const ancestors = ancestorPositionKeys(node);
      // Mask illegal + cycle-creating actions, then renormalize.
      const legal = [];
      let priorSum = 0;
      for (let i = 0; i < ACTIONS.length; i++) {
        const a = ACTIONS[i];
        const next = this.maze.step(node.position, a);
        if (!this.maze.isOpen(next)) continue;
        if (ancestors.has(next.join(","))) continue;
        legal.push({ a, idx: i, prior: p[i] });
        priorSum += p[i];
      }
      // If all priors zero (numerical) fall back to uniform.
      if (priorSum <= 1e-12) {
        for (const item of legal) item.prior = 1 / Math.max(1, legal.length);
      } else {
        for (const item of legal) item.prior = item.prior / priorSum;
      }
      for (const { a, prior } of legal) {
        const child = new AZNode(this.maze.step(node.position, a), node, a, prior);
        node.children.set(actionKey(a), child);
      }
    }

    // 3. Backprop: add leafValue to every node on the path, increment visits.
    for (const n of path) {
      n.visits += 1;
      n.totalReward += leafValue;
    }
    this.iterationCount += 1;
  }

  // Visit-count distribution over the 4 ACTIONS at the root.
  // Illegal/missing actions get 0; result sums to 1 (uniform fallback if empty).
  rootPolicy() {
    const pi = new Float32Array(ACTIONS.length);
    let total = 0;
    for (let i = 0; i < ACTIONS.length; i++) {
      const key = actionKey(ACTIONS[i]);
      const child = this.root.children.get(key);
      if (child) {
        pi[i] = child.visits;
        total += child.visits;
      }
    }
    if (total === 0) {
      // No children expanded (root is terminal). Return uniform.
      pi.fill(1 / ACTIONS.length);
    } else {
      for (let i = 0; i < pi.length; i++) pi[i] /= total;
    }
    return pi;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/az-mcts.test.js`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/az-mcts.js tests/az-mcts.test.js
git commit -m "feat(az): PUCT MCTS variant (AZNode, AZMCTS)"
```

---

## Task 7: az-selfplay.js — replay buffer + playOneGame (TDD)

**Files:**
- Create: `tests/az-selfplay.test.js`
- Create: `src/az-selfplay.js`

- [ ] **Step 1: Write the failing test file**

Create `/Users/brandon/mcts-maze-demo/tests/az-selfplay.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { ReplayBuffer, playOneGame } from "../src/az-selfplay.js";
import { Maze } from "../src/maze.js";
import { encodeState } from "../src/az-encode.js";

test("ReplayBuffer.append + size", () => {
  const buf = new ReplayBuffer(100);
  assert.equal(buf.size(), 0);
  buf.append({ state: new Float32Array(3), pi: new Float32Array(4), z: 1 });
  assert.equal(buf.size(), 1);
});

test("ReplayBuffer enforces FIFO capacity", () => {
  const buf = new ReplayBuffer(3);
  for (let i = 0; i < 5; i++) {
    buf.append({ state: new Float32Array([i]), pi: new Float32Array(4), z: i });
  }
  assert.equal(buf.size(), 3);
  // Oldest two (i=0, i=1) should have been dropped.
  const items = buf.items();
  assert.deepEqual(items.map((t) => t.z), [2, 3, 4]);
});

test("ReplayBuffer.sampleBatch returns minibatch arrays of correct shape", () => {
  const buf = new ReplayBuffer(100);
  const inputDim = 3;
  for (let i = 0; i < 20; i++) {
    buf.append({
      state: new Float32Array([i, i, i]),
      pi: new Float32Array([0.25, 0.25, 0.25, 0.25]),
      z: i / 20,
    });
  }
  const batch = buf.sampleBatch(8, inputDim);
  assert.equal(batch.states.length, 8 * inputDim);
  assert.equal(batch.pis.length, 8 * 4);
  assert.equal(batch.zs.length, 8);
});

// playOneGame integration with a stub network.
// Stub gives uniform prior + value 0; the agent therefore behaves close to
// uniform random under MCTS, which is enough to test the loop terminates and
// produces well-shaped trajectories.
const GRID = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);
const STUB = { predict: () => ({ p: new Float32Array([0.25, 0.25, 0.25, 0.25]), v: 0 }) };

// z formula: 1 if reached goal, 0 otherwise.
const Z_FN = (ctx) => (ctx.isGoal ? 1 : 0);

test("playOneGame terminates within step cap and returns trajectory + z", () => {
  const result = playOneGame({
    maze, network: STUB, zFn: Z_FN, simsPerMove: 20, cPuct: 1.0,
    maxSteps: 8, temperatureMoves: 0, rng: Math.random,
  });
  assert.ok(Array.isArray(result.trajectory));
  assert.ok(result.trajectory.length > 0);
  assert.ok(result.trajectory.length <= 8);
  assert.equal(typeof result.z, "number");
  // Each tuple shape:
  for (const t of result.trajectory) {
    assert.ok(t.state instanceof Float32Array);
    assert.equal(t.pi.length, 4);
  }
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/az-selfplay.test.js 2>&1 | head -10`
Expected: FAIL with "Cannot find module 'src/az-selfplay.js'"

- [ ] **Step 3: Write src/az-selfplay.js**

Create `/Users/brandon/mcts-maze-demo/src/az-selfplay.js`:

```js
import { AZMCTS } from "./az-mcts.js";
import { ACTIONS } from "./maze.js";
import { encodeState, CHANNELS } from "./az-encode.js";

// Replay buffer with FIFO eviction.
export class ReplayBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this._buf = [];
  }
  append(tuple) {
    this._buf.push(tuple);
    while (this._buf.length > this.capacity) this._buf.shift();
  }
  size() { return this._buf.length; }
  items() { return this._buf.slice(); }

  // Sample a minibatch with replacement. Flattens to typed-array form for TF.js.
  sampleBatch(B, inputDim) {
    const states = new Float32Array(B * inputDim);
    const pis = new Float32Array(B * 4);
    const zs = new Float32Array(B);
    for (let i = 0; i < B; i++) {
      const j = Math.floor(Math.random() * this._buf.length);
      const t = this._buf[j];
      states.set(t.state, i * inputDim);
      pis.set(t.pi, i * 4);
      zs[i] = t.z;
    }
    return { states, pis, zs };
  }
}

// Sample an action index in [0..3] from a distribution `pi`.
// temperature=0 → argmax. temperature=1 → sample by raw probabilities.
function sampleAction(pi, temperature, rng) {
  if (temperature === 0) {
    let best = 0;
    for (let i = 1; i < pi.length; i++) if (pi[i] > pi[best]) best = i;
    return best;
  }
  // For other temperatures we exponentiate: pi^(1/T).
  const adj = new Float64Array(pi.length);
  let sum = 0;
  for (let i = 0; i < pi.length; i++) {
    adj[i] = Math.pow(pi[i], 1 / temperature);
    sum += adj[i];
  }
  if (sum <= 0) return 0;
  const r = rng() * sum;
  let acc = 0;
  for (let i = 0; i < adj.length; i++) {
    acc += adj[i];
    if (r <= acc) return i;
  }
  return adj.length - 1;
}

// Run one full self-play game from maze.start to either the goal or maxSteps.
// Returns { trajectory: [{state, pi}], z: number, steps: number, reachedGoal: boolean }.
//
// opts:
//   maze, network, zFn, simsPerMove, cPuct, maxSteps,
//   temperatureMoves (number of leading moves to sample at T=1; rest at T=0),
//   rng (optional, defaults to Math.random)
export function playOneGame(opts) {
  const {
    maze, network, zFn, simsPerMove, cPuct,
    maxSteps, temperatureMoves, rng = Math.random,
  } = opts;
  let pos = [...maze.start];
  const trajectory = [];
  let t = 0;
  let reachedGoal = false;
  for (; t < maxSteps; t++) {
    if (maze.isGoal(pos)) { reachedGoal = true; break; }
    const mcts = new AZMCTS(maze, network, { cPuct, startPos: pos });
    for (let s = 0; s < simsPerMove; s++) mcts.iterate();
    const pi = mcts.rootPolicy();
    trajectory.push({ state: encodeState(maze, pos), pi });
    const aIdx = sampleAction(pi, t < temperatureMoves ? 1 : 0, rng);
    pos = maze.step(pos, ACTIONS[aIdx]);
  }
  if (maze.isGoal(pos)) reachedGoal = true;
  const ctx = {
    isGoal: reachedGoal,
    dist: maze.manhattan(pos, maze.goal),
    maxDist: maze.maxDistance(),
    steps: trajectory.length,
    startDist: maze.manhattan(maze.start, maze.goal),
  };
  const z = zFn(ctx);
  return { trajectory, z, steps: trajectory.length, reachedGoal };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/az-selfplay.test.js`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/az-selfplay.js tests/az-selfplay.test.js
git commit -m "feat(az): replay buffer + self-play game loop"
```

---

## Task 8: az-charts.js — line chart + probe panel (manual smoke)

**Files:**
- Create: `src/az-charts.js`

- [ ] **Step 1: Write az-charts.js**

Create `/Users/brandon/mcts-maze-demo/src/az-charts.js`:

```js
// Minimal time-series line chart on a 2D canvas.
// Multiple "series" can be added by name; addPoint(name, y) appends.
// X-axis auto-scales by point index across all series.
export class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = new Map(); // name → {data: [], color: string}
    this.yMin = opts.yMin ?? null; // null = auto
    this.yMax = opts.yMax ?? null;
    this.title = opts.title ?? "";
  }
  addSeries(name, color) {
    if (!this.series.has(name)) this.series.set(name, { data: [], color });
  }
  addPoint(name, y) {
    const s = this.series.get(name);
    if (!s) return;
    s.data.push(y);
  }
  clear() { for (const s of this.series.values()) s.data.length = 0; }
  render() {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    const PAD = 24;
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue("--bg-elev") || "#222";
    ctx.fillRect(0, 0, W, H);

    // Determine axes.
    let maxLen = 0, yMin = +Infinity, yMax = -Infinity;
    for (const s of this.series.values()) {
      if (s.data.length > maxLen) maxLen = s.data.length;
      for (const y of s.data) {
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
    if (!isFinite(yMin) || !isFinite(yMax)) { yMin = -1; yMax = 1; }
    if (this.yMin !== null) yMin = this.yMin;
    if (this.yMax !== null) yMax = this.yMax;
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }

    // Title.
    ctx.fillStyle = "#bbb";
    ctx.font = "11px monospace";
    ctx.fillText(this.title, PAD, 14);

    // Axes box.
    ctx.strokeStyle = "#555";
    ctx.strokeRect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
    // y=0 reference line.
    if (yMin < 0 && yMax > 0) {
      const y0 = PAD + (H - 2 * PAD) * (1 - (0 - yMin) / (yMax - yMin));
      ctx.strokeStyle = "#666";
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(PAD, y0);
      ctx.lineTo(W - PAD, y0);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // y range labels.
    ctx.fillStyle = "#888";
    ctx.fillText(yMax.toFixed(2), 2, PAD + 4);
    ctx.fillText(yMin.toFixed(2), 2, H - PAD + 4);

    // Series lines.
    const xFor = (i) => PAD + (W - 2 * PAD) * (maxLen <= 1 ? 0 : i / (maxLen - 1));
    const yFor = (v) => PAD + (H - 2 * PAD) * (1 - (v - yMin) / (yMax - yMin));
    for (const s of this.series.values()) {
      if (s.data.length === 0) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xFor(0), yFor(s.data[0]));
      for (let i = 1; i < s.data.length; i++) ctx.lineTo(xFor(i), yFor(s.data[i]));
      ctx.stroke();
    }
  }
}

// Tiny "probe" panel: 4 bars for p plus scalar v.
export class ProbePanel {
  constructor(container) {
    this.container = container;
    this.container.innerHTML = `
      <div class="probe-title">Probe @ start</div>
      <div class="probe-bars">
        ${["up","down","left","right"].map((lbl, i) => `
          <div class="probe-row">
            <span class="probe-label">${lbl}</span>
            <div class="probe-bar"><div class="probe-fill" data-i="${i}"></div></div>
            <span class="probe-pct" data-i="${i}">0.00</span>
          </div>
        `).join("")}
      </div>
      <div class="probe-v">v = <span id="probe-v-num">0.00</span></div>
    `;
  }
  update({ p, v }) {
    for (let i = 0; i < 4; i++) {
      const fill = this.container.querySelector(`.probe-fill[data-i="${i}"]`);
      const pct = this.container.querySelector(`.probe-pct[data-i="${i}"]`);
      fill.style.width = `${(p[i] * 100).toFixed(1)}%`;
      pct.textContent = p[i].toFixed(2);
    }
    this.container.querySelector("#probe-v-num").textContent = v.toFixed(3);
  }
}
```

- [ ] **Step 2: No test yet — used in Task 11 wiring**

Skip running tests. This file's smoke happens once it is wired into the AZ tab.

- [ ] **Step 3: Commit**

```bash
git add src/az-charts.js
git commit -m "feat(az): canvas line chart + probe panel"
```

---

## Task 9: AZ tab markup + styles

**Files:**
- Modify: `index.html`
- Modify: `styles.css`

- [ ] **Step 1: Replace AZ tab placeholder with full markup**

In `/Users/brandon/mcts-maze-demo/index.html`, replace the existing AZ-tab placeholder block:

```html
  <div id="tab-az" class="tab-pane">
    <p style="padding: 2rem; text-align: center;">AlphaZero tab content — wired in later tasks.</p>
  </div>
```

with:

```html
  <div id="tab-az" class="tab-pane">
    <header class="az-header">
      <div id="az-status-bar">
        <span>Games: <span id="az-games">0</span></span>
        <span>Latest z: <span id="az-latest-z">—</span></span>
        <span>z̄(20): <span id="az-avg-z">—</span></span>
        <span>Loss: <span id="az-loss">—</span></span>
        <span>Buffer: <span id="az-buf">0</span></span>
      </div>
    </header>

    <div id="az-overlay-toggle">
      Overlay:
      <label><input type="radio" name="az-overlay" value="v" checked /> v(s) heatmap</label>
      <label><input type="radio" name="az-overlay" value="p" /> p(a|s) arrows</label>
      <label><input type="radio" name="az-overlay" value="visits" /> MCTS visits</label>
      <label><input type="radio" name="az-overlay" value="off" /> off</label>
    </div>

    <main class="az-main">
      <section class="az-maze-pane">
        <canvas id="az-maze-canvas" width="600" height="600"></canvas>
      </section>
      <section class="az-tree-pane">
        <svg id="az-tree-svg" width="600" height="600"></svg>
      </section>
    </main>

    <section class="az-charts">
      <canvas id="az-chart-z" width="380" height="180"></canvas>
      <canvas id="az-chart-loss" width="380" height="180"></canvas>
      <div id="az-probe" class="az-probe"></div>
    </section>

    <footer class="az-controls">
      <div class="control-row">
        <label>Maze size:
          <select id="az-ctl-size">
            <option value="10" selected>10×10</option>
            <option value="15">15×15</option>
            <option value="20">20×20</option>
          </select>
        </label>
        <label class="formula-group">z formula:
          <input type="text" id="az-ctl-z-formula" spellcheck="false" autocomplete="off" />
          <select id="az-ctl-z-preset" title="Load a preset">
            <option value="" disabled selected>Presets ▾</option>
            <option value="Pure win/loss">Pure win/loss</option>
            <option value="Distance-shaped">Distance-shaped</option>
            <option value="Step penalty">Step penalty</option>
          </select>
        </label>
      </div>
      <div class="control-row">
        <label>c_puct: <input type="range" id="az-ctl-cpuct" min="0.1" max="4" step="0.05" value="1.0" />
          <span id="az-ctl-cpuct-val">1.00</span>
        </label>
        <label>MCTS sims/move: <input type="range" id="az-ctl-sims" min="10" max="400" step="10" value="50" />
          <span id="az-ctl-sims-val">50</span>
        </label>
        <label>Learning rate (log10): <input type="range" id="az-ctl-lr" min="-5" max="-2" step="0.1" value="-3" />
          <span id="az-ctl-lr-val">1e-3</span>
        </label>
      </div>
      <div class="control-row">
        <button id="az-btn-step">Self-play step</button>
        <button id="az-btn-run10">Run 10</button>
        <button id="az-btn-runcont">Run continuously</button>
        <button id="az-btn-reset-mcts">Reset MCTS</button>
        <button id="az-btn-reset-net">Reset network</button>
      </div>
    </footer>
  </div>
```

- [ ] **Step 2: Append AZ tab styles to styles.css**

Append to `/Users/brandon/mcts-maze-demo/styles.css`:

```css
.az-header {
  padding: 0.5rem 1rem;
}
#az-status-bar {
  display: flex;
  gap: 1.25rem;
  font-family: monospace;
  font-size: 0.9rem;
}
#az-overlay-toggle {
  padding: 0 1rem 0.5rem;
  font-size: 0.85rem;
  display: flex;
  gap: 0.75rem;
  align-items: center;
}
#az-overlay-toggle label {
  display: inline-flex;
  gap: 0.25rem;
  align-items: center;
}
.az-main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  padding: 0 1rem;
}
.az-maze-pane, .az-tree-pane {
  background: var(--bg-elev, #222);
  border: 1px solid var(--border, #444);
  border-radius: 6px;
}
.az-charts {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
}
.az-charts canvas,
.az-probe {
  background: var(--bg-elev, #222);
  border: 1px solid var(--border, #444);
  border-radius: 6px;
  padding: 0.5rem;
}
.az-probe { font-family: monospace; font-size: 0.85rem; }
.probe-title { color: #888; margin-bottom: 0.5rem; }
.probe-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem; }
.probe-label { width: 3.5em; }
.probe-bar { flex: 1; background: #333; height: 0.75em; border-radius: 3px; overflow: hidden; }
.probe-fill { height: 100%; background: #6ab0ff; width: 0; transition: width 120ms; }
.probe-pct { width: 2.75em; text-align: right; color: #ccc; }
.probe-v { margin-top: 0.5rem; color: #ccc; }
.az-controls {
  padding: 0.5rem 1rem 1rem;
}
.az-controls .control-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  align-items: center;
}
```

- [ ] **Step 3: Manual smoke**

Run: `python3 -m http.server 8000`. Open `http://localhost:8000`. Click `AlphaZero MCTS`. Page renders the maze canvas (blank), tree SVG (blank), three chart canvases (blank), all controls visible. Console shows the lazy-import 404 for `az-tab.js` — still expected. Classic tab still works.

Stop the server.

- [ ] **Step 4: Commit**

```bash
git add index.html styles.css
git commit -m "feat(az): AZ tab markup + styles"
```

---

## Task 10: az-tab.js — wiring without visuals (manual smoke)

**Files:**
- Create: `src/az-tab.js`

This task connects controls and runs one self-play step on click. v-heatmap, p-arrows, tree, and charts are added in later tasks.

- [ ] **Step 1: Write az-tab.js**

Create `/Users/brandon/mcts-maze-demo/src/az-tab.js`:

```js
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
```

- [ ] **Step 2: Manual smoke**

Run: `python3 -m http.server 8000`. Open `http://localhost:8000`. Click AZ tab. Click `Self-play step`. The status bar updates: Games = 1, Latest z shows a number (typically -1 for an early-untrained run on a 10×10), Buffer = trajectory length. Click again — Games = 2, Buffer grows. After Buffer ≥ 32, Loss starts showing a value. No errors in the console.

Click `Run 10` — Games advances by 10 over several seconds.
Click `Run continuously` — button changes to `Stop`; Games keeps advancing. Click `Stop` — button returns.
Click `Reset network` — confirm. Status bar zeros out.

Stop the server.

- [ ] **Step 3: Commit**

```bash
git add src/az-tab.js
git commit -m "feat(az): wire AZ tab controls and self-play step (no viz)"
```

---

## Task 11: Wire charts + probe into az-tab.js (manual smoke)

**Files:**
- Modify: `src/az-tab.js`

- [ ] **Step 1: Add chart imports and instances**

In `/Users/brandon/mcts-maze-demo/src/az-tab.js`, add to the imports at the top:

```js
import { LineChart, ProbePanel } from "./az-charts.js";
import { ACTIONS } from "./maze.js";
```

Then inside `freshState`, add chart instances. Since charts depend on DOM that exists once, build them once at init and store on `state` rather than re-creating per reset; rework `initAzTab` to construct charts there and pass them through. Concretely, replace the lines from `state = freshState(...)` through `mazeRenderer.render(currentMaze(), {})` with:

```js
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
  state.zFn = compileReward(zInput.value);
  mazeRenderer.render(currentMaze(), {});
  renderCharts();
```

Then in the `Reset network` handler, after `state = freshState(...)` re-attach the existing charts and clear them:

```js
    const lr2 = Math.pow(10, parseFloat(lrSlider.value));
    state = freshState(state.size, lr2);
    state.zChart = zChart;
    state.lossChart = lossChart;
    state.probe = probe;
    zChart.clear();
    lossChart.clear();
    state.zFn = compileReward(zInput.value);
```

And in the `size change` handler do the same — `state.zChart = zChart; state.lossChart = lossChart; state.probe = probe; zChart.clear(); lossChart.clear();`

- [ ] **Step 2: Add renderCharts and probe update**

At the bottom of the file, add:

```js
function renderCharts() {
  state.zChart.render();
  state.lossChart.render();
}

function updateProbe() {
  const maze = currentMaze();
  const { p, v } = predict(state.model, maze, maze.start);
  state.probe.update({ p, v });
}
```

- [ ] **Step 3: Hook chart updates into runSelfPlayStep**

At the end of `runSelfPlayStep`, after `updateStatusBar();`, add:

```js
  state.zChart.addPoint("raw", result.z);
  const w = state.zHistory.slice(-20);
  state.zChart.addPoint("ma", w.reduce((a, b) => a + b, 0) / w.length);
  if (state.lastLosses) {
    state.lossChart.addPoint("total", state.lastLosses.total);
    state.lossChart.addPoint("policy", state.lastLosses.policy);
    state.lossChart.addPoint("value", state.lastLosses.value);
  }
  renderCharts();
  updateProbe();
```

- [ ] **Step 4: Manual smoke**

Run: `python3 -m http.server 8000`. Open `http://localhost:8000`. AZ tab. Click `Self-play step` a few times. The z chart shows dots/lines updating; once buffer reaches 32, loss chart starts populating. Probe panel shows 4 bars and v reading at the start cell, updating each step.

Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/az-tab.js
git commit -m "feat(az): wire z curve, loss curves, and probe panel"
```

---

## Task 12: render-maze.js — v-heatmap overlay (manual smoke)

**Files:**
- Modify: `src/render-maze.js`

- [ ] **Step 1: Insert v-heatmap branch into MazeRenderer.render**

In `/Users/brandon/mcts-maze-demo/src/render-maze.js`, the `render(maze, state = {})` method draws cells (the `for r / for c` grid loop) ending at the closing brace right before the `// Selection path outlines` comment. Insert this block **immediately after** that closing brace and **before** the `// Selection path outlines` comment:

```js
    // v(s) heatmap (AZ tab). state.vHeatmap is Float32Array(size*size) row-major.
    if (state.vHeatmap) {
      for (let r = 0; r < maze.size; r++) {
        for (let c = 0; c < maze.size; c++) {
          if (maze.grid[r][c] === 1) continue;
          const v = state.vHeatmap[r * maze.size + c];
          const tt = Math.max(-1, Math.min(1, v));
          const red = tt < 0 ? 255 : Math.round(255 * (1 - tt));
          const blue = tt > 0 ? 255 : Math.round(255 * (1 + tt));
          const green = Math.round(120 * (1 - Math.abs(tt)));
          ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, 0.45)`;
          ctx.fillRect(offX + c * cell, offY + r * cell, cell, cell);
        }
      }
    }
```

The `cell`, `offX`, `offY`, and `ctx` identifiers already exist in scope from the top of the method.

- [ ] **Step 2: Wire heatmap toggle into az-tab.js**

At the top of `src/az-tab.js`, add to imports:

```js
import { predictBatch } from "./az-network.js";
```

Add a helper at the bottom of the file:

```js
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
```

Replace the `mazeRenderer.render(currentMaze(), {})` calls with overlay-aware rendering. Add at the bottom of the file:

```js
function renderMaze(mazeRenderer) {
  const maze = currentMaze();
  const mode = document.querySelector('input[name="az-overlay"]:checked').value;
  const opts = {};
  if (mode === "v") opts.vHeatmap = computeVHeatmap();
  mazeRenderer.render(maze, opts);
}
```

Replace existing `mazeRenderer.render(currentMaze(), {})` calls in `initAzTab`, the reset handlers, the size handler, and the end of `runSelfPlayStep` with `renderMaze(mazeRenderer)`.

Also wire the radio toggle and a theme-change re-render (inside `initAzTab`):

```js
  for (const r of document.querySelectorAll('input[name="az-overlay"]')) {
    r.addEventListener("change", () => renderMaze(mazeRenderer));
  }
  // classic-tab owns the theme toggle and flips data-theme; we just re-render.
  document.getElementById("theme-toggle").addEventListener("click", () => {
    renderMaze(mazeRenderer);
  });
```

- [ ] **Step 3: Manual smoke**

Run: `python3 -m http.server 8000`. AZ tab. With `v(s) heatmap` selected (default), the maze cells show a faint color overlay. Click `Self-play step` a few times — the heatmap colors shift. After running 100+ games (`Run continuously` for a minute), with `Distance-shaped` z and a 10×10 maze, the cells closer to the goal should trend blue and far cells red.

Stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/render-maze.js src/az-tab.js
git commit -m "feat(az): v(s) heatmap overlay"
```

---

## Task 13: render-maze.js — p-arrows overlay (manual smoke)

**Files:**
- Modify: `src/render-maze.js`
- Modify: `src/az-tab.js`

- [ ] **Step 1: Insert p-arrows branch into MazeRenderer.render**

In `/Users/brandon/mcts-maze-demo/src/render-maze.js`, find the v-heatmap block added in Task 12. **Immediately after** its closing brace and **before** the `// Selection path outlines` comment, insert:

```js
    // p(a|s) arrows (AZ tab). state.pArrows is Float32Array(size*size*4) in
    // row-major × ACTIONS order (up, down, left, right).
    if (state.pArrows) {
      ctx.strokeStyle = "#6ab0ff";
      ctx.lineWidth = 1.5;
      const ACTS = [[-1,0],[1,0],[0,-1],[0,1]];
      for (let r = 0; r < maze.size; r++) {
        for (let c = 0; c < maze.size; c++) {
          if (maze.grid[r][c] === 1) continue;
          const ccx = offX + c * cell + cell / 2;
          const ccy = offY + r * cell + cell / 2;
          for (let i = 0; i < 4; i++) {
            const p = state.pArrows[(r * maze.size + c) * 4 + i];
            const [dr, dc] = ACTS[i];
            const len = (cell * 0.4) * p;
            ctx.beginPath();
            ctx.moveTo(ccx, ccy);
            ctx.lineTo(ccx + dc * len, ccy + dr * len);
            ctx.stroke();
          }
        }
      }
    }
```

The `cell`, `offX`, `offY`, and `ctx` identifiers already exist in scope.

- [ ] **Step 2: Add p-arrows compute in az-tab.js**

Add a helper to `src/az-tab.js`:

```js
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
```

Extend `renderMaze`:

```js
function renderMaze(mazeRenderer) {
  const maze = currentMaze();
  const mode = document.querySelector('input[name="az-overlay"]:checked').value;
  const opts = {};
  if (mode === "v") opts.vHeatmap = computeVHeatmap();
  else if (mode === "p") opts.pArrows = computePArrows();
  mazeRenderer.render(maze, opts);
}
```

- [ ] **Step 3: Manual smoke**

Run: `python3 -m http.server 8000`. AZ tab. Select `p(a|s) arrows`. Maze cells show small 4-arrow widgets per cell. Run a few self-play steps — arrows update. With `Distance-shaped` z over 100+ games, arrows near the start should bias toward the down/right diagonal.

Stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/render-maze.js src/az-tab.js
git commit -m "feat(az): p(a|s) arrow overlay"
```

---

## Task 14: Reset MCTS button + size-change behavior fully wired

**Files:**
- Modify: `src/az-tab.js`

- [ ] **Step 1: Make Reset MCTS clear charts (network preserved)**

The AZ MCTS tree is constructed fresh per move inside `playOneGame`, so there is no persistent in-progress tree to clear; `Reset MCTS` in the AZ tab is therefore only useful as "clear the displayed search-tree state". Repurpose it to clear the tree pane drawing buffer. In `src/az-tab.js`, replace the placeholder `resetMctsBtn` handler:

```js
  resetMctsBtn.addEventListener("click", () => {
    document.getElementById("az-tree-svg").innerHTML = "";
  });
```

- [ ] **Step 2: Manual smoke**

Run server, switch to AZ tab. Click `Reset MCTS` — tree SVG remains blank (it never had content yet). Click `Reset network` — confirm — z chart and loss chart clear; status bar zeros out; v-heatmap (if active) updates to reflect re-initialized weights.

- [ ] **Step 3: Commit**

```bash
git add src/az-tab.js
git commit -m "feat(az): reset MCTS button"
```

---

## Task 15: Render search tree for the *current move* into the AZ tree pane (manual smoke)

This task makes the existing tree renderer show the AZ-MCTS tree of the *last* MCTS search performed during self-play, so students can see what PUCT explored.

**Files:**
- Modify: `src/az-selfplay.js`
- Modify: `src/az-tab.js`

- [ ] **Step 1: Have playOneGame expose the last MCTS instance**

In `src/az-selfplay.js`, change `playOneGame` so it returns the last `AZMCTS` instance constructed (i.e. the one used for the final move) by adding `lastMcts` to the returned object. Specifically, hoist `mcts` outside the loop:

```js
  let lastMcts = null;
  for (; t < maxSteps; t++) {
    if (maze.isGoal(pos)) { reachedGoal = true; break; }
    const mcts = new AZMCTS(maze, network, { cPuct, startPos: pos });
    for (let s = 0; s < simsPerMove; s++) mcts.iterate();
    lastMcts = mcts;
    // ... rest unchanged
  }
```

And at the bottom of the function, change the return to:

```js
  return { trajectory, z, steps: trajectory.length, reachedGoal, lastMcts };
```

- [ ] **Step 2: Add compatibility methods to AZMCTS / AZNode for TreeRenderer**

`TreeRenderer` (in `src/render-tree.js`) walks `mcts.root` for layout and reads `mcts.config.C` plus `n.ucb1(C, parentVisits)` from a mouseover tooltip handler. AZMCTS as written has none of those. Add them as a thin compatibility layer.

In `/Users/brandon/mcts-maze-demo/src/az-mcts.js`, modify the `AZMCTS` constructor to also expose `config` (a shallow object with `C: cPuct` aliasing), and add `principalVariation` and `treeStats` methods. Replace the constructor and append the methods so the file ends with the class definition shown below — only the constructor body and the three new methods are changed; the imports, `puctScore`, `ancestorPositionKeys`, and `iterate`/`rootPolicy` methods are unchanged.

Constructor — replace the existing constructor body with:

```js
  constructor(maze, network, config = {}) {
    this.maze = maze;
    this.network = network;
    this.cPuct = config.cPuct ?? 1.0;
    // TreeRenderer reads mcts.config.C; expose it as an alias of cPuct so the
    // same renderer works for both Classic and AZ trees. The tooltip label
    // still reads "UCB1" but the value shown is PUCT score for AZ nodes.
    this.config = { C: this.cPuct };
    const startPos = config.startPos ?? maze.start;
    this.root = new AZNode([...startPos], null, null, 1);
    this.iterationCount = 0;
  }
```

Append these methods to the `AZMCTS` class (inside the closing brace, after `rootPolicy`):

```js
  principalVariation() {
    const path = [this.root];
    let node = this.root;
    while (node.children.size > 0) {
      let best = null;
      let bestVisits = -1;
      for (const child of node.children.values()) {
        if (child.visits > bestVisits) { best = child; bestVisits = child.visits; }
      }
      if (!best) break;
      path.push(best);
      node = best;
    }
    return path;
  }

  treeStats() {
    let totalNodes = 0, maxDepth = 0, maxVisits = 0;
    const walk = (n, d) => {
      totalNodes++;
      if (d > maxDepth) maxDepth = d;
      if (n.visits > maxVisits) maxVisits = n.visits;
      for (const c of n.children.values()) walk(c, d + 1);
    };
    walk(this.root, 0);
    return { totalNodes, maxDepth, maxVisits };
  }

  cellVisits() {
    const map = new Map();
    const walk = (n) => {
      const key = `${n.position[0]},${n.position[1]}`;
      map.set(key, (map.get(key) || 0) + n.visits);
      for (const c of n.children.values()) walk(c);
    };
    walk(this.root);
    return map;
  }
```

Then in the `AZNode` class, add a `ucb1(C, parentVisits)` method that returns the PUCT score (so the tooltip in `render-tree.js` doesn't throw on hover). Append inside the AZNode class, after `meanReward`:

```js
  // Compatibility shim for TreeRenderer's tooltip. Returns PUCT score, not
  // UCB1; treat `C` as the c_puct coefficient.
  ucb1(C, parentVisits) {
    if (parentVisits == null || parentVisits === 0) return Infinity;
    return puctScore(this, parentVisits, C);
  }
```

Note: this references `puctScore`, defined at module top — JavaScript hoists module-level `function` declarations, but `puctScore` is an `export function`, which is also hoisted, so the order is fine.

- [ ] **Step 3: Wire TreeRenderer into az-tab.js**

In `src/az-tab.js` top imports, add:

```js
import { TreeRenderer } from "./render-tree.js";
```

In `initAzTab`, before `mazeRenderer` construction, add:

```js
  const treeRenderer = new TreeRenderer(document.getElementById("az-tree-svg"));
```

At the end of `runSelfPlayStep`, render the tree from `result.lastMcts`:

```js
  if (result.lastMcts) treeRenderer.render(result.lastMcts);
```

- [ ] **Step 4: Manual smoke**

Run server, AZ tab. Click `Self-play step`. The tree SVG renders a small tree representing the last per-move MCTS search. Run 10. Tree keeps redrawing for the last move of each game.

- [ ] **Step 5: Commit**

```bash
git add src/az-selfplay.js src/az-mcts.js src/az-tab.js
git commit -m "feat(az): render last-move PUCT tree in AZ tab"
```

---

## Task 16: Existing-test regression sweep + final integration smoke

**Files:**
- (read-only) all

- [ ] **Step 1: Run all tests**

Run: `cd /Users/brandon/mcts-maze-demo && npm test`
Expected: every test passes, including the new az-encode, az-mcts, and az-selfplay suites.

- [ ] **Step 2: Manual end-to-end smoke**

Run: `python3 -m http.server 8000`.

Classic tab:
- Maze renders. Step, Run 10, Run to end, Reset, New maze all work as before. Size dropdown switches mazes.

AZ tab:
- Page renders without console errors.
- Default z formula is `isGoal ? 1 : -1`. Status bar shows zeros.
- Click `Self-play step` once. Status bar updates. v-heatmap colors visible.
- Switch to `p(a|s) arrows` — arrows appear.
- Switch to `MCTS visits` — the existing classic-style visit heatmap renders (uses the same code path as Classic; if it errors, note as a known limitation — the AZ tab's renderer is shared but the visit data passed is for the current AZMCTS root, which may be empty between self-play steps).
- Switch the z preset to `Distance-shaped`. Click `Run 10`. z chart and loss chart populate; probe panel updates.
- Click `Run continuously`, wait ~60s. Button reads `Stop`. Click `Stop`. Charts show ~30+ points; loss is finite; status bar `z̄(20)` is closer to 0 or positive (training is making *some* progress; it does not need to "win").
- Click `Reset network`. Confirm. Charts clear, status zeros.
- Change maze size to 15. Confirm. Network and buffer reset.

Stop the server.

- [ ] **Step 3: Commit if any smoke-driven tweaks were needed**

If any small fix was applied in this task, commit:

```bash
git add -A
git commit -m "fix(az): smoke-test follow-ups"
```

If no fix was needed, skip this step.

---

## Verification checklist

When the plan completes, the following should be true:

- All 47 prior tests + 3 new test suites (az-encode, az-mcts, az-selfplay) pass.
- Classic MCTS tab behaves identically to before.
- AlphaZero tab supports: size dropdown, z formula + presets, c_puct slider, sims slider, learning rate slider, Self-play step, Run 10, Run continuously / Stop, Reset MCTS, Reset network.
- The maze pane shows toggleable overlays: v(s) heatmap, p(a|s) arrows, MCTS visits, off.
- The charts panel shows: z curve, loss curves, probe panel.
- The tree pane renders the last per-move PUCT search.
- Status bar shows Games, Latest z, z̄(20), Loss, Buffer.
- Switching tabs preserves state; switching maze sizes resets the AZ network with confirmation.

## Out of scope (deferred)

Per the spec §11, the following are explicitly not in this plan:

- Dirichlet noise on the root prior
- Sub-tree reuse between self-play moves
- Curriculum learning across maze sizes
- CNN architecture (Conv2D) instead of MLP
- Save/load weights to local storage
- BFS-oracle "fake AZ" as a comparison baseline
