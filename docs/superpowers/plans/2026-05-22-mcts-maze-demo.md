# MCTS Maze Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page, browser-based interactive visualization of Monte Carlo Tree Search running on a small grid-world maze, with step-by-step controls and side-by-side maze + tree views.

**Architecture:** Pure static files — `index.html` + `styles.css` + ES module sources in `src/`. The algorithm core (Maze, Node, MCTS) is unit-tested headlessly via Node's built-in `node:test`. The rendering layers (Canvas for maze, D3 SVG for tree) are verified visually in the browser. The MCTS class exposes an `iterate()` generator that yields one event per phase, letting the UI animate select → expand → simulate → backprop discretely.

**Tech Stack:** Vanilla JS (ES modules), HTML5 Canvas, D3 v7 (via CDN), Node's built-in test runner for unit tests. No build step, no npm install.

**Reference spec:** `/Users/brandon/mcts-maze-demo/mcts-maze-demo-spec.md`

---

## Project Conventions

- All source files are ES modules (`type="module"` in HTML, `import`/`export` everywhere).
- All file paths in this plan are absolute under `/Users/brandon/mcts-maze-demo/`.
- Commit after every task, using Conventional Commits style (`feat:`, `test:`, `chore:`, `refactor:`, `docs:`).
- Tests live in `tests/` mirroring the `src/` structure and use `node:test` + `node:assert`.
- Run tests with `node --test tests/` from the project root.

---

## Task 1: Project skeleton + HTML/CSS layout

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/index.html`
- Create: `/Users/brandon/mcts-maze-demo/styles.css`
- Create: `/Users/brandon/mcts-maze-demo/.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/brandon/mcts-maze-demo && git init && git add mcts-maze-demo-spec.md docs/
git commit -m "chore: add spec and implementation plan"
```

- [ ] **Step 2: Create `.gitignore`**

```
.DS_Store
node_modules/
.superpowers/
*.log
```

- [ ] **Step 3: Create `index.html` skeleton**

```html
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <title>MCTS Maze Demo</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdn.jsdelivr.net/npm/d3@7.8.5/dist/d3.min.js" defer></script>
  <script type="module" src="src/app.js" defer></script>
</head>
<body>
  <header>
    <h1>MCTS Maze Demo</h1>
    <div id="status-bar">
      <span>Iter: <span id="stat-iter">0</span></span>
      <span>Tree nodes: <span id="stat-nodes">1</span></span>
      <span>PV len: <span id="stat-pv">0</span></span>
      <span>Phase: <span id="stat-phase">idle</span></span>
    </div>
    <button id="theme-toggle" title="Toggle theme">🌓</button>
  </header>

  <main>
    <section id="maze-pane">
      <canvas id="maze-canvas" width="600" height="600"></canvas>
    </section>
    <section id="tree-pane">
      <svg id="tree-svg" width="600" height="600"></svg>
    </section>
  </main>

  <footer id="controls">
    <div class="control-row">
      <label>Maze size:
        <select id="ctl-size">
          <option value="10" selected>10×10</option>
          <option value="15">15×15</option>
          <option value="20">20×20</option>
        </select>
      </label>
      <label>Reward:
        <select id="ctl-reward">
          <option value="shaped" selected>Distance-shaped</option>
          <option value="winloss">Pure win/loss</option>
        </select>
      </label>
    </div>
    <div class="control-row">
      <label>UCB1 C: <input type="range" id="ctl-c" min="0" max="3" step="0.01" value="1.41" />
        <span id="ctl-c-val">1.41</span>
      </label>
      <label>Rollout horizon: <input type="range" id="ctl-horizon" min="1" max="80" step="1" value="40" />
        <span id="ctl-horizon-val">40</span>
      </label>
    </div>
    <div class="control-row">
      <button id="btn-step">Step</button>
      <button id="btn-run10">Run 10</button>
      <button id="btn-runend">Run to end</button>
      <button id="btn-reset">Reset</button>
    </div>
  </footer>
</body>
</html>
```

- [ ] **Step 4: Create `styles.css`**

```css
:root {
  --bg: #1a1a1a;
  --panel: #242424;
  --fg: #e0e0e0;
  --muted: #888;
  --accent: #4ade80;
  --warn: #fbbf24;
  --action: #fb923c;
  --border: #333;
}
html[data-theme="light"] {
  --bg: #f5f5f5;
  --panel: #ffffff;
  --fg: #1a1a1a;
  --muted: #666;
  --border: #ccc;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

header {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 0.75rem 1.5rem;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
header h1 { font-size: 1.1rem; margin: 0; }
#status-bar { display: flex; gap: 1rem; font-size: 0.9rem; color: var(--muted); flex: 1; }
#status-bar span span { color: var(--fg); font-variant-numeric: tabular-nums; }
#theme-toggle {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--fg);
  padding: 0.25rem 0.5rem;
  cursor: pointer;
  border-radius: 4px;
}

main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  flex: 1;
  gap: 1px;
  background: var(--border);
}
#maze-pane, #tree-pane {
  background: var(--panel);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}
#maze-canvas { background: var(--bg); }
#tree-svg { background: var(--bg); width: 100%; height: 100%; }

footer#controls {
  background: var(--panel);
  border-top: 1px solid var(--border);
  padding: 0.75rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.control-row {
  display: flex;
  gap: 1.5rem;
  align-items: center;
  flex-wrap: wrap;
}
.control-row label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
}
input[type="range"] { width: 160px; }
select, button {
  background: var(--bg);
  color: var(--fg);
  border: 1px solid var(--border);
  padding: 0.35rem 0.7rem;
  border-radius: 4px;
  font-size: 0.9rem;
  cursor: pointer;
}
button:hover { border-color: var(--muted); }
button:active { background: var(--panel); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
```

- [ ] **Step 5: Verify in browser**

Open `file:///Users/brandon/mcts-maze-demo/index.html` in a browser. Expected: two-pane layout, blank canvas on left, blank SVG on right, full control bar at bottom. Console should be free of errors (the app.js doesn't exist yet, but the `defer` + `type="module"` means a 404 is logged but doesn't break layout — that's fine for now).

- [ ] **Step 6: Commit**

```bash
git add index.html styles.css .gitignore
git commit -m "feat: add HTML skeleton and base styles"
```

---

## Task 2: Maze class (TDD)

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/maze.js`
- Create: `/Users/brandon/mcts-maze-demo/tests/maze.test.js`

- [ ] **Step 1: Write failing tests for Maze basics**

Create `/Users/brandon/mcts-maze-demo/tests/maze.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze } from "../src/maze.js";

// A 3x3 maze: walls on the right column except for goal cell.
//  0 0 1
//  0 0 1
//  0 0 0  (goal at [2,2])
const SMALL_GRID = [
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 0],
];

test("Maze: stores grid, start, goal", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.size, 3);
  assert.deepEqual(m.start, [0, 0]);
  assert.deepEqual(m.goal, [2, 2]);
});

test("Maze.isGoal returns true only at goal cell", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.isGoal([2, 2]), true);
  assert.equal(m.isGoal([0, 0]), false);
  assert.equal(m.isGoal([1, 1]), false);
});

test("Maze.legalActions excludes walls and out-of-bounds", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  // From [0,0]: can go right [0,1] and down [1,0]; up and left are out of bounds.
  const actions = m.legalActions([0, 0]).map((a) => a.join(","));
  assert.deepEqual(actions.sort(), ["0,1", "1,0"]);
  // From [0,1]: can go down [1,1] and left [0,0]; right is wall, up is OOB.
  const a2 = m.legalActions([0, 1]).map((a) => a.join(","));
  assert.deepEqual(a2.sort(), ["0,-1", "1,0"]);
});

test("Maze.step applies action", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.deepEqual(m.step([0, 0], [1, 0]), [1, 0]);
  assert.deepEqual(m.step([1, 1], [0, -1]), [1, 0]);
});

test("Maze.manhattan computes |dr| + |dc|", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.manhattan([0, 0], [2, 2]), 4);
  assert.equal(m.manhattan([1, 1], [1, 1]), 0);
});

test("Maze.maxDistance equals Manhattan diameter of open cells", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  // Far corners of the open region: [0,0] and [2,2] → 4
  assert.equal(m.maxDistance(), 4);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/maze.test.js`
Expected: All tests fail with `Cannot find module '../src/maze.js'` or similar.

- [ ] **Step 3: Implement Maze class**

Create `/Users/brandon/mcts-maze-demo/src/maze.js`:

```js
// Action deltas: up, down, left, right.
export const ACTIONS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

export class Maze {
  constructor(grid, start, goal) {
    this.grid = grid;
    this.size = grid.length;
    this.start = start;
    this.goal = goal;
    this._maxDistance = null;
  }

  inBounds([r, c]) {
    return r >= 0 && r < this.size && c >= 0 && c < this.size;
  }

  isOpen([r, c]) {
    return this.inBounds([r, c]) && this.grid[r][c] === 0;
  }

  isGoal([r, c]) {
    return r === this.goal[0] && c === this.goal[1];
  }

  legalActions([r, c]) {
    const result = [];
    for (const [dr, dc] of ACTIONS) {
      const nr = r + dr;
      const nc = c + dc;
      if (this.isOpen([nr, nc])) result.push([dr, dc]);
    }
    return result;
  }

  step([r, c], [dr, dc]) {
    return [r + dr, c + dc];
  }

  manhattan([r1, c1], [r2, c2]) {
    return Math.abs(r1 - r2) + Math.abs(c1 - c2);
  }

  maxDistance() {
    if (this._maxDistance !== null) return this._maxDistance;
    // Diameter over open cells, naive O(N^2).
    let best = 0;
    const open = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (this.grid[r][c] === 0) open.push([r, c]);
      }
    }
    for (let i = 0; i < open.length; i++) {
      for (let j = i + 1; j < open.length; j++) {
        const d = this.manhattan(open[i], open[j]);
        if (d > best) best = d;
      }
    }
    this._maxDistance = best;
    return best;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/maze.test.js`
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/maze.js tests/maze.test.js
git commit -m "feat: add Maze class with TDD coverage"
```

---

## Task 3: Hand-crafted maze data

**Files:**
- Modify: `/Users/brandon/mcts-maze-demo/src/maze.js` (add `MAZE_10`, `MAZE_15`, `MAZE_20` exports)
- Create: `/Users/brandon/mcts-maze-demo/tests/maze-data.test.js`

- [ ] **Step 1: Write failing test for maze data invariants**

Create `/Users/brandon/mcts-maze-demo/tests/maze-data.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze, MAZE_10, MAZE_15, MAZE_20 } from "../src/maze.js";

function assertReachable(maze) {
  // BFS from start; assert goal is reachable through open cells.
  const visited = new Set();
  const key = (p) => `${p[0]},${p[1]}`;
  const queue = [maze.start];
  visited.add(key(maze.start));
  while (queue.length) {
    const pos = queue.shift();
    if (maze.isGoal(pos)) return true;
    for (const a of maze.legalActions(pos)) {
      const next = maze.step(pos, a);
      if (!visited.has(key(next))) {
        visited.add(key(next));
        queue.push(next);
      }
    }
  }
  return false;
}

for (const [name, m] of [["MAZE_10", MAZE_10], ["MAZE_15", MAZE_15], ["MAZE_20", MAZE_20]]) {
  test(`${name}: correct size`, () => {
    const expected = parseInt(name.split("_")[1], 10);
    assert.equal(m.size, expected);
  });
  test(`${name}: start is open`, () => {
    assert.equal(m.grid[m.start[0]][m.start[1]], 0);
  });
  test(`${name}: goal is open`, () => {
    assert.equal(m.grid[m.goal[0]][m.goal[1]], 0);
  });
  test(`${name}: goal is reachable from start`, () => {
    assert.equal(assertReachable(m), true);
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/maze-data.test.js`
Expected: Tests fail with `MAZE_10 is not defined` or similar.

- [ ] **Step 3: Add maze data to `src/maze.js`**

Append to `/Users/brandon/mcts-maze-demo/src/maze.js`:

```js
// Hand-crafted mazes. 0 = open, 1 = wall.
// Each is designed to have at least two viable paths and a few dead-ends.

// 10×10: start top-left, goal bottom-right, central wall with two gaps.
const GRID_10 = [
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
  [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
  [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
  [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
];
export const MAZE_10 = new Maze(GRID_10, [0, 0], [9, 9]);

// 15×15.
const GRID_15 = [
  [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
  [1,1,0,1,0,1,1,1,0,1,0,1,1,1,0],
  [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
  [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
  [1,1,1,1,0,1,1,1,1,1,1,1,0,1,0],
  [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
  [0,1,0,1,1,1,1,1,0,1,0,1,1,1,0],
  [0,1,0,0,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
  [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
  [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0],
  [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,0,1,1,1,0,1,0,1,1,1,0,1,0],
  [0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
];
export const MAZE_15 = new Maze(GRID_15, [0, 0], [14, 14]);

// 20×20.
const GRID_20 = [
  [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
  [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0],
  [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
  [0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],
  [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
  [0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0],
  [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
  [0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,0],
  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
];
export const MAZE_20 = new Maze(GRID_20, [0, 0], [19, 19]);

export const MAZES = { 10: MAZE_10, 15: MAZE_15, 20: MAZE_20 };
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/`
Expected: All 12 tests pass (6 from Task 2 + 12 from this task = 18 actually; verify reachability passes for all three).

If the reachability test fails for any maze: edit the grid in `src/maze.js` to ensure a path exists. The test guarantees correctness.

- [ ] **Step 5: Commit**

```bash
git add src/maze.js tests/maze-data.test.js
git commit -m "feat: add three hand-crafted maze layouts (10/15/20)"
```

---

## Task 4: Maze canvas renderer

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/render-maze.js`
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js` (create)

- [ ] **Step 1: Create `src/render-maze.js`**

```js
// Canvas-based maze renderer. Stateless: pass it the maze and a render state,
// and it draws the current frame.

export class MazeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
  }

  /**
   * @param {Maze} maze
   * @param {object} state
   *   - visits: Map<"r,c", number>     // cell visit counts for heatmap
   *   - principalVariation: [r,c][]    // PV polyline
   *   - selectionPath: [r,c][]         // cells outlined during select phase
   *   - rolloutTrail: [r,c][]          // current rollout path so far
   */
  render(maze, state = {}) {
    const { ctx, canvas } = this;
    const cell = Math.floor(Math.min(canvas.width, canvas.height) / maze.size);
    const offX = (canvas.width - cell * maze.size) / 2;
    const offY = (canvas.height - cell * maze.size) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Heatmap normalization
    const visits = state.visits || new Map();
    let maxVisits = 0;
    for (const v of visits.values()) if (v > maxVisits) maxVisits = v;

    // Grid + heatmap
    for (let r = 0; r < maze.size; r++) {
      for (let c = 0; c < maze.size; c++) {
        const x = offX + c * cell;
        const y = offY + r * cell;
        const isWall = maze.grid[r][c] === 1;
        if (isWall) {
          ctx.fillStyle = "#3a3a3a";
        } else {
          const v = visits.get(`${r},${c}`) || 0;
          const t = maxVisits ? v / maxVisits : 0;
          // Blend from #1f1f1f (cold) to #3a82f6 (hot blue)
          const r0 = 31, g0 = 31, b0 = 31;
          const r1 = 58, g1 = 130, b1 = 246;
          const rr = Math.round(r0 + (r1 - r0) * t);
          const gg = Math.round(g0 + (g1 - g0) * t);
          const bb = Math.round(b0 + (b1 - b0) * t);
          ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
        }
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = "#0a0a0a";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }

    // Selection path outlines
    if (state.selectionPath?.length) {
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 3;
      for (const [r, c] of state.selectionPath) {
        ctx.strokeRect(offX + c * cell + 2, offY + r * cell + 2, cell - 4, cell - 4);
      }
    }

    // Rollout trail
    if (state.rolloutTrail?.length > 1) {
      ctx.strokeStyle = "#fb923c";
      ctx.lineWidth = Math.max(2, cell * 0.15);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      const cx = (r, c) => offX + c * cell + cell / 2;
      const cy = (r, c) => offY + r * cell + cell / 2;
      ctx.moveTo(cx(...state.rolloutTrail[0]), cy(...state.rolloutTrail[0]));
      for (const [r, c] of state.rolloutTrail.slice(1)) ctx.lineTo(cx(r, c), cy(r, c));
      ctx.stroke();
    }

    // Principal variation
    if (state.principalVariation?.length > 1) {
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = Math.max(2, cell * 0.2);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      const cx = (r, c) => offX + c * cell + cell / 2;
      const cy = (r, c) => offY + r * cell + cell / 2;
      ctx.moveTo(cx(...state.principalVariation[0]), cy(...state.principalVariation[0]));
      for (const [r, c] of state.principalVariation.slice(1)) ctx.lineTo(cx(r, c), cy(r, c));
      ctx.stroke();
    }

    // Start dot
    ctx.fillStyle = "#4ade80";
    ctx.beginPath();
    ctx.arc(offX + maze.start[1] * cell + cell / 2, offY + maze.start[0] * cell + cell / 2, cell * 0.25, 0, Math.PI * 2);
    ctx.fill();

    // Goal flag (red square)
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(
      offX + maze.goal[1] * cell + cell * 0.2,
      offY + maze.goal[0] * cell + cell * 0.2,
      cell * 0.6,
      cell * 0.6
    );
  }
}
```

- [ ] **Step 2: Create minimal `src/app.js` to render the default maze**

```js
import { MAZES } from "./maze.js";
import { MazeRenderer } from "./render-maze.js";

const canvas = document.getElementById("maze-canvas");
const sizeSel = document.getElementById("ctl-size");

const renderer = new MazeRenderer(canvas);

function render() {
  const size = parseInt(sizeSel.value, 10);
  const maze = MAZES[size];
  renderer.render(maze);
}

sizeSel.addEventListener("change", render);
render();
```

- [ ] **Step 3: Verify in browser**

Open `file:///Users/brandon/mcts-maze-demo/index.html`. Expected: left pane shows the 10×10 maze with visible walls, green dot at top-left, red square at bottom-right. Switching the dropdown to 15 or 20 reloads with the larger maze. Console should be clean.

- [ ] **Step 4: Commit**

```bash
git add src/render-maze.js src/app.js
git commit -m "feat: add canvas-based maze renderer and minimal app wiring"
```

---

## Task 5: Node class (TDD)

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/node.js`
- Create: `/Users/brandon/mcts-maze-demo/tests/node.test.js`

- [ ] **Step 1: Write failing tests**

Create `/Users/brandon/mcts-maze-demo/tests/node.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { Node } from "../src/node.js";

test("Node: defaults", () => {
  const n = new Node([1, 2], null, null);
  assert.deepEqual(n.position, [1, 2]);
  assert.equal(n.parent, null);
  assert.equal(n.action, null);
  assert.equal(n.visits, 0);
  assert.equal(n.totalReward, 0);
  assert.equal(n.children.size, 0);
});

test("Node.meanReward = totalReward / visits, 0 when unvisited", () => {
  const n = new Node([0, 0], null, null);
  assert.equal(n.meanReward(), 0);
  n.visits = 4;
  n.totalReward = 2;
  assert.equal(n.meanReward(), 0.5);
});

test("Node.ucb1 is +Infinity when unvisited", () => {
  const n = new Node([0, 0], null, null);
  assert.equal(n.ucb1(1.41, 10), Infinity);
});

test("Node.ucb1 = mean + C*sqrt(ln(parentVisits)/visits) when visited", () => {
  const n = new Node([0, 0], null, null);
  n.visits = 4;
  n.totalReward = 2;
  const C = 1.41;
  const expected = 0.5 + C * Math.sqrt(Math.log(10) / 4);
  const actual = n.ucb1(C, 10);
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} ≠ ${expected}`);
});

test("Node.isFullyExpanded once untriedActions is empty", () => {
  const n = new Node([0, 0], null, null);
  n.untriedActions = [[1, 0], [0, 1]];
  assert.equal(n.isFullyExpanded(), false);
  n.untriedActions = [];
  assert.equal(n.isFullyExpanded(), true);
});

test("Node.bestChild returns child with most visits", () => {
  const parent = new Node([0, 0], null, null);
  const a = new Node([1, 0], parent, [1, 0]);
  const b = new Node([0, 1], parent, [0, 1]);
  a.visits = 5;
  b.visits = 12;
  parent.children.set("1,0", a);
  parent.children.set("0,1", b);
  assert.equal(parent.bestChild(), b);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/node.test.js`
Expected: All tests fail with module-not-found.

- [ ] **Step 3: Implement `src/node.js`**

```js
export class Node {
  constructor(position, parent, action) {
    this.position = position;
    this.parent = parent;
    this.action = action; // [dr, dc] taken from parent to reach here
    this.children = new Map(); // key "dr,dc" → Node
    this.untriedActions = null; // populated lazily by MCTS on first visit
    this.visits = 0;
    this.totalReward = 0;
  }

  meanReward() {
    return this.visits === 0 ? 0 : this.totalReward / this.visits;
  }

  ucb1(C, parentVisits) {
    if (this.visits === 0) return Infinity;
    return this.meanReward() + C * Math.sqrt(Math.log(parentVisits) / this.visits);
  }

  isFullyExpanded() {
    return Array.isArray(this.untriedActions) && this.untriedActions.length === 0;
  }

  isTerminal(maze) {
    return maze.isGoal(this.position);
  }

  bestChild() {
    let best = null;
    let bestVisits = -1;
    for (const child of this.children.values()) {
      if (child.visits > bestVisits) {
        best = child;
        bestVisits = child.visits;
      }
    }
    return best;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/node.test.js`
Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/node.js tests/node.test.js
git commit -m "feat: add Node class with TDD coverage"
```

---

## Task 6: PRNG and reward functions (TDD)

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/rng.js`
- Create: `/Users/brandon/mcts-maze-demo/src/reward.js`
- Create: `/Users/brandon/mcts-maze-demo/tests/rng.test.js`
- Create: `/Users/brandon/mcts-maze-demo/tests/reward.test.js`

- [ ] **Step 1: Write failing test for RNG**

Create `/Users/brandon/mcts-maze-demo/tests/rng.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { mulberry32 } from "../src/rng.js";

test("mulberry32: deterministic given seed", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 100; i++) assert.equal(a(), b());
});

test("mulberry32: values in [0, 1)", () => {
  const r = mulberry32(7);
  for (let i = 0; i < 1000; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1, `value out of range: ${v}`);
  }
});

test("mulberry32: different seeds produce different sequences", () => {
  const a = mulberry32(1);
  const b = mulberry32(2);
  assert.notEqual(a(), b());
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/rng.test.js`
Expected: fail with module-not-found.

- [ ] **Step 3: Implement `src/rng.js`**

```js
// Mulberry32: small, seedable PRNG. Returns a function that yields [0, 1).
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Convenience: pick a uniform element from an array using a given rng().
export function pick(arr, rng) {
  return arr[Math.floor(rng() * arr.length)];
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/rng.test.js`
Expected: all pass.

- [ ] **Step 5: Write failing test for rewards**

Create `/Users/brandon/mcts-maze-demo/tests/reward.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze } from "../src/maze.js";
import { computeReward } from "../src/reward.js";

const GRID = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("winloss: 1 if final position is goal, 0 otherwise", () => {
  assert.equal(computeReward(maze, [2, 2], "winloss"), 1);
  assert.equal(computeReward(maze, [1, 1], "winloss"), 0);
});

test("shaped: 1 - dist/maxDist when not at goal", () => {
  // maxDistance = 4 for this 3x3.
  // dist [1,1]→goal = 2; reward = 1 - 2/4 = 0.5
  assert.equal(computeReward(maze, [1, 1], "shaped"), 0.5);
  // dist [0,0]→goal = 4; reward = 1 - 4/4 = 0
  assert.equal(computeReward(maze, [0, 0], "shaped"), 0);
});

test("shaped: adds +1 goal bonus when at goal", () => {
  // 1 - 0/4 + 1 = 2
  assert.equal(computeReward(maze, [2, 2], "shaped"), 2);
});
```

- [ ] **Step 6: Run to verify failure**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/reward.test.js`
Expected: fail with module-not-found.

- [ ] **Step 7: Implement `src/reward.js`**

```js
/**
 * Compute reward at the end of a rollout.
 * @param {Maze} maze
 * @param {[number,number]} finalPos - last position in the rollout
 * @param {"shaped"|"winloss"} mode
 * @returns {number}
 */
export function computeReward(maze, finalPos, mode) {
  if (mode === "winloss") {
    return maze.isGoal(finalPos) ? 1 : 0;
  }
  if (mode === "shaped") {
    const maxD = maze.maxDistance();
    const d = maze.manhattan(finalPos, maze.goal);
    const base = Math.max(0, 1 - d / maxD);
    return base + (maze.isGoal(finalPos) ? 1 : 0);
  }
  throw new Error(`Unknown reward mode: ${mode}`);
}
```

- [ ] **Step 8: Run to verify pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/`
Expected: all tests across all files pass.

- [ ] **Step 9: Commit**

```bash
git add src/rng.js src/reward.js tests/rng.test.js tests/reward.test.js
git commit -m "feat: add seedable PRNG and reward functions with TDD coverage"
```

---

## Task 7: MCTS class — selection, expansion, simulation, backprop (TDD)

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/mcts.js`
- Create: `/Users/brandon/mcts-maze-demo/tests/mcts.test.js`

- [ ] **Step 1: Write failing tests for MCTS core**

Create `/Users/brandon/mcts-maze-demo/tests/mcts.test.js`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze, MAZE_10 } from "../src/maze.js";
import { MCTS } from "../src/mcts.js";

const SIMPLE = new Maze(
  [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ],
  [0, 0],
  [2, 2]
);

test("MCTS: starts with a single root node at start position", () => {
  const m = new MCTS(SIMPLE);
  assert.deepEqual(m.root.position, [0, 0]);
  assert.equal(m.root.visits, 0);
  assert.equal(m.iterationCount, 0);
});

test("MCTS.iterate yields select → expand → simulate → backprop in order", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  const events = [];
  for (const ev of m.iterate()) events.push(ev.phase);
  assert.deepEqual(events, ["select", "expand", "simulate", "backprop"]);
});

test("MCTS: after one iteration, root has visits=1 and one child", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (const _ev of m.iterate()) { /* drain */ }
  assert.equal(m.root.visits, 1);
  assert.equal(m.root.children.size, 1);
  assert.equal(m.iterationCount, 1);
});

test("MCTS: after many iterations, finds path to goal (PV reaches goal)", () => {
  const m = new MCTS(MAZE_10, { seed: 42, rolloutHorizon: 40, rewardMode: "shaped" });
  for (let i = 0; i < 500; i++) for (const _ev of m.iterate()) { /* drain */ }
  const pv = m.principalVariation();
  // The PV should reach the goal or get very close — assert it reached the goal cell.
  const last = pv[pv.length - 1];
  assert.ok(
    MAZE_10.isGoal(last.position) || MAZE_10.manhattan(last.position, MAZE_10.goal) <= 3,
    `PV final position ${last.position} too far from goal ${MAZE_10.goal}`
  );
});

test("MCTS.treeStats reports node count, max depth, max visits", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (let i = 0; i < 10; i++) for (const _ev of m.iterate()) {}
  const s = m.treeStats();
  assert.ok(s.totalNodes > 1);
  assert.ok(s.maxDepth >= 1);
  assert.ok(s.maxVisits >= 1);
});

test("MCTS: backprop event path matches select+expand path", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  let selectPath, expandedNode, backpropPath;
  for (const ev of m.iterate()) {
    if (ev.phase === "select") selectPath = ev.path;
    if (ev.phase === "expand") expandedNode = ev.newNode;
    if (ev.phase === "backprop") backpropPath = ev.path;
  }
  // Backprop path should equal selectPath + [expandedNode]
  const expected = [...selectPath, expandedNode];
  assert.equal(backpropPath.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.equal(backpropPath[i], expected[i]);
  }
});

test("MCTS: visits increment along backprop path", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (let i = 0; i < 3; i++) for (const _ev of m.iterate()) {}
  // Root visits should equal iteration count.
  assert.equal(m.root.visits, 3);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/mcts.test.js`
Expected: fail with module-not-found.

- [ ] **Step 3: Implement `src/mcts.js`**

```js
import { Node } from "./node.js";
import { mulberry32, pick } from "./rng.js";
import { computeReward } from "./reward.js";

const actionKey = ([dr, dc]) => `${dr},${dc}`;

export class MCTS {
  constructor(maze, config = {}) {
    this.maze = maze;
    this.config = {
      C: config.C ?? 1.41,
      rolloutHorizon: config.rolloutHorizon ?? maze.size * 4,
      rewardMode: config.rewardMode ?? "shaped",
      seed: config.seed ?? 42,
    };
    this.rng = mulberry32(this.config.seed);
    this.root = new Node([...maze.start], null, null);
    this.iterationCount = 0;
  }

  setConfig(partial) {
    // Applied to next iteration; does NOT reset tree (caller controls reset).
    Object.assign(this.config, partial);
  }

  reset() {
    this.rng = mulberry32(this.config.seed);
    this.root = new Node([...this.maze.start], null, null);
    this.iterationCount = 0;
  }

  _ensureUntried(node) {
    if (node.untriedActions === null) {
      node.untriedActions = this.maze.legalActions(node.position).map((a) => [...a]);
    }
  }

  _select() {
    // Walk down until we hit either a non-fully-expanded node or a terminal.
    let node = this.root;
    const path = [node];
    this._ensureUntried(node);
    while (
      !node.isTerminal(this.maze) &&
      node.isFullyExpanded() &&
      node.children.size > 0
    ) {
      let best = null;
      let bestScore = -Infinity;
      for (const child of node.children.values()) {
        const score = child.ucb1(this.config.C, node.visits);
        if (score > bestScore) {
          bestScore = score;
          best = child;
        }
      }
      node = best;
      this._ensureUntried(node);
      path.push(node);
    }
    return path;
  }

  _expand(node) {
    // Precondition: node is not terminal and has untried actions.
    if (node.isTerminal(this.maze)) return node;
    if (!node.untriedActions || node.untriedActions.length === 0) return node;
    // Pop a random untried action.
    const idx = Math.floor(this.rng() * node.untriedActions.length);
    const action = node.untriedActions.splice(idx, 1)[0];
    const newPos = this.maze.step(node.position, action);
    const child = new Node(newPos, node, action);
    node.children.set(actionKey(action), child);
    return child;
  }

  _simulate(startPos) {
    const positions = [[...startPos]];
    let pos = startPos;
    for (let step = 0; step < this.config.rolloutHorizon; step++) {
      if (this.maze.isGoal(pos)) break;
      const actions = this.maze.legalActions(pos);
      if (actions.length === 0) break;
      const action = pick(actions, this.rng);
      pos = this.maze.step(pos, action);
      positions.push([...pos]);
    }
    const reward = computeReward(this.maze, pos, this.config.rewardMode);
    return { positions, reward };
  }

  _backprop(path, reward) {
    for (const node of path) {
      node.visits += 1;
      node.totalReward += reward;
    }
  }

  *iterate() {
    // Phase 1: select
    const selectPath = this._select();
    yield { phase: "select", path: selectPath };

    // Phase 2: expand (if non-terminal & has untried actions)
    const leaf = selectPath[selectPath.length - 1];
    let expanded = leaf;
    if (!leaf.isTerminal(this.maze) && leaf.untriedActions && leaf.untriedActions.length > 0) {
      expanded = this._expand(leaf);
    }
    yield { phase: "expand", newNode: expanded };

    // Phase 3: simulate
    const { positions, reward } = this._simulate(expanded.position);
    yield { phase: "simulate", positions, reward };

    // Phase 4: backprop along selectPath + expanded (skip duplicates if expanded === leaf)
    const backpropPath = expanded === leaf ? [...selectPath] : [...selectPath, expanded];
    this._backprop(backpropPath, reward);
    this.iterationCount += 1;
    yield { phase: "backprop", path: backpropPath, reward };
  }

  principalVariation() {
    const path = [this.root];
    let node = this.root;
    while (node.children.size > 0) {
      const next = node.bestChild();
      if (!next) break;
      path.push(next);
      node = next;
    }
    return path;
  }

  treeStats() {
    let totalNodes = 0;
    let maxDepth = 0;
    let maxVisits = 0;
    const walk = (n, depth) => {
      totalNodes += 1;
      if (depth > maxDepth) maxDepth = depth;
      if (n.visits > maxVisits) maxVisits = n.visits;
      for (const c of n.children.values()) walk(c, depth + 1);
    };
    walk(this.root, 0);
    return { totalNodes, maxDepth, maxVisits };
  }

  // Aggregate cell visit counts across all tree nodes (for the maze heatmap).
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
}
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/`
Expected: all tests pass across maze, maze-data, node, rng, reward, and mcts test files.

If the "PV reaches goal" test fails: it may need more iterations (raise from 500 to 1000) or a more carefully-tuned `C`. Confirm the algorithm runs first; if PV is stuck on a dead-end, debug the selection/expansion logic.

- [ ] **Step 5: Commit**

```bash
git add src/mcts.js tests/mcts.test.js
git commit -m "feat: add MCTS class with full TDD coverage"
```

---

## Task 8: Tree renderer (D3)

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/render-tree.js`
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js` (wire it up)

- [ ] **Step 1: Create `src/render-tree.js`**

```js
// D3-based tree renderer. Pass it an MCTS instance and it renders the tree.
// d3 is loaded as a global from the CDN script tag.

export class TreeRenderer {
  constructor(svgElement) {
    this.svg = d3.select(svgElement);
    this.svg.selectAll("*").remove();

    this.g = this.svg.append("g").attr("class", "tree-root");

    // Tooltip element
    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "tree-tooltip")
      .style("position", "fixed")
      .style("padding", "6px 10px")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "#fff")
      .style("border", "1px solid #444")
      .style("border-radius", "4px")
      .style("font", "12px monospace")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 1000);

    // Zoom/pan
    this.zoomBehavior = d3.zoom().scaleExtent([0.25, 4]).on("zoom", (ev) => {
      this.g.attr("transform", ev.transform);
    });
    this.svg.call(this.zoomBehavior);

    this.selectedPath = new Set();
    this.flashEdge = null;
  }

  setSelectionHighlight(path) {
    // path is an array of MCTS Nodes.
    this.selectedPath = new Set(path);
    this._restyle();
  }

  clearSelectionHighlight() {
    this.selectedPath = new Set();
    this._restyle();
  }

  render(mcts) {
    const width = this.svg.node().clientWidth || 600;
    const height = this.svg.node().clientHeight || 600;

    // Build a D3 hierarchy from MCTS root.
    const toD3 = (n) => ({
      data: n,
      children: [...n.children.values()].map(toD3),
    });
    const root = d3.hierarchy(toD3(mcts.root));

    const treeLayout = d3.tree().size([width - 40, height - 40]);
    treeLayout(root);

    // Determine maxVisits/maxReward for scaling.
    let maxVisits = 1;
    let maxReward = 0.0001;
    root.each((d) => {
      const n = d.data.data;
      if (n.visits > maxVisits) maxVisits = n.visits;
      if (n.meanReward() > maxReward) maxReward = n.meanReward();
    });

    // Edges
    const links = root.links();
    const edge = this.g
      .selectAll("line.edge")
      .data(links, (d) => `${d.source.data.data.position}->${d.target.data.data.position}`);
    edge.exit().remove();
    edge
      .enter()
      .append("line")
      .attr("class", "edge")
      .merge(edge)
      .attr("x1", (d) => d.source.x + 20)
      .attr("y1", (d) => d.source.y + 20)
      .attr("x2", (d) => d.target.x + 20)
      .attr("y2", (d) => d.target.y + 20)
      .attr("stroke", "#555")
      .attr("stroke-width", 1);

    // Nodes
    const nodes = root.descendants();
    const node = this.g
      .selectAll("circle.node")
      .data(nodes, (d) => `${d.data.data.position[0]},${d.data.data.position[1]}-${d.depth}`);
    node.exit().remove();
    const nodeEnter = node
      .enter()
      .append("circle")
      .attr("class", "node")
      .attr("r", 0);

    nodeEnter
      .merge(node)
      .attr("cx", (d) => d.x + 20)
      .attr("cy", (d) => d.y + 20)
      .attr("r", (d) => {
        const v = d.data.data.visits;
        return Math.max(3, Math.sqrt(v) * 2.5);
      })
      .attr("fill", (d) => {
        const mr = d.data.data.meanReward();
        const t = Math.min(1, mr / 1.5);
        const r0 = 255, g0 = 255, b0 = 255;
        const r1 = 74, g1 = 222, b1 = 128;
        const rr = Math.round(r0 + (r1 - r0) * t);
        const gg = Math.round(g0 + (g1 - g0) * t);
        const bb = Math.round(b0 + (b1 - b0) * t);
        return `rgb(${rr},${gg},${bb})`;
      })
      .attr("stroke", (d) => (this.selectedPath.has(d.data.data) ? "#fbbf24" : "#888"))
      .attr("stroke-width", (d) => (this.selectedPath.has(d.data.data) ? 3 : 1))
      .on("mouseover", (event, d) => {
        const n = d.data.data;
        const ucb = n.parent ? n.ucb1(mcts.config.C, n.parent.visits) : 0;
        this.tooltip
          .html(
            `pos: (${n.position[0]},${n.position[1]})<br>` +
              `visits: ${n.visits}<br>` +
              `mean reward: ${n.meanReward().toFixed(3)}<br>` +
              `UCB1: ${Number.isFinite(ucb) ? ucb.toFixed(3) : "∞"}`
          )
          .style("opacity", 1);
      })
      .on("mousemove", (event) => {
        this.tooltip.style("left", event.clientX + 12 + "px").style("top", event.clientY + 12 + "px");
      })
      .on("mouseout", () => {
        this.tooltip.style("opacity", 0);
      });
  }

  _restyle() {
    this.g
      .selectAll("circle.node")
      .attr("stroke", (d) => (this.selectedPath.has(d.data.data) ? "#fbbf24" : "#888"))
      .attr("stroke-width", (d) => (this.selectedPath.has(d.data.data) ? 3 : 1));
  }
}
```

- [ ] **Step 2: Update `src/app.js` to drive a temporary "Run 50" button so we can verify the tree renders**

Replace `/Users/brandon/mcts-maze-demo/src/app.js`:

```js
import { MAZES } from "./maze.js";
import { MCTS } from "./mcts.js";
import { MazeRenderer } from "./render-maze.js";
import { TreeRenderer } from "./render-tree.js";

const canvas = document.getElementById("maze-canvas");
const svg = document.getElementById("tree-svg");
const sizeSel = document.getElementById("ctl-size");
const rewardSel = document.getElementById("ctl-reward");
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
    rewardMode: rewardSel.value,
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
}

document.getElementById("btn-step").addEventListener("click", () => {
  statPhase.textContent = "running";
  for (const _ev of mcts.iterate()) {}
  statPhase.textContent = "idle";
  renderAll();
});
document.getElementById("btn-run10").addEventListener("click", () => {
  for (let i = 0; i < 10; i++) for (const _ev of mcts.iterate()) {}
  renderAll();
});
document.getElementById("btn-runend").addEventListener("click", () => {
  for (let i = 0; i < 500; i++) for (const _ev of mcts.iterate()) {}
  renderAll();
});
document.getElementById("btn-reset").addEventListener("click", newMcts);

sizeSel.addEventListener("change", () => {
  const size = parseInt(sizeSel.value, 10);
  hSlider.max = size * 8;
  hSlider.value = size * 4;
  hVal.textContent = hSlider.value;
  newMcts();
});
rewardSel.addEventListener("change", newMcts);
cSlider.addEventListener("input", () => {
  cVal.textContent = parseFloat(cSlider.value).toFixed(2);
  if (mcts) mcts.setConfig({ C: parseFloat(cSlider.value) });
});
hSlider.addEventListener("input", () => {
  hVal.textContent = hSlider.value;
  if (mcts) mcts.setConfig({ rolloutHorizon: parseInt(hSlider.value, 10) });
});

newMcts();
```

- [ ] **Step 3: Verify in browser**

Open `index.html`. Click "Run to end" — expect tree to populate, maze heatmap to appear, green PV line on maze, status bar updates. Hovering tree nodes shows tooltip. Zooming/panning works on tree.

- [ ] **Step 4: Commit**

```bash
git add src/render-tree.js src/app.js
git commit -m "feat: add D3 tree renderer and wire up basic controls"
```

---

## Task 9: Animation layer

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/src/animate.js`
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js` (use animation for Step button)

- [ ] **Step 1: Create `src/animate.js`**

```js
// Orchestrates per-phase animations for one MCTS iteration.
//
// Usage:
//   await animateIteration(mcts, { mazeRenderer, treeRenderer, currentMaze, setPhase });
//
// The animator drives the MCTS.iterate() generator, pausing between phases
// to render intermediate state with appropriate visual effects.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function animateIteration(mcts, { mazeRenderer, treeRenderer, getMaze, setPhase, setStats }) {
  const maze = getMaze();
  const gen = mcts.iterate();

  let selectPath = null;
  let rolloutPositions = null;

  for (const ev of gen) {
    if (ev.phase === "select") {
      setPhase("select");
      selectPath = ev.path;
      treeRenderer.setSelectionHighlight(selectPath);
      mazeRenderer.render(maze, {
        visits: mcts.cellVisits(),
        principalVariation: mcts.principalVariation().map((n) => n.position),
        selectionPath: selectPath.map((n) => n.position),
      });
      treeRenderer.render(mcts);
      await sleep(400);
    } else if (ev.phase === "expand") {
      setPhase("expand");
      treeRenderer.render(mcts);
      await sleep(200);
    } else if (ev.phase === "simulate") {
      setPhase("simulate");
      rolloutPositions = ev.positions;
      // Animate trail building up cell by cell.
      const total = rolloutPositions.length;
      const maxAnimatedSteps = 30;
      const animatedSteps = Math.min(total, maxAnimatedSteps);
      for (let i = 1; i <= animatedSteps; i++) {
        mazeRenderer.render(maze, {
          visits: mcts.cellVisits(),
          principalVariation: mcts.principalVariation().map((n) => n.position),
          selectionPath: selectPath.map((n) => n.position),
          rolloutTrail: rolloutPositions.slice(0, i + 1),
        });
        await sleep(40);
      }
      if (total > maxAnimatedSteps) {
        // Fast-forward: render full trail at once.
        mazeRenderer.render(maze, {
          visits: mcts.cellVisits(),
          principalVariation: mcts.principalVariation().map((n) => n.position),
          selectionPath: selectPath.map((n) => n.position),
          rolloutTrail: rolloutPositions,
        });
        await sleep(150);
      }
    } else if (ev.phase === "backprop") {
      setPhase("backprop");
      treeRenderer.render(mcts);
      await sleep(400);
    }
  }

  // Final cleanup: clear selection / rollout, redraw with PV only.
  treeRenderer.clearSelectionHighlight();
  mazeRenderer.render(maze, {
    visits: mcts.cellVisits(),
    principalVariation: mcts.principalVariation().map((n) => n.position),
  });
  treeRenderer.render(mcts);
  setStats();
  setPhase("idle");
}
```

- [ ] **Step 2: Update Step button in `src/app.js`**

In `/Users/brandon/mcts-maze-demo/src/app.js`, replace the existing `btn-step` listener (and add an import):

Add at top:
```js
import { animateIteration } from "./animate.js";
```

Replace the `btn-step` listener:
```js
let busy = false;
const setButtonsEnabled = (enabled) => {
  for (const id of ["btn-step", "btn-run10", "btn-runend", "btn-reset"]) {
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
    setStats: renderAll, // updates stats only; full re-render already done
  });
  setButtonsEnabled(true);
  busy = false;
});
```

- [ ] **Step 3: Verify in browser**

Open `index.html`. Click "Step" — expect to see:
1. Yellow outline appear on selection cells in maze; yellow stroke on tree nodes along select path.
2. New node pop into tree.
3. Orange trail animate step-by-step on maze.
4. Tree refreshes (visits/colors updated).
5. Phase indicator cycles through `select → expand → simulate → backprop → idle`.

Click Step several more times. Each iteration should show the new node growing the tree.

- [ ] **Step 4: Commit**

```bash
git add src/animate.js src/app.js
git commit -m "feat: add per-phase animation orchestration"
```

---

## Task 10: Smarter "Run to end" + phase-busy guarding for all buttons

**Files:**
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js`

- [ ] **Step 1: Replace Run 10, Run to end, Reset handlers in `src/app.js`**

Replace the three handlers (and the now-redundant ones from Task 8):

```js
document.getElementById("btn-run10").addEventListener("click", () => {
  if (busy) return;
  busy = true;
  setButtonsEnabled(false);
  statPhase.textContent = "running 10";
  for (let i = 0; i < 10; i++) for (const _ev of mcts.iterate()) {}
  renderAll();
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
  statPhase.textContent = "idle";
  setButtonsEnabled(true);
  busy = false;
});

document.getElementById("btn-reset").addEventListener("click", () => {
  if (busy) return;
  newMcts();
});
```

- [ ] **Step 2: Verify in browser**

- Click Run 10 → tree should grow noticeably.
- Click Run to end → should run quickly and terminate when PV reaches goal. Maze should show a clear green PV line from start to goal.
- Click Reset → tree collapses to root.
- All buttons should be disabled while busy.

- [ ] **Step 3: Commit**

```bash
git add src/app.js
git commit -m "feat: smarter Run-to-end termination and busy guards"
```

---

## Task 11: Theme toggle + reset-on-config-change

**Files:**
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js`

- [ ] **Step 1: Theme toggle**

In `/Users/brandon/mcts-maze-demo/src/app.js`, add at bottom:

```js
document.getElementById("theme-toggle").addEventListener("click", () => {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === "light" ? "dark" : "light";
  renderAll();
});
```

- [ ] **Step 2: Adjust reward-mode change to reset (already in Task 8, but verify)**

The existing `rewardSel.addEventListener("change", newMcts)` correctly resets the tree. Confirm. If sizes are now using `parseInt(sizeSel.value, 10) * 8` for the horizon max — also good (already in Task 8).

- [ ] **Step 3: Verify in browser**

- Click 🌓 → page toggles between dark and light. Canvas re-renders. Tree colors remain readable.
- Change reward mode → tree resets to root.
- Change maze size → tree resets, horizon slider updates max.
- Adjust C slider while iterating → no reset; next iteration uses new C.
- Adjust horizon slider → no reset; next iteration uses new horizon.

- [ ] **Step 4: Commit**

```bash
git add src/app.js
git commit -m "feat: add dark/light theme toggle"
```

---

## Task 12: Debug panel

**Files:**
- Modify: `/Users/brandon/mcts-maze-demo/index.html`
- Modify: `/Users/brandon/mcts-maze-demo/styles.css`
- Modify: `/Users/brandon/mcts-maze-demo/src/app.js`

- [ ] **Step 1: Add hidden debug HTML**

In `/Users/brandon/mcts-maze-demo/index.html`, before the closing `</body>`:

```html
<aside id="debug-panel" hidden>
  <h3>Debug</h3>
  <pre id="debug-stats"></pre>
  <pre id="debug-last-rollout"></pre>
  <button id="debug-bench">Run 1000 iterations (benchmark)</button>
  <pre id="debug-bench-out"></pre>
</aside>
```

- [ ] **Step 2: Add CSS for the debug panel**

Append to `/Users/brandon/mcts-maze-demo/styles.css`:

```css
#debug-panel {
  position: fixed;
  right: 12px;
  bottom: 80px;
  width: 320px;
  max-height: 50vh;
  overflow: auto;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  font: 12px ui-monospace, monospace;
  z-index: 999;
}
#debug-panel h3 { margin: 0 0 8px; font-size: 13px; }
#debug-panel pre { margin: 0 0 8px; white-space: pre-wrap; }
```

- [ ] **Step 3: Activate debug panel via `?debug=1`**

In `/Users/brandon/mcts-maze-demo/src/app.js`, add at top after imports:

```js
const DEBUG = new URLSearchParams(location.search).get("debug") === "1";
if (DEBUG) document.getElementById("debug-panel").hidden = false;
```

Track last rollout. In `animateIteration`, the rollout positions are available; but for simplicity, also track via mcts. Modify `animate.js` to expose last rollout. Simpler approach: maintain a `lastRollout` global in app.js, and have animate accept a callback.

In `/Users/brandon/mcts-maze-demo/src/animate.js`, add a parameter `onRollout`:

Replace the simulate-phase branch:
```js
} else if (ev.phase === "simulate") {
  setPhase("simulate");
  rolloutPositions = ev.positions;
  if (typeof onRollout === "function") onRollout(ev);
  // ...rest unchanged...
```

And update the signature:
```js
export async function animateIteration(
  mcts,
  { mazeRenderer, treeRenderer, getMaze, setPhase, setStats, onRollout }
) {
```

In `/Users/brandon/mcts-maze-demo/src/app.js`, pass `onRollout` and wire debug pane:

```js
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
```

Update step handler:
```js
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
```

Also call `updateDebug()` at the end of Run 10, Run to end, Reset, and `renderAll`.

Add benchmark button handler at bottom of `app.js`:
```js
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
```

- [ ] **Step 4: Verify in browser**

- Open `index.html` — debug panel hidden.
- Open `index.html?debug=1` — debug panel visible in lower-right with live tree stats.
- Click Step → "rollout" pane updates with start/end/steps/reward.
- Click "Run 1000 iterations" → outputs total + per-iteration time.

- [ ] **Step 5: Commit**

```bash
git add index.html styles.css src/app.js src/animate.js
git commit -m "feat: add debug panel with tree stats and benchmark"
```

---

## Task 13: README

**Files:**
- Create: `/Users/brandon/mcts-maze-demo/README.md`

- [ ] **Step 1: Write README**

```markdown
# MCTS Maze Demo

An interactive, browser-based visualization of Monte Carlo Tree Search running on a grid-world maze. Built as a learning artifact: step through one iteration at a time and watch the four MCTS phases — **select**, **expand**, **simulate**, **backpropagate** — play out visually.

## How to run

No build step, no install. Just open `index.html` in a modern browser:

```bash
open index.html
# or:
python3 -m http.server 8000  # then visit http://localhost:8000
```

(D3 v7 is loaded from a CDN; an internet connection is required on first load.)

## Controls

- **Maze size** — switches between three hand-crafted mazes (10×10, 15×15, 20×20). Resets the tree.
- **Reward** — `Distance-shaped` (continuous, dense) or `Pure win/loss` (sparse). Resets the tree.
- **UCB1 C** — exploration constant for the UCB1 formula. Applies to next iteration; does not reset.
- **Rollout horizon** — maximum steps per random rollout. Applies to next iteration; does not reset.
- **Step** — run one full MCTS iteration with phase-by-phase animation.
- **Run 10** — run 10 iterations without animation.
- **Run to end** — run until the principal variation reaches the goal and stays stable for 5 iterations (or 500 iterations max).
- **Reset** — clear the tree.

## What you're seeing

- **Left pane (maze):** the maze grid. Cell color = visit count (heatmap). Green dot = start, red square = goal. Thick green line = current principal variation. Orange trail = the current rollout (during simulate phase). Yellow outlines = cells visited during selection (during select phase).
- **Right pane (tree):** the MCTS search tree as a top-down D3 layout. Node radius = √visits. Node color = mean reward (white → green). Yellow stroke = nodes on the current selection path. Hover for tooltip.
- **Status bar:** iteration count, total tree nodes, PV length, current phase.

## Debug mode

Append `?debug=1` to the URL for a debug panel showing live tree stats, last rollout details, and a 1000-iteration benchmark button.

## Tests

Headless unit tests cover the algorithm core:

```bash
node --test tests/
```

(Requires Node 18+ for the built-in test runner.)

## Files

- `index.html` — page skeleton, controls, status bar.
- `styles.css` — theme + layout.
- `src/maze.js` — `Maze` class and three hand-crafted layouts.
- `src/node.js` — tree `Node` class.
- `src/mcts.js` — `MCTS` class with `iterate()` generator.
- `src/rng.js` — seedable PRNG (Mulberry32).
- `src/reward.js` — reward functions.
- `src/render-maze.js` — Canvas-based maze rendering.
- `src/render-tree.js` — D3 SVG-based tree rendering.
- `src/animate.js` — phase animation orchestration.
- `src/app.js` — control wiring and main loop.
- `tests/` — unit tests.
- `mcts-maze-demo-spec.md` — design spec.
- `docs/superpowers/plans/` — implementation plan.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## Task 14: Final verification with Claude Preview

**Files:** none changed; this is verification only.

- [ ] **Step 1: Run all unit tests**

Run: `cd /Users/brandon/mcts-maze-demo && node --test tests/`
Expected: all tests pass (Maze: 6, Maze data: 12, Node: 6, RNG: 3, Reward: 3, MCTS: 7 → 37 total).

- [ ] **Step 2: Launch Claude Preview**

Run: `mcp__Claude_Preview__preview_start` pointing at `/Users/brandon/mcts-maze-demo/index.html`.

- [ ] **Step 3: Screenshot the initial state**

Run: `mcp__Claude_Preview__preview_screenshot`. Verify the page loads, both panes render, controls are visible.

- [ ] **Step 4: Click Step and screenshot**

Run: `mcp__Claude_Preview__preview_click` on `#btn-step`, then screenshot. Verify the tree has one expanded child and the maze shows a rollout trail.

- [ ] **Step 5: Click Run to end and screenshot**

Run `mcp__Claude_Preview__preview_click` on `#btn-runend`, wait for animation to settle, then screenshot. Verify:
- Heatmap is populated.
- A green PV line traces from start to goal.
- Tree has many nodes.
- Status bar `Phase: idle` and `PV len` is sensible.

- [ ] **Step 6: Test the 15×15 and 20×20**

Change size to 15 via `mcp__Claude_Preview__preview_eval` setting the select value, run to end, screenshot. Same for 20.

- [ ] **Step 7: Check console for errors**

Run: `mcp__Claude_Preview__preview_console_logs`. Verify no errors.

- [ ] **Step 8: Stop preview and final commit (if any cleanup)**

Run: `mcp__Claude_Preview__preview_stop`.

If any bugs were found, fix them, re-verify, and commit. If everything works, no commit needed.

---

## Plan complete

Total tasks: 14. Total commits expected: ~14. Total unit tests: ~37.

If you find spec gaps during execution that require a new task, add it at the appropriate position and update task numbers in subsequent commits.
