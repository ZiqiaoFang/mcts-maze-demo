# Editable Reward Formula Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded reward-mode dropdown with a user-editable JavaScript formula box, with safe compilation, runtime guards, and three stock presets.

**Architecture:** A new `compileReward(formulaString)` factory in `src/reward.js` turns a user expression into a callable via `new Function(...)`, with two sentinel test calls at compile time. The compiled callable is stored as `mcts.config.rewardFn`. `_simulate` builds a context object (`isGoal`, `dist`, `maxDist`, `steps`, `startDist`) and the new `computeReward(compiled, ctx)` wraps the call in a try/catch + `isFinite` guard. The UI replaces the existing `<select id="ctl-reward">` with a formula input + presets menu; bad formulas show inline error and leave the previous compiled formula active.

**Tech Stack:** Vanilla JavaScript (ES modules), `node:test`, Node 26+ for tests, `python3 -m http.server` for browser smoke testing.

**Spec:** `docs/superpowers/specs/2026-06-03-editable-reward-formula-design.md`

---

## File Map

- **Modify**: `src/reward.js` — fully replace with `compileReward`, `PRESETS`, new `computeReward`
- **Modify**: `src/mcts.js` — rename config field `rewardMode` → `rewardFn`, capture `startPos`/`steps`/`startDist`, default to compiled distance-shaped formula for backward compat
- **Modify**: `tests/reward.test.js` — fully replace (remove 3 obsolete tests, add 11 new)
- **Modify**: `tests/mcts.test.js` — update test #4 and add new smoke test
- **Modify**: `index.html` — replace `<select id="ctl-reward">` with formula input + presets `<select>`
- **Modify**: `styles.css` — append `#ctl-reward-formula` and `.formula-group` rules
- **Modify**: `src/app.js` — replace `rewardSel` binding with formula input + presets handlers

---

## Task 1: Rewrite `reward.js` with compiled-formula API (TDD)

**Files:**
- Modify: `tests/reward.test.js` (fully replace)
- Modify: `src/reward.js` (fully replace)

- [ ] **Step 1: Write the failing test file**

Fully replace `/Users/brandon/mcts-maze-demo/tests/reward.test.js` with:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { compileReward, computeReward, PRESETS } from "../src/reward.js";

// --- compileReward ---

test("compileReward returns a callable for a valid formula", () => {
  const fn = compileReward("isGoal ? 1 : 0");
  assert.equal(typeof fn, "function");
  assert.equal(fn(true, 0, 20, 10, 8), 1);
  assert.equal(fn(false, 5, 20, 10, 8), 0);
});

test("compileReward throws on syntax error", () => {
  assert.throws(() => compileReward("isGoal ?? 1 :"), /Bad formula/);
});

test("compileReward throws when sentinel returns non-finite", () => {
  // 1/0 → Infinity for any sentinel
  assert.throws(() => compileReward("1/0"), /non-finite/);
});

test("compileReward throws when sentinel returns non-number", () => {
  assert.throws(() => compileReward("'hello'"), /non-number/);
});

test("compileReward throws when formula references undefined identifier", () => {
  // bogus is not in scope, so this throws at sentinel-call time
  assert.throws(() => compileReward("bogus"), /Formula threw/);
});

// --- Preset behaviour ---

test("PRESETS Distance-shaped: goal=2, halfway~0.5, far=0", () => {
  const fn = compileReward(PRESETS["Distance-shaped"]);
  // goal: isGoal=true, dist=0, maxDist=20 → 1 + max(0, 1-0/20) = 1 + 1 = 2
  assert.equal(fn(true, 0, 20, 10, 8), 2);
  // halfway: dist=10, maxDist=20 → 0 + max(0, 1-10/20) = 0.5
  assert.equal(fn(false, 10, 20, 10, 8), 0.5);
  // far: dist=20, maxDist=20 → 0 + max(0, 1-20/20) = 0
  assert.equal(fn(false, 20, 20, 10, 8), 0);
});

test("PRESETS Pure win/loss: 1 at goal, 0 otherwise", () => {
  const fn = compileReward(PRESETS["Pure win/loss"]);
  assert.equal(fn(true, 0, 20, 10, 8), 1);
  assert.equal(fn(false, 5, 20, 10, 8), 0);
});

test("PRESETS Step penalty: goal=1, non-goal at 50 steps = -0.5", () => {
  const fn = compileReward(PRESETS["Step penalty"]);
  assert.equal(fn(true, 0, 20, 50, 8), 1);
  assert.equal(fn(false, 5, 20, 50, 8), -0.5);
});

// --- Variable plumbing ---

test("formula referencing steps returns correct value", () => {
  const fn = compileReward("steps * 2");
  assert.equal(fn(false, 5, 20, 7, 8), 14);
  assert.equal(fn(false, 5, 20, 100, 8), 200);
});

test("formula referencing startDist returns correct value", () => {
  const fn = compileReward("startDist + 1");
  assert.equal(fn(false, 5, 20, 10, 8), 9);
  assert.equal(fn(false, 5, 20, 10, 0), 1);
});

// --- computeReward runtime safety ---

test("computeReward returns 0 when compiled returns NaN", () => {
  // A compile-time-valid formula that returns NaN on this specific input.
  // (Math.sqrt(-1) → NaN; sentinels use dist=5 and dist=0 so they pass.)
  const fn = compileReward("dist === 7 ? Math.sqrt(-1) : 1");
  assert.equal(computeReward(fn, { isGoal: false, dist: 7, maxDist: 20, steps: 10, startDist: 8 }), 0);
  // Sanity: passes through valid value otherwise.
  assert.equal(computeReward(fn, { isGoal: false, dist: 5, maxDist: 20, steps: 10, startDist: 8 }), 1);
});

test("computeReward returns 0 when compiled throws at runtime", () => {
  // Compile-time-valid (sentinels avoid dist=7), throws when dist=7.
  const fn = compileReward("dist === 7 ? (null).x : 1");
  assert.equal(computeReward(fn, { isGoal: false, dist: 7, maxDist: 20, steps: 10, startDist: 8 }), 0);
  // Sanity: passes through valid value otherwise.
  assert.equal(computeReward(fn, { isGoal: false, dist: 5, maxDist: 20, steps: 10, startDist: 8 }), 1);
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/brandon/mcts-maze-demo
node --test tests/reward.test.js
```

Expected: tests fail with "compileReward is not exported" or similar.

- [ ] **Step 3: Fully replace `src/reward.js`**

Fully replace `/Users/brandon/mcts-maze-demo/src/reward.js` with:

```js
/**
 * Three stock formulas for the UI presets menu. Each is a single JS expression
 * referencing the variables documented in compileReward().
 */
export const PRESETS = {
  "Distance-shaped": "(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)",
  "Pure win/loss":   "isGoal ? 1 : 0",
  "Step penalty":    "isGoal ? 1 : 0 - 0.01 * steps",
};

/**
 * Compile a reward formula string into a callable.
 *
 * The formula is a single JS expression with access to:
 *   isGoal    - boolean: rollout ended at goal cell
 *   dist      - number:  Manhattan distance from rollout end to goal
 *   maxDist   - number:  Manhattan diameter of the maze
 *   steps     - number:  rollout steps actually taken
 *   startDist - number:  Manhattan distance from rollout start to goal
 *
 * Plus the standard Math object.
 *
 * Two sentinel calls are made at compile time; if either throws or returns
 * a non-finite/non-number value, compileReward throws with a friendly message.
 *
 * @param {string} formulaString
 * @returns {(isGoal, dist, maxDist, steps, startDist) => number}
 * @throws {Error} on syntax errors or sentinel-run failures
 */
export function compileReward(formulaString) {
  let fn;
  try {
    fn = new Function(
      "isGoal", "dist", "maxDist", "steps", "startDist",
      `"use strict"; return (${formulaString});`
    );
  } catch (err) {
    throw new Error(`Bad formula: ${err.message}`);
  }

  // Sentinel inputs: cover non-goal + nonzero dist, and goal + zero dist.
  const sentinels = [
    [false, 5, 20, 10, 8],
    [true, 0, 20, 10, 8],
  ];
  for (const args of sentinels) {
    let r;
    try {
      r = fn(...args);
    } catch (err) {
      throw new Error(`Formula threw: ${err.name}: ${err.message}`);
    }
    if (typeof r !== "number") {
      throw new Error(`Formula returned a non-number (${typeof r})`);
    }
    if (!Number.isFinite(r)) {
      throw new Error(`Formula returned a non-finite value`);
    }
  }
  return fn;
}

/**
 * Call a compiled reward function with the per-rollout context.
 * Any thrown exception or non-finite return is silently turned into 0.
 *
 * @param {(isGoal, dist, maxDist, steps, startDist) => number} compiled
 * @param {{isGoal: boolean, dist: number, maxDist: number, steps: number, startDist: number}} ctx
 * @returns {number}
 */
export function computeReward(compiled, ctx) {
  let r;
  try {
    r = compiled(ctx.isGoal, ctx.dist, ctx.maxDist, ctx.steps, ctx.startDist);
  } catch {
    return 0;
  }
  return Number.isFinite(r) ? r : 0;
}
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
cd /Users/brandon/mcts-maze-demo
node --test tests/reward.test.js
```

Expected: 11 tests pass, 0 fail.

- [ ] **Step 5: Commit**

```bash
cd /Users/brandon/mcts-maze-demo
git add src/reward.js tests/reward.test.js
git commit -m "feat: replace mode-switch reward with compiled-formula API"
```

---

## Task 2: Update `mcts.js` to use compiled `rewardFn` (TDD)

**Files:**
- Modify: `tests/mcts.test.js` (update existing test #4, add new smoke test)
- Modify: `src/mcts.js` (rename `rewardMode` → `rewardFn`, capture rollout context, import PRESETS for default)

- [ ] **Step 1: Update `tests/mcts.test.js`**

Edit `/Users/brandon/mcts-maze-demo/tests/mcts.test.js`. Apply both of the following changes:

**Change 1**: Replace the existing import line:

```js
import { Maze, MAZE_10 } from "../src/maze.js";
import { MCTS } from "../src/mcts.js";
```

with:

```js
import { Maze, MAZE_10 } from "../src/maze.js";
import { MCTS } from "../src/mcts.js";
import { compileReward, PRESETS } from "../src/reward.js";
```

**Change 2**: Replace test #4 (the one that says `"MCTS: after many iterations, finds path to goal (PV reaches goal)"`). Replace exactly this block:

```js
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
```

with:

```js
test("MCTS: after many iterations, finds path to goal (PV reaches goal)", () => {
  const m = new MCTS(MAZE_10, {
    seed: 42,
    rolloutHorizon: 40,
    rewardFn: compileReward(PRESETS["Distance-shaped"]),
  });
  for (let i = 0; i < 500; i++) for (const _ev of m.iterate()) { /* drain */ }
  const pv = m.principalVariation();
  // The PV should reach the goal or get very close — assert it reached the goal cell.
  const last = pv[pv.length - 1];
  assert.ok(
    MAZE_10.isGoal(last.position) || MAZE_10.manhattan(last.position, MAZE_10.goal) <= 3,
    `PV final position ${last.position} too far from goal ${MAZE_10.goal}`
  );
});

test("MCTS: configurable rewardFn — runs without error and grows tree", () => {
  // Smoke test for the new rewardFn API on MAZE_10.
  const m = new MCTS(MAZE_10, {
    seed: 7,
    rolloutHorizon: 40,
    rewardFn: compileReward(PRESETS["Distance-shaped"]),
  });
  for (let i = 0; i < 50; i++) for (const _ev of m.iterate()) { /* drain */ }
  const s = m.treeStats();
  assert.ok(s.totalNodes > 1, `expected tree to grow, got ${s.totalNodes} nodes`);
});
```

- [ ] **Step 2: Run tests to verify the smoke test fails**

```bash
cd /Users/brandon/mcts-maze-demo
node --test tests/mcts.test.js
```

Expected: the smoke test (and possibly test #4) fails because `rewardFn` is not a recognized config field yet.

- [ ] **Step 3: Update `src/mcts.js`**

Apply three changes to `/Users/brandon/mcts-maze-demo/src/mcts.js`.

**Change 1**: Update the import line at the top of the file. Replace:

```js
import { computeReward } from "./reward.js";
```

with:

```js
import { compileReward, computeReward, PRESETS } from "./reward.js";

const DEFAULT_REWARD_FN = compileReward(PRESETS["Distance-shaped"]);
```

**Change 2**: Update the constructor config block. Replace:

```js
    this.config = {
      C: config.C ?? 1.41,
      rolloutHorizon: config.rolloutHorizon ?? maze.size * 4,
      rewardMode: config.rewardMode ?? "shaped",
      seed: config.seed ?? 42,
    };
```

with:

```js
    this.config = {
      C: config.C ?? 1.41,
      rolloutHorizon: config.rolloutHorizon ?? maze.size * 4,
      rewardFn: config.rewardFn ?? DEFAULT_REWARD_FN,
      seed: config.seed ?? 42,
    };
```

**Change 3**: Update the `_simulate` method body. Replace:

```js
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
```

with:

```js
  _simulate(startPos) {
    const positions = [[...startPos]];
    let pos = startPos;
    let steps = 0;
    for (let step = 0; step < this.config.rolloutHorizon; step++) {
      if (this.maze.isGoal(pos)) break;
      const actions = this.maze.legalActions(pos);
      if (actions.length === 0) break;
      const action = pick(actions, this.rng);
      pos = this.maze.step(pos, action);
      positions.push([...pos]);
      steps++;
    }
    const ctx = {
      isGoal: this.maze.isGoal(pos),
      dist: this.maze.manhattan(pos, this.maze.goal),
      maxDist: this.maze.maxDistance(),
      steps,
      startDist: this.maze.manhattan(startPos, this.maze.goal),
    };
    const reward = computeReward(this.config.rewardFn, ctx);
    return { positions, reward };
  }
```

- [ ] **Step 4: Run full test suite to verify everything passes**

```bash
cd /Users/brandon/mcts-maze-demo
node --test tests/*.test.js
```

Expected: all tests pass (Maze: 6, maze data: 12, Node: existing count, RNG: existing count, reward: 11, MCTS: 8 — both the updated #4 and the new smoke test).

- [ ] **Step 5: Commit**

```bash
cd /Users/brandon/mcts-maze-demo
git add src/mcts.js tests/mcts.test.js
git commit -m "feat(mcts): use compiled rewardFn and per-rollout context"
```

---

## Task 3: Update `index.html` and `styles.css` for formula input

**Files:**
- Modify: `index.html`
- Modify: `styles.css`

- [ ] **Step 1: Edit `index.html`**

In `/Users/brandon/mcts-maze-demo/index.html`, find this exact block:

```html
      <label>Reward:
        <select id="ctl-reward">
          <option value="shaped" selected>Distance-shaped</option>
          <option value="winloss">Pure win/loss</option>
        </select>
      </label>
```

Replace with:

```html
      <label class="formula-group">Reward:
        <input type="text" id="ctl-reward-formula"
               value="(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)"
               spellcheck="false" autocomplete="off" />
        <select id="ctl-reward-preset" title="Load a preset">
          <option value="" disabled selected>Presets ▾</option>
          <option value="Distance-shaped">Distance-shaped</option>
          <option value="Pure win/loss">Pure win/loss</option>
          <option value="Step penalty">Step penalty</option>
        </select>
      </label>
```

- [ ] **Step 2: Append CSS rules to `styles.css`**

Append to the end of `/Users/brandon/mcts-maze-demo/styles.css`:

```css
#ctl-reward-formula {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  width: 360px;
  background: var(--bg);
  color: var(--fg);
  border: 1px solid var(--border);
  padding: 0.35rem 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}
#ctl-reward-formula.invalid { border-color: #ef4444; }
.formula-group { gap: 0.5rem; }
```

- [ ] **Step 3: Verify HTML still parses**

```bash
cd /Users/brandon/mcts-maze-demo
xmllint --html --noout index.html 2>&1 || echo "WARN: xmllint not available or HTML had warnings"
```

Expected: exit code 0 (or `xmllint not available` if the tool is missing). HTML warnings about unclosed `<input>` or `<meta>` are normal and OK.

- [ ] **Step 4: Commit**

```bash
cd /Users/brandon/mcts-maze-demo
git add index.html styles.css
git commit -m "feat(ui): formula input and presets menu markup + styles"
```

---

## Task 4: Update `app.js` to wire up formula input

**Files:**
- Modify: `src/app.js`

- [ ] **Step 1: Update imports**

In `/Users/brandon/mcts-maze-demo/src/app.js`, find the existing import block at the top:

```js
import { animateIteration } from "./animate.js";
import { MAZES } from "./maze.js";
import { MCTS } from "./mcts.js";
import { MazeRenderer } from "./render-maze.js";
import { TreeRenderer } from "./render-tree.js";
```

Replace with:

```js
import { animateIteration } from "./animate.js";
import { MAZES } from "./maze.js";
import { MCTS } from "./mcts.js";
import { MazeRenderer } from "./render-maze.js";
import { TreeRenderer } from "./render-tree.js";
import { compileReward, PRESETS } from "./reward.js";
```

- [ ] **Step 2: Replace the `rewardSel` binding**

Find this line:

```js
const rewardSel = document.getElementById("ctl-reward");
```

Replace with:

```js
const formulaInput = document.getElementById("ctl-reward-formula");
const presetSel = document.getElementById("ctl-reward-preset");
let currentRewardFn = null;
```

- [ ] **Step 3: Update `newMcts()` to use `currentRewardFn`**

Find this block:

```js
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
```

Replace with:

```js
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
```

- [ ] **Step 4: Add `applyFormula` helper and event handlers**

Find this line:

```js
rewardSel.addEventListener("change", newMcts);
```

Replace with:

```js
function applyFormula(formulaString) {
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
```

- [ ] **Step 5: Replace the bootstrap call**

Find this line near the bottom:

```js
newMcts();
```

(It appears once, after all event-listener wiring, before the theme-toggle handler.)

Replace with:

```js
applyFormula(formulaInput.value);
```

This compiles the default formula from the input value, stores `currentRewardFn`, and calls `newMcts()` to build the initial MCTS — replacing the old direct `newMcts()` bootstrap.

- [ ] **Step 6: Run full test suite (sanity check that nothing JS-side broke)**

```bash
cd /Users/brandon/mcts-maze-demo
node --test tests/*.test.js
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/brandon/mcts-maze-demo
git add src/app.js
git commit -m "feat(app): wire formula input + presets handlers"
```

---

## Task 5: Manual browser verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the local server**

```bash
cd /Users/brandon/mcts-maze-demo
python3 -m http.server 8765 &
SERVER_PID=$!
sleep 1
echo "Server up at http://localhost:8765 (PID $SERVER_PID)"
```

- [ ] **Step 2: Open in browser**

Open http://localhost:8765 in any modern browser.

- [ ] **Step 3: Verify default formula is active**

The Reward input should show:

```
(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)
```

The presets dropdown should read "Presets ▾". Click **Step** several times; the tree should grow and the PV should start trending toward the goal — same behavior as the old "Distance-shaped" default.

- [ ] **Step 4: Verify presets populate the input and reset the tree**

Click the Presets menu and pick "Pure win/loss". Verify:
- The input box now reads `isGoal ? 1 : 0`.
- The presets menu label returns to "Presets ▾".
- The tree was reset (status bar shows Iter: 0 → 1 → ... as you click Step).

Then pick "Step penalty" and verify the input reads `isGoal ? 1 : 0 - 0.01 * steps`.

- [ ] **Step 5: Verify syntax-error feedback**

Type `isGoal ?? 1 :` into the input and press Enter. Verify:
- The input gets a red border.
- The status bar shows `reward error: Bad formula: ...`.
- Hovering the input shows the same message as a tooltip.
- The PV continues to update (the previously-valid formula is still active) when you click Step.

- [ ] **Step 6: Verify sentinel-failure feedback**

Type `1/0` into the input and press Enter. Verify:
- Red border + `reward error: Formula returned a non-finite value`.

- [ ] **Step 7: Verify recovery from error**

With the red-border error still showing, type `isGoal ? 1 : 0` and press Enter. Verify:
- Red border clears.
- Status bar returns to `idle`.
- Tree resets and starts growing under the new formula.

- [ ] **Step 8: Stop the server**

```bash
kill $SERVER_PID 2>/dev/null
echo "Server stopped"
```

- [ ] **Step 9: Commit nothing (manual verification only)**

No commit. If any of steps 3–7 fail, fix the underlying issue in the relevant file and re-run the failing step.
