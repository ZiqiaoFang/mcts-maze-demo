# Editable Reward Formula — Design

## 1. Purpose

The MCTS Maze Demo currently exposes two hard-coded reward modes (`shaped` and `winloss`) behind a dropdown. This change replaces the dropdown with a free-form formula input so a learner can experiment with arbitrary reward shapes (step penalties, exponential shaping, progress-from-start, etc.) without editing code or reloading the page.

The change is pedagogical: seeing how UCB1 + rollout-based learning responds to different reward signals is one of the central insights MCTS is supposed to teach.

## 2. Scope

### In scope

- Replace the `<select id="ctl-reward">` dropdown with a single-line formula input + a "Presets ▾" menu.
- Define a small expression language: one JavaScript expression with a fixed set of variables (`isGoal`, `dist`, `maxDist`, `steps`, `startDist`) plus the standard `Math` object.
- Compile formulas via `new Function(...)`, with two sentinel test calls at compile time to reject formulas that don't return finite numbers.
- Defensive runtime: rollouts whose formula throws or returns non-finite values are silently scored 0; the search continues.
- Three stock presets in the Presets menu: *Distance-shaped*, *Pure win/loss*, *Step penalty*.
- Plumb the new variables (`steps`, `startDist`) through `mcts.js` to `reward.js`.
- Update `tests/reward.test.js` and add an MCTS integration smoke test.

### Out of scope

- A formal expression grammar / non-JS language.
- Saving/loading custom formulas across reloads.
- Headless DOM tests for the input UI (manual browser verification).
- Per-formula `C` (UCB1 exploration constant) auto-tuning.
- Surfacing runtime formula crashes to the UI.

## 3. UX

### Layout

The existing `.control-row` containing `[Maze size] [Reward]` keeps its place. The Reward control changes from a single `<select>` to a labeled group:

```
Reward: [ (isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist) ] [ Presets ▾ ]
```

- The formula input is `~360px` wide, monospace, single-line.
- The Presets menu is a small `<select>` that resets its label back to `Presets ▾` after each pick (so it acts like an action menu rather than persistent state).
- The group is inside a `<label class="formula-group">` so the prefix "Reward:" stays attached.
- On narrow widths the row wraps; the formula input wraps to its own line.

### Apply behavior

- Pressing **Enter** in the input fires `change`.
- **Blur** (focus loss) also fires `change`.
- Picking a preset populates the input and fires `change`.
- On every `change`: try to compile and apply.

### Error feedback

- **Compile error or sentinel failure** — the input gets a red `1px` border (`.invalid` class), its `title` attribute is set to the error message (visible on hover), and the status bar's "Phase" field shows `reward error: <message>` until the formula is fixed. The previously-valid compiled formula stays active.
- **Runtime errors during rollouts** are NOT surfaced (the rollout silently scores 0).

### Tree lifecycle

A successful formula change resets the MCTS tree (same as today's reward dropdown change). The visit counts and rewards from the old formula are not transferable.

## 4. Formula language

A formula is a single JavaScript expression — no statements, no `return`, no semicolons.

### Variables exposed

| Name        | Type    | Meaning                                                           |
|-------------|---------|-------------------------------------------------------------------|
| `isGoal`    | boolean | True if the rollout's final position is the goal                  |
| `dist`      | number  | Manhattan distance from the final rollout position to the goal    |
| `maxDist`   | number  | Manhattan diameter of the maze (`maze.maxDistance()`)             |
| `steps`     | number  | Number of rollout steps actually taken (≤ `rolloutHorizon`)       |
| `startDist` | number  | Manhattan distance from the rollout's start position to the goal  |

### Globals exposed

- `Math` — the standard `Math` object (so `Math.exp`, `Math.sqrt`, etc. work).
- All other JS globals technically reachable from `new Function`-built code are present (`window`, `document`, …) but we do not document them or rely on them. The threat model is "user accidentally crashes the page", not malicious input.

### Stock presets

The Presets menu has three entries; each is the literal formula string that will be loaded into the input box:

- **Distance-shaped** — `(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)`
- **Pure win/loss** — `isGoal ? 1 : 0`
- **Step penalty** — `isGoal ? 1 : 0 - 0.01 * steps`

### Reward range

Rewards are not clamped or normalized. UCB1 only requires that rewards be bounded; very different magnitudes will require the user to retune the `C` slider. This is intentional — the UI does not surface it.

## 5. Evaluation & safety

### Compilation

```js
const fn = new Function(
  "isGoal", "dist", "maxDist", "steps", "startDist",
  `"use strict"; return (${formulaString});`
);
```

The wrapping parens around `${formulaString}` ensure a single expression. `"use strict"` rejects accidental `with` blocks and silent globals.

### Sentinel test runs at compile time

Immediately after `new Function` succeeds, call the function twice:

- `fn(false, 5, 20, 10, 8)` — a generic non-goal rollout context.
- `fn(true, 0, 20, 10, 8)` — a generic goal-reaching rollout context.

Both calls must return finite numbers (`Number.isFinite`). Otherwise `compileReward` throws:

- A thrown exception → `"Formula threw: <name>: <message>"`
- A non-finite return → `"Formula returned a non-finite value"`
- A non-number return → `"Formula returned a non-number (<typeof>)"`

The two sentinel inputs are chosen to cover the goal/non-goal branch of common ternaries and to give `dist` both a nonzero value and zero (catching `1/dist`-style bugs).

### Runtime safety

The active rollout reward call is:

```js
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

- A thrown error returns `0` for that one rollout.
- A NaN or Infinity return returns `0` for that one rollout.
- The search loop is never aborted by a bad formula.

### Failure-mode summary

| Failure                               | Caught where        | UI feedback                       | Search effect          |
|---------------------------------------|---------------------|-----------------------------------|------------------------|
| Syntax error                          | Compile time        | Red border + status bar msg       | Old formula stays active |
| Bad sentinel output                   | Compile time        | Red border + status bar msg       | Old formula stays active |
| Runtime throw on some rollouts        | Per-rollout try/catch | None                            | Bad rollouts → 0       |
| NaN/Infinity on some rollouts         | `isFinite` guard    | None                              | Bad rollouts → 0       |
| Always-zero / nonsense valid formula  | N/A — valid         | None                              | Search runs but won't learn |

## 6. Wiring

### `src/reward.js` — replace `mode`-switch with compiled-formula API

The current file (one function `computeReward(maze, finalPos, mode)`) is fully replaced with:

```js
export const PRESETS = {
  "Distance-shaped": "(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)",
  "Pure win/loss":   "isGoal ? 1 : 0",
  "Step penalty":    "isGoal ? 1 : 0 - 0.01 * steps",
};

export function compileReward(formulaString) { /* see Section 5 */ }

export function computeReward(compiled, ctx) { /* see Section 5 */ }
```

### `src/mcts.js`

Two changes:

1. **Config field rename**: `rewardMode` (string) → `rewardFn` (compiled callable). The `setConfig` partial-update mechanism is unchanged.
2. **Per-rollout context capture**: in the rollout/simulate path, capture the start position before the random walk and the actual number of steps taken. At rollout end, build a context object and call `computeReward(this.config.rewardFn, ctx)`:

```js
const ctx = {
  isGoal: maze.isGoal(finalPos),
  dist: maze.manhattan(finalPos, maze.goal),
  maxDist: maze.maxDistance(),
  steps: rolloutSteps,
  startDist: maze.manhattan(rolloutStartPos, maze.goal),
};
```

The `iterate()` event stream shape is unchanged.

### `index.html`

Inside the existing Maze-size/Reward `.control-row`, replace the reward `<select>` with:

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

### `styles.css`

Add (no existing rules are removed):

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

### `src/app.js`

- Replace `rewardSel` and its handler with `formulaInput` and `presetSel`.
- On page load: compile the default formula, store the compiled `rewardFn`, pass it into `newMcts()`.
- On formula `change` (Enter, blur, or preset selection): try `compileReward`. On success → clear `.invalid`, clear status-bar error, store new `currentRewardFn`, call `newMcts()`. On failure → add `.invalid`, set `input.title`, set status bar to `reward error: <msg>`.
- Picking a preset populates the input AND resets the preset menu's selected value to `""` so its label returns to "Presets ▾".

### Files NOT touched

`src/maze.js`, `src/node.js`, `src/rng.js`, `src/animate.js`, `src/render-maze.js`, `src/render-tree.js`. None of them know what a reward is.

## 7. Tests

All test changes are in `tests/reward.test.js` (replaced wholesale) and `tests/mcts.test.js` (one new smoke test).

### Removed (existing 3)

The current `computeReward(maze, finalPos, "shaped" | "winloss")` tests no longer apply — the signature is changing.

### New in `tests/reward.test.js`

1. `compileReward` returns a callable for a valid formula.
2. `compileReward` throws on a syntax error (e.g., `"isGoal ?? 1 :"`).
3. `compileReward` throws on a sentinel-failing formula (e.g., `"1/0"` returns Infinity).
4. `compileReward` throws on a non-number return (e.g., `"'hello'"`).
5. `PRESETS["Distance-shaped"]` — goal at dist=0 returns 2; halfway returns ~0.5; far returns 0.
6. `PRESETS["Pure win/loss"]` — goal returns 1; non-goal returns 0.
7. `PRESETS["Step penalty"]` — goal at 50 steps returns 1; non-goal at 50 steps returns `-0.5`.
8. A formula referencing `steps` returns the right value for different step counts.
9. A formula referencing `startDist` returns the right value for different start distances.
10. `computeReward(compiled, ctx)` returns `0` when the compiled function returns `NaN`.
11. `computeReward(compiled, ctx)` returns `0` when the compiled function throws at runtime.

### New in `tests/mcts.test.js`

12. Smoke test: configure MCTS with a compiled `PRESETS["Distance-shaped"]` callable, run 50 iterations on `MAZE_10`, assert `treeStats().totalNodes > 1` and no exception.

### Manual verification (not automated)

- Type each preset into the input, press Enter, verify the tree resets and starts growing.
- Type an invalid formula (`"isGoal ?"`), verify red border + status bar message + previous formula still active.
- Type a formula that always returns 0, verify no error and tree fills with low-confidence visits.

## 8. Migration

This is an irreversible API change to `reward.js` and a config-field rename in `mcts.js`. No persisted state exists (the demo doesn't save anything across reloads), so no migration is needed.

## 9. Out-of-scope but worth flagging for future work

- Persist the user's last-used custom formula in `localStorage`.
- Add a "share this formula" URL parameter so a learner can send a colleague the exact reward function and maze.
- Surface mean reward / PV-changed-this-iteration in the status bar to make formula effects more legible.
- A small "Help" tooltip near the formula input listing the five variables and their meanings.
