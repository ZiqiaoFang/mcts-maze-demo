# MCTS Maze Demo — Spec

## 1. Purpose

A single-page, browser-based interactive visualization of Monte Carlo Tree Search (MCTS) running on a small grid-world maze. Built to make the four MCTS phases — **select**, **expand**, **simulate**, **backpropagate** — concretely visible, so a learner can step through one iteration at a time and see how the search tree grows, how rollouts produce signal, and how that signal flows back up the tree.

Pedagogical, not production. The deliverable is a learning artifact, not a library.

## 2. Scope

### In scope

- Interactive HTML+JS visualization runnable by opening `index.html` in any modern browser.
- Three hand-crafted mazes (10×10, 15×15, 20×20), each with one start cell, one goal cell, and at least two viable paths.
- Step-by-step controls: step one iteration, run 10, run to completion, reset.
- Configurable algorithm parameters: UCB1 exploration constant `C`, rollout horizon, reward mode (distance-shaped or pure win/loss).
- Side-by-side rendering: maze on the left (Canvas), MCTS search tree on the right (D3 SVG).
- Animated transitions for each MCTS phase, with phase indicator in a status bar.
- Heatmap overlay on the maze showing visit counts, and a principal-variation overlay showing the current best path.
- Hidden debug panel toggled by `?debug=1`.

### Out of scope

- Mobile/touch layout — desktop browser only.
- A formal test suite. Sanity-checking via the debug panel only.
- Multiple agents, multi-step games, or anything beyond a single-agent grid-world.
- Server-side anything. Pure static files.
- Persistence of sessions — every reload starts fresh.

## 3. User experience

### Page layout

```
┌─────────────────────────────────────────────────────────┐
│  MCTS Maze Demo                          [dark/light]   │
│  Iter: 23 • Tree nodes: 47 • PV len: 18 • Phase: ...    │
├──────────────────────────┬──────────────────────────────┤
│                          │                              │
│       MAZE (canvas)      │     SEARCH TREE (SVG)        │
│                          │                              │
│                          │                              │
├──────────────────────────┴──────────────────────────────┤
│ Controls                                                │
│   Maze size: [10▼]   Reward: [distance-shaped▼]         │
│   UCB1 C:  [─●─] 1.41    Rollout horizon: [─●─] 40      │
│   [ Step ]  [ Run 10 ]  [ Run to end ]  [ Reset ]       │
└─────────────────────────────────────────────────────────┘
```

### Controls behavior

- **Maze size dropdown**: switching size resets the MCTS state and loads the corresponding hand-crafted maze.
- **Reward mode dropdown**: changing this resets the tree (rewards from the old mode are not comparable).
- **UCB1 C slider** (range 0–3, default ≈1.41): applies to the next selection; does not invalidate the tree.
- **Rollout horizon slider** (range 1 to 8×maze_size, default 4×maze_size): applies to the next iteration; does not invalidate the tree. Label notes "applies to next iteration".
- **Step**: run one full MCTS iteration with all four phase animations (~2.5s total).
- **Run 10**: run 10 iterations without per-phase animation; brief loading shimmer, then renders final state.
- **Run to end**: run until either the principal variation reaches the goal AND has been stable for 5 iterations, or a hard cap of 500 iterations is hit. Same no-animation rendering as Run 10.
- **Reset**: clear the tree, return to root-only state.

### Phase animations (Step mode)

| Phase     | Duration | What you see                                                     |
|-----------|----------|------------------------------------------------------------------|
| select    | 400ms    | Tree path flashes yellow from root downward; maze cells outline  |
| expand    | 200ms    | New tree node pops in (scale 0→1) with a soft halo               |
| simulate  | up to 1.5s | Orange trail walks the maze cell by cell (fast-forwards if >30 steps) |
| backprop  | 400ms    | Reward number flies up the tree; nodes pulse as they update      |

Status bar's "Phase" field updates in real time and is visible during animation.

## 4. Algorithm

### MCTS loop, per iteration

1. **Select**: starting at the root, repeatedly choose the child with the highest UCB1 score until reaching either (a) a node with at least one untried action or (b) a terminal node (position is goal).
2. **Expand**: if the selected node has untried actions, pick one, apply it to get a new position, create a child node, mark the action as tried.
3. **Simulate (rollout)**: from the expanded node's position, take random legal actions for up to `rolloutHorizon` steps, or until the goal is reached, whichever comes first. Compute reward.
4. **Backpropagate**: walk from the expanded node back up to the root, incrementing `visits` and adding the reward to `totalReward` at each node along the way.

### UCB1

```
UCB1(child) = meanReward(child) + C * sqrt(ln(parent.visits) / child.visits)
```

For unvisited children, UCB1 is treated as `+Infinity` so they are tried at least once before any UCB1 comparison.

### Reward modes

- **Distance-shaped**: reward = `clip(1 - finalManhattanDistanceToGoal / maxDistance, 0, 1)`, plus `+1` bonus if the rollout reached the goal. `maxDistance` is the Manhattan diameter of the maze. Range: `[0, 2]`.
- **Pure win/loss**: reward = `1` if rollout reached the goal, else `0`. Range: `{0, 1}`.

Mean reward (not win-rate) is used in UCB1's exploit term, because shaped reward is continuous. UCB1 still works since the value is bounded; only the constant `C` may need to be larger to compensate for the smaller value range when in pure-win-loss mode.

### Determinism

- Maze layouts are baked-in literals — not generated.
- Random rollouts use a seedable PRNG (default seed = 42; reset re-seeds with current seed). A future enhancement (not in scope here) could expose the seed in the UI.

## 5. Data model

### `Maze` (src/maze.js)

```js
class Maze {
  size;            // 10 | 15 | 20
  grid;            // number[][], 0 = open, 1 = wall
  start;           // [row, col]
  goal;            // [row, col]
  legalActions(pos);          // returns array of [dr, dc] excluding walls
  step(pos, action);          // returns [row+dr, col+dc]
  isGoal(pos);                // returns boolean
  manhattan(a, b);            // helper
  maxDistance();              // Manhattan diameter; cached
}
```

Three hand-crafted maze grids are exported as constants: `MAZE_10`, `MAZE_15`, `MAZE_20`.

### `Node` (src/node.js)

```js
class Node {
  position;          // [row, col]
  parent;            // Node | null
  action;            // [dr, dc] | null — action taken from parent
  children;          // Map<"dr,dc", Node>
  untriedActions;    // Array<[dr,dc]> — populated lazily on first visit
  visits = 0;
  totalReward = 0;

  meanReward();       // totalReward / visits, or 0 if visits == 0
  ucb1(C, parentVisits);  // returns +Infinity if visits == 0
  isFullyExpanded();
  isTerminal(maze);
  bestChild();         // most-visited child (for PV extraction)
}
```

### `MCTS` (src/mcts.js)

```js
class MCTS {
  maze;
  root;
  config;              // { C, rolloutHorizon, rewardMode, rng }
  iterationCount = 0;

  *iterate();          // generator yielding MCTSEvent per phase
  principalVariation();  // returns array of Nodes from root to leaf via bestChild
  treeStats();           // { totalNodes, maxDepth, maxVisits }
}

// Event shape (one per phase):
//   { phase: "select",    path: Node[] }
//   { phase: "expand",    newNode: Node }
//   { phase: "simulate",  positions: [r,c][], reward: number }
//   { phase: "backprop",  path: Node[], reward: number }
```

The generator design lets the rendering layer pause between phases for animation, while "Run 10" mode just consumes the generator without animating.

## 6. Rendering

### Maze pane (Canvas)

- Cell grid drawn with `fillRect`. Open cells = light gray (or near-black in dark mode); walls = dark slate (or light gray).
- Start = green dot at cell center; goal = red flag glyph.
- **Visit heatmap (always on)**: each open cell tinted by `visits / maxVisitsAmongCells`, white-to-blue gradient. Re-normalized every frame.
- **Principal variation overlay (always on)**: thick green polyline through the cells on the PV. Updated after each `backprop`.
- **Selection outline (during `select` phase)**: cells corresponding to the UCB1-chosen path get a yellow outline for the duration of the phase.
- **Rollout trail (during `simulate` phase)**: orange polyline animates cell-by-cell at ~40ms/step. If the rollout is longer than 30 steps, the remaining steps render in a single fast batch.

All canvas redraws are throttled via `requestAnimationFrame`.

### Tree pane (SVG, D3)

- Layout via `d3.tree()`, top-down. Re-laid out on every `expand` event (not on backprops).
- **Node visuals**:
  - Radius proportional to `Math.sqrt(visits) * baseRadius`.
  - Fill color interpolated by `meanReward`: white → deep green.
  - Stroke: gold during selection on selected nodes, gray otherwise.
- **Edges**: thin gray lines. Edge traversed during `select` flashes yellow then fades.
- **Hover tooltip**: shows `position`, `visits`, `mean reward`, `UCB1 score (relative to parent)`.
- **Zoom/pan**: `d3.zoom()` for mouse-wheel zoom and click-drag pan.

If tree node count exceeds 500 (likely only on 20×20 with many iterations), display a "compact tree" toggle that hides nodes with `visits < 2`.

### Status bar

```
Iter: 23  •  Tree nodes: 47  •  Best path length: 18  •  Phase: simulate
```

`Best path length` = depth of the PV. Turns green when PV ends at the goal.

### Theme

Dark mode by default. Light mode toggle in top right swaps a single CSS variable set. Color palette inspired by scientific viz (viridis-ish for the heatmap; orange/green/yellow accents).

## 7. File layout

```
mcts-maze-demo/
├── index.html
├── styles.css
├── src/
│   ├── maze.js          // Maze class + MAZE_10/15/20
│   ├── node.js          // Tree Node class
│   ├── mcts.js          // MCTS class, iterate() generator
│   ├── render-maze.js   // Canvas rendering
│   ├── render-tree.js   // D3 SVG rendering
│   ├── animate.js       // Phase animation orchestration, easing
│   └── app.js           // Glue: controls → MCTS → renderers
└── README.md
```

### Dependencies

- **D3 v7** via CDN `<script>` tag.

No build step, no npm install, no TypeScript, no framework.

## 8. Build order

1. **Skeleton** — `index.html` + `styles.css` two-pane layout with dummy controls.
2. **Maze + canvas** — `maze.js`, `render-maze.js`. Static maze renders; size dropdown switches mazes.
3. **MCTS core (no rendering)** — `node.js`, `mcts.js`. Headless test: log PV after 100 iterations.
4. **Tree renderer** — `render-tree.js`. Tree displays correctly post-search.
5. **Animation layer** — `animate.js` + wiring. Step button works with all four phase animations.
6. **All controls** — Run 10, Run to end, Reset, sliders, dropdowns.
7. **Polish** — heatmap, PV overlay, status bar, tooltips, debug panel, dark/light toggle, perf pass.

Each step is verified manually in the browser before moving on.

## 9. Verification

- After each build step, open in the browser and confirm the new functionality works as described.
- Final verification uses **Claude Preview** (`mcp__Claude_Preview__preview_start`) to launch the page in an automated browser; screenshots are captured and reviewed before handing off.
- Debug panel (`?debug=1`) exposes:
  - Tree statistics live (total nodes, max depth, max visits)
  - Last rollout details (start, end, reward, step count)
  - "Run 1000 iterations and log timing" button for a performance sanity check.

## 10. Out-of-scope but worth flagging for future work

- Seedable PRNG control in the UI (not just hard-coded default).
- "Compare two configurations side-by-side" view.
- Save/load tree state.
- Touch + mobile responsive layout.
- An evaluator network (AlphaZero-style) replacing the random rollout.
