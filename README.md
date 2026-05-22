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

## A note on the algorithm

This demo implements a common practical variant of MCTS for grid-world problems: during **expand**, actions that would revisit any cell already on the current root-to-leaf path are filtered out. The tree therefore stays acyclic by construction.

Textbook MCTS doesn't include this filter — each tree node represents a unique action sequence regardless of the resulting state, so two nodes at the same maze cell are valid distinct nodes. That works fine for game tree search (chess, go) where you can't revisit a position. On a maze, where you can walk back and forth, vanilla MCTS spends significant compute thrashing through cycles before useful structure emerges.

The no-revisit filter is the difference between "MCTS that converges visibly in 100 iterations" and "MCTS that needs thousands of iterations to escape early-cycle noise." For pedagogical clarity, this demo opts for the practical variant. The four phases — select, expand, simulate, backprop — and the UCB1 selection rule are unchanged.

## Debug mode

Append `?debug=1` to the URL for a debug panel showing live tree stats, last rollout details, and a 1000-iteration benchmark button.

## Tests

Headless unit tests cover the algorithm core:

```bash
npm test
# (runs: node --test tests/*.test.js)
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
