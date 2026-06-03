# AlphaZero MCTS Tab — Design

**Status:** Draft, pending implementation
**Date:** 2026-06-03
**Author:** brandonfang412@gmail.com (with Claude)

## 1. Motivation

The existing demo teaches **classic MCTS**: tree search with UCB1 selection and uniform-random rollouts. After working through it, a learner naturally asks two questions:

1. Why does MCTS waste iterations exploring obviously bad branches?
2. Why does the agent struggle on larger mazes even after many iterations?

The answers — UCB1's `+∞` for unvisited children, and noisy rollout signal under sparse reward — motivate **AlphaZero-style MCTS**, which replaces both the uniform exploration and the random rollout with a learned neural network providing a policy prior `p(a|s)` and a value estimate `v(s)`.

This spec adds a second tab to the demo so students can:

- Learn classic MCTS first (existing tab, unchanged).
- Then switch to the **AlphaZero MCTS** tab to see how a trained network changes the search behavior, and to watch the network itself learn from self-play.

## 2. Scope

**In scope:**

- New top-level tab switcher: `Classic MCTS` | `AlphaZero MCTS`.
- AZ tab with: maze pane, MCTS tree pane, charts panel, controls.
- TensorFlow.js-backed small MLP providing `p` and `v` from raw maze state.
- AZ-MCTS variant using **PUCT** selection and network-evaluated leaves (no rollouts).
- Self-play loop with replay buffer and online Adam training.
- User-editable `z` formula (the AZ-tab analog of the existing reward formula), with `Pure win/loss` as default.
- Visualizations: `v`-heatmap maze overlay, `p`-arrows maze overlay, `z`-per-iteration curve, training-loss curves, probe-state panel for the start cell.

**Out of scope:**

- Heavy playout variants (goal-biased, no-revisit) in the Classic tab — deferred to a future change.
- BFS-oracle "fake AZ" stand-in — the user opted for true training.
- CNN architectures (Conv2D) — start with a flat MLP; can extend later.
- Curriculum learning, intrinsic motivation, domain randomization across mazes — discussed in the conversation but not implemented in v1.
- Server-side training. Everything runs in the browser.

## 3. UI Architecture

### 3.1 Tab switcher

A new strip of tab buttons sits above the existing header:

```
┌──────────────────────────────────────────────────┐
│  [ Classic MCTS ]  [ AlphaZero MCTS ]            │
└──────────────────────────────────────────────────┘
│  (existing header: title, status bar, theme)     │
```

- Tabs are independent: switching does not reset the other tab's MCTS or network.
- The maze size dropdown is shared (mounts in both tabs) so the same maze is rendered in either view.
- Theme toggle stays in the global header.

### 3.2 AZ tab layout

```
┌────────────────────────┬────────────────────────┐
│   MAZE                 │   MCTS SEARCH TREE     │
│   (overlay-toggleable) │   (PUCT-based)         │
└────────────────────────┴────────────────────────┘
┌──────────────────────────────────────────────────┐
│  CHARTS PANEL                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │
│  │  z curve     │ │  loss curves │ │  probe   │ │
│  │  per game    │ │  per step    │ │  (start) │ │
│  └──────────────┘ └──────────────┘ └──────────┘ │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│  CONTROLS (z formula, sliders, buttons)          │
└──────────────────────────────────────────────────┘
```

### 3.3 Maze overlays (radio group above maze)

| Mode | Rendering |
|---|---|
| `v(s)` heatmap (default) | For each cell `c`, compute `v(state_with_agent_at_c)`; color cells by value, blue→red. |
| `p(a\|s)` arrows | For each cell `c`, evaluate `p(·\|state_with_agent_at_c)`; draw four arrows whose lengths are proportional to the four action probabilities. |
| MCTS visits | The existing classic-mode overlay (visit-count heatmap of the current search tree). |
| Off | No overlay; just walls and agent/goal. |

Overlays redraw after every self-play game ends. The cost is `size² ≈ 100–400` forward passes per redraw; at ~25µs/forward in TF.js for an 80k-param MLP, that is well under 20 ms — fine.

### 3.4 Charts panel

| Chart | x-axis | y-axis | Notes |
|---|---|---|---|
| **z curve** | Self-play game index | `z` value | Faint dots per game + bold moving-average (window 20). |
| **Loss curves** | Gradient step | Loss | Three lines: total, policy cross-entropy, value MSE. |
| **Probe (start cell)** | n/a (static panel) | n/a | 4-bar SVG of `p(action\|start)` + scalar `v(start)` shown below as text. Refreshed every iteration. |

Implementation: minimal canvas line-chart helper (~80 LOC). No d3 — the existing tree pane already uses d3, but adding more d3 charts is overkill for time-series plots.

### 3.5 Controls

```
[ z formula: __________________ ]  [ Presets ▾ ]

PUCT c_puct: [ slider ]   value: 1.00
MCTS sims/move: [ slider ]   value: 50
Learning rate: [ slider ]   value: 1e-3

[ Self-play step ] [ Run 10 ] [ Run continuously ] [ Reset MCTS ] [ Reset network ]
```

**`z` formula presets:**

- `Pure win/loss` (default) — `isGoal ? 1 : -1`
- `Distance-shaped` — `isGoal ? 1 : (1 - dist / maxDist)`
- `Step penalty` — `isGoal ? Math.max(0, 1 - steps / (maxDist * 4)) : -1`

The formula compiler from `reward.js` is reused; only the available identifiers change. The formula is evaluated **once per self-play game** at the terminal state, producing a single scalar `z` that labels every `(state, π)` tuple from that game.

**Slider ranges:**

- `c_puct`: 0.1 – 4.0, step 0.05
- MCTS sims/move: 10 – 400, step 10
- Learning rate: 1e-5 – 1e-2 on log scale

**Buttons:**

- **Self-play step**: run one full self-play game + one minibatch training pass.
- **Run 10**: ten self-play steps, yielding to the browser between games (`requestAnimationFrame`-paced) so the UI stays responsive and overlays update game-by-game.
- **Run continuously**: same as Run 10 but unbounded; loops until user clicks the stop button (which replaces this button while running).
- **Reset MCTS**: clear the in-progress search tree only. Network preserved.
- **Reset network**: confirmation dialog. Re-initializes weights, empties replay buffer, clears all charts. Network shape preserved.

### 3.6 Status bar additions (AZ tab)

The existing status bar gains AZ-specific fields when the AZ tab is active:

```
Games: 42   Latest z: +1.00   z̄(20): -0.31   Loss: 0.84   Buffer: 1280
```

## 4. Network

### 4.1 Architecture

```
input:  size × size × 3            (channels: wall mask, agent one-hot, goal one-hot)
        ↓ flatten
        ↓ dense(64, relu)
        ↓ dense(64, relu)
        ↓
   shared trunk
        ├── dense(4, softmax)  → p(a|s)
        └── dense(1, tanh)     → v(s)
```

- Built with `tf.layers` as a functional model with two outputs.
- Parameter count on 20×20: ≈ 400·3·64 + 64·64 + 64·4 + 64·1 ≈ 81 k.
- Channel encoding: every input is rendered fresh from `(maze, agent_position)`; the network is stateless.

### 4.2 Loss

```
L = MSE(v, z)  +  CategoricalCrossEntropy(π, p)  +  λ·‖θ‖²
```

with `λ = 1e-4`. Optimizer: Adam with default β values; learning rate from the slider.

### 4.3 Maze-size and re-init

The input dimension depends on the maze size, so switching the maze size requires reinitializing the network. Implementation: when the size dropdown changes while the AZ tab is active, prompt:

> "Changing maze size will reset the AlphaZero network and replay buffer. Continue?"

Switching tabs does not reset anything.

## 5. AZ-MCTS Variant

### 5.1 PUCT selection

At an internal node with parent visit count `N`:

```
PUCT(child) = Q(child) + c_puct · p(a|s) · sqrt(N) / (1 + N_child)
```

`Q(child)` is `W_child / N_child` (mean backed-up `v` over visits to this subtree). For a child with `N_child = 0` (no visits yet) we define `Q(child) = 0` in the formula — so the prior alone decides which unvisited child gets tried first. Unlike classic UCB1, unvisited children do **not** get `+∞`.

### 5.2 Expand

When MCTS selects a leaf for expansion:

1. Evaluate the network once on the leaf state. Get `v_leaf` and `p_leaf` (a 4-vector).
2. For every legal action `a` from the leaf, create a child node carrying the prior `p_leaf[a]` (re-normalized over legal actions only).
3. **No rollout.** Back up `v_leaf` along the selection path.

Illegal actions are masked from both the prior and the children — no priors are wasted on wall-bound moves.

### 5.3 Tree lifetime

Canonical AZ resets the search tree between self-play moves: the search tree is for the *current* root state only. We follow the canonical behavior; sub-tree reuse can be added as a future optimization.

### 5.4 Cycle handling

Classic-mode MCTS in this demo filters ancestor-revisiting actions to avoid infinite cycles in the tree. AZ-MCTS retains this filter for the same reason: a cycle in the tree breaks the policy-improvement property of MCTS.

## 6. Self-play Loop

One **Self-play step** does the following:

```
1. agent_pos = maze.start; trajectory = []
2. for t in 0..maxSteps:
       if maze.is_goal(agent_pos): break
       root = build_root(agent_pos)
       run PUCT-MCTS for `simsPerMove` simulations using current network
       π = visit_counts(root) / sum(visit_counts(root))
       trajectory.append((encode_state(agent_pos), π))
       a = sample(π, temperature = (1.0 if t < 5 else 0.0))
       agent_pos = maze.step(agent_pos, a)
3. compute terminal context:
       { isGoal, dist, maxDist, steps, startDist }     // same field names as reward.js
4. z = z_formula(terminal_context)
5. for (state, π) in trajectory:
       replay_buffer.append((state, π, z))
6. drop oldest entries from replay_buffer to keep its size ≤ 10000
7. sample minibatch of 32 from replay_buffer
8. one Adam update step
9. log z; append to z curve; append losses to loss curves
10. refresh probe panel + maze overlay
```

`maxSteps = maze.size * 4` (matches the classic-tab rollout horizon default).

### 6.1 Run continuously

Steps 1–10 are wrapped in a `requestAnimationFrame`-driven loop that yields to the browser between games so the UI stays responsive. The user can stop at any time.

## 7. Visualizations

### 7.1 v-heatmap overlay

For each cell `c`:

1. Build the input encoding with the agent at `c`.
2. Run a forward pass, take `v`.
3. Color the cell by `v` ∈ `[-1, 1]` using a diverging colormap (red = low, blue = high).

To minimize browser-JS overhead, batch all `size²` evaluations into a single forward pass with batch dimension `size²`.

### 7.2 p-arrows overlay

For each cell `c`:

1. Build the input encoding with the agent at `c`.
2. Run a forward pass, take `p`.
3. Draw four arrows whose lengths are proportional to `p[up], p[down], p[left], p[right]`.

Same batching as v-heatmap.

### 7.3 z curve

```
+1 ┤                    ●  ●●●●●●●●●●●●●
   │                ●●●●  ●●
 0 ┤        ● ●●●●●●●
   │   ●●●●●
-1 ┤●●●
   └──────────────────────────────────
     0           50           100   game #
```

Bold line = moving average (window 20). Faint dots = individual game outcomes.

### 7.4 Loss curves

Three lines on one axis. Linear y-axis if all losses are O(1); switch to log if value loss explodes.

### 7.5 Probe (start cell)

```
p(start):
  up    ▓▓░░░░░░  0.27
  down  ▓▓▓▓▓░░░  0.51
  left  ▓░░░░░░░  0.10
  right ▓░░░░░░░  0.12

v(start) = -0.34
```

Updated after every self-play game. Quick way to see "is the network learning to point toward the goal from the start?"

## 8. File Layout

```
mcts-maze-demo/
├── index.html              # tab switcher added; AZ tab markup added
├── styles.css              # tab styles + AZ-pane styles
├── src/
│   ├── app.js              # split: top-level tab routing
│   ├── classic-tab.js      # existing app.js logic, lightly refactored
│   ├── az-tab.js           # new: AZ tab wiring
│   ├── az-mcts.js          # new: PUCT-MCTS variant
│   ├── az-network.js       # new: TF.js model, loss, training step
│   ├── az-selfplay.js      # new: self-play loop + replay buffer
│   ├── az-encode.js        # new: maze + agent → input tensor
│   ├── az-charts.js        # new: line chart + probe panel renderers
│   ├── render-maze.js      # extended with v-heatmap and p-arrows overlays
│   ├── maze.js, node.js,   # unchanged
│   │   mcts.js, rng.js,
│   │   reward.js, animate.js,
│   │   render-tree.js
└── tests/                  # new tests for az-encode, az-mcts, az-selfplay
```

`reward.js`'s formula compiler is reused as the `z` formula compiler unchanged. The terminal-context fields it already supports (`isGoal`, `dist`, `maxDist`, `steps`, `startDist`) cover everything the `z` presets need; no compiler change is required. The only conceptual difference from classic mode: in classic the formula is evaluated once per *rollout*; in AZ mode it is evaluated once per *self-play game* at the played terminal state.

`index.html` gets a `<script>` tag for TensorFlow.js loaded from a CDN, deferred. The AZ tab only initializes after TF.js is loaded.

## 9. Performance Budget

- One forward pass at 80k params: ~0.5 ms in TF.js CPU backend.
- One MCTS step: 50 sims × ~0.5 ms = ~25 ms.
- One self-play move: 25 ms.
- One self-play game: ~80 moves × 25 ms = ~2 s.
- One training step (minibatch 32): ~10 ms.
- v-heatmap refresh: 400-batch forward pass ≈ ~5 ms.

A self-play step thus costs ~2 s. "Run 10" = ~20 s. Acceptable for a teaching demo. Users on slower machines will feel it; we will not over-optimize in v1.

## 10. Testing

- **Unit:** `az-encode` (channel correctness), PUCT formula, formula compiler with new identifiers, replay-buffer FIFO behavior.
- **Integration:** A single self-play step terminates within step cap on a 10×10 maze. Loss decreases on a hand-crafted overfit dataset.
- **Smoke:** `Run 10` on a 10×10 maze does not throw, produces monotonically growing replay buffer, produces a finite loss.

We will not assert that the agent solves the maze in v1 — convergence is too dependent on hyperparameters and random seed for a stable test.

## 11. Open Questions / Future Work

- **Sub-tree reuse** between self-play moves (canonical AZ does this, we don't in v1).
- **Dirichlet noise** at the root prior (AZ adds it for exploration in self-play; v1 uses temperature sampling instead).
- **Curriculum learning**: start training on small mazes, scale up.
- **CNN backbone**: replace flat MLP with a small Conv2D stack.
- **Save/load weights**: round-trip the trained network to local storage so a student can resume.
- **BFS-oracle "fake AZ"** as a comparison baseline alongside the trained network.

These are deferred until the v1 design lands and we see how it teaches.
