import { AZMCTS } from "./az-mcts.js";
import { ACTIONS } from "./maze.js";
import { encodeState, CHANNELS } from "./az-encode.js";
import { computeReward } from "./reward.js";

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
//   dirichletAlpha, dirichletEpsilon (optional, AlphaZero root-prior noise)
//   onProgress (optional, called before each move with {move, maxSteps, pos}).
//               If provided, the loop also yields to the event loop between
//               moves so the UI stays responsive and the callback can render.
export async function playOneGame(opts) {
  const {
    maze, network, zFn, simsPerMove, cPuct,
    maxSteps, temperatureMoves, rng = Math.random, onProgress = null,
    dirichletAlpha = null, dirichletEpsilon = 0.25,
  } = opts;
  let pos = [...maze.start];
  const trajectory = [];
  let t = 0;
  let reachedGoal = false;
  let lastMcts = null;
  // Visit counts across every per-move PUCT tree in this game, summed by cell.
  // Lets the AZ tab's "MCTS visits" overlay light up the agent's actual path
  // through the maze (start cell brightest, fading along the trajectory) —
  // not just the very last move's tiny local tree.
  const cumulativeVisits = new Map();
  for (; t < maxSteps; t++) {
    if (maze.isGoal(pos)) { reachedGoal = true; break; }
    if (onProgress) {
      onProgress({ move: t, maxSteps, pos });
      // Yield once per move so progress UI repaints. Cheap: setTimeout(0)
      // gives the browser a turn without dropping into rAF throttling.
      await new Promise((r) => setTimeout(r, 0));
    }
    const mcts = new AZMCTS(maze, network, {
      cPuct, startPos: pos, dirichletAlpha, dirichletEpsilon, rng,
    });
    for (let s = 0; s < simsPerMove; s++) mcts.iterate();
    lastMcts = mcts;
    // Fold this move's visits into the running per-cell totals.
    for (const [key, v] of mcts.cellVisits()) {
      cumulativeVisits.set(key, (cumulativeVisits.get(key) || 0) + v);
    }
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
  // compileReward returns a function taking POSITIONAL args
  // (isGoal, dist, maxDist, steps, startDist). Calling zFn(ctx) would bind
  // the whole ctx object to `isGoal` (truthy), making every game collapse to
  // the goal branch. Route through computeReward, which destructures ctx and
  // guards against NaN — same path classic-tab uses.
  const z = computeReward(zFn, ctx);
  return { trajectory, z, steps: trajectory.length, reachedGoal, lastMcts, cumulativeVisits };
}
