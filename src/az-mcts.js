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
    const pi = new Array(ACTIONS.length).fill(0);
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
      return pi.fill(1 / ACTIONS.length);
    }
    // Normalize in float64 to avoid Float32 rounding accumulation.
    return pi.map(v => v / total);
  }
}
