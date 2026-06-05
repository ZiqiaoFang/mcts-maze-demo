import { ACTIONS } from "./maze.js";

const actionKey = ([dr, dc]) => `${dr},${dc}`;

// Standard normal via Box-Muller. rng() must return uniform [0,1).
function randn(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Sample Gamma(shape=α, scale=1) via Marsaglia-Tsang. For α<1 boost
// by U^(1/α) per the standard trick (Marsaglia & Tsang 2000 §6).
function sampleGamma(alpha, rng) {
  if (alpha < 1) {
    const g = sampleGamma(alpha + 1, rng);
    const u = Math.max(1e-12, rng());
    return g * Math.pow(u, 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      x = randn(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// Dirichlet(α, α, …, α) sample of length k. Uses the standard ratio-of-Gammas
// construction: y_i ~ Gamma(α, 1), x_i = y_i / Σ y_j.
function sampleDirichlet(alpha, k, rng) {
  const y = new Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    y[i] = sampleGamma(alpha, rng);
    sum += y[i];
  }
  for (let i = 0; i < k; i++) y[i] /= sum;
  return y;
}

// Monotonic id so TreeRenderer's d3 key function can distinguish nodes.
// Without this every node's id is undefined → d3 dedups them and the tree
// pane renders only the root group with no children.
let _azNodeIdCounter = 0;

export class AZNode {
  constructor(position, parent, action, prior) {
    this.id = _azNodeIdCounter++;
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
  // Compatibility shim for TreeRenderer's tooltip. Returns PUCT score, not
  // UCB1; treat `C` as the c_puct coefficient.
  ucb1(C, parentVisits) {
    if (parentVisits == null || parentVisits === 0) return Infinity;
    return puctScore(this, parentVisits, C);
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
    this.network = network;
    this.cPuct = config.cPuct ?? 1.0;
    // TreeRenderer reads mcts.config.C; expose it as an alias of cPuct so the
    // same renderer works for both Classic and AZ trees. The tooltip label
    // still reads "UCB1" but the value shown is PUCT score for AZ nodes.
    this.config = { C: this.cPuct };
    const startPos = config.startPos ?? maze.start;
    this.root = new AZNode([...startPos], null, null, 1);
    this.iterationCount = 0;
    // Dirichlet noise at the root prior. AlphaZero adds this in self-play so
    // a saturated policy can't lock MCTS onto one branch. Default off so
    // evaluation/inference paths get no noise.
    this.dirichletAlpha = config.dirichletAlpha ?? null;
    this.dirichletEpsilon = config.dirichletEpsilon ?? 0.25;
    this.rng = config.rng ?? Math.random;
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
      // Dirichlet noise at the root only. legal.length > 1 because mixing
      // noise into a 1-element distribution is a no-op (1*α + (1-α)*1 = 1).
      if (node === this.root && this.dirichletAlpha !== null && legal.length > 1) {
        const noise = sampleDirichlet(this.dirichletAlpha, legal.length, this.rng);
        const eps = this.dirichletEpsilon;
        for (let i = 0; i < legal.length; i++) {
          legal[i].prior = (1 - eps) * legal[i].prior + eps * noise[i];
        }
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
}
