import { Node } from "./node.js";
import { mulberry32, pick } from "./rng.js";
import { computeReward } from "./reward.js";

const actionKey = ([dr, dc]) => `${dr},${dc}`;

// Return the set of position keys for node and all its ancestors.
function getAncestorPositions(node) {
  const positions = new Set();
  let n = node;
  while (n !== null) {
    positions.add(n.position.join(","));
    n = n.parent;
  }
  return positions;
}

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
      // Exclude actions that would revisit any ancestor (including self) to avoid tree cycles.
      const ancestorPositions = getAncestorPositions(node);
      node.untriedActions = this.maze
        .legalActions(node.position)
        .map((a) => [...a])
        .filter((a) => {
          const newPos = this.maze.step(node.position, a);
          return !ancestorPositions.has(newPos.join(","));
        });
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
    // Walk the tree by most-visited child, with cycle detection.
    const path = [this.root];
    const visited = new Set();
    visited.add(this.root.position.join(","));
    let node = this.root;
    while (node.children.size > 0) {
      let best = null;
      let bestVisits = -1;
      for (const child of node.children.values()) {
        const key = child.position.join(",");
        if (!visited.has(key) && child.visits > bestVisits) {
          best = child;
          bestVisits = child.visits;
        }
      }
      if (!best) break;
      visited.add(best.position.join(","));
      path.push(best);
      node = best;
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
