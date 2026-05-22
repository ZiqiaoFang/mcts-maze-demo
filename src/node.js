let _nodeIdCounter = 0;

export class Node {
  constructor(position, parent, action) {
    this.id = _nodeIdCounter++;
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
