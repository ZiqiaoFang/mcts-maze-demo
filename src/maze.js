// Action deltas: up, down, left, right.
export const ACTIONS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
];

export class Maze {
  constructor(grid, start, goal) {
    this.grid = grid;
    this.size = grid.length;
    this.start = start;
    this.goal = goal;
    this._maxDistance = null;
  }

  inBounds([r, c]) {
    return r >= 0 && r < this.size && c >= 0 && c < this.size;
  }

  isOpen([r, c]) {
    return this.inBounds([r, c]) && this.grid[r][c] === 0;
  }

  isGoal([r, c]) {
    return r === this.goal[0] && c === this.goal[1];
  }

  legalActions([r, c]) {
    const result = [];
    for (const [dr, dc] of ACTIONS) {
      const nr = r + dr;
      const nc = c + dc;
      if (this.isOpen([nr, nc])) result.push([dr, dc]);
    }
    return result;
  }

  step([r, c], [dr, dc]) {
    return [r + dr, c + dc];
  }

  manhattan([r1, c1], [r2, c2]) {
    return Math.abs(r1 - r2) + Math.abs(c1 - c2);
  }

  maxDistance() {
    if (this._maxDistance !== null) return this._maxDistance;
    // Diameter over open cells, naive O(N^2).
    let best = 0;
    const open = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        if (this.grid[r][c] === 0) open.push([r, c]);
      }
    }
    for (let i = 0; i < open.length; i++) {
      for (let j = i + 1; j < open.length; j++) {
        const d = this.manhattan(open[i], open[j]);
        if (d > best) best = d;
      }
    }
    this._maxDistance = best;
    return best;
  }
}

// Hand-crafted mazes. 0 = open, 1 = wall.
// Each is designed to have at least two viable paths and a few dead-ends.

// 10×10: start top-left, goal bottom-right, central wall with two gaps.
const GRID_10 = [
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
  [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
  [0, 1, 0, 1, 1, 0, 0, 0, 1, 0],
  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
  [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
];
export const MAZE_10 = new Maze(GRID_10, [0, 0], [9, 9]);

// 15×15.
const GRID_15 = [
  [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
  [1,1,0,1,0,1,1,1,0,1,0,1,1,1,0],
  [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
  [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
  [1,1,1,1,0,1,1,1,1,1,1,1,0,1,0],
  [0,0,0,1,0,0,0,0,0,1,0,0,0,1,0],
  [0,1,0,1,1,1,1,1,0,1,0,1,1,1,0],
  [0,1,0,0,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
  [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
  [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0],
  [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
  [0,1,0,1,1,1,0,1,0,1,1,1,0,1,0],
  [0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
];
export const MAZE_15 = new Maze(GRID_15, [0, 0], [14, 14]);

// 20×20.
const GRID_20 = [
  [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
  [1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0],
  [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
  [0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0],
  [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
  [0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0],
  [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
  [0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,0],
  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],
  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
  [1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0],
  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
];
export const MAZE_20 = new Maze(GRID_20, [0, 0], [19, 19]);

export const MAZES = { 10: MAZE_10, 15: MAZE_15, 20: MAZE_20 };
