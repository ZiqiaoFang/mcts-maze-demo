import { mulberry32 } from "./rng.js";

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

// Randomly generate a solvable maze of the given size.
// Uses randomized DFS (recursive backtracker) on cells at even coordinates,
// then knocks out a few extra walls so the maze has loops rather than being
// a pure spanning tree (keeps the "multiple viable paths" feel).
// Start defaults to top-left, goal to bottom-right; both are guaranteed open
// and connected to the rest of the maze.
export function generateMaze(
  size,
  seed,
  { start = [0, 0], goal = [size - 1, size - 1] } = {},
) {
  const rng = mulberry32(seed);
  const grid = Array.from({ length: size }, () => new Array(size).fill(1));

  // Snap a coord to the nearest even index inside [0, size).
  const snapEven = (v) => {
    const e = v - (v % 2);
    return e >= size ? e - 2 : e;
  };
  const carveStart = [snapEven(start[0]), snapEven(start[1])];

  // Randomized DFS over even-coord cells, carving the wall between.
  const stack = [carveStart];
  grid[carveStart[0]][carveStart[1]] = 0;
  const STEPS = [
    [-2, 0],
    [2, 0],
    [0, -2],
    [0, 2],
  ];
  while (stack.length) {
    const [r, c] = stack[stack.length - 1];
    const candidates = [];
    for (const [dr, dc] of STEPS) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < size && nc >= 0 && nc < size && grid[nr][nc] === 1) {
        candidates.push([nr, nc, dr, dc]);
      }
    }
    if (candidates.length === 0) {
      stack.pop();
      continue;
    }
    const [nr, nc, dr, dc] = candidates[Math.floor(rng() * candidates.length)];
    grid[r + dr / 2][c + dc / 2] = 0;
    grid[nr][nc] = 0;
    stack.push([nr, nc]);
  }

  // Knock out extra walls to create loops. Only walls with ≥2 open neighbors
  // qualify, so an opening always merges two existing corridors.
  const extraCount = Math.max(2, Math.floor(size * 0.6));
  const wallCells = [];
  for (let r = 1; r < size - 1; r++) {
    for (let c = 1; c < size - 1; c++) {
      if (grid[r][c] !== 1) continue;
      let openNbrs = 0;
      if (grid[r - 1][c] === 0) openNbrs++;
      if (grid[r + 1][c] === 0) openNbrs++;
      if (grid[r][c - 1] === 0) openNbrs++;
      if (grid[r][c + 1] === 0) openNbrs++;
      if (openNbrs >= 2) wallCells.push([r, c]);
    }
  }
  // Fisher-Yates shuffle, take prefix.
  for (let i = wallCells.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [wallCells[i], wallCells[j]] = [wallCells[j], wallCells[i]];
  }
  for (let i = 0; i < Math.min(extraCount, wallCells.length); i++) {
    const [r, c] = wallCells[i];
    grid[r][c] = 0;
  }

  // Make sure start and goal cells are open and reachable. For even-size
  // grids the bottom-right corner is off the even-coord lattice, so it ends
  // up isolated; tunnel it back to the carved region.
  const ensureOpenAndConnected = ([r, c]) => {
    grid[r][c] = 0;
    const hasOpenNbr = () =>
      (r > 0 && grid[r - 1][c] === 0) ||
      (r < size - 1 && grid[r + 1][c] === 0) ||
      (c > 0 && grid[r][c - 1] === 0) ||
      (c < size - 1 && grid[r][c + 1] === 0);
    if (hasOpenNbr()) return;
    // Tunnel inward until we reach an open cell.
    let cr = r;
    let cc = c;
    const dr = cr === size - 1 ? -1 : 1;
    const dc = cc === size - 1 ? -1 : 1;
    while (!hasOpenNbr()) {
      if (Math.abs(cr - r) <= Math.abs(cc - c) && cr + dr >= 0 && cr + dr < size) {
        cr += dr;
      } else if (cc + dc >= 0 && cc + dc < size) {
        cc += dc;
      } else {
        break;
      }
      grid[cr][cc] = 0;
    }
  };
  ensureOpenAndConnected(start);
  ensureOpenAndConnected(goal);

  return new Maze(grid, start, goal);
}

// Per-session seed: different maze on each page load, but stable within a
// session so the size dropdown returns to the same layout.
const SESSION_SEED =
  typeof Math !== "undefined" ? Math.floor(Math.random() * 0xffffffff) : 1;

export const MAZE_10 = generateMaze(10, SESSION_SEED ^ 10);
export const MAZE_15 = generateMaze(15, SESSION_SEED ^ 15);
export const MAZE_20 = generateMaze(20, SESSION_SEED ^ 20);

export const MAZES = { 10: MAZE_10, 15: MAZE_15, 20: MAZE_20 };

// Replace the cached maze for a given size with a freshly generated one.
// If seed is omitted, picks a new random seed.
export function regenerateMaze(size, seed) {
  const s =
    seed ?? (typeof Math !== "undefined" ? Math.floor(Math.random() * 0xffffffff) : size);
  const m = generateMaze(size, s);
  MAZES[size] = m;
  return m;
}
