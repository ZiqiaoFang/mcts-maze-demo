import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze, MAZE_10, MAZE_15, MAZE_20 } from "../src/maze.js";

function assertReachable(maze) {
  // BFS from start; assert goal is reachable through open cells.
  const visited = new Set();
  const key = (p) => `${p[0]},${p[1]}`;
  const queue = [maze.start];
  visited.add(key(maze.start));
  while (queue.length) {
    const pos = queue.shift();
    if (maze.isGoal(pos)) return true;
    for (const a of maze.legalActions(pos)) {
      const next = maze.step(pos, a);
      if (!visited.has(key(next))) {
        visited.add(key(next));
        queue.push(next);
      }
    }
  }
  return false;
}

for (const [name, m] of [["MAZE_10", MAZE_10], ["MAZE_15", MAZE_15], ["MAZE_20", MAZE_20]]) {
  test(`${name}: correct size`, () => {
    const expected = parseInt(name.split("_")[1], 10);
    assert.equal(m.size, expected);
  });
  test(`${name}: start is open`, () => {
    assert.equal(m.grid[m.start[0]][m.start[1]], 0);
  });
  test(`${name}: goal is open`, () => {
    assert.equal(m.grid[m.goal[0]][m.goal[1]], 0);
  });
  test(`${name}: goal is reachable from start`, () => {
    assert.equal(assertReachable(m), true);
  });
}
