import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze } from "../src/maze.js";

// A 3x3 maze: walls on the right column except for goal cell.
//  0 0 1
//  0 0 1
//  0 0 0  (goal at [2,2])
const SMALL_GRID = [
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 0],
];

test("Maze: stores grid, start, goal", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.size, 3);
  assert.deepEqual(m.start, [0, 0]);
  assert.deepEqual(m.goal, [2, 2]);
});

test("Maze.isGoal returns true only at goal cell", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.isGoal([2, 2]), true);
  assert.equal(m.isGoal([0, 0]), false);
  assert.equal(m.isGoal([1, 1]), false);
});

test("Maze.legalActions excludes walls and out-of-bounds", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  // From [0,0]: can go right [0,1] and down [1,0]; up and left are out of bounds.
  const actions = m.legalActions([0, 0]).map((a) => a.join(","));
  assert.deepEqual(actions.sort(), ["0,1", "1,0"]);
  // From [0,1]: can go down [1,1] and left [0,0]; right is wall, up is OOB.
  const a2 = m.legalActions([0, 1]).map((a) => a.join(","));
  assert.deepEqual(a2.sort(), ["0,-1", "1,0"]);
});

test("Maze.step applies action", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.deepEqual(m.step([0, 0], [1, 0]), [1, 0]);
  assert.deepEqual(m.step([1, 1], [0, -1]), [1, 0]);
});

test("Maze.manhattan computes |dr| + |dc|", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  assert.equal(m.manhattan([0, 0], [2, 2]), 4);
  assert.equal(m.manhattan([1, 1], [1, 1]), 0);
});

test("Maze.maxDistance equals Manhattan diameter of open cells", () => {
  const m = new Maze(SMALL_GRID, [0, 0], [2, 2]);
  // Far corners of the open region: [0,0] and [2,2] → 4
  assert.equal(m.maxDistance(), 4);
});
