import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze } from "../src/maze.js";
import { computeReward } from "../src/reward.js";

const GRID = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("winloss: 1 if final position is goal, 0 otherwise", () => {
  assert.equal(computeReward(maze, [2, 2], "winloss"), 1);
  assert.equal(computeReward(maze, [1, 1], "winloss"), 0);
});

test("shaped: 1 - dist/maxDist when not at goal", () => {
  // maxDistance = 4 for this 3x3.
  // dist [1,1]→goal = 2; reward = 1 - 2/4 = 0.5
  assert.equal(computeReward(maze, [1, 1], "shaped"), 0.5);
  // dist [0,0]→goal = 4; reward = 1 - 4/4 = 0
  assert.equal(computeReward(maze, [0, 0], "shaped"), 0);
});

test("shaped: adds +1 goal bonus when at goal", () => {
  // 1 - 0/4 + 1 = 2
  assert.equal(computeReward(maze, [2, 2], "shaped"), 2);
});
