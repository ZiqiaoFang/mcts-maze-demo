import { test } from "node:test";
import assert from "node:assert/strict";
import { AZNode, puctScore, AZMCTS } from "../src/az-mcts.js";
import { Maze } from "../src/maze.js";

test("AZNode initializes with prior, zero visits, zero W", () => {
  const n = new AZNode([0, 0], null, null, 0.25);
  assert.deepEqual(n.position, [0, 0]);
  assert.equal(n.parent, null);
  assert.equal(n.action, null);
  assert.equal(n.prior, 0.25);
  assert.equal(n.visits, 0);
  assert.equal(n.totalReward, 0);
  assert.equal(n.children.size, 0);
});

test("AZNode.meanReward returns 0 when unvisited", () => {
  const n = new AZNode([0, 0], null, null, 1.0);
  assert.equal(n.meanReward(), 0);
});

test("AZNode.meanReward returns W/N when visited", () => {
  const n = new AZNode([0, 0], null, null, 1.0);
  n.visits = 4;
  n.totalReward = 2.0;
  assert.equal(n.meanReward(), 0.5);
});

test("puctScore: unvisited child uses Q=0 and prior-scaled exploration", () => {
  const n = new AZNode([0, 0], null, null, 0.5);
  // c_puct=1, parentVisits=4, prior=0.5, N_child=0 → 0 + 1*0.5*sqrt(4)/(1+0) = 1
  assert.equal(puctScore(n, 4, 1), 1);
});

test("puctScore: visited child mixes Q with prior bonus", () => {
  const n = new AZNode([0, 0], null, null, 0.5);
  n.visits = 1;
  n.totalReward = 0.4;
  // Q=0.4, c_puct=1, parentVisits=4, prior=0.5, N_child=1
  //   → 0.4 + 1*0.5*sqrt(4)/(1+1) = 0.4 + 0.5 = 0.9
  assert.equal(Math.abs(puctScore(n, 4, 1) - 0.9) < 1e-9, true);
});

// AZMCTS integration with a stub network
const STUB = {
  // Always returns uniform prior and v=0.
  predict: () => ({ p: new Float32Array([0.25, 0.25, 0.25, 0.25]), v: 0 }),
};

const GRID = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("AZMCTS.iterate populates root children with priors on first call", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  assert.equal(m.root.children.size, 0);
  m.iterate();
  // From [0,0] legal actions are [right, down] (up/left out of bounds).
  // Uniform prior over those two = 0.5 each after re-normalization.
  assert.equal(m.root.children.size, 2);
  for (const child of m.root.children.values()) {
    assert.ok(Math.abs(child.prior - 0.5) < 1e-9);
  }
});

test("AZMCTS.iterate increments root visits by 1", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  m.iterate();
  assert.equal(m.root.visits, 1);
  m.iterate();
  assert.equal(m.root.visits, 2);
});

test("AZMCTS.rootPolicy returns visit-count distribution", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  for (let i = 0; i < 10; i++) m.iterate();
  const pi = m.rootPolicy();
  // pi has length 4 (one per ACTION); illegal actions get 0; sum = 1.
  assert.equal(pi.length, 4);
  const sum = pi.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `sum was ${sum}`);
});
