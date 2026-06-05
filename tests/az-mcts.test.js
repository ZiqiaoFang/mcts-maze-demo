import { test } from "node:test";
import assert from "node:assert/strict";
import { AZNode, puctScore, AZMCTS } from "../src/az-mcts.js";
import { Maze } from "../src/maze.js";
import { mulberry32 } from "../src/rng.js";

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

// Dirichlet noise tests. Standard AlphaZero adds noise to the root prior in
// self-play so a saturated network policy can still be challenged by other
// actions. Without it, an MCTS rooted at a near-one-hot prior burns all 50
// sims down the same branch and the policy can't escape a bad attractor.

test("AZMCTS without dirichletAlpha leaves root priors equal to renormalized network priors", () => {
  const m = new AZMCTS(maze, STUB, { cPuct: 1.0 });
  m.iterate();
  const priors = Array.from(m.root.children.values()).map((c) => c.prior);
  // Stub network gives uniform [0.25 × 4]. 2 legal actions from (0,0) →
  // each renormalized prior = 0.5. No noise should leave it exactly 0.5.
  assert.equal(priors.length, 2);
  for (const p of priors) assert.ok(Math.abs(p - 0.5) < 1e-9);
});

test("AZMCTS with dirichletAlpha perturbs root priors but keeps them on the simplex", () => {
  const m = new AZMCTS(maze, STUB, {
    cPuct: 1.0,
    dirichletAlpha: 0.3,
    dirichletEpsilon: 0.5,
    rng: mulberry32(123),
  });
  m.iterate();
  const priors = Array.from(m.root.children.values()).map((c) => c.prior);
  assert.equal(priors.length, 2);
  const sum = priors.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9, `priors did not sum to 1: ${sum}`);
  // With ε=0.5 noise from Dir(0.3), priors will almost surely deviate from 0.5.
  const allHalf = priors.every((p) => Math.abs(p - 0.5) < 1e-9);
  assert.ok(!allHalf, `expected noise to perturb priors, got ${priors}`);
  for (const p of priors) assert.ok(p > 0 && p < 1, `prior out of (0,1): ${p}`);
});

test("AZMCTS Dirichlet noise applies to root only, not to grandchildren", () => {
  // Need an open maze: the 3×3 fixture is so cramped that cycle prevention
  // leaves grandchildren with at most one legal action, so we can't even
  // observe a non-trivial child prior distribution. Use a 5×5 open maze
  // starting in the middle — gives multiple legal grandchildren.
  const OPEN = Array.from({ length: 5 }, () => new Array(5).fill(0));
  const openMaze = new Maze(OPEN, [2, 2], [4, 4]);
  const m = new AZMCTS(openMaze, STUB, {
    cPuct: 1.0,
    dirichletAlpha: 0.3,
    dirichletEpsilon: 0.5,
    rng: mulberry32(99),
  });
  for (let i = 0; i < 30; i++) m.iterate();
  // Find any depth-2 node with ≥2 children and verify its priors are still
  // pure renormalized network output (uniform from STUB, no noise).
  // Noise at deeper nodes would break the policy improvement guarantee.
  let checked = 0;
  for (const child of m.root.children.values()) {
    if (child.children.size <= 1) continue;
    const priors = Array.from(child.children.values()).map((c) => c.prior);
    const expected = 1 / priors.length;
    for (const p of priors) {
      assert.ok(
        Math.abs(p - expected) < 1e-9,
        `non-root child prior ${p} != uniform ${expected}`,
      );
    }
    checked++;
  }
  assert.ok(checked > 0, "expected at least one depth-2 expansion to check");
});
