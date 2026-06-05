import { test } from "node:test";
import assert from "node:assert/strict";
import { ReplayBuffer, playOneGame } from "../src/az-selfplay.js";
import { Maze } from "../src/maze.js";
import { encodeState } from "../src/az-encode.js";
import { compileReward } from "../src/reward.js";

test("ReplayBuffer.append + size", () => {
  const buf = new ReplayBuffer(100);
  assert.equal(buf.size(), 0);
  buf.append({ state: new Float32Array(3), pi: new Float32Array(4), z: 1 });
  assert.equal(buf.size(), 1);
});

test("ReplayBuffer enforces FIFO capacity", () => {
  const buf = new ReplayBuffer(3);
  for (let i = 0; i < 5; i++) {
    buf.append({ state: new Float32Array([i]), pi: new Float32Array(4), z: i });
  }
  assert.equal(buf.size(), 3);
  // Oldest two (i=0, i=1) should have been dropped.
  const items = buf.items();
  assert.deepEqual(items.map((t) => t.z), [2, 3, 4]);
});

test("ReplayBuffer.sampleBatch returns minibatch arrays of correct shape", () => {
  const buf = new ReplayBuffer(100);
  const inputDim = 3;
  for (let i = 0; i < 20; i++) {
    buf.append({
      state: new Float32Array([i, i, i]),
      pi: new Float32Array([0.25, 0.25, 0.25, 0.25]),
      z: i / 20,
    });
  }
  const batch = buf.sampleBatch(8, inputDim);
  assert.equal(batch.states.length, 8 * inputDim);
  assert.equal(batch.pis.length, 8 * 4);
  assert.equal(batch.zs.length, 8);
});

// playOneGame integration with a stub network.
// Stub gives uniform prior + value 0; the agent therefore behaves close to
// uniform random under MCTS, which is enough to test the loop terminates and
// produces well-shaped trajectories.
const GRID = [
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);
const STUB = { predict: () => ({ p: new Float32Array([0.25, 0.25, 0.25, 0.25]), v: 0 }) };

// z formula: 1 if reached goal, 0 otherwise.
const Z_FN = (ctx) => (ctx.isGoal ? 1 : 0);

test("playOneGame terminates within step cap and returns trajectory + z", async () => {
  const result = await playOneGame({
    maze, network: STUB, zFn: Z_FN, simsPerMove: 20, cPuct: 1.0,
    maxSteps: 8, temperatureMoves: 0, rng: Math.random,
  });
  assert.ok(Array.isArray(result.trajectory));
  assert.ok(result.trajectory.length > 0);
  assert.ok(result.trajectory.length <= 8);
  assert.equal(typeof result.z, "number");
  // Each tuple shape:
  for (const t of result.trajectory) {
    assert.ok(t.state instanceof Float32Array);
    assert.equal(t.pi.length, 4);
  }
});

test("playOneGame calls onProgress for each move and yields to event loop", async () => {
  const progressCalls = [];
  const result = await playOneGame({
    maze, network: STUB, zFn: Z_FN, simsPerMove: 5, cPuct: 1.0,
    maxSteps: 6, temperatureMoves: 0, rng: Math.random,
    onProgress: (ev) => progressCalls.push({ move: ev.move, maxSteps: ev.maxSteps }),
  });
  // Called once per move before its MCTS runs; never after the loop ends.
  assert.ok(progressCalls.length > 0);
  assert.ok(progressCalls.length <= result.trajectory.length);
  for (let i = 0; i < progressCalls.length; i++) {
    assert.equal(progressCalls[i].move, i);
    assert.equal(progressCalls[i].maxSteps, 6);
  }
});

test("playOneGame returns cumulativeVisits with the start cell present", async () => {
  const result = await playOneGame({
    maze, network: STUB, zFn: Z_FN, simsPerMove: 10, cPuct: 1.0,
    maxSteps: 6, temperatureMoves: 0, rng: Math.random,
  });
  assert.ok(result.cumulativeVisits instanceof Map);
  // The first move's tree is rooted at the start; every per-move tree
  // contributes its root's visits, so the start cell must be present
  // with at least simsPerMove visits.
  const startKey = `${maze.start[0]},${maze.start[1]}`;
  assert.ok(result.cumulativeVisits.has(startKey));
  assert.ok(result.cumulativeVisits.get(startKey) >= 10);
});

// Regression: zFn returned by compileReward takes positional args
// (isGoal, dist, maxDist, steps, startDist). playOneGame must invoke it
// with the ctx fields, not the ctx object — otherwise every formula of the
// form `isGoal ? A : B` short-circuits to A because the truthy object lands
// in the isGoal slot.
test("playOneGame computes z via positional args of compileReward result", async () => {
  // 3x3 with a wall at (1,1) — same as the GRID above. A 3-step cap can't
  // reach the goal at (2,2) from (0,0), so the non-goal branch must fire.
  const zFn = compileReward("isGoal ? 1 : (1 - dist / maxDist)");
  const result = await playOneGame({
    maze, network: STUB, zFn, simsPerMove: 5, cPuct: 1.0,
    maxSteps: 1, temperatureMoves: 0, rng: () => 0,
  });
  assert.equal(result.reachedGoal, false);
  assert.notEqual(result.z, 1, "z=1 means isGoal short-circuited on the ctx object");
  assert.ok(Number.isFinite(result.z));
  assert.ok(result.z >= 0 && result.z < 1);
});
