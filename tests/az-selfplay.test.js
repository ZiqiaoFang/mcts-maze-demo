import { test } from "node:test";
import assert from "node:assert/strict";
import { ReplayBuffer, playOneGame } from "../src/az-selfplay.js";
import { Maze } from "../src/maze.js";
import { encodeState } from "../src/az-encode.js";

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

test("playOneGame terminates within step cap and returns trajectory + z", () => {
  const result = playOneGame({
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
