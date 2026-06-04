import { test } from "node:test";
import assert from "node:assert/strict";
import { encodeState, CHANNELS } from "../src/az-encode.js";
import { Maze } from "../src/maze.js";

// 3x3 grid:
//  0 0 1
//  0 0 1
//  0 0 0   start [0,0], goal [2,2]
const GRID = [
  [0, 0, 1],
  [0, 0, 1],
  [0, 0, 0],
];
const maze = new Maze(GRID, [0, 0], [2, 2]);

test("encodeState returns Float32Array of length size*size*CHANNELS", () => {
  const v = encodeState(maze, [0, 0]);
  assert.ok(v instanceof Float32Array);
  assert.equal(v.length, 3 * 3 * CHANNELS);
});

test("encodeState channel layout: [walls, agent, goal] interleaved per cell", () => {
  // We use channel-last layout: index = (r * size + c) * CHANNELS + ch
  const v = encodeState(maze, [1, 0]);
  const at = (r, c, ch) => v[(r * 3 + c) * CHANNELS + ch];
  // walls channel: 1 where grid==1
  assert.equal(at(0, 0, 0), 0);
  assert.equal(at(0, 2, 0), 1);
  assert.equal(at(1, 2, 0), 1);
  assert.equal(at(2, 2, 0), 0);
  // agent channel: 1 only at agent_pos = [1,0]
  assert.equal(at(1, 0, 1), 1);
  assert.equal(at(0, 0, 1), 0);
  assert.equal(at(2, 2, 1), 0);
  // goal channel: 1 only at goal = [2,2]
  assert.equal(at(2, 2, 2), 1);
  assert.equal(at(0, 0, 2), 0);
});

test("encodeState: changing agent position only flips the agent channel", () => {
  const a = encodeState(maze, [0, 0]);
  const b = encodeState(maze, [2, 1]);
  // Walls and goal channels equal in both
  for (let i = 0; i < a.length; i += CHANNELS) {
    assert.equal(a[i], b[i]);       // walls
    assert.equal(a[i + 2], b[i + 2]); // goal
  }
  // Exactly two indices differ in agent channel
  let diffs = 0;
  for (let i = 1; i < a.length; i += CHANNELS) if (a[i] !== b[i]) diffs++;
  assert.equal(diffs, 2);
});
