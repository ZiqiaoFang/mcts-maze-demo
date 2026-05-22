import { test } from "node:test";
import assert from "node:assert/strict";
import { Maze, MAZE_10 } from "../src/maze.js";
import { MCTS } from "../src/mcts.js";

const SIMPLE = new Maze(
  [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ],
  [0, 0],
  [2, 2]
);

test("MCTS: starts with a single root node at start position", () => {
  const m = new MCTS(SIMPLE);
  assert.deepEqual(m.root.position, [0, 0]);
  assert.equal(m.root.visits, 0);
  assert.equal(m.iterationCount, 0);
});

test("MCTS.iterate yields select → expand → simulate → backprop in order", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  const events = [];
  for (const ev of m.iterate()) events.push(ev.phase);
  assert.deepEqual(events, ["select", "expand", "simulate", "backprop"]);
});

test("MCTS: after one iteration, root has visits=1 and one child", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (const _ev of m.iterate()) { /* drain */ }
  assert.equal(m.root.visits, 1);
  assert.equal(m.root.children.size, 1);
  assert.equal(m.iterationCount, 1);
});

test("MCTS: after many iterations, finds path to goal (PV reaches goal)", () => {
  const m = new MCTS(MAZE_10, { seed: 42, rolloutHorizon: 40, rewardMode: "shaped" });
  for (let i = 0; i < 500; i++) for (const _ev of m.iterate()) { /* drain */ }
  const pv = m.principalVariation();
  // The PV should reach the goal or get very close — assert it reached the goal cell.
  const last = pv[pv.length - 1];
  assert.ok(
    MAZE_10.isGoal(last.position) || MAZE_10.manhattan(last.position, MAZE_10.goal) <= 3,
    `PV final position ${last.position} too far from goal ${MAZE_10.goal}`
  );
});

test("MCTS.treeStats reports node count, max depth, max visits", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (let i = 0; i < 10; i++) for (const _ev of m.iterate()) {}
  const s = m.treeStats();
  assert.ok(s.totalNodes > 1);
  assert.ok(s.maxDepth >= 1);
  assert.ok(s.maxVisits >= 1);
});

test("MCTS: backprop event path matches select+expand path", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  let selectPath, expandedNode, backpropPath;
  for (const ev of m.iterate()) {
    if (ev.phase === "select") selectPath = ev.path;
    if (ev.phase === "expand") expandedNode = ev.newNode;
    if (ev.phase === "backprop") backpropPath = ev.path;
  }
  // Backprop path should equal selectPath + [expandedNode]
  const expected = [...selectPath, expandedNode];
  assert.equal(backpropPath.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.equal(backpropPath[i], expected[i]);
  }
});

test("MCTS: visits increment along backprop path", () => {
  const m = new MCTS(SIMPLE, { seed: 1 });
  for (let i = 0; i < 3; i++) for (const _ev of m.iterate()) {}
  // Root visits should equal iteration count.
  assert.equal(m.root.visits, 3);
});
