import { test } from "node:test";
import assert from "node:assert/strict";
import { Node } from "../src/node.js";

test("Node: defaults", () => {
  const n = new Node([1, 2], null, null);
  assert.deepEqual(n.position, [1, 2]);
  assert.equal(n.parent, null);
  assert.equal(n.action, null);
  assert.equal(n.visits, 0);
  assert.equal(n.totalReward, 0);
  assert.equal(n.children.size, 0);
});

test("Node.meanReward = totalReward / visits, 0 when unvisited", () => {
  const n = new Node([0, 0], null, null);
  assert.equal(n.meanReward(), 0);
  n.visits = 4;
  n.totalReward = 2;
  assert.equal(n.meanReward(), 0.5);
});

test("Node.ucb1 is +Infinity when unvisited", () => {
  const n = new Node([0, 0], null, null);
  assert.equal(n.ucb1(1.41, 10), Infinity);
});

test("Node.ucb1 = mean + C*sqrt(ln(parentVisits)/visits) when visited", () => {
  const n = new Node([0, 0], null, null);
  n.visits = 4;
  n.totalReward = 2;
  const C = 1.41;
  const expected = 0.5 + C * Math.sqrt(Math.log(10) / 4);
  const actual = n.ucb1(C, 10);
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} ≠ ${expected}`);
});

test("Node.isFullyExpanded once untriedActions is empty", () => {
  const n = new Node([0, 0], null, null);
  n.untriedActions = [[1, 0], [0, 1]];
  assert.equal(n.isFullyExpanded(), false);
  n.untriedActions = [];
  assert.equal(n.isFullyExpanded(), true);
});

test("Node.bestChild returns child with most visits", () => {
  const parent = new Node([0, 0], null, null);
  const a = new Node([1, 0], parent, [1, 0]);
  const b = new Node([0, 1], parent, [0, 1]);
  a.visits = 5;
  b.visits = 12;
  parent.children.set("1,0", a);
  parent.children.set("0,1", b);
  assert.equal(parent.bestChild(), b);
});
