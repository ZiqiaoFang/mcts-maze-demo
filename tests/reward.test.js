import { test } from "node:test";
import assert from "node:assert/strict";
import { compileReward, computeReward, PRESETS } from "../src/reward.js";

// --- compileReward ---

test("compileReward returns a callable for a valid formula", () => {
  const fn = compileReward("isGoal ? 1 : 0");
  assert.equal(typeof fn, "function");
  assert.equal(fn(true, 0, 20, 10, 8), 1);
  assert.equal(fn(false, 5, 20, 10, 8), 0);
});

test("compileReward throws on syntax error", () => {
  assert.throws(() => compileReward("isGoal ?? 1 :"), /Bad formula/);
});

test("compileReward throws when sentinel returns non-finite", () => {
  // 1/0 → Infinity for any sentinel
  assert.throws(() => compileReward("1/0"), /non-finite/);
});

test("compileReward throws when sentinel returns non-number", () => {
  assert.throws(() => compileReward("'hello'"), /non-number/);
});

test("compileReward throws when formula references undefined identifier", () => {
  // bogus is not in scope, so this throws at sentinel-call time
  assert.throws(() => compileReward("bogus"), /Formula threw/);
});

// --- Preset behaviour ---

test("PRESETS Distance-shaped: goal=2, halfway~0.5, far=0", () => {
  const fn = compileReward(PRESETS["Distance-shaped"]);
  // goal: isGoal=true, dist=0, maxDist=20 → 1 + max(0, 1-0/20) = 1 + 1 = 2
  assert.equal(fn(true, 0, 20, 10, 8), 2);
  // halfway: dist=10, maxDist=20 → 0 + max(0, 1-10/20) = 0.5
  assert.equal(fn(false, 10, 20, 10, 8), 0.5);
  // far: dist=20, maxDist=20 → 0 + max(0, 1-20/20) = 0
  assert.equal(fn(false, 20, 20, 10, 8), 0);
});

test("PRESETS Pure win/loss: 1 at goal, 0 otherwise", () => {
  const fn = compileReward(PRESETS["Pure win/loss"]);
  assert.equal(fn(true, 0, 20, 10, 8), 1);
  assert.equal(fn(false, 5, 20, 10, 8), 0);
});

test("PRESETS Step penalty: goal=1, non-goal at 50 steps = -0.5", () => {
  const fn = compileReward(PRESETS["Step penalty"]);
  assert.equal(fn(true, 0, 20, 50, 8), 1);
  assert.equal(fn(false, 5, 20, 50, 8), -0.5);
});

// --- Variable plumbing ---

test("formula referencing steps returns correct value", () => {
  const fn = compileReward("steps * 2");
  assert.equal(fn(false, 5, 20, 7, 8), 14);
  assert.equal(fn(false, 5, 20, 100, 8), 200);
});

test("formula referencing startDist returns correct value", () => {
  const fn = compileReward("startDist + 1");
  assert.equal(fn(false, 5, 20, 10, 8), 9);
  assert.equal(fn(false, 5, 20, 10, 0), 1);
});

// --- computeReward runtime safety ---

test("computeReward returns 0 when compiled returns NaN", () => {
  // A compile-time-valid formula that returns NaN on this specific input.
  // (Math.sqrt(-1) → NaN; sentinels use dist=5 and dist=0 so they pass.)
  const fn = compileReward("dist === 7 ? Math.sqrt(-1) : 1");
  assert.equal(computeReward(fn, { isGoal: false, dist: 7, maxDist: 20, steps: 10, startDist: 8 }), 0);
  // Sanity: passes through valid value otherwise.
  assert.equal(computeReward(fn, { isGoal: false, dist: 5, maxDist: 20, steps: 10, startDist: 8 }), 1);
});

test("computeReward returns 0 when compiled throws at runtime", () => {
  // Compile-time-valid (sentinels avoid dist=7), throws when dist=7.
  const fn = compileReward("dist === 7 ? (null).x : 1");
  assert.equal(computeReward(fn, { isGoal: false, dist: 7, maxDist: 20, steps: 10, startDist: 8 }), 0);
  // Sanity: passes through valid value otherwise.
  assert.equal(computeReward(fn, { isGoal: false, dist: 5, maxDist: 20, steps: 10, startDist: 8 }), 1);
});
