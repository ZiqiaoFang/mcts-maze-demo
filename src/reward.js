/**
 * Three stock formulas for the UI presets menu. Each is a single JS expression
 * referencing the variables documented in compileReward().
 */
export const PRESETS = {
  "Distance-shaped": "(isGoal ? 1 : 0) + Math.max(0, 1 - dist / maxDist)",
  "Pure win/loss":   "isGoal ? 1 : 0",
  "Step penalty":    "isGoal ? 1 : 0 - 0.01 * steps",
};

/**
 * Compile a reward formula string into a callable.
 *
 * The formula is a single JS expression with access to:
 *   isGoal    - boolean: rollout ended at goal cell
 *   dist      - number:  Manhattan distance from rollout end to goal
 *   maxDist   - number:  Manhattan diameter of the maze
 *   steps     - number:  rollout steps actually taken
 *   startDist - number:  Manhattan distance from rollout start to goal
 *
 * Plus the standard Math object.
 *
 * Two sentinel calls are made at compile time; if either throws or returns
 * a non-finite/non-number value, compileReward throws with a friendly message.
 *
 * @param {string} formulaString
 * @returns {(isGoal, dist, maxDist, steps, startDist) => number}
 * @throws {Error} on syntax errors or sentinel-run failures
 */
export function compileReward(formulaString) {
  let fn;
  try {
    fn = new Function(
      "isGoal", "dist", "maxDist", "steps", "startDist",
      `"use strict"; return (${formulaString});`
    );
  } catch (err) {
    throw new Error(`Bad formula: ${err.message}`);
  }

  // Sentinel inputs: cover non-goal + nonzero dist, and goal + zero dist.
  const sentinels = [
    [false, 5, 20, 10, 8],
    [true, 0, 20, 10, 8],
  ];
  for (const args of sentinels) {
    let r;
    try {
      r = fn(...args);
    } catch (err) {
      throw new Error(`Formula threw: ${err.name}: ${err.message}`);
    }
    if (typeof r !== "number") {
      throw new Error(`Formula returned a non-number (${typeof r})`);
    }
    if (!Number.isFinite(r)) {
      throw new Error(`Formula returned a non-finite value`);
    }
  }
  return fn;
}

/**
 * Call a compiled reward function with the per-rollout context.
 * Any thrown exception or non-finite return is silently turned into 0.
 *
 * @param {(isGoal, dist, maxDist, steps, startDist) => number} compiled
 * @param {{isGoal: boolean, dist: number, maxDist: number, steps: number, startDist: number}} ctx
 * @returns {number}
 */
export function computeReward(compiled, ctx) {
  let r;
  try {
    r = compiled(ctx.isGoal, ctx.dist, ctx.maxDist, ctx.steps, ctx.startDist);
  } catch {
    return 0;
  }
  return Number.isFinite(r) ? r : 0;
}
