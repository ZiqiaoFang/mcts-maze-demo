/**
 * Compute reward at the end of a rollout.
 * @param {Maze} maze
 * @param {[number,number]} finalPos - last position in the rollout
 * @param {"shaped"|"winloss"} mode
 * @returns {number}
 */
export function computeReward(maze, finalPos, mode) {
  if (mode === "winloss") {
    return maze.isGoal(finalPos) ? 1 : 0;
  }
  if (mode === "shaped") {
    const maxD = maze.maxDistance();
    const d = maze.manhattan(finalPos, maze.goal);
    const base = Math.max(0, 1 - d / maxD);
    return base + (maze.isGoal(finalPos) ? 1 : 0);
  }
  throw new Error(`Unknown reward mode: ${mode}`);
}
