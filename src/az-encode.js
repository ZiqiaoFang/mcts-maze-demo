// Number of input channels: walls, agent one-hot, goal one-hot.
export const CHANNELS = 3;

// Encode a maze + agent position into a flat Float32Array suitable for
// reshaping to [size, size, CHANNELS] on the TF.js side.
// Layout: index = (r * size + c) * CHANNELS + ch
//   ch 0: 1.0 where the cell is a wall (grid[r][c] === 1), else 0.
//   ch 1: 1.0 at the agent position, else 0.
//   ch 2: 1.0 at the goal position, else 0.
export function encodeState(maze, agentPos) {
  const size = maze.size;
  const out = new Float32Array(size * size * CHANNELS);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const base = (r * size + c) * CHANNELS;
      if (maze.grid[r][c] === 1) out[base] = 1;
    }
  }
  const [ar, ac] = agentPos;
  out[(ar * size + ac) * CHANNELS + 1] = 1;
  const [gr, gc] = maze.goal;
  out[(gr * size + gc) * CHANNELS + 2] = 1;
  return out;
}
