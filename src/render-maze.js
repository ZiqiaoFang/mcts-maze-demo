// Canvas-based maze renderer. Stateless: pass it the maze and a render state,
// and it draws the current frame.

export class MazeRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
  }

  /**
   * @param {Maze} maze
   * @param {object} state
   *   - visits: Map<"r,c", number>     // cell visit counts for heatmap
   *   - principalVariation: [r,c][]    // PV polyline
   *   - selectionPath: [r,c][]         // cells outlined during select phase
   *   - rolloutTrail: [r,c][]          // current rollout path so far
   */
  render(maze, state = {}) {
    const { ctx, canvas } = this;
    const cell = Math.floor(Math.min(canvas.width, canvas.height) / maze.size);
    const offX = (canvas.width - cell * maze.size) / 2;
    const offY = (canvas.height - cell * maze.size) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Heatmap normalization
    const visits = state.visits || new Map();
    let maxVisits = 0;
    for (const v of visits.values()) if (v > maxVisits) maxVisits = v;

    // Grid + heatmap
    for (let r = 0; r < maze.size; r++) {
      for (let c = 0; c < maze.size; c++) {
        const x = offX + c * cell;
        const y = offY + r * cell;
        const isWall = maze.grid[r][c] === 1;
        if (isWall) {
          ctx.fillStyle = "#3a3a3a";
        } else {
          const v = visits.get(`${r},${c}`) || 0;
          const t = maxVisits ? v / maxVisits : 0;
          // Blend from #1f1f1f (cold) to #3a82f6 (hot blue)
          const r0 = 31, g0 = 31, b0 = 31;
          const r1 = 58, g1 = 130, b1 = 246;
          const rr = Math.round(r0 + (r1 - r0) * t);
          const gg = Math.round(g0 + (g1 - g0) * t);
          const bb = Math.round(b0 + (b1 - b0) * t);
          ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
        }
        ctx.fillRect(x, y, cell, cell);
        ctx.strokeStyle = "#0a0a0a";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }

    // Selection path outlines
    if (state.selectionPath?.length) {
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 3;
      for (const [r, c] of state.selectionPath) {
        ctx.strokeRect(offX + c * cell + 2, offY + r * cell + 2, cell - 4, cell - 4);
      }
    }

    // Rollout trail
    if (state.rolloutTrail?.length > 1) {
      ctx.strokeStyle = "#fb923c";
      ctx.lineWidth = Math.max(2, cell * 0.15);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      const cx = (r, c) => offX + c * cell + cell / 2;
      const cy = (r, c) => offY + r * cell + cell / 2;
      ctx.moveTo(cx(...state.rolloutTrail[0]), cy(...state.rolloutTrail[0]));
      for (const [r, c] of state.rolloutTrail.slice(1)) ctx.lineTo(cx(r, c), cy(r, c));
      ctx.stroke();
    }

    // Principal variation
    if (state.principalVariation?.length > 1) {
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = Math.max(2, cell * 0.2);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      const cx = (r, c) => offX + c * cell + cell / 2;
      const cy = (r, c) => offY + r * cell + cell / 2;
      ctx.moveTo(cx(...state.principalVariation[0]), cy(...state.principalVariation[0]));
      for (const [r, c] of state.principalVariation.slice(1)) ctx.lineTo(cx(r, c), cy(r, c));
      ctx.stroke();
    }

    // Start dot
    ctx.fillStyle = "#4ade80";
    ctx.beginPath();
    ctx.arc(offX + maze.start[1] * cell + cell / 2, offY + maze.start[0] * cell + cell / 2, cell * 0.25, 0, Math.PI * 2);
    ctx.fill();

    // Goal flag (red square)
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(
      offX + maze.goal[1] * cell + cell * 0.2,
      offY + maze.goal[0] * cell + cell * 0.2,
      cell * 0.6,
      cell * 0.6
    );
  }
}
