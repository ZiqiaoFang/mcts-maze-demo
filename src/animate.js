// Orchestrates per-phase animations for one MCTS iteration.
//
// Usage:
//   await animateIteration(mcts, { mazeRenderer, treeRenderer, currentMaze, setPhase });
//
// The animator drives the MCTS.iterate() generator, pausing between phases
// to render intermediate state with appropriate visual effects.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function animateIteration(
  mcts,
  { mazeRenderer, treeRenderer, getMaze, setPhase, setStats, onRollout }
) {
  const maze = getMaze();
  const gen = mcts.iterate();

  let selectPath = null;
  let rolloutPositions = null;

  for (const ev of gen) {
    if (ev.phase === "select") {
      setPhase("select");
      selectPath = ev.path;
      treeRenderer.setSelectionHighlight(selectPath);
      mazeRenderer.render(maze, {
        visits: mcts.cellVisits(),
        principalVariation: mcts.principalVariation().map((n) => n.position),
        selectionPath: selectPath.map((n) => n.position),
      });
      treeRenderer.render(mcts);
      await sleep(400);
    } else if (ev.phase === "expand") {
      setPhase("expand");
      treeRenderer.render(mcts);
      await sleep(200);
    } else if (ev.phase === "simulate") {
      setPhase("simulate");
      rolloutPositions = ev.positions;
      if (typeof onRollout === "function") onRollout(ev);
      // Animate trail building up cell by cell.
      const total = rolloutPositions.length;
      const maxAnimatedSteps = 30;
      const animatedSteps = Math.min(total, maxAnimatedSteps);
      for (let i = 1; i <= animatedSteps; i++) {
        mazeRenderer.render(maze, {
          visits: mcts.cellVisits(),
          principalVariation: mcts.principalVariation().map((n) => n.position),
          selectionPath: selectPath.map((n) => n.position),
          rolloutTrail: rolloutPositions.slice(0, i + 1),
        });
        await sleep(40);
      }
      if (total > maxAnimatedSteps) {
        // Fast-forward: render full trail at once.
        mazeRenderer.render(maze, {
          visits: mcts.cellVisits(),
          principalVariation: mcts.principalVariation().map((n) => n.position),
          selectionPath: selectPath.map((n) => n.position),
          rolloutTrail: rolloutPositions,
        });
        await sleep(150);
      }
    } else if (ev.phase === "backprop") {
      setPhase("backprop");
      treeRenderer.render(mcts);
      await sleep(400);
    }
  }

  // Final cleanup: clear selection / rollout, redraw with PV only.
  treeRenderer.clearSelectionHighlight();
  mazeRenderer.render(maze, {
    visits: mcts.cellVisits(),
    principalVariation: mcts.principalVariation().map((n) => n.position),
  });
  treeRenderer.render(mcts);
  setStats();
  setPhase("idle");
}
