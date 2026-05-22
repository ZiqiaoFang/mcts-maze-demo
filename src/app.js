import { animateIteration } from "./animate.js";
import { MAZES } from "./maze.js";
import { MCTS } from "./mcts.js";
import { MazeRenderer } from "./render-maze.js";
import { TreeRenderer } from "./render-tree.js";

const canvas = document.getElementById("maze-canvas");
const svg = document.getElementById("tree-svg");
const sizeSel = document.getElementById("ctl-size");
const rewardSel = document.getElementById("ctl-reward");
const cSlider = document.getElementById("ctl-c");
const hSlider = document.getElementById("ctl-horizon");
const cVal = document.getElementById("ctl-c-val");
const hVal = document.getElementById("ctl-horizon-val");
const statIter = document.getElementById("stat-iter");
const statNodes = document.getElementById("stat-nodes");
const statPV = document.getElementById("stat-pv");
const statPhase = document.getElementById("stat-phase");

const mazeRenderer = new MazeRenderer(canvas);
const treeRenderer = new TreeRenderer(svg);

let mcts = null;

function currentMaze() {
  return MAZES[parseInt(sizeSel.value, 10)];
}

function newMcts() {
  const maze = currentMaze();
  mcts = new MCTS(maze, {
    C: parseFloat(cSlider.value),
    rolloutHorizon: parseInt(hSlider.value, 10),
    rewardMode: rewardSel.value,
    seed: 42,
  });
  renderAll();
}

function renderAll() {
  const maze = currentMaze();
  const pv = mcts.principalVariation().map((n) => n.position);
  mazeRenderer.render(maze, {
    visits: mcts.cellVisits(),
    principalVariation: pv,
  });
  treeRenderer.render(mcts);
  const s = mcts.treeStats();
  statIter.textContent = mcts.iterationCount;
  statNodes.textContent = s.totalNodes;
  statPV.textContent = pv.length - 1;
  if (maze.isGoal(pv[pv.length - 1])) statPV.style.color = "#4ade80";
  else statPV.style.color = "";
}

let busy = false;
const setButtonsEnabled = (enabled) => {
  for (const id of ["btn-step", "btn-run10", "btn-runend", "btn-reset"]) {
    document.getElementById(id).disabled = !enabled;
  }
};

document.getElementById("btn-step").addEventListener("click", async () => {
  if (busy) return;
  busy = true;
  setButtonsEnabled(false);
  await animateIteration(mcts, {
    mazeRenderer,
    treeRenderer,
    getMaze: currentMaze,
    setPhase: (p) => (statPhase.textContent = p),
    setStats: renderAll, // updates stats only; full re-render already done
  });
  setButtonsEnabled(true);
  busy = false;
});
document.getElementById("btn-run10").addEventListener("click", () => {
  if (busy) return;
  busy = true;
  setButtonsEnabled(false);
  statPhase.textContent = "running 10";
  for (let i = 0; i < 10; i++) for (const _ev of mcts.iterate()) {}
  renderAll();
  statPhase.textContent = "idle";
  setButtonsEnabled(true);
  busy = false;
});
document.getElementById("btn-runend").addEventListener("click", () => {
  if (busy) return;
  busy = true;
  setButtonsEnabled(false);
  statPhase.textContent = "running to end";
  const maze = currentMaze();
  let stableCount = 0;
  let lastPVKey = "";
  for (let i = 0; i < 500; i++) {
    for (const _ev of mcts.iterate()) {}
    const pv = mcts.principalVariation();
    const last = pv[pv.length - 1].position;
    const pvKey = pv.map((n) => n.position.join(",")).join("|");
    if (maze.isGoal(last) && pvKey === lastPVKey) stableCount++;
    else stableCount = 0;
    lastPVKey = pvKey;
    if (stableCount >= 5) break;
  }
  renderAll();
  statPhase.textContent = "idle";
  setButtonsEnabled(true);
  busy = false;
});
document.getElementById("btn-reset").addEventListener("click", () => {
  if (busy) return;
  newMcts();
});

sizeSel.addEventListener("change", () => {
  const size = parseInt(sizeSel.value, 10);
  hSlider.max = size * 8;
  hSlider.value = size * 4;
  hVal.textContent = hSlider.value;
  newMcts();
});
rewardSel.addEventListener("change", newMcts);
cSlider.addEventListener("input", () => {
  cVal.textContent = parseFloat(cSlider.value).toFixed(2);
  if (mcts) mcts.setConfig({ C: parseFloat(cSlider.value) });
});
hSlider.addEventListener("input", () => {
  hVal.textContent = hSlider.value;
  if (mcts) mcts.setConfig({ rolloutHorizon: parseInt(hSlider.value, 10) });
});

newMcts();
