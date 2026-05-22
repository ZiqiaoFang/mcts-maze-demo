import { MAZES } from "./maze.js";
import { MazeRenderer } from "./render-maze.js";

const canvas = document.getElementById("maze-canvas");
const sizeSel = document.getElementById("ctl-size");

const renderer = new MazeRenderer(canvas);

function render() {
  const size = parseInt(sizeSel.value, 10);
  const maze = MAZES[size];
  renderer.render(maze);
}

sizeSel.addEventListener("change", render);
render();
