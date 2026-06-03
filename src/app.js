import { initClassicTab } from "./classic-tab.js";

initClassicTab();

let azInitialized = false;
function activate(tabName) {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  }
  for (const pane of document.querySelectorAll(".tab-pane")) {
    pane.classList.toggle("active", pane.id === `tab-${tabName}`);
  }
  if (tabName === "az" && !azInitialized) {
    azInitialized = true;
    import("./az-tab.js").then((m) => m.initAzTab());
  }
}
for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => activate(btn.dataset.tab));
}
