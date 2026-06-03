// Minimal time-series line chart on a 2D canvas.
// Multiple "series" can be added by name; addPoint(name, y) appends.
// X-axis auto-scales by point index across all series.
export class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = new Map(); // name → {data: [], color: string}
    this.yMin = opts.yMin ?? null; // null = auto
    this.yMax = opts.yMax ?? null;
    this.title = opts.title ?? "";
  }
  addSeries(name, color) {
    if (!this.series.has(name)) this.series.set(name, { data: [], color });
  }
  addPoint(name, y) {
    const s = this.series.get(name);
    if (!s) return;
    s.data.push(y);
  }
  clear() { for (const s of this.series.values()) s.data.length = 0; }
  render() {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    const PAD = 24;
    ctx.fillStyle = getComputedStyle(canvas).getPropertyValue("--bg-elev") || "#222";
    ctx.fillRect(0, 0, W, H);

    // Determine axes.
    let maxLen = 0, yMin = +Infinity, yMax = -Infinity;
    for (const s of this.series.values()) {
      if (s.data.length > maxLen) maxLen = s.data.length;
      for (const y of s.data) {
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
    if (!isFinite(yMin) || !isFinite(yMax)) { yMin = -1; yMax = 1; }
    if (this.yMin !== null) yMin = this.yMin;
    if (this.yMax !== null) yMax = this.yMax;
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }

    // Title.
    ctx.fillStyle = "#bbb";
    ctx.font = "11px monospace";
    ctx.fillText(this.title, PAD, 14);

    // Axes box.
    ctx.strokeStyle = "#555";
    ctx.strokeRect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
    // y=0 reference line.
    if (yMin < 0 && yMax > 0) {
      const y0 = PAD + (H - 2 * PAD) * (1 - (0 - yMin) / (yMax - yMin));
      ctx.strokeStyle = "#666";
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(PAD, y0);
      ctx.lineTo(W - PAD, y0);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // y range labels.
    ctx.fillStyle = "#888";
    ctx.fillText(yMax.toFixed(2), 2, PAD + 4);
    ctx.fillText(yMin.toFixed(2), 2, H - PAD + 4);

    // Series lines.
    const xFor = (i) => PAD + (W - 2 * PAD) * (maxLen <= 1 ? 0 : i / (maxLen - 1));
    const yFor = (v) => PAD + (H - 2 * PAD) * (1 - (v - yMin) / (yMax - yMin));
    for (const s of this.series.values()) {
      if (s.data.length === 0) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xFor(0), yFor(s.data[0]));
      for (let i = 1; i < s.data.length; i++) ctx.lineTo(xFor(i), yFor(s.data[i]));
      ctx.stroke();
    }
  }
}

// Tiny "probe" panel: 4 bars for p plus scalar v.
export class ProbePanel {
  constructor(container) {
    this.container = container;
    this.container.innerHTML = `
      <div class="probe-title">Probe @ start</div>
      <div class="probe-bars">
        ${["up","down","left","right"].map((lbl, i) => `
          <div class="probe-row">
            <span class="probe-label">${lbl}</span>
            <div class="probe-bar"><div class="probe-fill" data-i="${i}"></div></div>
            <span class="probe-pct" data-i="${i}">0.00</span>
          </div>
        `).join("")}
      </div>
      <div class="probe-v">v = <span id="probe-v-num">0.00</span></div>
    `;
  }
  update({ p, v }) {
    for (let i = 0; i < 4; i++) {
      const fill = this.container.querySelector(`.probe-fill[data-i="${i}"]`);
      const pct = this.container.querySelector(`.probe-pct[data-i="${i}"]`);
      fill.style.width = `${(p[i] * 100).toFixed(1)}%`;
      pct.textContent = p[i].toFixed(2);
    }
    this.container.querySelector("#probe-v-num").textContent = v.toFixed(3);
  }
}
