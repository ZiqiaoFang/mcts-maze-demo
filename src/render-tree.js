// D3-based tree renderer. Pass it an MCTS instance and it renders the tree.
// d3 is loaded as a global from the CDN script tag.

export class TreeRenderer {
  constructor(svgElement) {
    this.svg = d3.select(svgElement);
    this.svg.selectAll("*").remove();

    this.g = this.svg.append("g").attr("class", "tree-root");

    // Tooltip element
    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "tree-tooltip")
      .style("position", "fixed")
      .style("padding", "6px 10px")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "#fff")
      .style("border", "1px solid #444")
      .style("border-radius", "4px")
      .style("font", "12px monospace")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 1000);

    // Zoom/pan
    this.zoomBehavior = d3.zoom().scaleExtent([0.25, 4]).on("zoom", (ev) => {
      this.g.attr("transform", ev.transform);
    });
    this.svg.call(this.zoomBehavior);

    this.selectedPath = new Set();
    this.flashEdge = null;
  }

  setSelectionHighlight(path) {
    // path is an array of MCTS Nodes.
    this.selectedPath = new Set(path);
    this._restyle();
  }

  clearSelectionHighlight() {
    this.selectedPath = new Set();
    this._restyle();
  }

  render(mcts) {
    const width = this.svg.node().clientWidth || 600;
    const height = this.svg.node().clientHeight || 600;

    // Build a D3 hierarchy from MCTS root.
    const toD3 = (n) => ({
      data: n,
      children: [...n.children.values()].map(toD3),
    });
    const root = d3.hierarchy(toD3(mcts.root));

    const treeLayout = d3.tree().size([width - 40, height - 40]);
    treeLayout(root);

    // Determine maxVisits/maxReward for scaling.
    let maxVisits = 1;
    let maxReward = 0.0001;
    root.each((d) => {
      const n = d.data.data;
      if (n.visits > maxVisits) maxVisits = n.visits;
      if (n.meanReward() > maxReward) maxReward = n.meanReward();
    });

    // Edges
    const links = root.links();
    const edge = this.g
      .selectAll("line.edge")
      .data(links, (d) => `${d.source.data.data.id}->${d.target.data.data.id}`);
    edge.exit().remove();
    edge
      .enter()
      .append("line")
      .attr("class", "edge")
      .merge(edge)
      .attr("x1", (d) => d.source.x + 20)
      .attr("y1", (d) => d.source.y + 20)
      .attr("x2", (d) => d.target.x + 20)
      .attr("y2", (d) => d.target.y + 20)
      .attr("stroke", "#555")
      .attr("stroke-width", 1);

    // Nodes
    const nodes = root.descendants();
    const node = this.g
      .selectAll("circle.node")
      .data(nodes, (d) => d.data.data.id);
    node.exit().remove();
    const nodeEnter = node
      .enter()
      .append("circle")
      .attr("class", "node")
      .attr("r", 0);

    nodeEnter
      .merge(node)
      .attr("cx", (d) => d.x + 20)
      .attr("cy", (d) => d.y + 20)
      .attr("r", (d) => {
        const v = d.data.data.visits;
        return Math.max(3, Math.sqrt(v) * 2.5);
      })
      .attr("fill", (d) => {
        const mr = d.data.data.meanReward();
        const t = Math.min(1, mr / 1.5);
        const r0 = 255, g0 = 255, b0 = 255;
        const r1 = 74, g1 = 222, b1 = 128;
        const rr = Math.round(r0 + (r1 - r0) * t);
        const gg = Math.round(g0 + (g1 - g0) * t);
        const bb = Math.round(b0 + (b1 - b0) * t);
        return `rgb(${rr},${gg},${bb})`;
      })
      .attr("stroke", (d) => (this.selectedPath.has(d.data.data) ? "#fbbf24" : "#888"))
      .attr("stroke-width", (d) => (this.selectedPath.has(d.data.data) ? 3 : 1))
      .on("mouseover", (event, d) => {
        const n = d.data.data;
        const ucb = n.parent ? n.ucb1(mcts.config.C, n.parent.visits) : 0;
        this.tooltip
          .html(
            `pos: (${n.position[0]},${n.position[1]})<br>` +
              `visits: ${n.visits}<br>` +
              `mean reward: ${n.meanReward().toFixed(3)}<br>` +
              `UCB1: ${Number.isFinite(ucb) ? ucb.toFixed(3) : "∞"}`
          )
          .style("opacity", 1);
      })
      .on("mousemove", (event) => {
        this.tooltip.style("left", event.clientX + 12 + "px").style("top", event.clientY + 12 + "px");
      })
      .on("mouseout", () => {
        this.tooltip.style("opacity", 0);
      });
  }

  _restyle() {
    this.g
      .selectAll("circle.node")
      .attr("stroke", (d) => (this.selectedPath.has(d.data.data) ? "#fbbf24" : "#888"))
      .attr("stroke-width", (d) => (this.selectedPath.has(d.data.data) ? 3 : 1));
  }
}
