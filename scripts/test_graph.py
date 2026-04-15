import argparse
import html
import json
from pathlib import Path


def format_attributes(attributes: dict) -> str:
    if not attributes:
        return "N/A"

    parts = []
    for key, value in attributes.items():
        parts.append(f"{html.escape(str(key))}={html.escape(str(value))}")
    return "; ".join(parts)


def format_tooltip_html(node: dict) -> str:
    metadata = node.get("metadata", {})

    lines = [
        f"<strong>ID:</strong> {html.escape(str(node.get('node_id', '')))}",
        f"<strong>Type:</strong> {html.escape(str(metadata.get('arg_type', node.get('node_type', ''))))}",
        f"<strong>Base score:</strong> {html.escape(str(node.get('base_score', '')))}",
    ]

    if node.get("node_type") == "root":
        target_item_name = metadata.get("target_item_name")
        if target_item_name is not None:
            lines.append(f"<strong>Target item:</strong> {html.escape(str(target_item_name))}")
        lines.append(f"<strong>Text:</strong> {html.escape(str(node.get('text', '')))}")
        return "<br>".join(lines)

    lines.append(f"<strong>Text:</strong> {html.escape(str(node.get('text', '')))}")

    evidence = metadata.get("evidence", [])
    if evidence:
        evidence_html = "<br>".join(
            f"- {html.escape(str(item))}" for item in evidence
        )
        lines.append(f"<strong>Evidence:</strong><br>{evidence_html}")

    llm_score = metadata.get("llm_score")
    if llm_score is not None:
        lines.append(f"<strong>LLM score:</strong> {html.escape(str(llm_score))}")

    llm_reason = metadata.get("llm_score_reason")
    if llm_reason:
        lines.append(f"<strong>LLM reason:</strong> {html.escape(str(llm_reason))}")

    mf_score = metadata.get("mf_score")
    if mf_score is not None:
        lines.append(f"<strong>MF score:</strong> {html.escape(str(mf_score))}")

    combined_score = metadata.get("combined_score")
    if combined_score is not None:
        lines.append(f"<strong>Combined score:</strong> {html.escape(str(combined_score))}")

    return "<br>".join(lines)


def build_layout(graph: dict):
    nodes = graph["nodes"]
    edges = graph["edges"]
    root_id = graph["root_id"]

    support_ids = []
    attack_ids = []

    for edge in edges:
        if edge["target_id"] != root_id:
            continue

        if edge["relation_type"] == "support":
            support_ids.append(edge["source_id"])
        elif edge["relation_type"] == "attack":
            attack_ids.append(edge["source_id"])

    positioned_nodes = {}

    width = 1400
    height = 900

    positioned_nodes[root_id] = {
        **nodes[root_id],
        "x": width / 2,
        "y": height / 2,
    }

    def place_vertical(node_ids, x):
        if not node_ids:
            return

        top_margin = 140
        bottom_margin = height - 140

        if len(node_ids) == 1:
            ys = [height / 2]
        else:
            step = (bottom_margin - top_margin) / (len(node_ids) - 1)
            ys = [top_margin + i * step for i in range(len(node_ids))]

        for node_id, y in zip(node_ids, ys):
            positioned_nodes[node_id] = {
                **nodes[node_id],
                "x": x,
                "y": y,
            }

    place_vertical(support_ids, x=260)
    place_vertical(attack_ids, x=1140)

    return positioned_nodes, edges, width, height


def build_history_html(context: dict) -> str:
    history = context.get("history", [])
    if not history:
        return "<div class='history-empty'>No history available.</div>"

    blocks = []
    for item in history:
        name = html.escape(str(item.get("name", "Unknown item")))
        rating = html.escape(str(item.get("user_stars", "N/A")))
        categories = ", ".join(item.get("categories", []))
        categories = html.escape(categories) if categories else "N/A"
        attributes = format_attributes(item.get("attributes", {}))

        blocks.append(
            f"""
            <div class="history-card">
                <div class="history-title">{name}</div>
                <div><strong>User rating:</strong> {rating}</div>
                <div><strong>Categories:</strong> {categories}</div>
                <div><strong>Attributes:</strong> {attributes}</div>
            </div>
            """
        )

    return "".join(blocks)


def build_context_panel(record: dict) -> str:
    context = record.get("context", {})
    target = context.get("target_item", {})

    if not context:
        return """
        <div class="side-panel">
            <h2>Context</h2>
            <div class="history-empty">No source context was saved in the input JSON.</div>
        </div>
        """

    target_name = html.escape(str(target.get("name", "Unknown target")))
    user_target_stars = html.escape(str(target.get("user_target_stars", "N/A")))
    normalized_score = html.escape(str(target.get("normalized_user_target_score", "N/A")))
    global_stars = html.escape(str(target.get("global_stars", "N/A")))
    categories = ", ".join(target.get("categories", []))
    categories = html.escape(categories) if categories else "N/A"
    attributes = format_attributes(target.get("attributes", {}))

    history_html = build_history_html(context)

    return f"""
    <div class="side-panel">
        <h2>Context</h2>

        <div class="target-box">
            <div class="target-title">{target_name}</div>
            <div><strong>User target rating:</strong> {user_target_stars}</div>
            <div><strong>User target score [0,1]:</strong> {normalized_score}</div>
            <div><strong>Global rating:</strong> {global_stars}</div>
            <div><strong>Categories:</strong> {categories}</div>
            <div><strong>Attributes:</strong> {attributes}</div>
        </div>

        <h3>User history</h3>
        <div class="history-list">
            {history_html}
        </div>
    </div>
    """


def build_html(record: dict) -> str:
    graph = record.get("argument_graph")
    if graph is None:
        raise ValueError("No 'argument_graph' found in the input JSON.")

    dfquad = record.get("dfquad", {})
    positioned_nodes, edges, width, height = build_layout(graph)

    node_divs = []
    edge_lines = []
    tooltip_data = {}

    for node_id, node in positioned_nodes.items():
        x = node["x"]
        y = node["y"]

        metadata = node.get("metadata", {})
        node_type = node.get("node_type", "")
        arg_type = metadata.get("arg_type", node_type)

        if node_id == graph["root_id"]:
            css_class = "node root"
            label = "ROOT"
            score_text = f"final={dfquad.get('final_score', 'N/A')}"
        else:
            css_class = "node support" if arg_type == "support" else "node attack"
            label = node_id
            combined_score = metadata.get("combined_score", node.get("base_score"))
            score_text = (
                f"{combined_score:.3f}"
                if isinstance(combined_score, (int, float))
                else str(combined_score)
            )

        tooltip_html = format_tooltip_html(node)
        tooltip_data[node_id] = tooltip_html

        safe_label = html.escape(label)
        safe_score = html.escape(score_text)

        node_divs.append(
            f"""
            <div class="{css_class}" id="node-{html.escape(node_id)}"
                 style="left:{x}px; top:{y}px;"
                 data-tooltip-id="{html.escape(node_id)}">
                <div class="node-label">{safe_label}</div>
                <div class="node-score">{safe_score}</div>
            </div>
            """
        )

    for edge in edges:
        source = positioned_nodes[edge["source_id"]]
        target = positioned_nodes[edge["target_id"]]
        relation_type = edge["relation_type"]

        color = "#2e7d32" if relation_type == "support" else "#c62828"
        dash = "0"

        x1, y1 = source["x"], source["y"]
        x2, y2 = target["x"], target["y"]

        edge_lines.append(
            f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                  stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"
                  marker-end="url(#arrow-{relation_type})" />
            """
        )

    tooltip_json = json.dumps(tooltip_data, ensure_ascii=False)
    target_name = html.escape(str(record.get("target_name", "Unknown target")))
    user_id = html.escape(str(record.get("user_id", "Unknown user")))
    dataset_index = html.escape(str(record.get("index", "N/A")))
    final_score = html.escape(str(dfquad.get("final_score", "N/A")))
    aggregated_support = html.escape(str(dfquad.get("aggregated_support", "N/A")))
    aggregated_attack = html.escape(str(dfquad.get("aggregated_attack", "N/A")))

    context_panel = build_context_panel(record)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Argument Graph - {target_name}</title>
<style>
    body {{
        margin: 0;
        font-family: Arial, sans-serif;
        background: #f7f7f9;
        color: #222;
    }}

    .page {{
        display: grid;
        grid-template-columns: 380px 1fr;
        gap: 20px;
        padding: 20px;
        min-height: 100vh;
        box-sizing: border-box;
    }}

    .side-panel, .main-panel, .header {{
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    .side-panel {{
        padding: 16px;
        height: fit-content;
        max-height: calc(100vh - 40px);
        overflow: auto;
        position: sticky;
        top: 20px;
    }}

    .side-panel h2, .side-panel h3 {{
        margin-top: 0;
    }}

    .target-box {{
        padding: 12px;
        border-radius: 10px;
        background: #eef4ff;
        margin-bottom: 16px;
        line-height: 1.5;
    }}

    .target-title {{
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .history-list {{
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}

    .history-card {{
        padding: 10px 12px;
        border-radius: 10px;
        background: #fafafa;
        border: 1px solid #e6e6e6;
        font-size: 13px;
        line-height: 1.45;
    }}

    .history-title {{
        font-weight: 700;
        margin-bottom: 6px;
    }}

    .history-empty {{
        color: #666;
        font-style: italic;
    }}

    .main-panel {{
        overflow: hidden;
        min-width: 0;
        display: flex;
        flex-direction: column;
    }}

    .header {{
        margin: 0 0 16px 0;
        padding: 16px 20px;
    }}

    .header h1 {{
        margin: 0 0 10px 0;
        font-size: 22px;
    }}

    .meta {{
        display: grid;
        grid-template-columns: repeat(2, minmax(240px, 1fr));
        gap: 8px 20px;
        font-size: 14px;
    }}

    .legend {{
        display: flex;
        gap: 18px;
        font-size: 14px;
        margin-top: 10px;
        flex-wrap: wrap;
    }}

    .legend span {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}

    .legend-box {{
        width: 14px;
        height: 14px;
        border-radius: 4px;
        display: inline-block;
    }}

    .graph-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 16px 12px 16px;
        flex-wrap: wrap;
    }}

    .graph-controls button {{
        border: none;
        background: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 14px;
        cursor: pointer;
    }}

    .graph-controls button:hover {{
        background: #163172;
    }}

    .graph-controls #zoom-level {{
        font-size: 14px;
        font-weight: 600;
        color: #333;
    }}

    .canvas-wrapper {{
        background: white;
        border-radius: 12px;
        overflow: auto;
        max-width: 100%;
        max-height: calc(100vh - 260px);
        padding: 12px;
        box-sizing: border-box;
        margin: 0 16px 16px 16px;
    }}

    .graph-stage {{
        transform-origin: top left;
        width: fit-content;
        height: fit-content;
    }}

    .graph-canvas {{
        position: relative;
        width: {width}px;
        height: {height}px;
        min-width: {width}px;
        min-height: {height}px;
        background: linear-gradient(180deg, #fcfcfd 0%, #f7f7f9 100%);
    }}

    svg.edges {{
        position: absolute;
        inset: 0;
        pointer-events: none;
    }}

    .node {{
        position: absolute;
        transform: translate(-50%, -50%);
        min-width: 110px;
        max-width: 180px;
        padding: 10px 12px;
        border-radius: 14px;
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        cursor: pointer;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        user-select: none;
    }}

    .node:hover {{
        transform: translate(-50%, -50%) scale(1.04);
        box-shadow: 0 8px 18px rgba(0,0,0,0.16);
    }}

    .node.root {{
        background: #1e3a8a;
        color: white;
        min-width: 200px;
        max-width: 240px;
    }}

    .node.support {{
        background: #e8f5e9;
        border: 2px solid #2e7d32;
        color: #1b5e20;
    }}

    .node.attack {{
        background: #ffebee;
        border: 2px solid #c62828;
        color: #8e0000;
    }}

    .node-label {{
        font-size: 14px;
        margin-bottom: 4px;
    }}

    .node-score {{
        font-size: 12px;
        opacity: 0.9;
    }}

    .tooltip {{
        position: fixed;
        z-index: 1000;
        max-width: 420px;
        background: rgba(20, 20, 24, 0.96);
        color: white;
        padding: 12px 14px;
        border-radius: 10px;
        font-size: 13px;
        line-height: 1.45;
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        display: none;
        pointer-events: none;
        white-space: normal;
    }}
</style>
</head>
<body>
<div class="page">
    {context_panel}

    <div class="main-panel">
        <div class="header">
            <h1>Interactive Argument Graph</h1>
            <div class="meta">
                <div><strong>Dataset index:</strong> {dataset_index}</div>
                <div><strong>User ID:</strong> {user_id}</div>
                <div><strong>Target item:</strong> {target_name}</div>
                <div><strong>Final DF-QuAD score:</strong> {final_score}</div>
                <div><strong>Aggregated support:</strong> {aggregated_support}</div>
                <div><strong>Aggregated attack:</strong> {aggregated_attack}</div>
            </div>

            <div class="legend">
                <span><span class="legend-box" style="background:#1e3a8a;"></span> Root claim</span>
                <span><span class="legend-box" style="background:#e8f5e9; border:2px solid #2e7d32;"></span> Support</span>
                <span><span class="legend-box" style="background:#ffebee; border:2px solid #c62828;"></span> Attack</span>
            </div>
        </div>

        <div class="graph-controls">
            <button type="button" id="zoom-out">−</button>
            <button type="button" id="zoom-in">+</button>
            <button type="button" id="zoom-reset">Reset</button>
            <span id="zoom-level">100%</span>
        </div>

        <div class="canvas-wrapper">
            <div class="graph-stage" id="graph-stage">
                <div class="graph-canvas">
                    <svg class="edges" width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">
                        <defs>
                            <marker id="arrow-support" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                                <path d="M0,0 L0,6 L9,3 z" fill="#2e7d32"></path>
                            </marker>
                            <marker id="arrow-attack" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
                                <path d="M0,0 L0,6 L9,3 z" fill="#c62828"></path>
                            </marker>
                        </defs>
                        {''.join(edge_lines)}
                    </svg>

                    {''.join(node_divs)}
                </div>
            </div>
        </div>
    </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const tooltipData = {tooltip_json};
const tooltip = document.getElementById("tooltip");

function moveTooltip(event) {{
    const offset = 18;
    const viewportPadding = 12;

    let left = event.clientX + offset;
    let top = event.clientY + offset;

    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    const tooltipRect = tooltip.getBoundingClientRect();

    const maxLeft = window.innerWidth - tooltipRect.width - viewportPadding;
    const maxTop = window.innerHeight - tooltipRect.height - viewportPadding;

    if (left > maxLeft) {{
        left = Math.max(viewportPadding, event.clientX - tooltipRect.width - offset);
    }}

    if (top > maxTop) {{
        top = Math.max(viewportPadding, event.clientY - tooltipRect.height - offset);
    }}

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
}}

function showTooltip(event, htmlContent) {{
    tooltip.innerHTML = htmlContent;
    tooltip.style.display = "block";
    moveTooltip(event);
}}

function hideTooltip() {{
    tooltip.style.display = "none";
}}

document.querySelectorAll(".node").forEach((node) => {{
    const tooltipId = node.dataset.tooltipId;
    const htmlContent = tooltipData[tooltipId] || "No details available.";

    node.addEventListener("mouseenter", (event) => {{
        showTooltip(event, htmlContent);
    }});

    node.addEventListener("mousemove", (event) => {{
        moveTooltip(event);
    }});

    node.addEventListener("mouseleave", () => {{
        hideTooltip();
    }});
}});

const graphStage = document.getElementById("graph-stage");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const zoomResetBtn = document.getElementById("zoom-reset");
const zoomLevel = document.getElementById("zoom-level");

let currentScale = 1.0;
const MIN_SCALE = 0.4;
const MAX_SCALE = 2.5;
const STEP = 0.1;

function updateZoom() {{
    graphStage.style.transform = `scale(${{currentScale}})`;
    zoomLevel.textContent = `${{Math.round(currentScale * 100)}}%`;
}}

zoomInBtn.addEventListener("click", () => {{
    currentScale = Math.min(MAX_SCALE, currentScale + STEP);
    updateZoom();
}});

zoomOutBtn.addEventListener("click", () => {{
    currentScale = Math.max(MIN_SCALE, currentScale - STEP);
    updateZoom();
}});

zoomResetBtn.addEventListener("click", () => {{
    currentScale = 1.0;
    updateZoom();
}});

updateZoom();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML visualization for an argumentative graph."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON file produced by test_dfquad.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output HTML file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    if "argument_graph" not in record:
        raise ValueError("No 'argument_graph' found in the input file.")

    html_content = build_html(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive graph saved to: {output_path}")


if __name__ == "__main__":
    main()