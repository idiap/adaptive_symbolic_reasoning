# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Utilities for converting execution traces into human-friendly HTML reports."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.meta_agents.planner import Plan
from base.helper import TracePersister


def _stringify(value: Any) -> Any:
    """Return JSON-serialisable data, falling back to string conversion."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def serialize_plan(plan: Optional[Plan]) -> Dict[str, Any]:
    if plan is None:
        return {}
    agents_snapshot: List[Dict[str, Any]] = []
    for name, agent in getattr(plan, "agents", {}).items():
        agents_snapshot.append(
            {
                "name": name,
                "goal": getattr(agent, "goal", ""),
                "predecessors": list(getattr(agent, "predecessors", []) or []),
                "problem_to_solve": list(getattr(agent, "problem_to_solve", []) or []),
            }
        )
    edges_snapshot = [f"{edge.source}->{edge.target}" for edge in getattr(plan, "edges", [])]
    return {"agents": agents_snapshot, "edges": edges_snapshot}


def snapshot_memory(memory) -> Dict[str, Any]:
    """Create a serialisable snapshot of the Scratchpad contents."""
    if memory is None:
        return {}
    snapshot: Dict[str, Any] = {}
    store = getattr(memory, "_store", {})
    for key, value in store.items():
        val, _expiry = value if isinstance(value, (tuple, list)) and len(value) == 2 else (value, None)
        snapshot[key] = _stringify(val)
    return snapshot


def build_trace_payload(
    plan: Optional[Plan],
    tracer: Optional[TracePersister],
    *,
    memory=None,
    metadata: Optional[Dict[str, Any]] = None,
    problem_statement: Optional[str] = None,
) -> Dict[str, Any]:
    """Bundle all useful artefacts into a single dictionary for prompting/rendering."""
    trace_json: List[Dict[str, Any]] = []
    if tracer is not None:
        raw = tracer.dump_json()
        try:
            trace_json = json.loads(raw)
        except json.JSONDecodeError:
            trace_json = [raw]
    payload: Dict[str, Any] = {
        "plan": serialize_plan(plan),
        "trace": trace_json,
        "memory": snapshot_memory(memory),
        "metadata": metadata or {},
    }
    if problem_statement:
        payload["problem_statement"] = problem_statement
    return payload


# ---------------------------------------------------------------------------
# Structured rendering helpers
# ---------------------------------------------------------------------------

AREA_CLASS_MAP = {
    "cv": "cv",
    "computer vision": "cv",
    "nlp": "nlp",
    "natural-language processing": "nlp",
    "sys": "sys",
    "systems": "sys",
    "rl": "rl",
    "reinforcement learning": "rl",
    "theory": "theory",
}


def _normalise_session_name(name: str | None) -> str:
    if not name:
        return ""
    return name.replace("_", "-")


def _session_symbol(name: str | None) -> str:
    if not name:
        return ""
    return name.replace("-", "_").lower()


def _prepare_context(trace_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract general information from trace payload.
    No problem-specific parsing - let LLM handle that.
    """
    metadata = trace_payload.get("metadata", {}) or {}
    scenario = metadata.get("scenario") or metadata.get("title") or "Trace Explanation"
    problem_statement = (trace_payload.get("problem_statement") or "").strip()
    plan_data = trace_payload.get("plan") or {}
    agents = plan_data.get("agents", [])
    edges = plan_data.get("edges", [])
    trace_events = trace_payload.get("trace", [])
    result_summary = metadata.get("result_summary") or []

    # Extract solver information (generic)
    solver_names = set()
    for item in result_summary:
        if item.get("solver_name"):
            solver_names.add(item.get("solver_name"))

    # Summarize trace events (generic)
    trace_highlights = []
    for event in trace_events[:10]:  # Show more events
        highlight = {
            "agent": event.get("agent"),
            "output_summary": _summarise_text(event.get("output"), limit=200),
        }
        trace_highlights.append(highlight)

    return {
        "scenario": scenario,
        "solver_names": list(solver_names),
        "problem_statement": problem_statement,
        "plan_agents": agents,
        "plan_edges": edges,
        "trace_events": trace_events,
        "trace_highlights": trace_highlights,
        "result_summary": result_summary,  # Pass through for LLM
        "metadata": metadata,
        "memory": trace_payload.get("memory", {}),  # Include raw memory for LLM
    }


def _summarise_text(value: Any, limit: int = 160) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return (text[: limit].rstrip() + ("…" if len(text) > limit else ""))


def _generate_commentary(
    generator,
    structured: Dict[str, Any],
    narrative_context: str,
    style_hint: Optional[str],
    extra_guidance: Optional[str],
    model_args: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate problem-specific HTML sections using LLM.
    Returns dict with 'html_sections' containing complete HTML.
    """
    # Prepare data for LLM (include everything that might be useful)
    context_blob = {
        "scenario": structured.get("scenario"),
        "problem_statement": structured.get("problem_statement"),
        "counts": structured.get("counts"),
        "plan_agents": structured.get("plan_agents"),
        "plan_edges": structured.get("plan_edges"),
        "result_entries": structured.get("result_entries"),
        "trace_events": structured.get("trace_events"),
        "memory_snapshot": structured.get("memory", {}),
    }

    replacements = {
        "context": json.dumps(context_blob, ensure_ascii=False, indent=2),
        "narrative_context": narrative_context.strip(),
        "style_hint": (style_hint or "").strip(),
        "extra_guidance": (extra_guidance or "").strip(),
    }

    commentary_args = dict(model_args or {})
    commentary_args.setdefault("temperature", 0.2)
    # Remove json_object format - we want HTML
    commentary_args.pop("response_format", None)

    response = generator.generate(
        model_prompt_dir="helpers",
        prompt_name="trace_visualization.txt",  # Use sections-only prompt
        model_args=commentary_args,
        **replacements,
    )

    # Clean up markdown wrappers if LLM added them
    html_content = response.strip()
    if html_content.startswith("```html"):
        html_content = html_content[7:]
    if html_content.startswith("```"):
        html_content = html_content[3:]
    if html_content.endswith("```"):
        html_content = html_content[:-3]
    html_content = html_content.strip()

    return {"html_sections": html_content}


def _render_html(structured: Dict[str, Any], commentary: Dict[str, List[str]]) -> str:
    """
    Render fixed sections only (header, overview, problem, plan, results).
    LLM-generated commentary contains problem-specific visualizations.
    """
    sections: List[str] = []
    # Fixed sections (controlled by code)
    sections.append(_render_header(structured))
    sections.append(_render_overview(structured))  # Includes results display
    sections.append(_render_problem_block(structured))
    sections.append(_render_plan_block(structured))

    # LLM-generated commentary (flexible visualizations)
    sections.append(_render_commentary_block(commentary))

    # Optional: raw trace for debugging (usually not needed)
    # sections.append(_render_trace_block(structured))

    body = "\n".join(section for section in sections if section)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{escape(structured.get('scenario', 'Trace Explanation'))}</title>
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap\" rel=\"stylesheet\">
{_BASE_STYLE}
</head>
<body>
<div class=\"container\">
{body}
  <footer>Generated on-demand from the execution trace · Formal solvers provide the guarantees, interpretations may not.</footer>
</div>
</body>
</html>"""


_BASE_STYLE = """
<style>
  :root{
    --bg:#0b1220;
    --panel:#111a2b;
    --soft:#1a2438;
    --ink:#e8eefc;
    --sub:#9fb3d9;
    --accent:#7cc4ff;
    --accent-2:#b892ff;
    --ok:#4ade80;
    --warn:#facc15;
    --bad:#fb7185;
    --cv:#8b5cf6;
    --nlp:#f59e0b;
    --sys:#10b981;
    --rl:#3b82f6;
    --theory:#ec4899;
    --code:#0f172a;
    --chip:#233251;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:linear-gradient(180deg,#0b1220 0%, #0a0f1c 60%, #0b1220 100%);color:var(--ink);font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,"Helvetica Neue",Arial,"Noto Sans",sans-serif;line-height:1.55}
  a{color:var(--accent)}
  .container{max-width:1200px;margin:32px auto;padding:0 20px}
  header{display:flex;gap:16px;align-items:center;margin-bottom:24px}
  .logo{width:46px;height:46px;border-radius:12px;background:radial-gradient(120% 120% at 20% 20%, #7cc4ff 0%, #b892ff 40%, #111a2b 100%);box-shadow:0 10px 30px rgba(124,196,255,.25), inset 0 0 14px rgba(184,146,255,.3)}
  h1{font-size:30px;margin:0;font-weight:800;letter-spacing:.2px}
  .subtitle{color:var(--sub);margin-top:2px;font-weight:500}
  .grid{display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
  .card{grid-column:span 12;background:linear-gradient(180deg,var(--panel), #0f172a);border:1px solid rgba(124,196,255,.15);border-radius:16px;padding:18px 20px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
  .card h2{margin:2px 0 10px 0;font-size:20px;letter-spacing:.2px}
  .card h3{margin:16px 0 8px 0;font-size:16px;color:#cbd5e1}
  .meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px}
  .chip{background:var(--chip);color:var(--ink);padding:6px 12px;border-radius:999px;border:1px solid rgba(124,196,255,.18);font-size:12px}
  code, pre{background:var(--code);color:#dbeafe;border-radius:10px}
  pre{padding:14px;overflow:auto;border:1px solid rgba(124,196,255,.18);font-size:12px}
  table{width:100%;border-collapse:separate;border-spacing:0 6px}
  th,td{text-align:left;padding:8px 10px;font-size:13px}
  th{color:#cbd5e1;font-weight:700;border-bottom:1px solid rgba(124,196,255,.25)}
  tr{background:rgba(26,36,56,.65)}
  tr td:first-child{border-top-left-radius:10px;border-bottom-left-radius:10px}
  tr td:last-child{border-top-right-radius:10px;border-bottom-right-radius:10px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;font-weight:700}
  .pill.ok{background:rgba(34,197,94,.12);color:#bbf7d0;border:1px solid rgba(34,197,94,.35)}
  .pill.warn{background:rgba(250,204,21,.12);color:#fde68a;border:1px solid rgba(250,204,21,.35)}
  .session-box{background:rgba(26,36,56,.8);border:1px solid rgba(124,196,255,.2);border-radius:12px;padding:14px}
  .session-box h4{margin:0 0 8px 0;font-size:15px;color:#cbd5e1;font-weight:700}
  .paper-badge{display:inline-block;padding:4px 8px;margin:3px;border-radius:6px;font-size:11px;font-weight:600;border:1px solid rgba(255,255,255,.12)}
  .paper-badge.cv{background:rgba(139,92,246,.2);color:#c4b5fd}
  .paper-badge.nlp{background:rgba(245,158,11,.2);color:#fcd34d}
  .paper-badge.sys{background:rgba(16,185,129,.2);color:#6ee7b7}
  .paper-badge.rl{background:rgba(59,130,246,.2);color:#93c5fd}
  .paper-badge.theory{background:rgba(236,72,153,.2);color:#f9a8d4}
  .oral-badge{background:rgba(124,196,255,.15);color:#7cc4ff;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:600;margin-left:8px}
  .poster-badge{background:rgba(184,146,255,.15);color:#b892ff;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:600;margin-left:8px}
  .plan-flow{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;padding:0}
  .plan-flow li{list-style:none;background:rgba(255,255,255,.04);padding:6px 12px;border-radius:999px;border:1px solid rgba(124,196,255,.18);font-size:12px}
  .dag-svg rect.module{fill:rgba(17,26,43,.95);stroke:rgba(124,196,255,.6);stroke-width:2.5}
  .dag-svg text{font-family:Inter,sans-serif}
  .trace-table{width:100%;margin-top:10px}
  .trace-table tr td{font-size:12px}
  details{margin-top:8px}
  summary{cursor:pointer;font-weight:600;color:var(--accent)}
  footer{margin:30px 0 12px 0;color:#8aa2c8;font-size:12px;text-align:center}
  @media (max-width: 900px){ .grid{grid-template-columns:1fr} }
</style>
"""


def _render_header(structured: Dict[str, Any]) -> str:
    solver = structured.get("engine_label") or "Adaptive Solver"
    return f"""
  <header>
    <div class=\"logo\" aria-hidden=\"true\"></div>
    <div>
      <h1>{escape(structured.get('scenario', 'Trace Explanation'))}</h1>
      <div class=\"subtitle\">Engine: <span class=\"mono\">{escape(solver)}</span></div>
    </div>
  </header>
"""


def _render_overview(structured: Dict[str, Any]) -> str:
    """Render scenario overview with complete results"""
    meta = structured.get("metadata", {})
    result_summary = structured.get("result_summary", [])
    solver_names = structured.get("solver_names", [])
    description = meta.get("scenario") or structured.get("scenario")

    # Generic solver info
    solver_chips = "".join(f'<span class="chip">Solver: {escape(s)}</span>' for s in solver_names) if solver_names else ""

    # Build complete results display
    result_items = []
    if result_summary:
        for entry in result_summary:
            problem_id = entry.get("problem_id") or "task"

            # Check if this is a CSP problem with assignments
            assignments = entry.get("assignments")
            if assignments:
                # CSP: Show ALL assignments
                assignment_dict = assignments.get("paper_session") or assignments
                if isinstance(assignment_dict, dict):
                    items = [f"{k}→{v}" for k, v in assignment_dict.items()]
                    result_items.append(f'<li><strong>{escape(problem_id)}:</strong> {escape(", ".join(items))}</li>')
                else:
                    result_items.append(f'<li><strong>{escape(problem_id)}:</strong> Assignment found</li>')
            else:
                # LP/FOL/SMT: Show parsed answer
                parsed_answer = entry.get("parsed_answer")
                if parsed_answer:
                    answer_display = "Yes" if parsed_answer == "A" else "No" if parsed_answer == "B" else parsed_answer
                    pill_class = "ok" if parsed_answer == "A" else "no"
                    result_items.append(f'<li><strong>{escape(problem_id)}:</strong> <span class="pill {pill_class}">{escape(answer_display)}</span></li>')
                else:
                    result_items.append(f'<li><strong>{escape(problem_id)}:</strong> No answer</li>')

    results_html = "".join(result_items) if result_items else '<li>No solver outputs captured yet.</li>'

    return f"""
  <section class=\"grid\">
    <div class=\"card\" style=\"grid-column:span 12\">
      <h2>Scenario Overview & Results</h2>
      <p class=\"small\">{escape(description or 'Adaptive reasoning trace')}</p>
      <div class=\"meta\">
        {solver_chips}
        <span class=\"chip\">Agents in plan: {len(structured.get('plan_agents', []))}</span>
        <span class=\"chip\">Trace events: {len(structured.get('trace_events', []))}</span>
      </div>
      <h3 style=\"margin-top:16px\">Results</h3>
      <ul class=\"small\" style=\"margin:0 0 0 18px\">
        {results_html}
      </ul>
    </div>
  </section>
"""


def _render_problem_block(structured: Dict[str, Any]) -> str:
    statement = structured.get("problem_statement", "")
    if not statement:
        return ""
    return f"""
  <section class=\"grid\" style=\"margin-top:8px\">
    <div class=\"card\" style=\"grid-column:span 12\">
      <h2>Original Problem Statement</h2>
      <details open>
        <summary>Show / hide</summary>
        <pre>{escape(statement)}</pre>
      </details>
    </div>
  </section>
"""


def _render_plan_block(structured: Dict[str, Any]) -> str:
    agents = structured.get("plan_agents", [])
    edges = structured.get("plan_edges", [])
    if not agents and not edges:
        return ""
    flow_items = []
    for agent in agents:
        flow_items.append(f"<li>{escape(agent.get('name', ''))}</li>")
    edge_list = "<br>".join(escape(edge) for edge in edges)
    agent_rows = "".join(
        f"<tr><td>{escape(agent.get('name',''))}</td><td>{escape(agent.get('goal',''))}</td></tr>" for agent in agents
    )
    svg = _render_plan_svg(agents, edges)
    return f"""
  <section class=\"grid\" style=\"margin-top:8px\">
    <div class=\"card\" style=\"grid-column:span 12\">
      <h2>Plan & DAG</h2>
      <ul class=\"plan-flow\">{''.join(flow_items)}</ul>
      {svg}
      <table>
        <thead><tr><th>Agent</th><th>Goal</th></tr></thead>
        <tbody>{agent_rows}</tbody>
      </table>
      {'<div class="small" style="margin-top:8px">Edges: ' + edge_list + '</div>' if edge_list else ''}
    </div>
  </section>
"""


def _render_plan_svg(agents: List[Dict[str, Any]], edges: List[str]) -> str:
    if not agents:
        return ""
    count = len(agents)
    width = max(900, 240 * count)
    height = 200
    step = width / (count + 1)
    node_positions = {}
    node_colors = {}
    nodes_svg = []
    colors = [
        "#7cc4ff", "#b892ff", "#4ade80", "#facc15", "#fb7185",
        "#38bdf8", "#f472b6", "#a3e635", "#f97316", "#c084fc"
    ]
    for idx, agent in enumerate(agents, start=1):
        x = step * idx
        y = height / 2
        node_positions[agent.get("name")] = (x, y)
        fill = colors[(idx - 1) % len(colors)]
        node_colors[agent.get("name")] = fill
        nodes_svg.append(
            f"<g class=\"dag-node\">"
            f"<rect class='module' x='{x - 60}' y='{y - 36}' width='120' height='72' rx='16' fill='rgba(17,26,43,.95)' stroke='{fill}' stroke-width='2.5' />"
            f"<text x='{x}' y='{y + 6}' text-anchor='middle' font-size='13' fill='#e2e8f0' font-weight='600'>{escape(agent.get('name',''))}</text>"
            f"</g>"
        )

    edges_svg = []
    half_w, half_h = 60, 36
    marker_lookup = {color: f"arrow_{idx}" for idx, color in enumerate(dict.fromkeys(node_colors.values()))}
    for idx, edge in enumerate(edges, start=1):
        if '->' not in edge:
            continue
        source, target = edge.split('->', 1)
        if source not in node_positions or target not in node_positions:
            continue
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        scale = max(abs(dx) / half_w if half_w else 0, abs(dy) / half_h if half_h else 0, 1e-6)
        trim_x = dx / scale
        trim_y = dy / scale
        start_x = x1 + trim_x
        start_y = y1 + trim_y
        end_x = x2 - trim_x
        end_y = y2 - trim_y
        color = node_colors.get(source, "rgba(124,196,255,.8)")
        marker_id = marker_lookup.get(color, "arrow")
        edges_svg.append(
            f"<line x1='{start_x}' y1='{start_y}' x2='{end_x}' y2='{end_y}' stroke='{color}' stroke-width='2.2' marker-end='url(#{marker_id})' />"
        )

    edges_markup = "".join(edges_svg)
    nodes_markup = "".join(nodes_svg)
    marker_defs = "".join(
        f"<marker id='{marker_lookup[color]}' markerWidth='12' markerHeight='12' refX='9' refY='4' orient='auto' markerUnits='strokeWidth'>"
        f"<path d='M0,0 L0,8 L10,4 z' fill='{color}' /></marker>"
        for color in marker_lookup
    )
    fallback_marker = "<marker id='arrow' markerWidth='12' markerHeight='12' refX='9' refY='4' orient='auto' markerUnits='strokeWidth'><path d='M0,0 L0,8 L10,4 z' fill='rgba(184,146,255,.9)' /></marker>"
    return f"""
      <div class="svg-wrap" style="margin-bottom:16px">
        <svg class="dag-svg" viewBox='0 0 {width} {height}' width='100%' height='100%' role='img' aria-label='Adaptive routing workflow'>
          <defs>{marker_defs or fallback_marker}</defs>
          {edges_markup}
          {nodes_markup}
        </svg>
      </div>
    """






def _render_commentary_block(commentary: Dict[str, List[str]]) -> str:
    """
    Render LLM-generated HTML sections directly.
    The commentary should contain complete HTML sections with problem-specific visualizations.
    """
    # If commentary contains raw HTML (new format)
    if "html_sections" in commentary:
        return commentary["html_sections"]

    # Fallback: old format with text lists (backward compatibility)
    highlights = commentary.get("reasoning_highlights") or []
    notes = commentary.get("interpretation_notes") or []
    if not highlights and not notes:
        return ""

    highlights_html = "".join(f"<li>{escape(str(item))}</li>" for item in highlights)
    notes_html = "".join(f"<li>Interpretation: {escape(str(item))}</li>" for item in notes)
    return f"""
  <section class=\"grid\" style=\"margin-top:8px\">
    <div class=\"card\" style=\"grid-column:span 12\">
      <h2>LLM Commentary</h2>
      <h3>Reasoning Highlights</h3>
      <ul class=\"small\" style=\"margin:0 0 12px 18px\">{highlights_html or '<li>No highlights generated.</li>'}</ul>
      <h3>Interpretation Notes</h3>
      <ul class=\"small\" style=\"margin:0 0 0 18px\">{notes_html or '<li>None.</li>'}</ul>
    </div>
  </section>
"""


def why(
    generator,
    trace_payload: Dict[str, Any],
    *,
    narrative_context: str,
    style_hint: Optional[str] = None,
    extra_guidance: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    prompt_name: str | None = None,
    prompt_dir: str | None = None,
    model_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Render an HTML explanation by combining deterministic sections with LLM commentary."""
    structured = _prepare_context(trace_payload)
    commentary = _generate_commentary(
        generator,
        structured,
        narrative_context,
        style_hint,
        extra_guidance,
        model_args,
    )
    html = _render_html(structured, commentary)
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
    return html


__all__ = [
    "build_trace_payload",
    "why",
    "snapshot_memory",
    "serialize_plan",
]
