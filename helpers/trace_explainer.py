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
from typing import Any, Dict, List, Optional, Sequence

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


def prepare_why_inputs(
    problem_ids: Sequence[str],
    memory,
    *,
    scenario: str,
    summary_fields: Sequence[str],
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Create ``result_summary`` and metadata dictionaries for ``why``.

    Args:
        problem_ids: Iterable of problem identifiers.
        memory: Scratchpad instance containing ``result_{pid}`` entries.
        scenario: Human-readable scenario name.
        summary_fields: Keys to copy from each ``result_{pid}``.
        metadata_overrides: Additional metadata entries to merge.

    Returns:
        Tuple ``(result_summary, metadata)`` ready to pass to ``why``.
    """
    result_summary: List[Dict[str, Any]] = []
    for pid in problem_ids:
        record = memory.read(f"result_{pid}") or {}
        summary = {"problem_id": pid}
        for field in summary_fields:
            summary[field] = record.get(field)
        result_summary.append(summary)

    metadata = {
        "scenario": scenario,
        "problem_ids": list(problem_ids),
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)

    return result_summary, metadata


MAX_EVENTS_PER_PROBLEM = 12
_TRACE_STRING_LIMIT = 1500
FORMAL_VALUE_KEYS = {
    "intermediate_form",
    "formalization",
    "formal_code",
    "Facts",
    "facts",
    "Rules",
    "rules",
    "premises",
    "conclusion",
    "code",
}
FORMAL_VALUE_PLACEHOLDER = "[formal code omitted – added in static section]"
FORMAL_KEY_TITLES = {
    "intermediate_form": "Formalization",
    "formalization": "Formalization",
    "formal_code": "Formal Code",
    "Facts": "Facts",
    "facts": "Facts",
    "Rules": "Rules",
    "rules": "Rules",
    "premises": "Premises",
    "conclusion": "Conclusion",
    "code": "Code",
}


def _event_contains_problem(event: Dict[str, Any], problem_id: str) -> bool:
    if not problem_id or not isinstance(event, dict):
        return False
    try:
        blob = json.dumps(event, ensure_ascii=False)
    except TypeError:
        return False
    return problem_id in blob


def _trim_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return event
    trimmed = dict(event)
    for field in ("input", "output"):
        value = trimmed.get(field)
        if isinstance(value, dict):
            trimmed[field] = _clean_mapping(value)
        elif isinstance(value, str) and len(value) > _TRACE_STRING_LIMIT:
            trimmed[field] = value[: _TRACE_STRING_LIMIT] + "..."
    return trimmed


def _clean_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in mapping.items():
        if key in FORMAL_VALUE_KEYS and isinstance(value, str):
            cleaned[key] = FORMAL_VALUE_PLACEHOLDER
            continue
        if isinstance(value, str) and len(value) > _TRACE_STRING_LIMIT:
            cleaned[key] = value[: _TRACE_STRING_LIMIT] + "..."
        else:
            cleaned[key] = value
    return cleaned


def _select_trace_events(events: List[Dict[str, Any]], problem_id: str) -> List[Dict[str, Any]]:
    if not events:
        return []
    matched: List[Dict[str, Any]] = []
    for event in events:
        if _event_contains_problem(event, problem_id):
            matched.append(_trim_event(event))
        if len(matched) >= MAX_EVENTS_PER_PROBLEM:
            break
    if not matched:
        return [_trim_event(event) for event in events[:MAX_EVENTS_PER_PROBLEM]]
    return matched


def _filter_memory_for_problem(memory_snapshot: Dict[str, Any], problem_id: str) -> Dict[str, Any]:
    if not memory_snapshot:
        return {}
    if not problem_id:
        return memory_snapshot
    filtered = {k: v for k, v in memory_snapshot.items() if problem_id in k}
    return filtered or {}


def _build_problem_runs(structured: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = structured.get("metadata", {}) or {}
    problem_ids = metadata.get("problem_ids") or []
    result_summary = structured.get("result_summary", []) or []
    summary_lookup = {
        item.get("problem_id"): item
        for item in result_summary
        if isinstance(item, dict) and item.get("problem_id")
    }
    trace_events = structured.get("trace_events", []) or []
    memory_snapshot = structured.get("memory", {}) or {}
    task_titles = metadata.get("task_titles", {}) or {}
    agent_problem_map = {
        agent.get("name"): agent.get("problem_to_solve", []) or []
        for agent in structured.get("plan_agents", []) or []
    }
    formalizations_by_problem = _gather_formalizations(trace_events, agent_problem_map)

    runs: List[Dict[str, Any]] = []

    if not problem_ids:
        derived = sorted({pid for ids in agent_problem_map.values() for pid in ids if pid})
        if derived:
            problem_ids = derived

    if not problem_ids:
        fallback_id = next(iter(summary_lookup.keys()), next(iter(formalizations_by_problem.keys()), "task"))
        fallback_summary = summary_lookup.get(fallback_id) or (result_summary[0] if result_summary else {})
        runs.append(
            {
                "problem_id": fallback_id,
                "label": task_titles.get(fallback_id) or fallback_summary.get("title") or fallback_id,
                "result_entry": fallback_summary,
                "trace_events": [_trim_event(event) for event in trace_events[:MAX_EVENTS_PER_PROBLEM]],
                "memory_snapshot": memory_snapshot,
                "formalizations": formalizations_by_problem.get(fallback_id, []),
            }
        )
        return runs

    for problem_id in problem_ids:
        entry = summary_lookup.get(problem_id, {})
        runs.append(
            {
                "problem_id": problem_id,
                "label": task_titles.get(problem_id) or entry.get("title") or problem_id,
                "result_entry": entry,
                "trace_events": _select_trace_events(trace_events, problem_id),
                "memory_snapshot": _filter_memory_for_problem(memory_snapshot, problem_id),
                "formalizations": formalizations_by_problem.get(problem_id, []),
            }
        )

    return runs


def _gather_formalizations(trace_events: List[Dict[str, Any]], agent_problem_map: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    bucket: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    for event in trace_events:
        agent = event.get("agent")
        if not agent:
            continue
        targets = agent_problem_map.get(agent) or []
        if not targets:
            continue
        containers = []
        for field in ("input", "output"):
            payload = event.get(field)
            if isinstance(payload, dict):
                containers.append(payload)
        for data in containers:
            for key, title in FORMAL_KEY_TITLES.items():
                snippet = data.get(key)
                if not isinstance(snippet, str) or not snippet.strip():
                    continue
                snippet = snippet.strip()
                for problem_id in targets:
                    dedup_key = (problem_id, key, agent, snippet)
                    if dedup_key in seen[problem_id]:
                        continue
                    seen[problem_id].add(dedup_key)
                    bucket[problem_id].append(
                        {
                            "agent": agent,
                            "title": f"{title} ({agent})" if agent else title,
                            "code": snippet,
                        }
                    )
    return bucket


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


def _normalise_model_html(content: Optional[str]) -> str:
    if not content:
        return ""
    html_content = content.strip()
    if html_content.startswith("```html"):
        html_content = html_content[7:]
    if html_content.startswith("```"):
        html_content = html_content[3:]
    if html_content.endswith("```"):
        html_content = html_content[:-3]
    return html_content.strip()


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
    scenario_context = {
        "scenario": structured.get("scenario"),
        "problem_statement": structured.get("problem_statement"),
        "plan_agents": structured.get("plan_agents"),
        "plan_edges": structured.get("plan_edges"),
        "solver_names": structured.get("solver_names"),
        "result_summary": structured.get("result_summary"),
        "trace_highlights": structured.get("trace_highlights"),
    }
    scenario_blob = json.dumps(scenario_context, ensure_ascii=False, indent=2)

    problem_runs = _build_problem_runs(structured)
    replacements_base = {
        "narrative_context": narrative_context.strip(),
        "style_hint": (style_hint or "").strip(),
        "extra_guidance": (extra_guidance or "").strip(),
        "scenario_context": scenario_blob,
    }

    commentary_args = dict(model_args or {})
    commentary_args.setdefault("temperature", 0.2)
    commentary_args.pop("response_format", None)

    html_sections: List[str] = []
    for run in problem_runs:
        context_blob = {
            "problem_id": run.get("problem_id"),
            "problem_label": run.get("label"),
            "result_entry": run.get("result_entry"),
            "trace_events": run.get("trace_events"),
            "memory_snapshot": run.get("memory_snapshot"),
            "plan_agents": structured.get("plan_agents"),
            "plan_edges": structured.get("plan_edges"),
        }
        replacements = dict(replacements_base)
        replacements.update(
            {
                "context": json.dumps(context_blob, ensure_ascii=False, indent=2),
                "problem_id": run.get("problem_id", ""),
                "problem_label": run.get("label", ""),
            }
        )

        response = generator.generate(
            model_prompt_dir="helpers",
            prompt_name="trace_visualization.txt",
            model_args=commentary_args,
            **replacements,
        )
        html_content = _normalise_model_html(response)
        formal_section = _render_formalization_sections(run.get("label"), run.get("formalizations"))
        combined = "\n".join(part for part in (html_content, formal_section) if part)
        if combined:
            html_sections.append(combined)

    return {"html_sections": "\n".join(html_sections)}


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
    --bg:#f7f9fc;
    --panel:#ffffff;
    --soft:#eef3fb;
    --ink:#1f2a37;
    --sub:#5f728d;
    --accent:#2563eb;
    --accent-2:#7c3aed;
    --ok:#15803d;
    --warn:#b45309;
    --bad:#b91c1c;
    --cv:#4c1d95;
    --nlp:#92400e;
    --sys:#065f46;
    --rl:#1d4ed8;
    --theory:#9d174d;
    --code:#f1f5f9;
    --chip:#e2e8f0;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,"Helvetica Neue",Arial,"Noto Sans",sans-serif;line-height:1.55}
  a{color:var(--accent)}
  .container{max-width:1100px;margin:24px auto;padding:0 16px}
  header{display:flex;gap:16px;align-items:center;margin-bottom:20px}
  .logo{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#2563eb,#7c3aed);box-shadow:0 6px 18px rgba(37,99,235,.25)}
  h1{font-size:28px;margin:0;font-weight:700;letter-spacing:.2px;color:var(--ink)}
  .subtitle{color:var(--sub);margin-top:2px;font-weight:500}
  .grid{display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
  .card{grid-column:span 12;background:var(--panel);border:1px solid #e4e7ec;border-radius:14px;padding:18px 20px;box-shadow:0 8px 24px rgba(15,23,42,.08)}
  .card h2{margin:2px 0 10px 0;font-size:20px;color:var(--ink)}
  .card h3{margin:16px 0 8px 0;font-size:15px;color:var(--sub)}
  .meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:8px}
  .chip{background:var(--chip);color:var(--ink);padding:6px 12px;border-radius:999px;border:1px solid #d5dae1;font-size:12px;font-weight:600}
  code, pre{background:var(--code);color:#0f172a;border-radius:10px}
  pre{padding:14px;overflow:auto;border:1px solid #d5dae1;font-size:12px}
  table{width:100%;border-collapse:separate;border-spacing:0 6px;background:#fefefe;border:1px solid #e4e7ec;border-radius:12px}
  th{color:#475467;font-weight:700;border-bottom:1px solid #e4e7ec;background:#f3f6fb;padding:10px 12px;font-size:13px}
  td{text-align:left;padding:10px 12px;font-size:13px;color:#1f2a37;background:#ffffff}
  tr td:first-child{border-top-left-radius:10px;border-bottom-left-radius:10px}
  tr td:last-child{border-top-right-radius:10px;border-bottom-right-radius:10px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;font-weight:700;font-size:12px;border:1px solid transparent}
  .pill.ok{background:rgba(21,128,61,.12);color:#166534;border-color:rgba(21,128,61,.4)}
  .pill.warn{background:rgba(180,83,9,.12);color:#92400e;border-color:rgba(180,83,9,.35)}
  .session-box{background:#f4f7fb;border:1px solid #e4e7ec;border-radius:12px;padding:14px}
  .session-box h4{margin:0 0 8px 0;font-size:15px;color:#1f2a37;font-weight:600}
  .paper-badge{display:inline-block;padding:4px 8px;margin:3px;border-radius:6px;font-size:11px;font-weight:600;border:1px solid #d5dae1;background:#eef2ff;color:#3730a3}
  .paper-badge.cv{background:#ede9fe;color:#4c1d95}
  .paper-badge.nlp{background:#fef3c7;color:#92400e}
  .paper-badge.sys{background:#d1fae5;color:#065f46}
  .paper-badge.rl{background:#e0f2fe;color:#1d4ed8}
  .paper-badge.theory{background:#fce7f3;color:#9d174d}
  .oral-badge{background:#dbeafe;color:#1d4ed8;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:600;margin-left:8px}
  .poster-badge{background:#e9d5ff;color:#7c3aed;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:600;margin-left:8px}
  .plan-flow{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;padding:0}
  .plan-flow li{list-style:none;background:#eef2ff;padding:6px 12px;border-radius:999px;border:1px solid #dbe4ff;font-size:12px;color:#1e3a8a}
  .dag-svg rect.module{fill:#ffffff;stroke:#2563eb;stroke-width:2.2;filter:drop-shadow(0 2px 6px rgba(15,23,42,.12))}
  .dag-svg text{font-family:Inter,sans-serif;fill:#1f2a37;font-weight:600}
  .trace-table{width:100%;margin-top:10px}
  .trace-table tr td{font-size:12px}
  details{margin-top:8px}
  summary{cursor:pointer;font-weight:600;color:var(--accent)}
  footer{margin:24px 0 12px 0;color:#64748b;font-size:12px;text-align:center}
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
    layer_map, ordered_layers = _compute_plan_layers(agents, edges)
    flow_items = []
    for idx, layer in enumerate(ordered_layers, start=1):
        label = ", ".join(escape(name) for name in layer)
        flow_items.append(f"<li>Stage {idx}: {label}</li>")
    edge_list = "<br>".join(escape(edge) for edge in edges)
    agent_rows = "".join(
        f"<tr><td>{escape(agent.get('name',''))}</td><td>{escape(agent.get('goal',''))}</td></tr>" for agent in agents
    )
    svg = _render_plan_svg(agents, edges, layer_map, ordered_layers)
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


def _compute_plan_layers(agents: List[Dict[str, Any]], edges: List[str]):
    names = [agent.get("name") for agent in agents if agent.get("name")]
    if not names:
        return {}, [names]
    index_lookup = {name: idx for idx, name in enumerate(names)}
    parents: Dict[str, set] = defaultdict(set)
    children: Dict[str, set] = defaultdict(set)
    indegree: Dict[str, int] = defaultdict(int)
    for name in names:
        indegree.setdefault(name, 0)
    for edge in edges:
        if "->" not in edge:
            continue
        source, target = edge.split("->", 1)
        children[source].add(target)
        parents[target].add(source)
        indegree[target] += 1
        indegree.setdefault(source, indegree.get(source, 0))
    indegree_copy = indegree.copy()
    from collections import deque

    def _sorted_queue(items):
        return deque(sorted(items, key=lambda n: index_lookup.get(n, len(names))))

    queue = _sorted_queue([node for node, deg in indegree_copy.items() if deg == 0])
    topo_order: List[str] = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in children.get(node, []):
            indegree_copy[child] -= 1
            if indegree_copy[child] == 0:
                queue.append(child)
        queue = _sorted_queue(queue)
    for name in names:
        if name not in topo_order:
            topo_order.append(name)
            parents.setdefault(name, set())

    layer: Dict[str, int] = {}
    for node in topo_order:
        parent_layers = [layer.get(parent, 0) for parent in parents.get(node, [])]
        layer[node] = (max(parent_layers) + 1) if parent_layers else 0

    ordered_layers: List[List[str]] = []
    seen_layers = sorted(set(layer.values())) if layer else [0]
    for lvl in seen_layers:
        nodes = [name for name in names if layer.get(name, 0) == lvl]
        if nodes:
            ordered_layers.append(nodes)
    if not ordered_layers and names:
        ordered_layers.append(names)
    return layer, ordered_layers


def _render_plan_svg(
    agents: List[Dict[str, Any]],
    edges: List[str],
    layer_map: Optional[Dict[str, int]] = None,
    ordered_layers: Optional[List[List[str]]] = None,
) -> str:
    if not agents:
        return ""
    names = [agent.get("name") for agent in agents if agent.get("name")]
    if not names:
        return ""
    if layer_map is None or ordered_layers is None:
        layer_map, ordered_layers = _compute_plan_layers(agents, edges)
    max_layer_width = max((len(layer) for layer in ordered_layers), default=1)
    layer_count = max(len(ordered_layers), 1)
    width = max(900, 240 * max_layer_width)
    height = max(320, 220 * layer_count)
    vertical_step = height / (layer_count + 1)
    node_positions = {}
    node_colors = {}
    nodes_svg = []
    colors = [
        "#7cc4ff", "#b892ff", "#4ade80", "#facc15", "#fb7185",
        "#38bdf8", "#f472b6", "#a3e635", "#f97316", "#c084fc"
    ]
    color_lookup = {name: idx for idx, name in enumerate(names)}
    for layer_idx, layer in enumerate(ordered_layers):
        horizontal_step = width / (len(layer) + 1)
        y = vertical_step * (layer_idx + 1)
        for pos, name in enumerate(layer, start=1):
            x = horizontal_step * pos
            node_positions[name] = (x, y)
            color = colors[color_lookup.get(name, 0) % len(colors)]
            node_colors[name] = color
            nodes_svg.append(
                f"<g class=\"dag-node\">"
                f"<rect class='module' x='{x - 60}' y='{y - 36}' width='120' height='72' rx='16' fill='rgba(17,26,43,.95)' stroke='{color}' stroke-width='2.5' />"
                f"<text x='{x}' y='{y + 6}' text-anchor='middle' font-size='13' fill='#e2e8f0' font-weight='600'>{escape(name)}</text>"
                f"</g>"
            )

    edges_svg = []
    half_w, half_h = 60, 36
    marker_lookup = {color: f"arrow_{idx}" for idx, color in enumerate(dict.fromkeys(node_colors.values()))}
    for edge in edges:
        if "->" not in edge:
            continue
        source, target = edge.split("->", 1)
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


def _render_formalization_sections(problem_label: str | None, snippets: Optional[List[Dict[str, str]]]) -> str:
    if not snippets:
        return ""
    label = problem_label or "Task"
    sections: List[str] = []
    for snippet in snippets:
        title = snippet.get("title") or "Formalization"
        code = snippet.get("code", "").strip()
        if not code:
            continue
        header = f"{label} · {title}"
        sections.append(
            f"""
  <section class=\"grid\" style=\"margin-top:8px\">
    <div class=\"card\" style=\"grid-column:span 12\">
      <h2>{escape(header)}</h2>
      <pre>{escape(code)}</pre>
    </div>
  </section>
"""
        )
    return "\n".join(sections)


def why(
    generator,
    trace_payload: Optional[Dict[str, Any]] = None,
    *,
    plan: Optional[Plan] = None,
    tracer: Optional[TracePersister] = None,
    memory=None,
    metadata: Optional[Dict[str, Any]] = None,
    result_summary: Optional[List[Dict[str, Any]]] = None,
    problem_statement: Optional[str] = None,
    narrative_context: str = None,
    style_hint: Optional[str] = None,
    extra_guidance: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    prompt_name: str | None = None,
    prompt_dir: str | None = None,
    model_args: Optional[Dict[str, Any]] = None,
    # New parameters for automatic prepare_why_inputs integration
    problem_ids: Optional[Sequence[str]] = None,
    scenario: Optional[str] = None,
    summary_fields: Optional[Sequence[str]] = None,
    task_titles: Optional[Dict[str, str]] = None,
) -> str:
    """Render an HTML explanation by combining deterministic sections with LLM commentary.

    Either provide a ready-made ``trace_payload`` or let this function build one by passing
    ``plan``/``tracer``/``memory``/``metadata``/``problem_statement``.

    **Simplified usage:** Pass ``problem_ids``, ``scenario``, and ``summary_fields`` to
    automatically prepare result_summary and metadata. This eliminates the need to call
    ``prepare_why_inputs`` separately.

    Args:
        generator: LLM generator for commentary.
        problem_ids: (Optional) List of problem IDs to include in the report.
        scenario: (Optional) Scenario description for the report.
        summary_fields: (Optional) Fields to extract from each result.
        task_titles: (Optional) Mapping of problem_id to human-readable titles.
        narrative_context: (Optional) Context for LLM commentary.
        style_hint: (Optional) Style guidance for LLM.
        extra_guidance: (Optional) Additional instructions for LLM.
        Other args: See original docstring.

    Returns:
        HTML string of the report.
    """
    # Auto-prepare inputs if simplified parameters provided
    if problem_ids is not None and scenario is not None and memory is not None:
        if summary_fields is None:
            summary_fields = ["parsed_answer", "solver_name"]

        auto_result_summary, auto_metadata = prepare_why_inputs(
            problem_ids,
            memory,
            scenario=scenario,
            summary_fields=summary_fields,
            metadata_overrides={
                "task_titles": task_titles,
                "narrative_context": narrative_context,
                "style_hint": style_hint,
                "extra_guidance": extra_guidance,
            } if any([task_titles, narrative_context, style_hint, extra_guidance]) else None,
        )

        # Merge with explicitly provided metadata
        if metadata is None:
            metadata = auto_metadata
        else:
            metadata = {**auto_metadata, **metadata}

        # Use auto-prepared result_summary if not explicitly provided
        if result_summary is None:
            result_summary = auto_result_summary

    meta_payload = dict(metadata or {})
    if result_summary is not None and "result_summary" not in meta_payload:
        meta_payload["result_summary"] = result_summary

    if not narrative_context:
        narrative_context = meta_payload.get("narrative_context", "")
    if not style_hint:
        style_hint = meta_payload.get("style_hint", "")
    if not extra_guidance:
        extra_guidance = meta_payload.get("extra_guidance", "")

    if trace_payload is None:
        if tracer is None:
            raise ValueError("Either trace_payload or tracer must be provided")
        trace_payload = build_trace_payload(
            plan,
            tracer,
            memory=memory,
            metadata=meta_payload,
            problem_statement=problem_statement,
        )
    else:
        # When caller passes a payload but still supplies metadata overrides,
        # merge them into the structured context by updating trace payload.
        if meta_payload:
            trace_payload = dict(trace_payload)
            trace_payload.setdefault("metadata", {})
            trace_payload["metadata"] = {**trace_payload["metadata"], **meta_payload}

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
    "prepare_why_inputs",
    "snapshot_memory",
    "serialize_plan",
]
