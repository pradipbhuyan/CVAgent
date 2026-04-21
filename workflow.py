from typing import Any, TypedDict
import time

from langgraph.graph import StateGraph, END

from core import (
    detect_document_type,
    extract_structured_json,
    build_resume,
    get_current_metrics_snapshot,
    diff_metrics_snapshot,
)


class IDPState(TypedDict, total=False):
    text: str
    filename: str
    template: bytes
    progress: Any
    event_callback: Any
    ocr_used: bool
    extraction_mode: str
    exception_reason: str

    doc_type: str
    data: dict
    result: dict
    error: str
    step_metrics: list
    validation: dict
    confidence: dict


def safe_progress(state: IDPState, percent: int, message: str) -> None:
    progress = state.get("progress")
    if progress:
        try:
            progress(percent, message)
        except Exception:
            pass


def emit_agent_event(state: IDPState, agent: str, status: str, message: str) -> None:
    callback = state.get("event_callback")
    if callback:
        try:
            callback(agent, status, message)
        except Exception:
            pass


def add_step_metric(
    state: IDPState,
    step_name: str,
    started_at: float,
    before: dict,
    note: str = "",
) -> None:
    after = get_current_metrics_snapshot()
    diff = diff_metrics_snapshot(before, after)

    if "step_metrics" not in state or state["step_metrics"] is None:
        state["step_metrics"] = []

    state["step_metrics"].append({
        "step": step_name,
        "duration_sec": round(time.time() - started_at, 2),
        "tokens": diff.get("tokens", 0),
        "input_tokens": diff.get("input_tokens", 0),
        "output_tokens": diff.get("output_tokens", 0),
        "cost": round(diff.get("cost", 0.0), 6),
        "calls": diff.get("calls", 0),
        "note": note,
    })


def detect_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Classification Agent", "running", "Classifying CV document")
    safe_progress(state, 40, "Classification Agent — classifying document")

    text = state.get("text", "") or ""
    state["doc_type"] = detect_document_type(text)

    emit_agent_event(
        state,
        "Classification Agent",
        "done" if state["doc_type"] == "resume" else "error",
        f"Document identified as {state.get('doc_type', 'other')}",
    )

    add_step_metric(
        state,
        "Detect document type",
        started_at,
        before,
        state.get("doc_type", "unknown"),
    )
    return state


def extract_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    doc_type = state.get("doc_type", "other")

    if doc_type == "resume":
        emit_agent_event(state, "Structuring Agent", "running", "Extracting candidate profile")
        safe_progress(state, 55, "Structuring Agent — extracting candidate profile")
        state["data"] = extract_structured_json(state.get("text", ""), "resume")
        emit_agent_event(state, "Structuring Agent", "done", "Candidate profile extracted")
        add_step_metric(state, "Extract structured resume data", started_at, before, "resume")
    else:
        state["data"] = {}
        add_step_metric(state, "Skip structured extraction", started_at, before, doc_type)

    return state


def resume_node(state: IDPState) -> IDPState:
    started_at = time.time()
    before = get_current_metrics_snapshot()

    emit_agent_event(state, "Output Agent", "running", "Generating resume in template")
    safe_progress(state, 75, "Output Agent — generating resume in template")

    data = state.get("data") or {}
    template_bytes = state.get("template")

    file_bytes = build_resume(data, template_bytes)

    emit_agent_event(state, "Output Agent", "done", "Resume generated")
    safe_progress(state, 95, "Output Agent — resume ready")

    candidate_name = (data.get("name") or "candidate").strip() if isinstance(data, dict) else "candidate"
    safe_name = "".join(ch for ch in candidate_name if ch not in '\\/*?:"<>|').strip() or "candidate"

    state["result"] = {
        "type": "resume",
        "file": file_bytes,
        "file_name": f"{safe_name}.docx",
        "data": data,
    }

    add_step_metric(state, "Build resume", started_at, before, state["result"]["file_name"])
    return state


def other_node(state: IDPState) -> IDPState:
    state["result"] = {
        "type": "other",
        "data": {},
    }
    return state


def route(state: IDPState) -> str:
    return "resume" if state.get("doc_type") == "resume" else "other"


def build_graph():
    builder = StateGraph(IDPState)

    builder.add_node("detect", detect_node)
    builder.add_node("extract", extract_node)
    builder.add_node("resume", resume_node)
    builder.add_node("other", other_node)

    builder.set_entry_point("detect")
    builder.add_edge("detect", "extract")

    builder.add_conditional_edges(
        "extract",
        route,
        {
            "resume": "resume",
            "other": "other",
        },
    )

    builder.add_edge("resume", END)
    builder.add_edge("other", END)

    return builder.compile()
