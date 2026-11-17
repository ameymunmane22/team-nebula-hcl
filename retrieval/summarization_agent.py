"""Summarization Agent for processing and summarizing enterprise knowledge.

Handles:
- PDF documents, reports, technical documentation, emails, and chat threads
- Key point extraction, critical context preservation (Asset IDs, Incident IDs)
- Concise, actionable summaries tailored for employees

Orchestrated using LangGraph.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

load_dotenv()


SYSTEM_PROMPT: str = """You are an Enterprise Knowledge Summarization Agent.
Your responsibility is to extract critical facts, identify key troubleshooting steps,
preserve enterprise context (e.g., Asset IDs like AST..., Incident IDs like INC..., Employee IDs like EMP...),
and produce clear, concise, and structured summaries for employees.
Avoid hallucinating, preserve exact IDs and references, and focus on actionable information."""


# ============================================================================
# LangGraph State Schema
# ============================================================================

class SummarizationState(TypedDict):
    """State schema for the Summarization Agent LangGraph workflow."""
    raw_content: Union[str, List[Any]]
    summary_type: str
    normalized_text: str
    key_points: List[str]
    summary: str
    metadata: Dict[str, Any]


# ============================================================================
# Node Implementations
# ============================================================================

def prepare_content_node(state: SummarizationState) -> Dict[str, Any]:
    """Normalize input documents or raw text into a single coherent text block."""
    raw = state["raw_content"]
    normalized_parts: List[str] = []
    entities = set()

    if isinstance(raw, str):
        normalized_parts.append(raw.strip())
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                content = item.get("content", item.get("text", str(item)))
                normalized_parts.append(str(content).strip())
            else:
                normalized_parts.append(str(item).strip())
    else:
        normalized_parts.append(str(raw).strip())

    combined_text = "\n\n---\n\n".join(part for part in normalized_parts if part)

    # Extract enterprise identifiers for context preservation
    asset_ids = re.findall(r"\bAST\d+\b", combined_text, re.IGNORECASE)
    incident_ids = re.findall(r"\bINC\d+\b", combined_text, re.IGNORECASE)
    emp_ids = re.findall(r"\bEMP\d+\b", combined_text, re.IGNORECASE)

    metadata = {
        "identified_assets": list(set(a.upper() for a in asset_ids)),
        "identified_incidents": list(set(i.upper() for i in incident_ids)),
        "identified_employees": list(set(e.upper() for e in emp_ids)),
        "character_count": len(combined_text)
    }

    return {
        "normalized_text": combined_text[:8000],  # Bound text size for performance
        "metadata": metadata
    }


def extract_key_points_node(state: SummarizationState) -> Dict[str, Any]:
    """Extract bulleted key technical points and operational observations."""
    text = state["normalized_text"]
    key_points: List[str] = []

    # Heuristic key point extraction
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]
    for line in lines:
        if any(marker in line.lower() for marker in ("error", "failed", "symptom", "procedure", "status", "action", "resolved")):
            clean_point = re.sub(r"^[#*\-\d.]+\s*", "", line)
            if clean_point not in key_points:
                key_points.append(clean_point)
            if len(key_points) >= 6:
                break

    if not key_points and lines:
        key_points = lines[:4]

    return {"key_points": key_points}


def generate_summary_node(state: SummarizationState) -> Dict[str, Any]:
    """Generate concise summary using ChatGroq or deterministic synthesis."""
    text = state["normalized_text"]
    key_points = state.get("key_points", [])
    summary_type = state.get("summary_type", "concise")

    if not text.strip():
        return {"summary": "No content provided to summarize."}

    # Attempt LLM-based summarization with Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_api_key,
                temperature=0.1
            )
            prompt = (
                f"Format Requirement: {summary_type.upper()} summary.\n\n"
                f"Enterprise Content:\n{text}\n\n"
                "Provide a clear, 3-5 sentence executive summary preserving all referenced Asset IDs, "
                "Incident IDs, and technical resolutions."
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            llm_summary = str(response.content).strip()
            if llm_summary:
                return {"summary": llm_summary}
        except Exception as e:
            print(f"LLM summarization fallback: {e}")

    # Fallback deterministic summary
    meta = state.get("metadata", {})
    assets = ", ".join(meta.get("identified_assets", [])) or "N/A"
    incidents = ", ".join(meta.get("identified_incidents", [])) or "N/A"

    points_formatted = "\n".join(f"- {p}" for p in key_points) if key_points else "- Content analyzed."
    fallback_summary = (
        f"Summary (Assets: {assets} | Incidents: {incidents}):\n"
        f"{points_formatted}\n\n"
        f"Consolidated from {meta.get('character_count', 0)} characters of enterprise records."
    )

    return {"summary": fallback_summary}


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

builder = StateGraph(SummarizationState)
builder.add_node("prepare_content", prepare_content_node)
builder.add_node("extract_key_points", extract_key_points_node)
builder.add_node("generate_summary", generate_summary_node)

builder.add_edge(START, "prepare_content")
builder.add_edge("prepare_content", "extract_key_points")
builder.add_edge("extract_key_points", "generate_summary")
builder.add_edge("generate_summary", END)

summarization_agent_graph = builder.compile()


def summarize_content(
    content: Union[str, List[Any]],
    summary_type: str = "concise"
) -> Dict[str, Any]:
    """Summarize enterprise documents, retrieved chunks, or technical discussions.

    Args:
        content: Raw text or list of retrieved documents.
        summary_type: Type of summary - 'concise', 'technical', or 'action_items'.

    Returns:
        Dictionary containing summary text, extracted key points, and metadata.
    """
    initial_state: SummarizationState = {
        "raw_content": content,
        "summary_type": summary_type,
        "normalized_text": "",
        "key_points": [],
        "summary": "",
        "metadata": {}
    }
    return summarization_agent_graph.invoke(initial_state)
