"""Compliance Agent for evaluating retrieved evidence, detecting contradictions, and query rewriting.

Workflow:
1. Evaluates chunks retrieved from hybrid search for:
   - Evidence sufficiency
   - Relevance
   - Query coverage
   - Contradiction detection (verifying conflicting claims, context, and applicability)
2. Uses Pydantic structured output to route decisions:
   - 'GOOD': Evidence is sufficient, relevant, and consistent -> Proceed.
   - 'WEAK': Evidence is insufficient or incomplete -> Rewrite missing query parts and retry (max 3 attempts).
   - Unresolved Contradictions or exhausted retries -> Output 'evidence not found'.
3. Built with LangGraph.
"""

from enum import Enum
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from retrieval.hybrid_search import hybrid_search

load_dotenv()


SYSTEM_PROMPT: str = """You are an Enterprise Compliance and Evidence Verification Agent.

Your responsibility is to critically evaluate retrieved evidence chunks against a user query and determine whether the evidence is GOOD or WEAK.

Evaluation Criteria:
1. Evidence Sufficiency: Do the chunks contain enough concrete facts to reliably answer the query?
2. Relevance: Are the chunks directly on-topic and applicable?
3. Query Coverage: Does the evidence cover all required aspects of the query?
4. Contradiction Detection: Check for conflicting or contradictory claims between chunks.
   - If conflicting claims exist, examine context, asset versions, dates, and applicability to see if the contradiction is resolved.
   - If conflicting claims cannot be resolved reliably, mark contradiction_detected=True and contradiction_resolved=False.

Decision Logic:
- Rate as 'GOOD' if evidence is relevant, sufficient, covers the query, and contains no unresolved contradictions.
- Rate as 'WEAK' if evidence is incomplete, tangential, partially missing, or lacks sufficient factual depth.
- If WEAK, identify the exact missing information and provide a focused rewritten query targeting solely the missing details."""


# ============================================================================
# Pydantic Schemas
# ============================================================================

class EvaluationDecision(str, Enum):
    """Permitted routing decisions for compliance evaluation."""
    GOOD = "GOOD"
    WEAK = "WEAK"


class ComplianceEvaluation(BaseModel):
    """Pydantic structured output model for evidence evaluation."""
    decision: EvaluationDecision = Field(
        description="Routing decision: 'GOOD' if evidence is complete, relevant, and consistent; otherwise 'WEAK'."
    )
    evidence_sufficiency: bool = Field(
        description="True if retrieved chunks contain sufficient factual depth to answer the query."
    )
    relevance: bool = Field(
        description="True if the chunks are directly relevant to the user's information requirement."
    )
    query_coverage: float = Field(
        description="Score between 0.0 and 1.0 indicating fraction of query requirements covered."
    )
    contradiction_detected: bool = Field(
        description="True if conflicting or contradictory claims are detected across retrieved sources."
    )
    contradiction_resolved: bool = Field(
        default=True,
        description="True if context/applicability reconciles the conflict; False if conflict is irreconcilable."
    )
    unresolved_reason: Optional[str] = Field(
        default=None,
        description="Detailed explanation if contradiction cannot be resolved."
    )
    missing_information: Optional[str] = Field(
        default=None,
        description="Specific missing facts or gaps if decision is WEAK."
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Refined and focused query targeting the missing information if decision is WEAK."
    )


# ============================================================================
# LangGraph State Schema
# ============================================================================

class ComplianceState(TypedDict):
    """State schema for the Compliance Agent LangGraph workflow."""
    original_query: str
    current_query: str
    attempt: int
    max_attempts: int
    chunks: List[Dict[str, Any]]
    evaluation: Optional[ComplianceEvaluation]
    status: str
    final_output: str


# ============================================================================
# Node Implementations
# ============================================================================

def retrieve_chunks_node(state: ComplianceState) -> Dict[str, Any]:
    """Retrieve top candidate chunks using hybrid search."""
    query = state["current_query"]
    results = hybrid_search(query, top_k=5)
    return {"chunks": results}


def evaluate_evidence_node(state: ComplianceState) -> Dict[str, Any]:
    """Evaluate retrieved chunks for sufficiency, relevance, coverage, and contradictions."""
    query = state["original_query"]
    chunks = state.get("chunks", [])

    if not chunks:
        # No chunks retrieved -> automatic WEAK evaluation
        weak_eval = ComplianceEvaluation(
            decision=EvaluationDecision.WEAK,
            evidence_sufficiency=False,
            relevance=False,
            query_coverage=0.0,
            contradiction_detected=False,
            contradiction_resolved=True,
            missing_information="No evidence chunks were retrieved.",
            rewritten_query=f"{query} technical documentation"
        )
        return {"evaluation": weak_eval}

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_api_key,
                temperature=0.0
            )
            structured_llm = llm.with_structured_output(ComplianceEvaluation)

            context_snippets = "\n\n".join(
                f"[{idx+1}] ID: {c.get('id', 'N/A')}\n{c.get('content', '')}"
                for idx, c in enumerate(chunks[:5])
            )

            prompt = (
                f"User Query:\n{query}\n\n"
                f"Retrieved Evidence Chunks:\n{context_snippets}\n\n"
                "Evaluate the evidence against the query following the system guidelines."
            )

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            evaluation: ComplianceEvaluation = structured_llm.invoke(messages)
            return {"evaluation": evaluation}
        except Exception as e:
            print(f"Compliance LLM evaluation fallback: {e}")

    # Fallback heuristic evaluation
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    corpus_text = " ".join(c.get("content", "").lower() for c in chunks)
    matched = [t for t in query_terms if t in corpus_text]
    coverage = len(matched) / len(query_terms) if query_terms else 1.0

    is_sufficient = coverage >= 0.5 and len(chunks) > 0
    decision = EvaluationDecision.GOOD if is_sufficient else EvaluationDecision.WEAK
    missing = "Specific details missing from retrieved excerpts" if not is_sufficient else None
    rewritten = f"{query} procedure guide" if not is_sufficient else None

    fallback_eval = ComplianceEvaluation(
        decision=decision,
        evidence_sufficiency=is_sufficient,
        relevance=coverage > 0.3,
        query_coverage=round(coverage, 2),
        contradiction_detected=False,
        contradiction_resolved=True,
        missing_information=missing,
        rewritten_query=rewritten
    )
    return {"evaluation": fallback_eval}


def route_compliance_decision(state: ComplianceState) -> Literal["proceed", "rewrite", "evidence_not_found"]:
    """Conditional router based on compliance evaluation and retry limits."""
    evaluation = state.get("evaluation")
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 3)

    if not evaluation:
        return "evidence_not_found"

    # If contradiction is detected and cannot be resolved -> evidence not found
    if evaluation.contradiction_detected and not evaluation.contradiction_resolved:
        return "evidence_not_found"

    # If decision is GOOD -> proceed
    if evaluation.decision == EvaluationDecision.GOOD:
        return "proceed"

    # If decision is WEAK: retry if attempts remaining, otherwise fail
    if attempt < max_attempts:
        return "rewrite"

    return "evidence_not_found"


def rewrite_query_node(state: ComplianceState) -> Dict[str, Any]:
    """Increment attempt counter and update current query with refined rewrite."""
    evaluation = state.get("evaluation")
    attempt = state.get("attempt", 1) + 1

    new_query = state["current_query"]
    if evaluation and evaluation.rewritten_query:
        new_query = evaluation.rewritten_query.strip()
    elif evaluation and evaluation.missing_information:
        new_query = f"{state['original_query']} {evaluation.missing_information}".strip()

    return {
        "attempt": attempt,
        "current_query": new_query
    }


def proceed_success_node(state: ComplianceState) -> Dict[str, Any]:
    """Finalize successful compliance validation with verified chunks."""
    chunks = state.get("chunks", [])
    evaluation = state.get("evaluation")
    summary = (
        f"Compliance verification PASSED (Coverage: {evaluation.query_coverage if evaluation else 1.0}). "
        f"Verified {len(chunks)} chunks without irreconcilable contradictions."
    )
    return {
        "status": "GOOD",
        "final_output": summary
    }


def proceed_failure_node(state: ComplianceState) -> Dict[str, Any]:
    """Finalize failed compliance verification when evidence is insufficient or contradicted."""
    return {
        "status": "WEAK",
        "final_output": "evidence not found"
    }


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

builder = StateGraph(ComplianceState)
builder.add_node("retrieve_chunks", retrieve_chunks_node)
builder.add_node("evaluate_evidence", evaluate_evidence_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("proceed_success", proceed_success_node)
builder.add_node("proceed_failure", proceed_failure_node)

builder.add_edge(START, "retrieve_chunks")
builder.add_edge("retrieve_chunks", "evaluate_evidence")

builder.add_conditional_edges(
    "evaluate_evidence",
    route_compliance_decision,
    {
        "proceed": "proceed_success",
        "rewrite": "rewrite_query",
        "evidence_not_found": "proceed_failure"
    }
)

builder.add_edge("rewrite_query", "retrieve_chunks")
builder.add_edge("proceed_success", END)
builder.add_edge("proceed_failure", END)

compliance_agent_graph = builder.compile()


# ============================================================================
# Public API Function
# ============================================================================

def evaluate_compliance(
    query: str,
    max_attempts: int = 3
) -> Dict[str, Any]:
    """Evaluate retrieval compliance, resolve contradictions, and execute up to 3 rewrite retries.

    Args:
        query: User query to evaluate against knowledge sources.
        max_attempts: Maximum retrieval rewrite attempts before failing (default 3).

    Returns:
        Dictionary with final status, verified chunks, compliance evaluation, and final_output.
    """
    initial_state: ComplianceState = {
        "original_query": query.strip(),
        "current_query": query.strip(),
        "attempt": 1,
        "max_attempts": max_attempts,
        "chunks": [],
        "evaluation": None,
        "status": "",
        "final_output": ""
    }

    result = compliance_agent_graph.invoke(initial_state, {"recursion_limit": 25})
    return {
        "status": result["status"],
        "final_output": result["final_output"],
        "attempts_made": result["attempt"],
        "chunks": result["chunks"],
        "evaluation": result["evaluation"]
    }
