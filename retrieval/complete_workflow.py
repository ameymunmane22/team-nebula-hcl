"""Complete Enterprise Agentic RAG Workflow.

Coordinates the full end-to-end agentic lifecycle:
1. Input Guardrails: Detects prompt injections, harmful content, and masks sensitive data.
2. Query Classification: Uses Pydantic structured output to classify queries as SIMPLE or COMPLEX.
3. Orchestration Agent: Allocates subtasks across specialized agents (Search, Expert Finder, Compliance, Summarization).
4. Sub-Agent Execution: Dispatches and gathers evidence from hybrid search, employee directories, and compliance verification.
5. Output LLM: Synthesizes high-fidelity, grounded answers using ChatGroq.
6. Output Guardrails: Sanitizes final responses before presenting to the user.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, TypedDict

# Ensure project root is in sys.path when running script directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Avoid network timeouts when offline
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from retrieval.complaince_agent import evaluate_compliance
from retrieval.expert_finder import find_experts
from retrieval.guardrails import (
    check_guardrails,
    detect_harmful_content,
    detect_prompt_injection,
    is_irrelevant_or_greeting,
    mask_sensitive_info,
)
from retrieval.orchestration_agent import (
    ExecutionStrategy,
    SubAgentType,
    TaskAllocation,
    orchestrate_query,
)
from retrieval.query_decomposition import classify_query
from retrieval.search_agent import search_enterprise
from retrieval.summarization_agent import summarize_content

load_dotenv()


SYNTHESIS_SYSTEM_PROMPT: str = """You are the Senior Enterprise AI Knowledge Assistant for Northstar Systems.
Your mission is to generate a comprehensive, accurate, and fluent natural language answer synthesized directly from the retrieved enterprise evidence chunks.

CRITICAL CITATION & SYNTHESIS RULES:
1. Synthesize a complete, fluent explanation in natural language answering the user's query directly. Do NOT output raw chunk copies or isolated fragments.
2. Every factual statement, diagnosis, procedure, command, threshold, or policy assertion MUST include an inline citation in square brackets referencing the specific chunk ID from which it originates (e.g. [SP-001_chunk_0], [CF-002_chunk_1], [INC002]).
3. Synthesize multiple chunks together into a coherent narrative with actionable steps. When multiple chunks support a statement, cite them together: [SP-001_chunk_0, SP-001_chunk_1].
4. Retain all enterprise identifiers accurately (Asset IDs like AST..., Incident IDs like INC..., Employee IDs like EMP...).
5. If subject-matter experts are identified, weave them naturally into an expert escalation section with their names, titles, and email addresses.
6. Conclude your response with a dedicated '### Sources & Citations' section listing every cited chunk:
   - **[Chunk-ID]**: Document Title / System — Summary of referenced content.
7. If verified evidence is insufficient or compliance reports 'evidence not found', state clearly: "evidence not found: Sufficient verified enterprise records could not be found to answer your request."
8. Never invent or hallucinate facts outside the provided evidence."""


# ============================================================================
# State Schema
# ============================================================================

class CompleteWorkflowState(TypedDict):
    """Global state for the complete Agentic RAG workflow."""
    user_query: str
    sanitized_query: str
    is_safe: bool
    is_irrelevant: Optional[bool]
    security_violation_reason: Optional[str]
    query_classification: Optional[str]
    task_allocation: Optional[TaskAllocation]
    retrieved_evidence: List[Dict[str, Any]]
    expert_results: Optional[Dict[str, Any]]
    compliance_results: Optional[Dict[str, Any]]
    summary_results: Optional[Dict[str, Any]]
    raw_answer: str
    final_answer: str
    execution_steps: List[str]


# ============================================================================
# Node Implementations
# ============================================================================

def input_guardrail_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Execute input security checks: prompt injection, harmful content, and PII masking."""
    query = state["user_query"]
    steps = list(state.get("execution_steps", []))
    steps.append("1. Input Guardrails: Validating safety and masking sensitive information.")

    # 0. Check for casual greetings or irrelevant non-queries
    if is_irrelevant_or_greeting(query):
        return {
            "is_safe": True,
            "is_irrelevant": True,
            "sanitized_query": query,
            "security_violation_reason": None,
            "final_answer": "use this chatbot for query purpose only",
            "execution_steps": steps + ["Input identified as greeting/non-query. Guidance returned."]
        }

    # 1. Prompt Injection Check
    inj_result = detect_prompt_injection(query)
    if inj_result["detected"]:
        reason = f"Prompt injection blocked: detected pattern ({', '.join(inj_result['categories'])})"
        return {
            "is_safe": False,
            "is_irrelevant": False,
            "security_violation_reason": reason,
            "final_answer": f"Security Alert: Request rejected. {reason}",
            "execution_steps": steps
        }

    # 2. Harmful Content Check
    harm_result = detect_harmful_content(query)
    if harm_result["detected"]:
        reason = f"Prohibited request blocked: detected {', '.join(harm_result['categories'])}"
        return {
            "is_safe": False,
            "is_irrelevant": False,
            "security_violation_reason": reason,
            "final_answer": f"Safety Alert: Request rejected. {reason}",
            "execution_steps": steps
        }

    # 3. Mask Sensitive Enterprise Information
    sanitized = mask_sensitive_info(query)

    return {
        "is_safe": True,
        "is_irrelevant": False,
        "sanitized_query": sanitized,
        "security_violation_reason": None,
        "execution_steps": steps
    }


def query_classification_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Classify incoming sanitized query as SIMPLE or COMPLEX."""
    steps = list(state.get("execution_steps", []))
    steps.append("2. Query Classification: Assessing query complexity.")

    classification = classify_query(state["sanitized_query"])
    return {
        "query_classification": classification,
        "execution_steps": steps
    }


def orchestration_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Generate structured task allocation plan across specialized sub-agents."""
    steps = list(state.get("execution_steps", []))
    steps.append("3. Orchestration: Allocating subtasks across sub-agents.")

    plan = orchestrate_query(
        query=state["sanitized_query"],
        classification=state.get("query_classification")
    )
    return {
        "task_allocation": plan,
        "execution_steps": steps
    }


def subagent_execution_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Execute allocated subtasks by delegating to specialized agents."""
    steps = list(state.get("execution_steps", []))
    plan: Optional[TaskAllocation] = state.get("task_allocation")
    query = state["sanitized_query"]

    retrieved_evidence: List[Dict[str, Any]] = []
    expert_results: Optional[Dict[str, Any]] = None
    compliance_results: Optional[Dict[str, Any]] = None
    summary_results: Optional[Dict[str, Any]] = None

    if not plan or not plan.tasks:
        # Default fallback to Search Agent
        steps.append("4. Execution: Running default Search Agent.")
        search_res = search_enterprise(query, top_k=5)
        retrieved_evidence = search_res.get("final_results", [])
    else:
        steps.append(f"4. Execution: Dispatching {len(plan.tasks)} task(s) ({plan.execution_strategy.value} strategy).")
        for task in plan.tasks:
            agent_type = task.assigned_agent

            if agent_type == SubAgentType.SEARCH_AGENT:
                search_res = search_enterprise(task.description, top_k=5)
                retrieved_evidence.extend(search_res.get("final_results", []))

            elif agent_type == SubAgentType.EXPERT_FINDER:
                expert_results = find_experts(task.description, top_n=3)

            elif agent_type == SubAgentType.COMPLIANCE_AGENT:
                compliance_results = evaluate_compliance(task.description, max_attempts=2)
                if compliance_results.get("chunks"):
                    retrieved_evidence.extend(compliance_results["chunks"])

            elif agent_type == SubAgentType.SUMMARIZATION_AGENT:
                # Summarize currently gathered evidence
                if retrieved_evidence:
                    summary_results = summarize_content(retrieved_evidence, summary_type="concise")

    # Deduplicate retrieved evidence by ID
    unique_evidence: List[Dict[str, Any]] = []
    seen_ids = set()
    for doc in retrieved_evidence:
        doc_id = doc.get("id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_evidence.append(doc)

    return {
        "retrieved_evidence": unique_evidence,
        "expert_results": expert_results,
        "compliance_results": compliance_results,
        "summary_results": summary_results,
        "execution_steps": steps
    }


def _synthesize_deterministic_natural_language_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    experts: Optional[Dict[str, Any]] = None,
    compliance: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Synthesize fluent natural language answer with inline citations and a reference list."""
    import re

    query_lower = query.lower()
    is_asking_expert = any(w in query_lower for w in ["expert", "experts", "who is", "who are", "lead", "owner", "contact", "engineer", "dba", "sme"])
    is_asking_incident = any(w in query_lower for w in ["incident", "incidents", "ticket", "tickets", "open", "active", "servicenow", "outage"])
    is_asking_procedure = any(w in query_lower for w in ["procedure", "troubleshoot", "troubleshooting", "guide", "how to", "steps", "solve", "fix", "resolve", "checklist", "policy", "validation"])

    # Extract topic terms for relevance validation
    query_tokens = [re.sub(r"[^\w\-]", "", w) for w in query_lower.split()]
    stopwords = {"who", "what", "where", "when", "why", "how", "the", "and", "for", "with", "are", "is", "about", "find", "show", "tell", "give", "open", "incidents", "expert"}
    topic_terms = {w for w in query_tokens if len(w) > 2 and w not in stopwords}
    expanded_terms = set(topic_terms)
    for t in topic_terms:
        if t in ("database", "databases", "db"):
            expanded_terms.update(["dba", "sql", "backup", "backups", "database"])
        elif t in ("network", "networking"):
            expanded_terms.update(["netops", "vpn", "dns", "wi-fi", "network"])
        elif t in ("vpn", "ast001"):
            expanded_terms.update(["netops", "vpn", "ast001"])
        elif t in ("iam", "identity", "auth", "sso", "mfa"):
            expanded_terms.update(["iam", "mfa", "sso", "identity"])
        elif t in ("security", "secops"):
            expanded_terms.update(["secops", "incident response", "audit"])
        elif t in ("itops", "laptop", "endpoint", "hardware"):
            expanded_terms.update(["itops", "email", "endpoints", "storage"])

    sections: List[str] = [f"### Resolution & Guidance for: {query}\n"]
    citations: List[str] = []
    seen_citations = set()

    # 1. Subject-Matter Experts (Prioritized when query asks for experts)
    ranked_experts = experts.get("ranked_experts", []) if experts else []
    if ranked_experts and is_asking_expert:
        exp_lines = []
        for exp in ranked_experts[:3]:
            exp_id = exp.get("employee_id", "EMP")
            skills = ", ".join(exp.get("matched_skills", []))
            exp_lines.append(
                f"- **{exp.get('full_name')}** ({exp.get('job_title')} | Team: {exp.get('team')}) — "
                f"Contact: `{exp.get('email')}` [Skills: {skills}] [{exp_id}]"
            )
            if exp_id not in seen_citations:
                seen_citations.add(exp_id)
                citations.append(f"- **[{exp_id}]**: *Employee Directory (SQLite)* — {exp.get('full_name')}, {exp.get('job_title')} ({exp.get('team')})")
        sections.append("**Subject-Matter Experts & Ownership:**\n" + "\n".join(exp_lines) + "\n")

    # 2. Open / Associated Incidents
    incident_docs = [
        d for d in evidence
        if "servicenow" in str(d.get("source_type", "")).lower() or str(d.get("id", "")).startswith("INC")
    ]
    if incident_docs and is_asking_incident:
        inc_lines = []
        for inc in incident_docs[:6]:
            inc_id = inc.get("id")
            meta = inc.get("metadata") or {}
            desc = meta.get("summary") or inc.get("content", "").replace("\n", " ").strip()
            state_label = meta.get("state", "Open")
            inc_lines.append(f"- **[{inc_id}]**: {desc}")
            if inc_id not in seen_citations:
                seen_citations.add(inc_id)
                citations.append(f"- **[{inc_id}]**: *servicenow_incidents.json* (ServiceNow) — {desc}")
        sections.append("**Active & Open Incidents:**\n" + "\n".join(inc_lines) + "\n")

    # 3. Knowledge Procedures & Guidance (Filtered for topic relevance)
    overview_statements: List[str] = []
    action_steps: List[str] = []

    for doc in evidence:
        doc_id = doc.get("id", "REF")
        if str(doc_id).startswith("INC") and is_asking_incident:
            continue  # Already represented in Incidents section

        content = doc.get("content", "").strip()
        meta = doc.get("metadata") or {}
        file_name = meta.get("file_name", doc_id)
        category = meta.get("category", doc.get("source", "Knowledge Base"))

        if not content:
            continue

        # Relevance filter: Ensure document content or metadata shares at least one term with the query
        content_lower = content.lower()
        if expanded_terms and not any(term in content_lower or term in file_name.lower() for term in expanded_terms):
            continue

        # Clean noise headers
        cleaned = re.sub(r"NORTHSTAR SYSTEMS Internal IT knowledge \| [A-Z0-9_-]+", "", content, flags=re.I).strip()
        cleaned = re.sub(r"[\r\n]+", " ", cleaned)

        # Extract structured sections if present
        section_pattern = r"\b(Scope|Symptoms|Diagnostic Steps|Validation|Escalation|Procedure|Checklist|Overview|Policy)\b"
        parts = re.split(section_pattern, cleaned, flags=re.I)

        if len(parts) > 2:
            for i in range(1, len(parts), 2):
                heading = parts[i].strip()
                body = parts[i + 1].strip() if (i + 1) < len(parts) else ""
                body = re.sub(r"^[:\s\-\.]+", "", body).strip()
                if len(body) < 5:
                    continue

                if heading.lower() in ("scope", "symptoms", "overview", "policy"):
                    overview_statements.append(f"{body} [{doc_id}]")
                else:
                    action_steps.append(f"**{heading.capitalize()}**: {body} [{doc_id}]")
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", cleaned) if len(s.strip()) > 15]
            if sentences:
                overview_statements.append(f"{sentences[0]} [{doc_id}]")
                for s in sentences[1:3]:
                    action_steps.append(f"{s} [{doc_id}]")

        if doc_id not in seen_citations:
            seen_citations.add(doc_id)
            raw_summary = meta.get("summary")
            if not raw_summary:
                raw_summary = cleaned[:120] + ("..." if len(cleaned) > 120 else "")
            clean_summary = re.sub(r"\s+", " ", str(raw_summary)).strip()
            citations.append(f"- **[{doc_id}]**: *{file_name}* ({category}) — {clean_summary}")

    # Build Overview & Action Steps if procedures or relevant content exist
    if overview_statements and (is_asking_procedure or not is_asking_expert or not ranked_experts):
        combined_overview = " ".join(overview_statements[:3])
        sections.append(f"**Overview & Context:**\n{combined_overview}\n")
    elif summary and summary.get("summary") and is_asking_procedure:
        sections.append(f"**Summary of Retrieved Evidence:**\n{summary['summary']}\n")

    if action_steps and (is_asking_procedure or not is_asking_expert or not ranked_experts):
        formatted_steps = []
        for idx, step in enumerate(action_steps[:5]):
            formatted_steps.append(f"{idx + 1}. {step}")
        sections.append("**Recommended Procedures & Action Steps:**\n" + "\n".join(formatted_steps) + "\n")

    # If experts were found but wasn't explicitly asked first, append them at the end
    if ranked_experts and not is_asking_expert:
        exp_lines = []
        for exp in ranked_experts[:2]:
            exp_id = exp.get("employee_id", "EMP")
            skills = ", ".join(exp.get("matched_skills", []))
            exp_lines.append(
                f"- **{exp.get('full_name')}** ({exp.get('job_title')} | Team: {exp.get('team')}) — "
                f"Contact: `{exp.get('email')}` [Skills: {skills}] [{exp_id}]"
            )
            if exp_id not in seen_citations:
                seen_citations.add(exp_id)
                citations.append(f"- **[{exp_id}]**: *Employee Directory (SQLite)* — {exp.get('full_name')}, {exp.get('job_title')}")
        sections.append("**Subject-Matter Experts & Escalation Points:**\n" + "\n".join(exp_lines) + "\n")

    # Sources & Citations
    if citations:
        sections.append("### Sources & Citations\n" + "\n".join(citations))
    elif not overview_statements and not action_steps and not ranked_experts and not incident_docs:
        return "evidence not found: Sufficient verified enterprise records could not be found to answer your request."

    return "\n".join(sections).strip()


def output_llm_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Synthesize grounded, natural language response with inline citations using ChatGroq or deterministic synthesis."""
    steps = list(state.get("execution_steps", []))
    steps.append("5. Output Generation: Synthesizing natural language response with citations.")

    query = state["sanitized_query"]
    evidence = state.get("retrieved_evidence", [])
    experts = state.get("expert_results")
    compliance = state.get("compliance_results")
    summary = state.get("summary_results")

    # Check if compliance agent flagged lack of evidence
    if compliance and compliance.get("final_output") == "evidence not found" and not evidence:
        return {
            "raw_answer": "evidence not found: Sufficient verified enterprise records could not be found to answer your request.",
            "execution_steps": steps
        }

    # Format context for LLM with explicit chunk IDs and document sources
    context_sections: List[str] = []

    if evidence:
        doc_lines = []
        for idx, doc in enumerate(evidence[:6]):
            content = doc.get("content", str(doc)).strip()
            doc_id = doc.get("id", f"DOC-{idx+1}")
            meta = doc.get("metadata") or {}
            source_name = meta.get("file_name", doc.get("source", "Knowledge Base"))
            category = meta.get("category", "General")
            doc_lines.append(f"### [Chunk ID: {doc_id}] (Source: {source_name}, Category: {category})\n{content}")
        context_sections.append("### Retrieved Enterprise Evidence Chunks:\n\n" + "\n\n".join(doc_lines))

    if experts and experts.get("ranked_experts"):
        exp_lines = []
        for exp in experts["ranked_experts"][:2]:
            exp_id = exp.get("employee_id", "EMP")
            exp_lines.append(
                f"- {exp.get('full_name')} ({exp.get('job_title')}, {exp.get('team')}) - Email: {exp.get('email')} "
                f"[Employee ID: {exp_id}, Skills: {', '.join(exp.get('matched_skills', []))}]"
            )
        context_sections.append("### Subject-Matter Experts Identified:\n" + "\n".join(exp_lines))

    if summary and summary.get("summary"):
        context_sections.append(f"### Intermediate Summary:\n{summary['summary']}")

    context_prompt = "\n\n".join(context_sections)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and context_sections:
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=groq_api_key,
                temperature=0.2
            )
            user_prompt = (
                f"User Query:\n{query}\n\n"
                f"{context_prompt}\n\n"
                "Please synthesize a comprehensive, natural language answer with inline citations "
                "(e.g. [Chunk-ID]) referencing the provided evidence chunks, and conclude with a '### Sources & Citations' section."
            )
            messages = [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            llm_text = str(response.content).strip()
            if llm_text:
                return {
                    "raw_answer": llm_text,
                    "execution_steps": steps
                }
        except Exception as e:
            print(f"Output generation fallback: {e}")

    # Fallback to intelligent deterministic natural language synthesis with citations
    natural_answer = _synthesize_deterministic_natural_language_answer(
        query=query,
        evidence=evidence,
        experts=experts,
        compliance=compliance,
        summary=summary,
    )

    return {
        "raw_answer": natural_answer,
        "execution_steps": steps
    }


def output_guardrail_node(state: CompleteWorkflowState) -> Dict[str, Any]:
    """Execute output security guardrails: mask any leaked credentials or sensitive data."""
    steps = list(state.get("execution_steps", []))
    steps.append("6. Output Guardrails: Inspecting response for credential leaks or sensitive data.")

    raw = state.get("raw_answer") or state.get("final_answer") or ""
    sanitized = mask_sensitive_info(raw)

    return {
        "final_answer": sanitized,
        "execution_steps": steps
    }


def route_after_input_guardrail(state: CompleteWorkflowState) -> str:
    """Route to execution pipeline if safe and relevant; terminate early if unsafe or irrelevant."""
    if not state.get("is_safe", True) or state.get("is_irrelevant", False):
        return "block"
    return "proceed"


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

workflow = StateGraph(CompleteWorkflowState)

workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("query_classification", query_classification_node)
workflow.add_node("orchestration", orchestration_node)
workflow.add_node("subagent_execution", subagent_execution_node)
workflow.add_node("output_llm", output_llm_node)
workflow.add_node("output_guardrail", output_guardrail_node)

workflow.add_edge(START, "input_guardrail")

workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    {
        "proceed": "query_classification",
        "block": "output_guardrail"
    }
)

workflow.add_edge("query_classification", "orchestration")
workflow.add_edge("orchestration", "subagent_execution")
workflow.add_edge("subagent_execution", "output_llm")
workflow.add_edge("output_llm", "output_guardrail")
workflow.add_edge("output_guardrail", END)

complete_rag_graph = workflow.compile()


# ============================================================================
# Public API Function
# ============================================================================

def run_agentic_rag(query: str) -> Dict[str, Any]:
    """Run the complete end-to-end Enterprise Agentic RAG workflow.

    Args:
        query: The incoming user query.

    Returns:
        Dictionary containing final answer, execution steps, query classification,
        task allocation plan, and safety status.
    """
    initial_state: CompleteWorkflowState = {
        "user_query": query.strip(),
        "sanitized_query": "",
        "is_safe": True,
        "is_irrelevant": False,
        "security_violation_reason": None,
        "query_classification": None,
        "task_allocation": None,
        "retrieved_evidence": [],
        "expert_results": None,
        "compliance_results": None,
        "summary_results": None,
        "raw_answer": "",
        "final_answer": "",
        "execution_steps": []
    }

    result = complete_rag_graph.invoke(initial_state)
    return {
        "query": query,
        "final_answer": result["final_answer"],
        "classification": result.get("query_classification"),
        "is_safe": result.get("is_safe"),
        "is_irrelevant": result.get("is_irrelevant", False),
        "execution_steps": result.get("execution_steps", []),
        "task_allocation": result.get("task_allocation"),
        "evidence_count": len(result.get("retrieved_evidence", []))
    }


if __name__ == "__main__":
    sample_query = "What is the procedure for AST001 troubleshooting?"
    print(f"Running Agentic RAG for: '{sample_query}'\n")
    response = run_agentic_rag(sample_query)
    print("=== Execution Steps ===")
    for step in response["execution_steps"]:
        print(f"  {step}")
    print("\n=== Final Answer ===")
    print(response["final_answer"])

