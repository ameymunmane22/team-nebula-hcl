"""Expert Finder Agent for identifying employees and subject-matter experts.

Identifies personnel based on:
- Technical skills, proficiencies, and years of experience (employee_skills)
- Job title, team, department, and skills directory (employees)
- Knowledge article ownership (knowledge_articles)
- Technical incident/maintenance resolution history (maintenance_history)

Orchestrated using LangGraph.
"""

from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


DB_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "structured"
    / "sql"
    / "northstar_enterprise.sqlite"
)


# Technical domain and role synonyms for robust employee and SME matching
ROLE_SYNONYMS: Dict[str, List[str]] = {
    "database": ["dba", "sql", "backups", "performance", "database", "rdbms", "postgres", "mysql"],
    "databases": ["dba", "sql", "backups", "performance", "database"],
    "db": ["dba", "sql", "backups", "performance", "database"],
    "dba": ["dba", "sql", "backups", "performance", "database"],
    "sql": ["dba", "sql", "backups", "performance", "database"],
    "backup": ["dba", "sql", "backups", "performance", "storage"],
    "backups": ["dba", "sql", "backups", "performance", "storage"],
    "network": ["netops", "vpn", "dns", "wi-fi", "network", "firewall", "router"],
    "networking": ["netops", "vpn", "dns", "wi-fi", "network"],
    "vpn": ["netops", "vpn", "dns", "wi-fi", "ast001"],
    "dns": ["netops", "vpn", "dns", "wi-fi"],
    "wifi": ["netops", "wi-fi", "vpn", "dns"],
    "wi-fi": ["netops", "wi-fi", "vpn", "dns"],
    "iam": ["iam", "mfa", "sso", "identity", "authentication"],
    "identity": ["iam", "mfa", "sso", "identity", "authentication"],
    "sso": ["iam", "mfa", "sso", "identity"],
    "mfa": ["iam", "mfa", "sso", "identity"],
    "auth": ["iam", "mfa", "sso", "identity", "secops", "authentication"],
    "authentication": ["iam", "mfa", "sso", "identity", "secops", "authentication"],
    "itops": ["itops", "email", "endpoints", "storage"],
    "email": ["itops", "email", "endpoints", "storage"],
    "storage": ["itops", "email", "endpoints", "storage"],
    "endpoint": ["itops", "email", "endpoints", "storage"],
    "endpoints": ["itops", "email", "endpoints", "storage"],
    "laptop": ["itops", "email", "endpoints", "storage"],
    "desktop": ["itops", "email", "endpoints", "storage"],
    "hardware": ["itops", "email", "endpoints", "storage"],
    "security": ["secops", "incident response", "audit", "authentication", "security"],
    "secops": ["secops", "incident response", "audit", "authentication"],
    "audit": ["secops", "incident response", "audit"],
    "incident response": ["secops", "incident response", "audit"],
    "appdev": ["appdev", "erp", "apis", "web services"],
    "api": ["appdev", "erp", "apis", "web services"],
    "apis": ["appdev", "erp", "apis", "web services"],
    "erp": ["appdev", "erp", "apis", "web services"],
    "web": ["appdev", "erp", "apis", "web services"],
    "software": ["appdev", "erp", "apis", "web services"],
    "ast001": ["netops", "vpn", "ast001"],
    "ast002": ["iam", "mfa", "ast002"],
    "ast003": ["itops", "email", "ast003"],
    "ast004": ["dba", "sql", "ast004"],
    "ast005": ["appdev", "erp", "ast005"],
    "ast006": ["secops", "incident response", "ast006"]
}


# ============================================================================
# LangGraph State Schema
# ============================================================================

class ExpertFinderState(TypedDict):
    """State schema for the Expert Finder LangGraph workflow."""
    query: str
    top_n: int
    target_terms: List[str]
    candidate_experts: List[Dict[str, Any]]
    ranked_experts: List[Dict[str, Any]]
    summary: str


# ============================================================================
# Node Implementations
# ============================================================================

def extract_criteria_node(state: ExpertFinderState) -> Dict[str, Any]:
    """Extract search terms, skills, technologies, and asset identifiers from query."""
    raw_query = state["query"]
    terms = set()

    # Strip orchestrator wrapper prompts (e.g., "Identify subject matter experts or owners related to: ...")
    clean_query = re.sub(
        r"^(identify|find|search for|locate)\s+(subject\s+matter\s+)?(experts?|owners?|engineers?|team)\s+(related\s+to|for|on)?:\s*",
        "",
        raw_query,
        flags=re.IGNORECASE
    ).strip()

    # Extract asset codes like AST001, AST002
    asset_matches = re.findall(r"\bAST\d+\b", clean_query, re.IGNORECASE)
    for m in asset_matches:
        code = m.upper()
        terms.add(code)
        if code.lower() in ROLE_SYNONYMS:
            terms.update(ROLE_SYNONYMS[code.lower()])

    # Extract meaningful keywords (excluding stopwords) and expand via domain synonyms
    stopwords = {
        "who", "the", "for", "and", "with", "what", "how", "expert", "experts",
        "owner", "owners", "lead", "find", "can", "help", "about", "are", "is",
        "tell", "show", "give", "please", "subject", "matter", "related", "open",
        "incidents", "issues", "tickets", "work", "order", "orders", "or", "on", "in", "to", "of"
    }
    words = [re.sub(r"[^\w\-]", "", w.lower()) for w in clean_query.split()]
    for word in words:
        if len(word) > 1 and word not in stopwords:
            terms.add(word)
            if word in ROLE_SYNONYMS:
                terms.update(ROLE_SYNONYMS[word])

    return {"target_terms": list(terms)}


def query_directory_node(state: ExpertFinderState) -> Dict[str, Any]:
    """Query employee database, skill matrices, article ownership, and maintenance logs."""
    terms = state.get("target_terms", [])
    candidates: Dict[str, Dict[str, Any]] = {}

    if not DB_PATH.is_file() or not terms:
        return {"candidate_experts": []}

    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # 1. Query employee profiles and skills
            for term in terms:
                pattern = f"%{term}%"
                cur.execute(
                    """
                    SELECT employee_id, full_name, team, job_title, skills, email, active
                    FROM employees
                    WHERE active = 1 AND (skills LIKE ? OR job_title LIKE ? OR team LIKE ? OR full_name LIKE ?)
                    """,
                    (pattern, pattern, pattern, pattern)
                )
                for row in cur.fetchall():
                    emp_id = row["employee_id"]
                    if emp_id not in candidates:
                        candidates[emp_id] = {
                            "employee_id": emp_id,
                            "full_name": row["full_name"],
                            "team": row["team"],
                            "job_title": row["job_title"],
                            "skills": row["skills"],
                            "email": row["email"],
                            "score": 10,
                            "matched_skills": [term],
                            "authored_articles": [],
                            "resolved_work_orders": 0
                        }
                    else:
                        candidates[emp_id]["score"] += 10
                        if term not in candidates[emp_id]["matched_skills"]:
                            candidates[emp_id]["matched_skills"].append(term)

            # 2. Query detailed employee_skills table for proficiencies
            for term in terms:
                pattern = f"%{term}%"
                cur.execute(
                    """
                    SELECT es.employee_id, es.skill_name, es.proficiency, es.years_experience, e.full_name, e.team, e.job_title, e.email
                    FROM employee_skills es
                    JOIN employees e ON es.employee_id = e.employee_id
                    WHERE e.active = 1 AND es.skill_name LIKE ?
                    """,
                    (pattern,)
                )
                for row in cur.fetchall():
                    emp_id = row["employee_id"]
                    proficiency_weight = {"Expert": 20, "Advanced": 15, "Intermediate": 10, "Beginner": 5}.get(row["proficiency"], 10)
                    bonus = proficiency_weight + int(row["years_experience"])

                    if emp_id not in candidates:
                        candidates[emp_id] = {
                            "employee_id": emp_id,
                            "full_name": row["full_name"],
                            "team": row["team"],
                            "job_title": row["job_title"],
                            "skills": f"{row['skill_name']} ({row['proficiency']})",
                            "email": row["email"],
                            "score": bonus,
                            "matched_skills": [f"{row['skill_name']} ({row['proficiency']}, {row['years_experience']} yrs)"],
                            "authored_articles": [],
                            "resolved_work_orders": 0
                        }
                    else:
                        candidates[emp_id]["score"] += bonus
                        skill_desc = f"{row['skill_name']} ({row['proficiency']}, {row['years_experience']} yrs)"
                        if skill_desc not in candidates[emp_id]["matched_skills"]:
                            candidates[emp_id]["matched_skills"].append(skill_desc)

            # 3. Query knowledge article ownership
            for term in terms:
                pattern = f"%{term}%"
                cur.execute(
                    """
                    SELECT article_id, article_title, owner_employee, asset_ref
                    FROM knowledge_articles
                    WHERE article_title LIKE ? OR problem LIKE ? OR asset_ref LIKE ?
                    """,
                    (pattern, pattern, pattern)
                )
                for row in cur.fetchall():
                    emp_id = row["owner_employee"]
                    if emp_id in candidates:
                        candidates[emp_id]["score"] += 15
                        candidates[emp_id]["authored_articles"].append(row["article_title"])

            # 4. Query maintenance resolution history
            for term in terms:
                pattern = f"%{term}%"
                cur.execute(
                    """
                    SELECT technician_id, COUNT(*) as resolution_count
                    FROM maintenance_history
                    WHERE (fault_summary LIKE ? OR asset_ref LIKE ?) AND work_status = 'Resolved'
                    GROUP BY technician_id
                    """,
                    (pattern, pattern)
                )
                for row in cur.fetchall():
                    emp_id = row["technician_id"]
                    if emp_id in candidates:
                        candidates[emp_id]["score"] += int(row["resolution_count"]) * 5
                        candidates[emp_id]["resolved_work_orders"] += int(row["resolution_count"])

    except Exception as e:
        print(f"Expert directory query error: {e}")

    return {"candidate_experts": list(candidates.values())}


def rank_and_explain_node(state: ExpertFinderState) -> Dict[str, Any]:
    """Rank candidates by score and compile expert profiles with rationales."""
    candidates = state.get("candidate_experts", [])
    top_n = state.get("top_n", 3)

    # Sort descending by computed relevance score
    ranked = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

    if ranked:
        top_expert = ranked[0]
        summary_text = (
            f"Identified {len(ranked)} subject-matter expert(s) for query '{state['query']}'. "
            f"Top match: {top_expert['full_name']} ({top_expert['job_title']} in {top_expert['team']}), "
            f"email: {top_expert['email']}."
        )
    else:
        summary_text = f"No direct subject-matter experts found for query '{state['query']}'."

    return {
        "ranked_experts": ranked,
        "summary": summary_text
    }


# ============================================================================
# LangGraph Workflow Construction
# ============================================================================

builder = StateGraph(ExpertFinderState)
builder.add_node("extract_criteria", extract_criteria_node)
builder.add_node("query_directory", query_directory_node)
builder.add_node("rank_and_explain", rank_and_explain_node)

builder.add_edge(START, "extract_criteria")
builder.add_edge("extract_criteria", "query_directory")
builder.add_edge("query_directory", "rank_and_explain")
builder.add_edge("rank_and_explain", END)

expert_finder_graph = builder.compile()


def find_experts(query: str, top_n: int = 3) -> Dict[str, Any]:
    """Identify subject-matter experts for a given technical query or asset.

    Args:
        query: User query regarding technical issues, assets, or expertise needed.
        top_n: Number of top experts to return.

    Returns:
        Dictionary with ranked experts, candidate details, and recommendation summary.
    """
    initial_state: ExpertFinderState = {
        "query": query.strip(),
        "top_n": top_n,
        "target_terms": [],
        "candidate_experts": [],
        "ranked_experts": [],
        "summary": ""
    }
    return expert_finder_graph.invoke(initial_state)
