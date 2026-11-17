"""Search Agent for discovering relevant information across enterprise knowledge sources.

Supports keyword-based and semantic discovery across:
- Unstructured sources: SharePoint (PDF), Confluence (HTML), Teams (TXT), Outlook (EML) via ChromaDB
- Structured sources: SQLite database (knowledge_articles, assets, maintenance), Jira, and ServiceNow
Uses LangGraph for search coordination and result consolidation.
"""

import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Offline HF embeddings safeguard
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from langgraph.graph import END, START, StateGraph

from ingestion.ingestion import hf_embeddings, knowledge_collection


DB_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "structured"
    / "sql"
    / "northstar_enterprise.sqlite"
)
JIRA_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "structured"
    / "jira"
    / "jira_issues.json"
)
SERVICENOW_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "structured"
    / "servicenow"
    / "servicenow_incidents.json"
)


# ============================================================================
# LangGraph State Definition
# ============================================================================

class SearchState(TypedDict):
    """State schema for the Search Agent LangGraph workflow."""
    query: str
    top_k: int
    semantic_results: List[Dict[str, Any]]
    structured_results: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    summary: str


# ============================================================================
# Search Node Implementations
# ============================================================================

def semantic_search_node(state: SearchState) -> Dict[str, Any]:
    """Execute vector similarity search on ChromaDB across indexed enterprise documents."""
    query = state["query"]
    top_k = state.get("top_k", 5)
    semantic_results: List[Dict[str, Any]] = []

    try:
        if knowledge_collection.count() > 0:
            query_embedding = hf_embeddings.embed_query(query)
            results = knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, knowledge_collection.count())
            )

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)

            for doc_id, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
                score = round(1.0 - float(dist), 4) if dist is not None else 1.0
                # Filter out irrelevant vector matches with low similarity
                if dist is not None and dist > 0.68:
                    continue
                semantic_results.append({
                    "source_type": "semantic_vector",
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": meta,
                    "relevance_score": score
                })
    except Exception as e:
        print(f"Semantic search warning: {e}")

    return {"semantic_results": semantic_results}


def structured_search_node(state: SearchState) -> Dict[str, Any]:
    """Execute keyword search across SQLite tables, Jira issues, and ServiceNow incidents."""
    query = state["query"].lower()
    
    # Filter stopwords and intent/status words from general keywords
    stopwords = {
        "who", "what", "where", "when", "why", "how", "the", "and", "for", "with",
        "are", "is", "was", "were", "can", "could", "should", "would", "about",
        "find", "list", "show", "tell", "give", "search", "records", "related",
        "knowledge", "base", "from", "into", "that", "this", "these", "identify",
        "please", "expert", "experts"
    }
    status_intent_words = {
        "open", "active", "pending", "closed", "resolved", "incident", "incidents",
        "ticket", "tickets", "issue", "issues", "bug", "bugs", "work", "order"
    }
    
    raw_tokens = [w.strip("?,.:;!'\"()") for w in query.split()]
    all_keywords = [w for w in raw_tokens if len(w) > 2 and w not in stopwords]
    
    # Topic keywords exclude status/intent words so 'open' doesn't match all open issues indiscriminately
    topic_keywords = [w for w in all_keywords if w not in status_intent_words]
    
    # Domain synonym expansion for technical topics
    expanded_topic_keywords = set(topic_keywords)
    for kw in list(topic_keywords):
        if kw in ("database", "databases", "db"):
            expanded_topic_keywords.update(["dba", "sql", "backup", "backups", "database"])
        elif kw in ("network", "networking"):
            expanded_topic_keywords.update(["netops", "vpn", "dns", "wi-fi", "network"])
        elif kw in ("vpn", "ast001"):
            expanded_topic_keywords.update(["netops", "vpn", "ast001"])
        elif kw in ("iam", "identity", "auth", "sso", "mfa"):
            expanded_topic_keywords.update(["iam", "mfa", "sso", "identity"])
        elif kw in ("security", "secops"):
            expanded_topic_keywords.update(["secops", "audit", "incident response"])
        elif kw in ("itops", "laptop", "endpoint", "hardware"):
            expanded_topic_keywords.update(["itops", "email", "endpoints", "storage"])

    search_terms = list(expanded_topic_keywords) if expanded_topic_keywords else all_keywords

    wants_open = any(w in query for w in ("open", "active", "pending", "current", "unresolved"))
    wants_incidents = any(w in query for w in ("incident", "incidents", "servicenow", "ticket", "tickets", "outage"))
    wants_jira = any(w in query for w in ("jira", "issue", "issues", "backlog", "sprint", "story", "bug"))

    structured_results: List[Dict[str, Any]] = []

    # 1. Search SQLite Database
    if DB_PATH.is_file() and search_terms:
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                # Search knowledge articles
                like_clauses = " OR ".join(["article_title LIKE ? OR problem LIKE ? OR procedure LIKE ?"] * len(search_terms))
                params = []
                for kw in search_terms:
                    params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])
                sql = f"SELECT article_id, article_title, problem, procedure, asset_ref FROM knowledge_articles WHERE {like_clauses} LIMIT 5"
                cur.execute(sql, params)
                for row in cur.fetchall():
                    structured_results.append({
                        "source_type": "sql_knowledge_article",
                        "id": row["article_id"],
                        "content": f"Title: {row['article_title']}\nProblem: {row['problem']}\nProcedure: {row['procedure']}",
                        "metadata": {
                            "file_name": "northstar_enterprise.sqlite",
                            "category": "SQL (Knowledge Articles)",
                            "summary": f"{row['article_title']}: {row['problem']}",
                            "asset_ref": row["asset_ref"]
                        }
                    })

                # Search maintenance history
                like_maint = " OR ".join(["fault_summary LIKE ? OR action_taken LIKE ?"] * len(search_terms))
                params_maint = []
                for kw in search_terms:
                    params_maint.extend([f"%{kw}%", f"%{kw}%"])
                sql_maint = f"SELECT work_order_id, asset_ref, fault_summary, action_taken, work_status FROM maintenance_history WHERE {like_maint} LIMIT 5"
                cur.execute(sql_maint, params_maint)
                for row in cur.fetchall():
                    structured_results.append({
                        "source_type": "sql_maintenance_record",
                        "id": row["work_order_id"],
                        "content": f"Asset: {row['asset_ref']} | Fault: {row['fault_summary']} | Action: {row['action_taken']} ({row['work_status']})",
                        "metadata": {
                            "file_name": "northstar_enterprise.sqlite",
                            "category": "SQL (Maintenance Records)",
                            "summary": f"Work Order {row['work_order_id']}: {row['fault_summary']}",
                            "asset_ref": row["asset_ref"]
                        }
                    })
        except Exception as e:
            print(f"SQLite search warning: {e}")

    # 2. Search ServiceNow JSON (Prioritized for incident/ticket queries)
    if SERVICENOW_PATH.is_file():
        try:
            with open(SERVICENOW_PATH, "r", encoding="utf-8") as f:
                snow_data = json.load(f)
                incidents = snow_data.get("result", []) if isinstance(snow_data, dict) else (snow_data if isinstance(snow_data, list) else [])

                snow_matches = []
                for inc in incidents:
                    number = inc.get("number", "INC")
                    s_desc = inc.get("short_description", "")
                    desc = inc.get("description", "")
                    state_raw = str(inc.get("state", "1"))
                    priority = inc.get("priority", "3")
                    grp = inc.get("assignment_group", "IT")

                    state_map = {"1": "New", "2": "In Progress", "3": "On Hold", "6": "Resolved", "7": "Closed"}
                    state_label = state_map.get(state_raw, f"State {state_raw}")
                    is_open = state_raw in ("1", "2", "3")

                    content_blob = f"{number} {s_desc} {desc} {grp}".lower()

                    # Check topic relevance
                    has_topic_match = any(kw in content_blob for kw in search_terms) if search_terms else False

                    # Match criteria
                    match = False
                    if wants_incidents and wants_open:
                        if search_terms:
                            match = has_topic_match and is_open
                        else:
                            match = is_open
                    elif wants_incidents:
                        match = has_topic_match if search_terms else True
                    elif has_topic_match:
                        match = True

                    if match:
                        snow_matches.append({
                            "source_type": "servicenow_incident",
                            "id": number,
                            "content": f"ServiceNow Incident {number}: {s_desc} | Priority: P{priority} | State: {state_label} | Group: {grp}",
                            "metadata": {
                                "file_name": "servicenow_incidents.json",
                                "category": "ServiceNow (Incidents/JSON)",
                                "summary": f"{number}: {s_desc} ({state_label}, Group: {grp})",
                                "state": state_label,
                                "is_open": is_open
                            }
                        })
                        if len(snow_matches) >= 6:
                            break

                # If specific topic open incidents were 0 but user asked for open incidents generally
                if wants_incidents and wants_open and not snow_matches:
                    for inc in incidents:
                        state_raw = str(inc.get("state", "1"))
                        if state_raw in ("1", "2", "3"):
                            number = inc.get("number", "INC")
                            s_desc = inc.get("short_description", "")
                            grp = inc.get("assignment_group", "IT")
                            priority = inc.get("priority", "3")
                            state_map = {"1": "New", "2": "In Progress", "3": "On Hold"}
                            snow_matches.append({
                                "source_type": "servicenow_incident",
                                "id": number,
                                "content": f"ServiceNow Incident {number}: {s_desc} | Priority: P{priority} | State: {state_map.get(state_raw, 'Open')} | Group: {grp}",
                                "metadata": {
                                    "file_name": "servicenow_incidents.json",
                                    "category": "ServiceNow (Incidents/JSON)",
                                    "summary": f"{number}: {s_desc} ({state_map.get(state_raw, 'Open')})",
                                    "is_open": True
                                }
                            })
                            if len(snow_matches) >= 5:
                                break

                structured_results.extend(snow_matches)
        except Exception as e:
            print(f"ServiceNow search warning: {e}")

    # 3. Search Jira JSON
    if JIRA_PATH.is_file() and search_terms:
        try:
            with open(JIRA_PATH, "r", encoding="utf-8") as f:
                jira_data = json.load(f)
                issues = jira_data.get("issues", []) if isinstance(jira_data, dict) else (jira_data if isinstance(jira_data, list) else [])
                jira_matches = []
                for issue in issues:
                    fields = issue.get("fields", {}) if isinstance(issue, dict) else {}
                    summary = fields.get("summary") or issue.get("summary", "")
                    status = fields.get("status", {}).get("name") if isinstance(fields.get("status"), dict) else issue.get("status", "Open")
                    desc = str(fields.get("description") or issue.get("description", ""))
                    key = issue.get("key", "JIRA")

                    # Check topic keywords against summary and description (NOT against status)
                    content_blob = f"{key} {summary} {desc}".lower()
                    if any(kw in content_blob for kw in search_terms):
                        jira_matches.append({
                            "source_type": "jira_issue",
                            "id": key,
                            "content": f"Jira Issue {key}: {summary} | Status: {status} | Details: {desc[:200]}",
                            "metadata": {
                                "file_name": "jira_issues.json",
                                "category": "Jira (Issues/JSON)",
                                "summary": f"{key}: {summary} ({status})"
                            }
                        })
                        if len(jira_matches) >= (4 if wants_incidents else 6):
                            break
                structured_results.extend(jira_matches)
        except Exception as e:
            print(f"Jira search warning: {e}")

    return {"structured_results": structured_results}


def consolidate_results_node(state: SearchState) -> Dict[str, Any]:
    """Consolidate, deduplicate, and rank results from semantic and structured sources."""
    query = state["query"].lower()
    semantic = state.get("semantic_results", [])
    structured = state.get("structured_results", [])
    top_k = state.get("top_k", 8)

    wants_incidents = any(w in query for w in ("incident", "incidents", "servicenow", "ticket", "tickets"))
    wants_jira = any(w in query for w in ("jira", "issue", "issues", "bug"))

    def source_priority(item: Dict[str, Any]) -> int:
        src = item.get("source_type", "")
        if wants_incidents and "servicenow" in src:
            return 0
        if wants_jira and "jira" in src:
            return 0
        if "sql" in src:
            return 1
        if "servicenow" in src:
            return 2
        if "jira" in src:
            return 3
        return 4

    structured_sorted = sorted(structured, key=source_priority)

    combined: List[Dict[str, Any]] = []
    seen_ids = set()

    for item in structured_sorted + semantic:
        item_id = item.get("id")
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            combined.append(item)

    final_results = combined[:top_k]
    summary_text = (
        f"Search completed for '{state['query']}': "
        f"retrieved {len(final_results)} relevant enterprise records "
        f"({len(semantic)} semantic, {len(structured)} structured)."
    )

    return {
        "final_results": final_results,
        "summary": summary_text
    }


# ============================================================================
# LangGraph Workflow Compilation
# ============================================================================

builder = StateGraph(SearchState)
builder.add_node("semantic_search", semantic_search_node)
builder.add_node("structured_search", structured_search_node)
builder.add_node("consolidate_results", consolidate_results_node)

builder.add_edge(START, "semantic_search")
builder.add_edge(START, "structured_search")
builder.add_edge("semantic_search", "consolidate_results")
builder.add_edge("structured_search", "consolidate_results")
builder.add_edge("consolidate_results", END)

search_agent_graph = builder.compile()


def search_enterprise(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute enterprise search across structured and unstructured knowledge sources using LangGraph.

    Args:
        query: User search query.
        top_k: Maximum number of relevant records to return.

    Returns:
        Dictionary containing final results, semantic matches, structured matches, and summary.
    """
    initial_state: SearchState = {
        "query": query.strip(),
        "top_k": top_k,
        "semantic_results": [],
        "structured_results": [],
        "final_results": [],
        "summary": ""
    }
    return search_agent_graph.invoke(initial_state)
