#understand query from  query_decomposition agent and if its complex split the query into sub tasks and assign tasks to each sub agent
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

from retrieval.query_decomposition import classify_query

load_dotenv()

SYSTEM_PROMPT: str = """You are a Task Planning and Sub-Agent Orchestration Agent in an Enterprise Intelligent Knowledge Assistant.

Your responsibility is to understand the user's original query, use the classification received from the Query Classification Agent, and create an appropriate task allocation plan.

AVAILABLE SUB-AGENTS:

1. Search Agent:
   Retrieves relevant information from enterprise documents, databases, and knowledge sources.

2. Expert Finder Agent:
   Identifies employees, teams, and subject-matter experts based on their skills, experience, roles, and project involvement.

3. Summarization Agent:
   Summarizes documents, consolidates retrieved information, and generates concise summaries.

4. Compliance Agent:
   Evaluates information against company policies, compliance requirements, regulatory guidelines, and internal rules.

INSTRUCTIONS:

1. Understand the semantic intent and information requirements of the original user query.

2. Read the classification provided by the Query Classification Agent.

3. If classification is SIMPLE:
   - Preserve the original query.
   - Create one task assigned to the Search Agent.
   - Set execution_strategy to DIRECT.

4. If classification is COMPLEX:
   - Identify all distinct information requirements.
   - Break the original query into focused, independently actionable subtasks.
   - Assign each subtask to the most appropriate available sub-agent.
   - Specify a clear task description and expected output.
   - Identify dependencies between subtasks using task IDs.
   - Prefer parallel execution for independent tasks.
   - Use sequential execution when one task requires the output of another.
   - Use hybrid execution when the workflow contains both parallel and dependent tasks.

5. Do not create unnecessary subtasks or duplicate tasks.
6. Preserve the complete intent of the original query.
7. Do not attempt to answer the user's query yourself.
8. Do not assign tasks to agents outside the available sub-agent list.
9. Do not invent information or assume that a sub-agent has retrieved evidence.
10. Return the result strictly using the provided Pydantic TaskAllocation schema.

OUTPUT REQUIREMENTS:

Return:
- Original query
- Query classification
- Subtask IDs
- Task descriptions
- Assigned sub-agents
- Expected outputs
- Task dependencies
- Execution strategy

Your objective is to produce an accurate, minimal, and executable task allocation plan."""


class SubAgentType(str, Enum):
    """Permitted enterprise sub-agents."""
    SEARCH_AGENT = "Search Agent"
    EXPERT_FINDER = "Expert Finder Agent"
    SUMMARIZATION_AGENT = "Summarization Agent"
    COMPLIANCE_AGENT = "Compliance Agent"


class ExecutionStrategy(str, Enum):
    """Permitted execution strategies for sub-agent orchestration."""
    DIRECT = "DIRECT"
    PARALLEL = "PARALLEL"
    SEQUENTIAL = "SEQUENTIAL"
    HYBRID = "HYBRID"


class SubTask(BaseModel):
    """Pydantic model representing an individual orchestrated subtask."""
    task_id: str = Field(description="Unique identifier for the subtask (e.g., 'task_1', 'task_2').")
    description: str = Field(description="Clear, focused, and actionable task description.")
    assigned_agent: SubAgentType = Field(description="Assigned sub-agent responsible for executing this task.")
    expected_output: str = Field(description="Expected output or deliverables from this subtask.")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task can execute."
    )


class TaskAllocation(BaseModel):
    """Structured task allocation plan produced by the orchestration agent."""
    original_query: str = Field(description="The user's original query.")
    query_classification: Literal["SIMPLE", "COMPLEX"] = Field(
        description="Query classification category received from the Query Classification Agent."
    )
    tasks: List[SubTask] = Field(description="List of allocated subtasks.")
    execution_strategy: ExecutionStrategy = Field(
        description="Orchestration strategy: DIRECT, PARALLEL, SEQUENTIAL, or HYBRID."
    )



# LangGraph Workflow Definition


class OrchestrationState(TypedDict):
    """State definition for the LangGraph orchestration agent."""
    original_query: str
    query_classification: Optional[str]
    task_allocation: Optional[TaskAllocation]


def classify_node(state: OrchestrationState) -> Dict[str, Any]:
    """LangGraph node to verify or compute query classification."""
    classification = state.get("query_classification")
    if not classification or classification not in ("SIMPLE", "COMPLEX"):
        classification = classify_query(state["original_query"])
    return {"query_classification": classification}


def plan_node(state: OrchestrationState) -> Dict[str, Any]:
    """LangGraph node to generate the structured TaskAllocation plan."""
    query = state["original_query"]
    classification = state.get("query_classification", "SIMPLE")

    # 1. Simple query: exactly one task assigned to Search Agent with DIRECT strategy
    if classification == "SIMPLE":
        plan = TaskAllocation(
            original_query=query,
            query_classification="SIMPLE",
            tasks=[
                SubTask(
                    task_id="task_1",
                    description=f"Retrieve relevant documentation and knowledge sources answering: {query}",
                    assigned_agent=SubAgentType.SEARCH_AGENT,
                    expected_output="Direct factual answer and reference document excerpts.",
                    dependencies=[]
                )
            ],
            execution_strategy=ExecutionStrategy.DIRECT
        )
        return {"task_allocation": plan}

    # 2. Complex query: use LLM with structured output to break into subtasks
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
    structured_llm = llm.with_structured_output(TaskAllocation)

    prompt = (
        f"Original User Query: {query}\n"
        f"Query Classification: {classification}\n\n"
        "Generate a minimal, executable TaskAllocation plan splitting this complex query "
        "into distinct subtasks assigned to the available sub-agents."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        plan: TaskAllocation = structured_llm.invoke(messages)
        return {"task_allocation": plan}
    except Exception:
        # Fallback plan for complex queries if external call is unavailable
        subtasks: List[SubTask] = [
            SubTask(
                task_id="task_1",
                description=f"Search knowledge base and records for: {query}",
                assigned_agent=SubAgentType.SEARCH_AGENT,
                expected_output="Retrieved evidence and source documents.",
                dependencies=[]
            )
        ]

        normalized = query.lower()
        if any(w in normalized for w in ("who", "owner", "contact", "expert", "team")):
            subtasks.append(
                SubTask(
                    task_id="task_2",
                    description=f"Identify subject matter experts or owners related to: {query}",
                    assigned_agent=SubAgentType.EXPERT_FINDER,
                    expected_output="Relevant expert employee details and roles.",
                    dependencies=[]
                )
            )

        if any(w in normalized for w in ("policy", "compliance", "approved", "rule", "standard")):
            subtasks.append(
                SubTask(
                    task_id="task_3",
                    description=f"Check compliance policies and internal standards for: {query}",
                    assigned_agent=SubAgentType.COMPLIANCE_AGENT,
                    expected_output="Compliance verification status and guidelines.",
                    dependencies=["task_1"]
                )
            )

        subtasks.append(
            SubTask(
                task_id=f"task_{len(subtasks) + 1}",
                description="Synthesize and summarize findings from retrieved evidence into a consolidated answer.",
                assigned_agent=SubAgentType.SUMMARIZATION_AGENT,
                expected_output="Consolidated and concise summary resolving the complex query.",
                dependencies=[t.task_id for t in subtasks]
            )
        )

        has_deps = any(len(t.dependencies) > 0 for t in subtasks)
        has_indep = any(len(t.dependencies) == 0 for t in subtasks)
        strategy = (
            ExecutionStrategy.HYBRID
            if (has_deps and has_indep)
            else (ExecutionStrategy.SEQUENTIAL if has_deps else ExecutionStrategy.PARALLEL)
        )

        fallback_plan = TaskAllocation(
            original_query=query,
            query_classification="COMPLEX",
            tasks=subtasks,
            execution_strategy=strategy
        )
        return {"task_allocation": fallback_plan}


# Construct and compile the LangGraph workflow
graph_builder = StateGraph(OrchestrationState)
graph_builder.add_node("classify_query", classify_node)
graph_builder.add_node("plan_tasks", plan_node)

graph_builder.add_edge(START, "classify_query")
graph_builder.add_edge("classify_query", "plan_tasks")
graph_builder.add_edge("plan_tasks", END)

orchestration_graph = graph_builder.compile()


def orchestrate_query(
    query: str,
    classification: Optional[str] = None
) -> TaskAllocation:
    """Orchestrate query understanding and task allocation using LangGraph.

    Args:
        query: The user query string.
        classification: Optional pre-computed classification ('SIMPLE' or 'COMPLEX').

    Returns:
        TaskAllocation: Validated Pydantic model containing the complete task allocation plan.
    """
    initial_state: OrchestrationState = {
        "original_query": query.strip(),
        "query_classification": classification,
        "task_allocation": None
    }
    result = orchestration_graph.invoke(initial_state)
    return result["task_allocation"]
from enum import Enum
import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from retrieval.query_decomposition import classify_query

load_dotenv()

SYSTEM_PROMPT: str = """You are a Task Planning and Sub-Agent Orchestration Agent in an Enterprise Intelligent Knowledge Assistant.

Your responsibility is to understand the user's original query, use the classification received from the Query Classification Agent, and create an appropriate task allocation plan.

AVAILABLE SUB-AGENTS:

1. Search Agent:
   Retrieves relevant information from enterprise documents, databases, and knowledge sources.

2. Expert Finder Agent:
   Identifies employees, teams, and subject-matter experts based on their skills, experience, roles, and project involvement.

3. Summarization Agent:
   Summarizes documents, consolidates retrieved information, and generates concise summaries.

4. Compliance Agent:
   Evaluates information against company policies, compliance requirements, regulatory guidelines, and internal rules.

INSTRUCTIONS:

1. Understand the semantic intent and information requirements of the original user query.

2. Read the classification provided by the Query Classification Agent.

3. If classification is SIMPLE:
   - Preserve the original query.
   - Create one task assigned to the Search Agent.
   - Set execution_strategy to DIRECT.

4. If classification is COMPLEX:
   - Identify all distinct information requirements.
   - Break the original query into focused, independently actionable subtasks.
   - Assign each subtask to the most appropriate available sub-agent.
   - Specify a clear task description and expected output.
   - Identify dependencies between subtasks using task IDs.
   - Prefer parallel execution for independent tasks.
   - Use sequential execution when one task requires the output of another.
   - Use hybrid execution when the workflow contains both parallel and dependent tasks.

5. Do not create unnecessary subtasks or duplicate tasks.
6. Preserve the complete intent of the original query.
7. Do not attempt to answer the user's query yourself.
8. Do not assign tasks to agents outside the available sub-agent list.
9. Do not invent information or assume that a sub-agent has retrieved evidence.
10. Return the result strictly using the provided Pydantic TaskAllocation schema.

OUTPUT REQUIREMENTS:

Return:
- Original query
- Query classification
- Subtask IDs
- Task descriptions
- Assigned sub-agents
- Expected outputs
- Task dependencies
- Execution strategy

Your objective is to produce an accurate, minimal, and executable task allocation plan."""


class SubAgentType(str, Enum):
    """Permitted enterprise sub-agents."""
    SEARCH_AGENT = "Search Agent"
    EXPERT_FINDER = "Expert Finder Agent"
    SUMMARIZATION_AGENT = "Summarization Agent"
    COMPLIANCE_AGENT = "Compliance Agent"


class ExecutionStrategy(str, Enum):
    """Permitted execution strategies for sub-agent orchestration."""
    DIRECT = "DIRECT"
    PARALLEL = "PARALLEL"
    SEQUENTIAL = "SEQUENTIAL"
    HYBRID = "HYBRID"


class SubTask(BaseModel):
    """Pydantic model representing an individual orchestrated subtask."""
    task_id: str = Field(description="Unique identifier for the subtask (e.g., 'task_1', 'task_2').")
    description: str = Field(description="Clear, focused, and actionable task description.")
    assigned_agent: SubAgentType = Field(description="Assigned sub-agent responsible for executing this task.")
    expected_output: str = Field(description="Expected output or deliverables from this subtask.")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task can execute."
    )


class TaskAllocation(BaseModel):
    """Structured task allocation plan produced by the orchestration agent."""
    original_query: str = Field(description="The user's original query.")
    query_classification: Literal["SIMPLE", "COMPLEX"] = Field(
        description="Query classification category received from the Query Classification Agent."
    )
    tasks: List[SubTask] = Field(description="List of allocated subtasks.")
    execution_strategy: ExecutionStrategy = Field(
        description="Orchestration strategy: DIRECT, PARALLEL, SEQUENTIAL, or HYBRID."
    )


# ============================================================================
# LangGraph Workflow Definition
# ============================================================================

class OrchestrationState(TypedDict):
    """State definition for the LangGraph orchestration agent."""
    original_query: str
    query_classification: Optional[str]
    task_allocation: Optional[TaskAllocation]


def classify_node(state: OrchestrationState) -> Dict[str, Any]:
    """LangGraph node to verify or compute query classification."""
    classification = state.get("query_classification")
    if not classification or classification not in ("SIMPLE", "COMPLEX"):
        classification = classify_query(state["original_query"])
    return {"query_classification": classification}


def plan_node(state: OrchestrationState) -> Dict[str, Any]:
    """LangGraph node to generate the structured TaskAllocation plan."""
    query = state["original_query"]
    classification = state.get("query_classification", "SIMPLE")

    # 1. Simple query: exactly one task assigned to Search Agent with DIRECT strategy
    if classification == "SIMPLE":
        plan = TaskAllocation(
            original_query=query,
            query_classification="SIMPLE",
            tasks=[
                SubTask(
                    task_id="task_1",
                    description=f"Retrieve relevant documentation and knowledge sources answering: {query}",
                    assigned_agent=SubAgentType.SEARCH_AGENT,
                    expected_output="Direct factual answer and reference document excerpts.",
                    dependencies=[]
                )
            ],
            execution_strategy=ExecutionStrategy.DIRECT
        )
        return {"task_allocation": plan}

    # 2. Complex query: use LLM with structured output to break into subtasks
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
    structured_llm = llm.with_structured_output(TaskAllocation)

    prompt = (
        f"Original User Query: {query}\n"
        f"Query Classification: {classification}\n\n"
        "Generate a minimal, executable TaskAllocation plan splitting this complex query "
        "into distinct subtasks assigned to the available sub-agents."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        plan: TaskAllocation = structured_llm.invoke(messages)
        return {"task_allocation": plan}
    except Exception:
        # Fallback plan for complex queries if external call is unavailable
        subtasks: List[SubTask] = [
            SubTask(
                task_id="task_1",
                description=f"Search knowledge base and records for: {query}",
                assigned_agent=SubAgentType.SEARCH_AGENT,
                expected_output="Retrieved evidence and source documents.",
                dependencies=[]
            )
        ]

        normalized = query.lower()
        if any(w in normalized for w in ("who", "owner", "contact", "expert", "team")):
            subtasks.append(
                SubTask(
                    task_id="task_2",
                    description=f"Identify subject matter experts or owners related to: {query}",
                    assigned_agent=SubAgentType.EXPERT_FINDER,
                    expected_output="Relevant expert employee details and roles.",
                    dependencies=[]
                )
            )

        if any(w in normalized for w in ("policy", "compliance", "approved", "rule", "standard")):
            subtasks.append(
                SubTask(
                    task_id="task_3",
                    description=f"Check compliance policies and internal standards for: {query}",
                    assigned_agent=SubAgentType.COMPLIANCE_AGENT,
                    expected_output="Compliance verification status and guidelines.",
                    dependencies=["task_1"]
                )
            )

        subtasks.append(
            SubTask(
                task_id=f"task_{len(subtasks) + 1}",
                description="Synthesize and summarize findings from retrieved evidence into a consolidated answer.",
                assigned_agent=SubAgentType.SUMMARIZATION_AGENT,
                expected_output="Consolidated and concise summary resolving the complex query.",
                dependencies=[t.task_id for t in subtasks]
            )
        )

        has_deps = any(len(t.dependencies) > 0 for t in subtasks)
        has_indep = any(len(t.dependencies) == 0 for t in subtasks)
        strategy = (
            ExecutionStrategy.HYBRID
            if (has_deps and has_indep)
            else (ExecutionStrategy.SEQUENTIAL if has_deps else ExecutionStrategy.PARALLEL)
        )

        fallback_plan = TaskAllocation(
            original_query=query,
            query_classification="COMPLEX",
            tasks=subtasks,
            execution_strategy=strategy
        )
        return {"task_allocation": fallback_plan}


# Construct and compile the LangGraph workflow
graph_builder = StateGraph(OrchestrationState)
graph_builder.add_node("classify_query", classify_node)
graph_builder.add_node("plan_tasks", plan_node)

graph_builder.add_edge(START, "classify_query")
graph_builder.add_edge("classify_query", "plan_tasks")
graph_builder.add_edge("plan_tasks", END)

orchestration_graph = graph_builder.compile()


def orchestrate_query(
    query: str,
    classification: Optional[str] = None
) -> TaskAllocation:
    """Orchestrate query understanding and task allocation using LangGraph.

    Args:
        query: The user query string.
        classification: Optional pre-computed classification ('SIMPLE' or 'COMPLEX').

    Returns:
        TaskAllocation: Validated Pydantic model containing the complete task allocation plan.
    """
    initial_state: OrchestrationState = {
        "original_query": query.strip(),
        "query_classification": classification,
        "task_allocation": None
    }
    result = orchestration_graph.invoke(initial_state)
    return result["task_allocation"]
