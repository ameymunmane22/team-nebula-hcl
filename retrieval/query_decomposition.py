#Query decommposition agent , use pydantic for structred classification to classify query as simple or complex query and return the classification as a string. The agent should be able to understand the query
from enum import Enum
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

SYSTEM_PROMPT: str = """You are a Query Classification Agent in an Enterprise Intelligent Knowledge Assistant.

Your responsibility is to understand incoming user queries and classify them into exactly one of two categories: SIMPLE or COMPLEX.

Classification Rules:

Classify a query as SIMPLE when:

It asks for a single, specific piece of information.

It requires straightforward information retrieval.

It can generally be answered using one relevant document or source.

It does not require comparisons, multi-step reasoning, or combining multiple pieces of evidence.

Classify a query as COMPLEX when:

It contains multiple information requirements or sub-questions.

It requires comparing information across documents, departments, policies, or entities.

It requires multi-step reasoning or combining evidence from multiple sources.

It involves dependencies between different questions.

It requires summarizing, aggregating, or synthesizing information from multiple sources to produce an answer.

Decision Guidelines:

Understand the semantic intent of the entire query before classifying it.

Evaluate the number of information requirements, reasoning steps, dependencies, and retrieval requirements.

Do not classify a query as complex merely because it is long or contains technical terminology.

Do not classify a query as simple merely because it is short.

When a query contains multiple independent or dependent information requirements, classify it as COMPLEX.

When the query is ambiguous but requires clarification rather than multi-step retrieval, do not automatically classify it as COMPLEX.

Output Requirements:

Use the provided Pydantic structured output schema.

Return exactly one classification: SIMPLE or COMPLEX.

Do not provide explanations, reasoning, additional fields, or conversational responses.

Never generate information that is not part of the classification task."""


class QueryCategory(str, Enum):
    """Permitted categories for query classification."""
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class QueryClassification(BaseModel):
    """Pydantic schema for structured query classification output."""
    classification: QueryCategory = Field(
        description="The classification category of the query: strictly 'SIMPLE' or 'COMPLEX'."
    )


def classify_query(
    query: str,
    llm: Optional[ChatGroq] = None,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """Classify an incoming query as either 'SIMPLE' or 'COMPLEX' using structured LLM output.

    Args:
        query: The user query string to understand and classify.
        llm: Optional pre-configured ChatGroq instance.
        model_name: Groq model name to use if llm is not provided.

    Returns:
        Classification result as a string: 'SIMPLE' or 'COMPLEX'.
    """
    if not query or not query.strip():
        return QueryCategory.SIMPLE.value

    client = llm or ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

    structured_llm = client.with_structured_output(QueryClassification)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query.strip())
    ]

    try:
        result: QueryClassification = structured_llm.invoke(messages)
        return result.classification.value
    except Exception:
        # Heuristic fallback if LLM call is unavailable
        complex_indicators = (" and ", " vs ", "compare", "difference", "also", "steps to", "how do both")
        normalized = query.lower()
        if any(indicator in normalized for indicator in complex_indicators):
            return QueryCategory.COMPLEX.value
        return QueryCategory.SIMPLE.value
