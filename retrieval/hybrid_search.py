#implement hybrid search that is vector search with keyword search BM25 for retrival
from collections import defaultdict
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi

# Set offline mode for local cached embeddings to avoid network round-trips
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ingestion.ingestion import hf_embeddings, knowledge_collection

load_dotenv()


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25 indexing."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ============================================================================
# Hybrid Search Engine
# ============================================================================

class HybridSearchEngine:
    """Hybrid Search Engine combining dense vector similarity and sparse BM25 keyword retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge and rerank candidate documents.
    """

    def __init__(self, collection: Optional[Any] = None) -> None:
        self.collection = collection or knowledge_collection
        self.corpus_ids: List[str] = []
        self.corpus_docs: List[str] = []
        self.corpus_metadatas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Fetch all documents from ChromaDB and construct the BM25Okapi index."""
        try:
            if self.collection.count() > 0:
                data = self.collection.get()
                self.corpus_ids = data.get("ids", [])
                self.corpus_docs = data.get("documents", [])
                self.corpus_metadatas = data.get("metadatas", [])

                tokenized_corpus = [tokenize(doc) for doc in self.corpus_docs]
                if tokenized_corpus:
                    self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception as e:
            print(f"Warning: Failed to initialize BM25 index from ChromaDB: {e}")

    def refresh_index(self) -> None:
        """Re-sync the BM25 index with the latest state of the ChromaDB collection."""
        self._build_bm25_index()

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform dense semantic search using Hugging Face embeddings and ChromaDB.

        Args:
            query: User search query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of ranked document dictionaries with similarity metrics.
        """
        results: List[Dict[str, Any]] = []
        if self.collection.count() == 0:
            return results

        try:
            query_vector = hf_embeddings.embed_query(query)
            n_results = min(top_k, self.collection.count())
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )

            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0] if "distances" in raw else [0.0] * len(ids)

            for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
                # Convert cosine distance to approximate similarity score
                similarity = round(1.0 - float(dist), 4) if dist is not None else 1.0
                results.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": meta,
                    "vector_score": similarity,
                    "vector_rank": rank + 1
                })
        except Exception as e:
            print(f"Vector search warning: {e}")

        return results

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform sparse keyword search using BM25Okapi.

        Args:
            query: User search query.
            top_k: Number of top scoring documents to return.

        Returns:
            List of ranked document dictionaries with BM25 scores.
        """
        results: List[Dict[str, Any]] = []
        tokens = tokenize(query)
        if not self.bm25 or not tokens or not self.corpus_docs:
            return results

        try:
            scores = self.bm25.get_scores(tokens)
            # Filter non-zero scores and sort descending
            scored_indices = [
                (idx, float(score)) for idx, score in enumerate(scores) if score > 0.0
            ]
            scored_indices.sort(key=lambda x: x[1], reverse=True)

            for rank, (idx, score) in enumerate(scored_indices[:top_k]):
                results.append({
                    "id": self.corpus_ids[idx],
                    "content": self.corpus_docs[idx],
                    "metadata": self.corpus_metadatas[idx] if idx < len(self.corpus_metadatas) else {},
                    "bm25_score": round(score, 4),
                    "bm25_rank": rank + 1
                })
        except Exception as e:
            print(f"BM25 search warning: {e}")

        return results

    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 5,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Combine dense and sparse search results using Reciprocal Rank Fusion (RRF).

        Formula: RRF_Score(d) = sum( weight_m / (rrf_k + rank_m(d)) )

        Args:
            vector_results: Ranked results from semantic vector search.
            bm25_results: Ranked results from BM25 keyword search.
            top_k: Number of fused results to return.
            rrf_k: Smoothing constant (standard default is 60).
            vector_weight: Relative weight for vector ranking.
            bm25_weight: Relative weight for BM25 ranking.

        Returns:
            Consolidated, deduplicated, and reranked list of documents.
        """
        doc_store: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        # 1. Accumulate vector search RRF scores
        for item in vector_results:
            doc_id = item["id"]
            rank = item["vector_rank"]
            rrf_scores[doc_id] += vector_weight / (rrf_k + rank)
            doc_store[doc_id] = {
                "id": doc_id,
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_rank": rank,
                "vector_score": item.get("vector_score", 0.0),
                "bm25_rank": None,
                "bm25_score": 0.0
            }

        # 2. Accumulate BM25 search RRF scores
        for item in bm25_results:
            doc_id = item["id"]
            rank = item["bm25_rank"]
            rrf_scores[doc_id] += bm25_weight / (rrf_k + rank)
            if doc_id in doc_store:
                doc_store[doc_id]["bm25_rank"] = rank
                doc_store[doc_id]["bm25_score"] = item.get("bm25_score", 0.0)
            else:
                doc_store[doc_id] = {
                    "id": doc_id,
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "vector_rank": None,
                    "vector_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": item.get("bm25_score", 0.0)
                }

        # 3. Sort by total RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

        fused_results: List[Dict[str, Any]] = []
        for did in sorted_ids[:top_k]:
            entry = doc_store[did]
            entry["hybrid_score"] = round(rrf_scores[did], 6)
            fused_results.append(entry)

        return fused_results


# Default global instance of HybridSearchEngine
hybrid_engine = HybridSearchEngine()


# ============================================================================
# LangGraph Workflow Definition
# ============================================================================

class HybridSearchState(TypedDict):
    """State schema for the LangGraph Hybrid Search pipeline."""
    query: str
    top_k: int
    vector_results: List[Dict[str, Any]]
    bm25_results: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]


def vector_search_step(state: HybridSearchState) -> Dict[str, Any]:
    """LangGraph node for dense vector retrieval."""
    results = hybrid_engine.vector_search(
        query=state["query"],
        top_k=state.get("top_k", 5) * 2
    )
    return {"vector_results": results}


def bm25_search_step(state: HybridSearchState) -> Dict[str, Any]:
    """LangGraph node for sparse BM25 keyword retrieval."""
    results = hybrid_engine.bm25_search(
        query=state["query"],
        top_k=state.get("top_k", 5) * 2
    )
    return {"bm25_results": results}


def fusion_step(state: HybridSearchState) -> Dict[str, Any]:
    """LangGraph node combining vector and BM25 rankings via Reciprocal Rank Fusion."""
    fused = hybrid_engine.reciprocal_rank_fusion(
        vector_results=state.get("vector_results", []),
        bm25_results=state.get("bm25_results", []),
        top_k=state.get("top_k", 5)
    )
    return {"final_results": fused}


# Compile LangGraph graph
builder = StateGraph(HybridSearchState)
builder.add_node("vector_search", vector_search_step)
builder.add_node("bm25_search", bm25_search_step)
builder.add_node("reciprocal_rank_fusion", fusion_step)

builder.add_edge(START, "vector_search")
builder.add_edge(START, "bm25_search")
builder.add_edge("vector_search", "reciprocal_rank_fusion")
builder.add_edge("bm25_search", "reciprocal_rank_fusion")
builder.add_edge("reciprocal_rank_fusion", END)

hybrid_search_graph = builder.compile()


# ============================================================================
# Public API Function
# ============================================================================

def hybrid_search(
    query: str,
    top_k: int = 5,
    rrf_k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """Execute hybrid retrieval combining dense vector similarity with BM25 keyword search.

    Args:
        query: Search query string.
        top_k: Number of fused results to return.
        rrf_k: Constant for Reciprocal Rank Fusion formula (default 60).
        vector_weight: Weight for semantic vector matches.
        bm25_weight: Weight for BM25 exact keyword matches.

    Returns:
        List of reranked document records with hybrid scores and provenance metadata.
    """
    if not query or not query.strip():
        return []

    initial_state: HybridSearchState = {
        "query": query.strip(),
        "top_k": top_k,
        "vector_results": [],
        "bm25_results": [],
        "final_results": []
    }

    result = hybrid_search_graph.invoke(initial_state)
    return result["final_results"]
