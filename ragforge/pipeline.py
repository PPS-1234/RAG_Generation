"""Core runtime RAG orchestration."""
from __future__ import annotations
import os
from ragforge import vectorstore
from ragforge.agent import QueryAgent
from ragforge.chunker import chunk_documents
from ragforge.embeddings import get_embedder
from ragforge.generator import generate_answer
from ragforge.grounding import validate_citations
from ragforge.loaders import load_directory
from ragforge.reranker import rerank

AGENT = QueryAgent()


def ingest(docs_dir: str, collection_name: str, replace: bool = True) -> dict:
    raw_docs = load_directory(docs_dir)
    if not raw_docs:
        raise ValueError(f"No supported documents found in {docs_dir}")
    chunks = chunk_documents(raw_docs)
    if not chunks:
        raise ValueError("Documents were loaded but produced zero chunks - check file contents.")
    embedder = get_embedder()
    embedder.fit([c.text for c in chunks])
    embeddings = embedder.embed_documents([c.text for c in chunks])
    if replace:
        collection = vectorstore.reset_collection(collection_name)
    else:
        try:
            collection = vectorstore.get_collection(collection_name)
        except ValueError:
            collection = vectorstore.create_collection(collection_name)
    vectorstore.add_chunks(collection, chunks, embeddings)
    store_path = os.path.join(vectorstore.STORE_DIR, collection_name)
    embedder.save(store_path)
    return {
        "collection": collection_name,
        "documents_loaded": len(raw_docs),
        "chunks_indexed": len(chunks),
        "embedder": embedder.name,
        "embedder_dim": embedder.dim,
    }


def ask(collection_name: str, question: str, top_k: int = 5, max_distance: float = 0.97) -> dict:
    collection = vectorstore.get_collection(collection_name)
    plan = AGENT.plan(question, top_k)
    embedder = get_embedder()
    store_path = os.path.join(vectorstore.STORE_DIR, collection_name)
    embedder.load(store_path)
    query_embedding = embedder.embed_query(plan.query)
    # Retrieve a larger candidate set before reranking.
    candidates = vectorstore.query(collection, query_embedding, top_k=min(plan.top_k * 3, 20))
    if not candidates:
        return _refusal(collection_name, plan, "No indexed evidence was found.")
    if candidates[0]["distance"] >= max_distance:
        return _refusal(collection_name, plan, "The retrieved evidence was below the relevance threshold.")
    hits = rerank(plan.query, candidates, plan.top_k)
    answer, provider = generate_answer(plan.query, hits)
    valid, validation = validate_citations(answer, len(hits))
    if not valid and provider != "extractive-fallback":
        answer = _grounded_fallback(hits)
        provider = "extractive-fallback"
        validation = "generator citation validation failed; returned retrieved evidence instead"
    return {
        "question": question,
        "answer": answer,
        "provider": provider,
        "agent": {"strategy": plan.strategy, "rationale": plan.rationale, "rewritten_query": plan.query},
        "grounding": {"validated": True, "validation": validation, "threshold": max_distance},
        "sources": [
            {"source": h["source"], "distance": round(h["distance"], 4), "rerank_score": h.get("rerank_score"), "excerpt": h["text"][:500]}
            for h in hits
        ],
    }


def _grounded_fallback(hits: list[dict]) -> str:
    return "Relevant evidence from the uploaded documents:\n\n" + "\n\n".join(
        f"[S{i}] {h['text']}" for i, h in enumerate(hits[:3], 1)
    )


def _refusal(collection_name: str, plan, reason: str) -> dict:
    return {
        "question": plan.query,
        "answer": f"I don't have relevant information to answer this question in the '{collection_name}' collection.",
        "provider": "groundedness-gate",
        "agent": {"strategy": plan.strategy, "rationale": plan.rationale, "rewritten_query": plan.query},
        "grounding": {"validated": True, "validation": reason},
        "sources": [],
    }
