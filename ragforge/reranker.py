import re

STOPWORDS = {"the","a","an","is","are","was","were","what","why","how","does","do","of","to","in","on","for","and","or","with","from","this","that","it"}

def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t not in STOPWORDS}

def rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """Small lexical tie-breaker over vector candidates.

    It never introduces a document that the vector search did not retrieve.
    The score is diagnostic and combines normalized vector similarity with
    query-token overlap.
    """
    q = _tokens(question)
    scored = []
    for hit in hits:
        overlap = len(q & _tokens(hit["text"])) / max(len(q), 1)
        vector_score = max(0.0, 1.0 - float(hit.get("distance", 1.0)))
        score = 0.75 * vector_score + 0.25 * overlap
        item = dict(hit)
        item["rerank_score"] = round(score, 6)
        scored.append(item)
    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
