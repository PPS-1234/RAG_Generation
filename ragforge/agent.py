from dataclasses import dataclass

@dataclass(frozen=True)
class QueryPlan:
    query: str
    top_k: int
    strategy: str
    rationale: str

class QueryAgent:
    """Deterministic query-planning agent.

    It selects a retrieval strategy from the query shape while keeping the
    planning interface replaceable by an LLM/tool-calling planner later.
    """
    def plan(self, question: str, top_k: int = 5) -> QueryPlan:
        q = " ".join(question.split()).strip()
        if not q:
            raise ValueError("Question cannot be empty.")
        low = q.lower()
        if any(x in low for x in ("compare", "difference", "versus", " vs ")):
            strategy = "comparison"
            rationale = "Comparison intent detected; retrieve a broader evidence set."
            k = max(top_k, 6)
        elif any(x in low for x in ("how", "why", "steps", "process")):
            strategy = "procedural"
            rationale = "Procedural intent detected; preserve multiple supporting chunks."
            k = max(top_k, 5)
        else:
            strategy = "semantic"
            rationale = "General information request; use standard semantic retrieval."
            k = top_k
        return QueryPlan(q, min(k, 10), strategy, rationale)
