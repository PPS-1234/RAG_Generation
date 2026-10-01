"""
Chunking: word-count based with overlap, splitting on paragraph boundaries where
possible so we don't cut mid-sentence unnecessarily.

Kept deliberately simple and dependency-free rather than reaching for
langchain's text splitters - for an assessment, being able to explain every
line of your chunking logic beats importing a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ragforge.loaders import RawDoc


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)  # rough word->token heuristic


def chunk_document(doc: RawDoc, min_tokens: int = 60, max_tokens: int = 250,
                    overlap_tokens: int = 30) -> list[Chunk]:
    paragraphs = [p.strip() for p in doc.text.split("\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush():
        if current:
            chunks.append(Chunk(text="\n".join(current), source=doc.source, metadata=dict(doc.metadata)))

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_tokens >= min_tokens:
            flush()
            overlap_words = " ".join(current).split()[-int(overlap_tokens / 1.3):]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = _estimate_tokens(" ".join(current))

        current.append(para)
        current_tokens += para_tokens

    flush()
    return chunks


def chunk_documents(docs: list[RawDoc], **kwargs) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks
