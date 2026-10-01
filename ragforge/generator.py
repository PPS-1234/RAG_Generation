"""LLM generation with grounded citation instructions and safe fallback."""
from __future__ import annotations
import os

SYSTEM = """You are a grounded RAG assistant. Use ONLY the supplied CONTEXT. Do not use outside knowledge. Every factual claim must be supported by one or more context items and cited as [S1], [S2], etc. If the context is insufficient, say so. Never invent citations."""


def generate_answer(question: str, hits: list[dict]) -> tuple[str, str]:
    if not hits:
        return "I don't have enough evidence in the uploaded documents to answer that.", "groundedness-gate"
    context = "\n\n".join(f"[S{i}] {h['text']}" for i, h in enumerate(hits, 1))
    groq = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if groq:
        try:
            from groq import Groq
            client = Groq(api_key=groq)
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0,
                messages=[{"role":"system","content":SYSTEM}, {"role":"user","content":f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"}],
            )
            return response.choices[0].message.content or "", "groq"
        except Exception:
            pass
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[{"role":"system","content":SYSTEM}, {"role":"user","content":f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"}],
            )
            return response.choices[0].message.content or "", "openai"
        except Exception:
            pass
    return _extractive(hits), "extractive-fallback"


def _extractive(hits: list[dict]) -> str:
    return "Relevant evidence from the uploaded documents:\n\n" + "\n\n".join(
        f"[S{i}] {h['text']}" for i, h in enumerate(hits[:3], 1)
    )
