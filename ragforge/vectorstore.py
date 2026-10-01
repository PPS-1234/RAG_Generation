"""
Vector storage via Chroma, running embedded/persistent (no server process,
no Docker) - the whole DB is a folder on disk. We pass our OWN precomputed
embeddings (from embeddings.py) rather than letting Chroma download its own
default embedding model, which is what keeps this runnable with zero internet
access on the TF-IDF path.

Collections are the mechanism for "different document sets without code
changes": each document set gets its own named collection. Ingesting a new
folder never touches the code - it just creates (or replaces) a collection.
"""
from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings

from ragforge.chunker import Chunk

STORE_DIR = ".ragforge_store"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=STORE_DIR, settings=Settings(anonymized_telemetry=False)
    )


def reset_collection(collection_name: str):
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def create_collection(collection_name: str):
    client = get_client()
    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def get_collection(collection_name: str):
    client = get_client()
    try:
        return client.get_collection(collection_name)
    except Exception:
        raise ValueError(
            f"No such collection: '{collection_name}'. "
            f"Run `ingest --docs <dir> --collection {collection_name}` first."
        )


def add_chunks(collection, chunks: list[Chunk], embeddings: list[list[float]]):
    if not chunks:
        return
    collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, **c.metadata} for c in chunks],
    )


def query(collection, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({"text": text, "source": meta.get("source"), "metadata": meta, "distance": dist})
    return hits


def list_collections() -> list[str]:
    client = get_client()
    return [c.name for c in client.list_collections()]
