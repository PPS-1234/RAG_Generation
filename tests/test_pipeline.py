"""
Run with: pytest tests/ -v

Uses temporary collections (prefixed test_) so these never collide with your
real document sets, and cleans up after itself.
"""
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragforge import pipeline, vectorstore

SAMPLE_A = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "set_a_kubernetes")
SAMPLE_B = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "set_b_coffee")


@pytest.fixture(scope="session", autouse=True)
def clean_store():
    # Chroma's PersistentClient caches open connections by path for the life of
    # the process, so wiping the store dir between individual tests (while a
    # cached connection still points at the old files) corrupts that cache and
    # produces "attempt to write a readonly database". Clean up once, at the
    # end of the whole test session, instead - test_* collections are isolated
    # from each other by name either way.
    yield
    if os.path.exists(vectorstore.STORE_DIR):
        shutil.rmtree(vectorstore.STORE_DIR)


def test_ingest_returns_nonzero_chunks():
    result = pipeline.ingest(SAMPLE_A, "test_k8s")
    assert result["documents_loaded"] >= 1
    assert result["chunks_indexed"] >= 1


def test_ingest_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        pipeline.ingest("./this_directory_does_not_exist", "test_bad")


def test_ask_unindexed_collection_raises():
    with pytest.raises(ValueError):
        pipeline.ask("test_never_ingested", "anything")


def test_on_topic_question_is_grounded_and_cites_source():
    pipeline.ingest(SAMPLE_A, "test_k8s")
    result = pipeline.ask("test_k8s", "What is the difference between a liveness and readiness probe?")
    assert result["provider"] != "groundedness-gate"
    assert len(result["sources"]) > 0
    assert "kubernetes_basics" in result["sources"][0]["source"]


def test_off_topic_question_is_refused_not_hallucinated():
    """The core 'grounded answers' requirement: asking a Kubernetes collection
    about coffee must NOT return a fabricated answer."""
    pipeline.ingest(SAMPLE_A, "test_k8s")
    result = pipeline.ask("test_k8s", "What grind size should I use for pour-over coffee?")
    assert result["provider"] == "groundedness-gate"
    assert result["sources"] == []


def test_two_document_sets_stay_isolated():
    """The core 'different document sets without code changes' requirement."""
    pipeline.ingest(SAMPLE_A, "test_k8s")
    pipeline.ingest(SAMPLE_B, "test_coffee")

    k8s_result = pipeline.ask("test_k8s", "What temperature should brewing water be?")
    coffee_result = pipeline.ask("test_coffee", "What temperature should brewing water be?")

    assert k8s_result["provider"] == "groundedness-gate"
    assert coffee_result["provider"] != "groundedness-gate"
    assert "coffee_brewing" in coffee_result["sources"][0]["source"]


def test_reingesting_same_collection_replaces_not_duplicates():
    pipeline.ingest(SAMPLE_A, "test_k8s")
    first = pipeline.ingest(SAMPLE_A, "test_k8s")
    second = pipeline.ingest(SAMPLE_A, "test_k8s")
    assert first["chunks_indexed"] == second["chunks_indexed"]
