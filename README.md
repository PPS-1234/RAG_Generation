# RAGForge — strengthened runtime RAG generator

RAGForge is an assessment-ready RAG application that accepts document sets at runtime, creates isolated queryable collections, retrieves evidence, applies a relevance gate and reranking, and returns grounded answers with structured sources. It supports CLI and FastAPI usage.

## Assessment fit

The candidate brief requires runtime document ingestion, a RAG application over those documents, grounded Q&A, and support for different document sets without code changes. RAGForge addresses those requirements through collection-based ingestion and query APIs/CLI.

## Architecture

```text
                       CLI / Browser / API Client
                                  |
                                  v
                              FastAPI
                                  |
                         +--------+--------+
                         |                 |
                    Ingestion          Query Agent
                         |                 |
                    load/parse        plan strategy
                         |                 |
                       chunk               |
                         |                 v
                      embed          candidate retrieval
                         |                 |
                         v              rerank
                    Chroma                |
                         |                 v
                         +----------> grounding gate
                                           |
                                      +----+----+
                                      |         |
                                    reject    generate
                                      |         |
                                      |    citation validator
                                      |         |
                                      +----+----+
                                           |
                                      answer+sources
```

### Why this is stronger than a basic RAG demo

- **Runtime collections:** new document sets are selected by collection name/path; the orchestration code is unchanged.
- **Pluggable retrieval:** TF-IDF is the deterministic offline backend; Sentence Transformers can be enabled without changing the pipeline.
- **Agentic query planning:** a query agent selects a retrieval strategy (semantic/comparison/procedural) and controls candidate depth. The interface can later be backed by an LLM tool-calling planner without changing downstream contracts.
- **Candidate reranking:** lexical overlap is used as a transparent tie-breaker over vector candidates; it does not introduce documents outside the retrieved candidate set.
- **Grounding gate:** irrelevant collections are refused before generation when the best candidate is below the configured relevance threshold.
- **Citation validation:** generated citations are checked against retrieved evidence; if validation fails, the system falls back to transparent extractive evidence rather than returning an unvalidated answer.
- **Collection isolation:** every document set has its own Chroma collection, with tests proving cross-set isolation.
- **CLI + REST API + browser UI:** the same core pipeline is exposed through multiple interfaces.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt

python main.py ingest --docs ./sample_docs/set_a_kubernetes --collection kubernetes
python main.py ask --collection kubernetes -q "What does a Deployment do?"

uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for Swagger. The browser UI can be added to the FastAPI root in a future cosmetic pass; the REST endpoints are the primary interface.

## Semantic embeddings

For the most realistic semantic retrieval path:

```bash
pip install sentence-transformers torch
EMBEDDER=sentence-transformers python main.py ingest --docs ./sample_docs/set_a_kubernetes --collection kubernetes_semantic
```

The offline TF-IDF backend remains useful for deterministic evaluation and environments where model downloads are unavailable. This is an explicit engineering trade-off, not a claim that TF-IDF is semantically equivalent to transformer embeddings.

## LLM generation

Set either `GROQ_API_KEY` or `OPENAI_API_KEY`. If neither is present, the application returns retrieved evidence in an extractive fallback mode. The same retrieval and grounding pipeline is used in all cases.

## API

`POST /api/v1/collections/{collection_id}/documents` — multipart file upload.

`POST /api/v1/collections/{collection_id}/query`

```json
{"question":"What does a Deployment do?","top_k":5}
```

The response includes `answer`, `sources`, `agent`, and `grounding` metadata.

## Tests

```bash
pytest tests/ -v
```

The suite covers ingestion, errors, grounded answers, off-topic refusal, cross-collection isolation and re-ingestion. Extend it with PDF/DOCX/CSV fixtures before final submission if the evaluator expects those formats explicitly.

## Production hardening

For production: authentication/authorization, tenant ACLs, object storage, asynchronous ingestion, OCR for scanned PDFs, content hashing for incremental updates, hybrid BM25+dense retrieval, a learned reranker, embedding/model versioning, evaluation/observability, rate limits, secret management, malware scanning and retention/deletion controls should be added.

## Submission requirement: AI transcript

The candidate brief explicitly asks for the complete AI-agent coding transcript. Add the actual exported Codex/Claude Code transcript to `submission/` before submission. Never fabricate this artifact and never commit secrets.
