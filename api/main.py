from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from ragforge import pipeline, vectorstore

app = FastAPI(title="RAGForge Generator", version="2.0.0")

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ragforge"}

@app.post("/api/v1/collections/{collection_id}/documents")
async def upload_document(collection_id: str, file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".pdf", ".docx", ".csv"}:
        raise HTTPException(400, "Unsupported file type")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "File exceeds 20 MB limit")
    source_dir = Path(".ragforge_sources") / collection_id
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.txt").name
    (source_dir / safe_name).write_bytes(data)
    # Rebuild the collection from the complete runtime document set. This is
    # important for fitted embedding backends such as TF-IDF: every document
    # must share one vocabulary/model state.
    result = pipeline.ingest(str(source_dir), collection_id, replace=True)
    return {**result, "filename": safe_name}

@app.get("/api/v1/collections")
def collections():
    return {"collections": vectorstore.list_collections()}

@app.get("/api/v1/collections/{collection_id}")
def collection(collection_id: str):
    try:
        c = vectorstore.get_collection(collection_id)
        return {"collection": collection_id, "chunks": c.count()}
    except ValueError as exc:
        raise HTTPException(404, str(exc))

@app.post("/api/v1/collections/{collection_id}/query")
def query(collection_id: str, request: QueryRequest):
    try:
        return pipeline.ask(collection_id, request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

from fastapi.responses import FileResponse

@app.get("/")
def ui():
    return FileResponse(Path("static/index.html"))
