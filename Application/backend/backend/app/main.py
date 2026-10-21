from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .services.generation import (
    ExtractiveFallbackProvider,
    GenerationContext,
    GenerationProvider,
    configured_provider,
    generate_with_fallback,
)
from .services.rag import Chunk, CrossEncoderReranker, RetrievalBackend, SentenceTransformerEncoder


ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / "data" / "raw"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Versatile RAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
retriever = RetrievalBackend()
documents: dict[str, dict] = {}
document_hashes: dict[str, str] = {}
request_metrics: dict[str, list[float]] = {"bm25": [], "dense": [], "hybrid": [], "rerank": []}
generation_init_error: str | None = None
try:
    generation_provider: GenerationProvider = configured_provider()
except Exception as exc:
    generation_init_error = str(exc)
    generation_provider = ExtractiveFallbackProvider()
_dense_load_error: str | None = None
_reranker_load_error: str | None = None
MAX_UPLOAD_BYTES = int(os.getenv("VERSATILE_RAG_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

HISTORICAL_PERFORMANCE = [
    {"method": "BM25", "doc_recall_at_10": 0.9600, "section_recall_at_10": 0.8767, "section_mrr_at_10": 0.6902, "mean_latency_ms": 128.88},
    {"method": "Dense", "doc_recall_at_10": 0.9967, "section_recall_at_10": 0.9200, "section_mrr_at_10": 0.6818, "mean_latency_ms": 34.62},
    {"method": "Hybrid RRF", "doc_recall_at_10": 1.0000, "section_recall_at_10": 0.9433, "section_mrr_at_10": 0.7272, "mean_latency_ms": 164.24},
    {"method": "Hybrid + reranker", "doc_recall_at_10": 1.0000, "section_recall_at_10": 0.9400, "section_mrr_at_10": 0.7646, "mean_latency_ms": 1419.97},
]


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    method: str = Field(default="hybrid", pattern="^(bm25|dense|hybrid|rerank)$")
    limit: int = Field(default=5, ge=1, le=20)


@app.get("/api/health")
def health():
    return {"status": "ok", "documents": len(documents), "chunks": len(retriever.chunks)}


@app.get("/api/documents")
def list_documents():
    return list(documents.values())


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF uploads are supported")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
    content_hash = hashlib.sha256(content).hexdigest()
    duplicate = next((item for item, digest in document_hashes.items() if digest == content_hash), None)
    if duplicate:
        raise HTTPException(409, f"This PDF is already indexed as {documents[duplicate]['filename']}")
    document_id = uuid.uuid4().hex
    target = UPLOADS / f"{document_id}.pdf"
    target.write_bytes(content)
    try:
        reader = PdfReader(str(target))
        chunks: list[Chunk] = []
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            words = text.split()
            for offset in range(0, len(words), 180):
                body = " ".join(words[offset : offset + 180])
                if body:
                    chunks.append(
                        Chunk(
                            id=f"{document_id}-{page_number}-{offset}",
                            document_id=document_id,
                            filename=file.filename or target.name,
                            page=page_number,
                            text=body,
                        )
                    )
        if not chunks:
            target.unlink(missing_ok=True)
            raise HTTPException(422, "PDF contains no extractable text")
    except HTTPException:
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, f"PDF processing failed: {exc}") from exc
    retriever.add_chunks(chunks)
    record = {
        "id": document_id,
        "filename": file.filename or target.name,
        "pages": len(reader.pages),
        "chunks": len(chunks),
        "size_bytes": len(content),
    }
    documents[document_id] = record
    document_hashes[document_id] = content_hash
    return record


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    record = documents.pop(document_id, None)
    if record is None:
        raise HTTPException(404, "Document not found")
    document_hashes.pop(document_id, None)
    removed = retriever.delete_document(document_id)
    (UPLOADS / f"{document_id}.pdf").unlink(missing_ok=True)
    return {"deleted": document_id, "chunks_removed": removed}


@app.post("/api/ask")
def ask(request: AskRequest):
    global _dense_load_error, _reranker_load_error
    if request.method != "bm25" and retriever.encoder is None:
        try:
            retriever.encoder = SentenceTransformerEncoder("BAAI/bge-small-en-v1.5")
        except Exception as exc:
            _dense_load_error = str(exc)
            raise HTTPException(503, f"Dense retrieval is unavailable: {_dense_load_error}") from exc
    if request.method == "rerank" and retriever.reranker is None:
        try:
            retriever.reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            _reranker_load_error = str(exc)
            raise HTTPException(503, f"Reranking is unavailable: {_reranker_load_error}") from exc
    try:
        results, latency_ms = retriever.search(request.question, request.method, request.limit)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not results:
        raise HTTPException(404, "No indexed evidence is available")
    request_metrics[request.method].append(latency_ms)
    contexts = [
        GenerationContext(
            text=item["text"],
            filename=item["filename"],
            page=item["page"],
            rank=item["rank"],
        )
        for item in results
    ]
    generated = generate_with_fallback(
        generation_provider,
        request.question,
        contexts,
        error=generation_init_error,
    )
    return {
        "answer": generated.answer,
        "answer_mode": generated.provider,
        "generation_status": generated.status,
        "generation_model": generated.model,
        "generation_latency_ms": generated.generation_latency_ms,
        "generation_initialization_ms": generated.initialization_ms,
        "generation_error": generated.error,
        "sources": results,
        "latency_ms": latency_ms,
        "method": request.method,
    }


@app.get("/api/capabilities")
def capabilities():
    return {
        "generation": {
            "available": generation_provider.name != "extractive_context_fallback",
            "mode": generation_provider.name,
            "model": generation_provider.model_name,
            "initialization_ms": generation_provider.initialization_ms,
            "initialization_error": generation_init_error,
        },
        "dense": {"available": retriever.encoder is not None, "model": "BAAI/bge-small-en-v1.5"},
        "reranker": {"available": retriever.reranker is not None, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        "note": "Dense and reranker models load lazily on first use and remain unavailable until successfully loaded.",
    }


@app.get("/api/performance")
def performance():
    current = []
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid RRF", "rerank": "Hybrid + reranker"}
    for key, values in request_metrics.items():
        current.append(
            {
                "method": labels[key],
                "requests": len(values),
                "mean_latency_ms": round(sum(values) / len(values), 2) if values else None,
                "last_latency_ms": round(values[-1], 2) if values else None,
            }
        )
    return {"historical_phase5_2": HISTORICAL_PERFORMANCE, "current_session": current}
