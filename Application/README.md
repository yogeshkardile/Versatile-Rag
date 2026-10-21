# Versatile RAG

This folder contains the runnable private-PDF application. It supports
page-aware extraction, chunk indexing, BM25 retrieval, lazy dense retrieval,
hybrid RRF retrieval, cross-encoder reranking, source display, deletion,
duplicate detection, measured request latency, and a configurable generation
provider. The default generation provider is an explicitly labeled extractive
fallback.

## Run the backend

From `Versatile-RAG\Application`:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

## Run the frontend

From the project root:

```powershell
cd Application\frontend
npm.cmd install
npm.cmd run dev
```

The backend intentionally returns an extractive context fallback until a local generator adapter is configured. It does not claim answer-generation quality or citation correctness.

## Verify

```powershell
python -m compileall -q backend/app tests
$env:PYTHONPATH="backend"
python -m unittest discover -s tests -v
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

`GET /api/performance` separates current-session timings from the immutable Phase 5.2 research comparison. Dense and reranker models load only when selected and return an explicit unavailable error if model initialization fails.
