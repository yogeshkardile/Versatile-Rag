# Versatile RAG — Completion Report

Last verified: 2026-10-17

## Current status

Phase 5.2 retrieval evaluation is preserved in:

`..\Experiments\phase5_2`

The validated 300-document result used 300 seeded, category/type-stratified queries on 25,618 text-only chunks. Historical Phase 5 and Phase 5.1 outputs were not modified.

| Method | Doc R@10 | Section R@10 | Section MRR@10 | Mean latency |
|---|---:|---:|---:|---:|
| BM25 | 0.9600 | 0.8767 | 0.6902 | 128.88 ms |
| Dense | 0.9967 | 0.9200 | 0.6818 | 34.62 ms |
| Hybrid RRF | 1.0000 | 0.9433 | 0.7272 | 164.24 ms |
| Hybrid + reranker | 1.0000 | 0.9400 | 0.7646 | 1,419.97 ms |

These metrics apply only to the fixed 300-document sampled corpus and queries. No answer-generation, citation-correctness, table/figure retrieval, held-out-domain, or full-corpus claim is supported.

## Application checkout

Created because no configured application project or existing checkout was found:

`C:\Users\HP\.copilot\chats\2026-10-15\cautious-succotash-d8457527\Versatile-RAG`

Created/changed:

- `backend/app/main.py`: FastAPI upload, extraction, indexing, retrieval, deletion, and ask endpoints.
- `backend/app/services/rag.py`: common BM25/dense/hybrid/rerank retrieval backend.
- `backend/requirements.txt`: runtime dependencies.
- `frontend/src/main.tsx`: real upload, question, method selection, source display, and deletion actions.
- `frontend/src/styles.css`: requested palette and responsive layout.
- `frontend` Vite/React/TypeScript configuration.
- `README.md`, `.gitignore`, `.env.example`, and this report.
- `backend/app/services/generation.py`: replaceable generation-provider boundary with an explicit extractive fallback.
- `tests/test_retrieval.py`: model-independent retrieval tests for BM25, dense, hybrid RRF, and rerank behavior.
- `tests/test_generation.py`: generation context, mocked provider, unavailable provider, and fallback tests.
- `frontend/src/vite-env.d.ts`: Vite TypeScript environment declarations.

## Actual validation

Completed:

- Phase 5.2 artifact presence and validation-log verification.
- Python syntax validation for the application backend (`backend_compile=PASS`).
- Frontend dependency installation and production build (`vite build` passed; 22 packages audited, 0 vulnerabilities).
- Live FastAPI integration test using a one-page scientific PDF:
  - PDF upload returned one page and one indexed chunk.
  - BM25 question returned the expected evidence with page `1`.
  - Response declared `answer_mode=extractive_context_fallback`.
  - Actual request latency was returned by the API.
  - Document deletion removed the indexed chunk.
  - Invalid non-PDF upload returned HTTP `400`.
- Real lazy dense and cross-encoder adapters were wired; unavailable model access returns an explicit HTTP `503`.
- Upload size limit, duplicate-content detection, empty-PDF rejection, and SHA-256 content tracking were added.
- `/api/capabilities` reports generation/model availability without triggering model downloads.
- `/api/performance` separates historical Phase 5.2 metrics from current-session request measurements.
- The frontend now includes deletion confirmation, model-unavailable messaging, loading/error/empty states, source page references, and historical/current performance comparison.
- Retrieval unit tests passed: 5/5.
- Generation and retrieval unit tests passed: 9/9.
- Frontend TypeScript check passed: `npm.cmd run typecheck`.
- Frontend production build passed: `npm.cmd run build`.
- Runtime inspection recorded Python 3.12.2, Windows 11, 16.47 GB RAM, PyTorch `2.6.0+cpu`, and CUDA unavailable.
- Cached live-model smoke test passed without downloading new files:
  - `BAAI/bge-small-en-v1.5`: initialization 187.42 ms; warm encode 79.07 ms; 384-dimensional embeddings; valid top result and cosine score 0.8330.
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`: initialization 1,041.00 ms; warm pair scoring 28.88 ms; real scores 9.2056 and -11.1304 on two passages.
  - `google/flan-t5-base`: initialization 465.57 ms; warm generation 1,811.94 ms; returned `recall and evidence latency`.
- Live FastAPI smoke test with `google/flan-t5-base` completed on one tiny scientific PDF:
  - provider initialization in the API process: 10,642.51 ms;
  - BM25, dense, hybrid RRF, and hybrid plus reranking each returned the valid document ID, page `1`, text, rank, and numeric score;
  - generation status was `generated` for all four methods;
  - warm generation latencies were 808.11 ms, 510.36 ms, 525.14 ms, and 555.58 ms respectively;
  - dense retrieval latency was 137.36 ms, hybrid 40.02 ms, and reranking 88.87 ms for this one-chunk fixture;
  - deletion removed the indexed chunk after testing.
- Unavailable-model behavior was verified with a non-cached model ID: `/api/capabilities` reported `available: false`, retained `extractive_context_fallback`, and returned the actionable Hugging Face offline/cache error.

Pending:

- The application defaults to `extractive_context_fallback`; the live test enabled `VERSATILE_RAG_GENERATION_PROVIDER=local_transformers` and used the already cached `google/flan-t5-base`.
- The live test was CPU-only and used one tiny fixture, so it is a capability smoke test, not an answer-quality evaluation.
- No paid or remote provider was introduced. No API key was used.
- Frontend browser automation was not run; API-level end-to-end behavior and production compilation were verified.

## Stage 3 bounded answer and citation evaluation

Stage 3 was run as a separate evaluation and did not modify Phase 5, 5.1, or 5.2 artifacts. The governing protocol is:

`..\Experiments\protocols\stage3_answer_evaluation_protocol.md`

Artifacts are under:

`..\Experiments\stage3`

The frozen manifest contains 30 query IDs selected as the first 30 entries of the immutable Phase 5.2 manifest, using the inherited seed `20261017`. The run evaluated all four historical methods, for 120 answer records total. Phase 5.2 per-query rankings were reused and resolved against the pinned dataset snapshot; retrieval indexing and the original benchmark were not rerun.

| Method | Generated | Non-empty | Source metadata valid | Evidence-token overlap* | Reference-token recall* | Mean generation |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 30/30 | 30/30 | 30/30 | 0.8486 | 0.1635 | 2,798.18 ms |
| Dense | 30/30 | 30/30 | 30/30 | 0.8471 | 0.1597 | 3,192.77 ms |
| Hybrid RRF | 30/30 | 30/30 | 30/30 | 0.8409 | 0.1863 | 3,350.66 ms |
| Hybrid + reranker | 30/30 | 30/30 | 30/30 | 0.9357 | 0.1967 | 3,263.77 ms |

`*` These are descriptive lexical diagnostics only. They are not semantic answer-quality scores, citation-faithfulness scores, or evidence of statistical superiority.

The live provider was cached `google/flan-t5-base` with local-files-only loading on CPU. Initialization was 11,238.11 ms and total generation wall time was 378,594.39 ms. All 120 records completed with `generation_status=generated`; no fallback or provider failure occurred. Structured source records had valid document, section, chunk, and rank metadata for all records. The dataset exposes section IDs rather than PDF page numbers, so page-level citation correctness was not claimed.

Semantic relevance, faithfulness, context utilization, and claim-level citation support remain explicitly unscored and are queued in `citation_review.csv` for human annotation. The pinned sample does not contain defensible unanswerable or conflicting-evidence labels, so abstention evaluation was not reported as benchmark evidence. The released reference answers are used only for the descriptive token-recall diagnostic; they are not treated as complete semantic ground truth.

Exact Stage 3 command:

```powershell
cd C:\Users\HP\.copilot\chats\2026-10-15\cautious-succotash-d8457527\Versatile-RAG
$env:PYTHONPATH="backend"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python ..\Experiments\stage3\run_stage3.py
```

This Stage 3 result is a bounded live-generation evaluation layered on historical retrieval rankings, not a new retrieval benchmark or full-corpus answer-quality claim.

## Launch commands

Backend:

```powershell
cd C:\Users\HP\.copilot\chats\2026-10-15\cautious-succotash-d8457527\Versatile-RAG
python -m uvicorn app.main:app --app-dir backend --reload
```

Frontend:

```powershell
cd C:\Users\HP\.copilot\chats\2026-10-15\cautious-succotash-d8457527\Versatile-RAG\frontend
npm.cmd install
npm.cmd run dev
```

## Exact verification commands

Backend syntax:

```powershell
python -m compileall -q backend/app tests
```

Model-independent retrieval tests:

```powershell
$env:PYTHONPATH="backend"
python -m unittest discover -s tests -v
```

Live cached-model smoke test:

```powershell
$env:VERSATILE_RAG_GENERATION_PROVIDER="local_transformers"
$env:VERSATILE_RAG_GENERATION_MODEL="google/flan-t5-base"
python -m uvicorn app.main:app --app-dir backend --port 8000
```

The live smoke test used only cached model files and was stopped after the bounded fixture run. The default launch remains fallback-only unless the generation environment variables are set.

Frontend checks:

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

Bounded API verification passed with:

- valid scientific PDF upload;
- BM25 retrieval with expected evidence on page 1;
- extractive fallback labeling;
- duplicate upload rejection: HTTP 409;
- empty PDF rejection: HTTP 422;
- invalid MIME rejection: HTTP 400;
- short-query validation: HTTP 422;
- document deletion removing one indexed chunk;
- empty document list after deletion;
- capability endpoint reporting dense/reranker unavailable without model download.

## Next action

For a normal local launch, leave `VERSATILE_RAG_GENERATION_PROVIDER=fallback`. To enable the cached FLAN-T5 provider, set the documented environment variables first. The next research action is manual review of the Stage 3 citation queue, if semantic claim-support labels are required; no full-corpus answer-generation benchmark should be started from this bounded result.
