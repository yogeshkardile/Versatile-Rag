# Research checkpoint: Phase 5.2

The validated retrieval checkpoint compares four text-only systems on a deterministic 300-document subset and 300 sampled queries from `vectara/open_ragbench`, revision `63f6b052ff83508b08e242db42263ee708815c26`.

| System | Document R@10 | Section R@10 | Section MRR@10 | Mean warm latency |
|---|---:|---:|---:|---:|
| E1 BM25 | 0.9600 | 0.8767 | 0.6902 | 128.88 ms |
| E2 BGE-small | 0.9967 | 0.9200 | 0.6818 | 34.62 ms |
| E3 Hybrid RRF | 1.0000 | 0.9433 | 0.7272 | 164.24 ms |
| E4 Hybrid + reranker | 1.0000 | 0.9400 | 0.7646 | 1,419.97 ms |

The run validated 3,600 per-query rows, ranking order, aggregate recomputation, latency statistics, and paired uncertainty analysis. E4 improved section MRR but added substantial latency. Results are exploratory evidence for the sampled corpus and query set, not full-corpus or held-out-domain generalization.

## Limitations and pending work

- The experiment indexed extracted text only; tables, figures, and multimodal evidence were excluded.
- The released dataset has no official held-out split.
- Answer generation, citation correctness, and end-to-end QA quality are not validated.
- The 1,000-document benchmark was not run because measured CPU costs made it impractical in this environment.

## Immutable source artifacts

- `..\..\Experiments\phase5_2\phase5_2_completion_report.md`
- `...\frozen_manifest.json`
- `...\per_query_results.csv`
- `...\aggregate_metrics_recomputed.csv`
- `...\latency_statistics_recomputed.csv`
- `...\paired_statistics.csv`
- `...\run_metadata.json`
- `...\validation.log`
