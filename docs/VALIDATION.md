# Release validation

Validated on 2026-11-21 with Ubuntu 24.04, Python 3.11.18, PyTorch
2.9.1+cu128, NVIDIA RTX 3090 hardware, and Node.js 20.18.1. A separate Python
environment was created for this copy; the source server's environment, datasets,
and model caches were not copied into it.

## Completed checks

- The documented `setup.sh --torch cu128 --dev` workflow installed dependencies
  and built the frontend. `pip check` reported no broken requirements; doctor
  detected CUDA and all four visible GPUs.
- Backend regression suite with Hub access disabled: **128 passed, 1 skipped**.
  The skipped upstream integration test requires a configured `.data` raw path;
  Studio's automatic path was exercised separately below.
- Frontend suite: **116 passed in 16 files**. The TypeScript/production build
  completed with Vite 6.4.6 and Vitest 4.1.12. `npm audit` reported **0 known
  vulnerabilities** at validation time. A large-bundle advisory remains a
  non-fatal build-size warning.
- Real HTTP service: built page, health, all 14 available templates, graph
  validation, YAML compilation verified by the real loader, graph save/reload.
  A browser connected through SSH forwarding loaded the component registry,
  template graph, and GPU environment details.
- With `.data` absent and no copied source data/cache, Studio downloaded real
  MIND-small and GloVe resources, prepared the NAML Demo, and completed CPU
  training and evaluation. The CPU run emitted GAUC 0.5654, MRR 0.1946,
  NDCG@5 0.2096, and NDCG@10 0.2671. These values demonstrate execution, not a
  full-dataset benchmark claim.
- A second NAML Demo run completed on CUDA using the prepared resources, with
  no download steps. It emitted GAUC 0.5629, MRR 0.1955, NDCG@5 0.2101, and
  NDCG@10 0.2676.
- Tokenization was also checked with NLTK restricted to the private project
  cache, so a user-home tokenization cache was not required.
- A temporary optional `.data` file pointing at the downloaded raw MIND files
  was read successfully and reused those files. The release root has no `.data`.
- Service stop/restart was exercised using this checkout's manager. Tests cover
  an absent manager, PID reuse, and missing frontend preflight.
- Source/history credential-pattern and historical blob-size checks found no
  matching private-key/AWS/GitHub/Hugging Face token patterns and no blobs at or
  above 100 MiB in the upstream history. This is a bounded pattern scan, not a
  claim that every possible secret format has been detected.

## Issues found and corrected during validation

The previous MIND entry required authorization and its old Azure fallback
returned HTTP 409. Acquisition now follows Recommenders' current public MIND
archive. GloVe uses Stanford NLP's published Hub archive first, with the Stanford
server as fallback. A cold export also revealed duplicate UniTok GloVe vocabulary
registration; export and tokenizer loading now reuse the canonical vocabulary,
with row-order checks and regression tests. Missing NLTK tokenization data is
prepared in a private project cache.

Vite/Vitest and their transitive dependencies were updated after advisories were
reported for the inherited development toolchain. The Vite configuration was
adjusted for the new test configuration types. Resource-state tests were isolated
from pre-existing embedding files so they also run in an empty checkout.

## Scope and remaining platform coverage

The real CPU training used the CUDA-capable PyTorch wheel on a CPU device; a
separate CPU-only wheel installation was not performed on this machine. The CPU
installation workflow is configured in CI, but GitHub Actions has not run because
this copy has not been pushed. Native Windows/macOS are unsupported; WSL2 and
other GPU/driver combinations were not exercised. Live downloads for every
RecBench/xMIND variant and gated models were not revalidated. Their availability
and access remain provider dependent.
