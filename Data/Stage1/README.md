# Stage 1 Frozen Artifacts

`artifacts/` is created on demand by the active Multi-Agent runtime.

Each file is addressed by the exact Stage 1 experiment identity:

```text
Data/Stage1/artifacts/<case_id>/<input_fingerprint>.json
```

The fingerprint binds the Stage 0 EvidencePack, prompt-facing forecast payload,
Stage 1 prompts and output schema, pinned Qwen3.8 model artifact, llama.cpp
runtime profile, reasoning/MTP/sampling settings, and seed.

An exact match is reused without another LLM call. Changed inputs or runtime
settings create a new fingerprint. Existing artifacts are never overwritten;
artifact/output hashes are validated on load.

## Main case set

`Multi-Agent/Stage1/cases.py` defines the frozen seven-case main evaluation set.
The full 303 Threat-PMT relation table remains the quantitative forecast/gap
population; expensive Agent critique is restricted to the predeclared seven-case
set. Selection is completed before Stage 1 generation and may use only Stage 0
forecast/evidence metadata, never Agent outputs.

The five regime representatives must satisfy at least 5/6 evidence-slot coverage
and at least four source families under the fixed ex-ante cutoff. The two anchors
are intentionally exempt so evidence-poor behavior is measured rather than hidden
by a coverage-only sampling rule. The exact versioned definition is materialized
under:

```text
Data/Stage1/case_sets/stage1-main-cases-v1.json
```

## Resume checkpoints

Before the paired final artifact exists, each successfully validated Critic is
checkpointed immediately at:

```text
Data/Stage1/checkpoints/<case_id>/<input_fingerprint>/<critic_type>.json
```

The checkpoint is bound to the same exact Stage 1 input fingerprint and carries
its own hash plus an assessment hash. It is not a compatibility fallback. If any
input/prompt/model/runtime/schema/seed component changes, the path changes and the
old checkpoint is ignored. After both critics are validated and the final immutable
artifact is frozen, redundant partial checkpoints are removed.

## Progress and live logs

`py -3 -m Pipeline.main` runs all seven main cases. It writes line-buffered live
logs under `Data/Stage1/logs/` and atomically updates the batch state under
`Data/Stage1/runs/` after every case. Qwen generation logs periodic llama.cpp slot
progress (elapsed time and available token counters) and final usage metadata; it
does not dump private reasoning text.

## Runtime calibration traces

`Data/Stage1/runtime_bench/` contains the pre-experiment hardware/runtime calibration traces used to freeze the main Stage 1 inference profile. Calibration is performed before main-case generation and is not adapted per case. The current procedure verifies 128K full-GPU fit, measures a non-speculative baseline, screens bounded MTP candidates, rejects candidates that fail a deterministic output-fingerprint check against the baseline, and rechecks the selected profile on a main-like long-prompt structured-output workload. The selected runtime configuration is then copied into `Stage1/runtime.py` / `Stage1/start_qwen38.ps1` and becomes part of the Stage 1 experiment fingerprint.

The current frozen profile is 131,072 context, all model layers resident on GPU, Q8 target/draft KV, batch/ubatch 2048/512, MTP draft depth 4 with `p_min=0.05`, xhigh reasoning, and one inference slot.
