# Multi-Agent Cybersecurity Analysis System

> The implemented runtime is now separated by responsibility. `Stage0/` owns canonical forecast/evidence preparation and deterministic retrieval, `Stage1/` owns Qwen inference, prompts, critic nodes and frozen Stage 1 artifacts, and `Pipeline/` contains only cross-stage orchestration. `Lagacy/` is archival and is never imported by the active pipeline. Neither implemented stage trains or reruns B-MTGNN.

`load_data_node` no longer reads B-MTGNN forecast TXT files directly. Agent runs consume an existing Stage 0 forecast dataset and frozen evidence snapshot. Configure the case with `STAGE0_SNAPSHOT_ID`, `STAGE0_THREAT`, `STAGE0_PMT`, and `STAGE0_ANALYSIS_CUTOFF`; optional `STAGE0_EVALUATION_MODE` and `STAGE0_CASE_ID` control case execution. `forecast_origin_date` is fixed by the canonical forecast manifest and cannot be overridden at runtime.

This repository contains the legacy multi-agent collaborative framework plus the redesigned Stage 0 data/evidence boundary and Stage 1 independent forecast critics. The active graph currently stops after Stage 1; legacy Stage 2+ code remains present but is not wired into the active graph until those contracts are redesigned.

### Stage 1 LLM runtime

The active Stage 1 runtime uses the local `Qwen3.8-27B-Q6_K_L` GGUF through a pinned llama.cpp CUDA server rather than Ollama. On an artifact miss, start the server with `Stage1/start_qwen38.ps1`; then run the implemented pipeline from the `Multi-Agent/` directory with `python -m Pipeline.main`. An exact frozen-artifact hit does not require the LLM server because no model call is made.

The main-experiment thinking profile is fixed to the Qwen3.8 recommended sampling values: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and `repeat_penalty=1.0`, with reasoning enabled at `xhigh` effort and preserved thinking. The llama.cpp server enables the model's embedded MTP path with `--spec-type draft-mtp`.

The server intentionally uses `--parallel 1`. Stage 1 remains logically parallel and independent in LangGraph, but the two HTTP generations are queued at the inference server so current CUDA/MTP concurrency issues cannot contaminate experiment output. This does not expose either Critic's assessment to the other.

The pinned main-run artifact is `Qwen3.8-27B-Q6_K_L.gguf`, size `24,958,908,128` bytes, SHA-256 `e8750bb81ba49f90eb68df99776b250dbc14666043098952de6422cbecd77a21`, from `bartowski/Qwen3.8-27B-GGUF` revision `e4cc3ff3b37f5aabf253d8583b0b0247e8b25b70`. The runtime is llama.cpp build `b10919`, commit `d3146f2b5`. The calibrated Stage 1 main profile reserves a **131,072-token context window**, keeps **all model layers on GPU**, uses Q8 target/draft KV caches, `batch=2048`, `ubatch=512`, MTP draft depth `4` with `p_min=0.05`, Flash Attention, and one inference slot.

### Pre-experiment runtime calibration

Inference acceleration is selected **before** main-case generation and then frozen as part of the experiment identity. Calibration is not allowed to adapt per Threat–PMT case. The procedure is: (1) verify the target 128K context and KV precision fit with full GPU residency, (2) measure a non-speculative baseline, (3) screen bounded MTP draft-depth / acceptance-threshold candidates under the same model, hardware, prompt, sampling profile and seed, (4) run a deterministic greedy probe against the non-speculative baseline as a correctness fingerprint gate, and (5) promote only a candidate that passes the correctness gate and then survives a main-like long-prompt recheck. A faster candidate is rejected if its deterministic output fingerprint differs from the non-speculative baseline.

On the current RTX 5000 Ada / llama.cpp build, the 128K full-GPU profile fit successfully. Screening measured approximately `20.37 tok/s` with MTP disabled, `29.09` at depth 1, `31.41` at depth 2, `31.18` at depth 3, and `32.52–32.55` at depth 4. In the deterministic correctness probe, the selected `depth=4, p_min=0.05` profile matched the non-speculative baseline fingerprint while tested depth-2/depth-3 candidates did not. A main-like Stage 1 request with a 15,747-token prompt, JSON-schema constraint and xhigh reasoning measured `38.47 tok/s` over a 2,048-token capped generation. These numbers are calibration measurements for this exact hardware/build, not universal model-speed claims. Calibration traces are stored under `Data/Stage1/runtime_bench/`.

### Frozen Stage 1 artifacts

Main-experiment Stage 1 output is generated once per exact experiment identity and frozen under `Data/Stage1/artifacts/<case_id>/<input_fingerprint>.json`. The fingerprint binds the Stage 0 `EvidencePack`, prompt-facing forecast payload, both Critic prompts, structured-output schema, model artifact/revision/hash, llama.cpp build, sampling/reasoning/MTP configuration, context profile, and seed. If the exact fingerprint already exists, both Critic LLM calls are skipped and the frozen pre-assessments are reused. A changed input, prompt, model, inference setting, schema, or seed creates a new fingerprint; the runtime never falls back to or overwrites another Stage 1 artifact.

Each frozen artifact carries its own SHA-256 plus a separate Stage 1 output SHA-256. Hash mismatch, unknown evidence references, wrong critic type, or an attempt to write different outputs under an already-frozen fingerprint is a hard error. Repeated-run robustness experiments therefore use different predeclared seeds/fingerprints and do not mutate the canonical seed-42 main artifact.

Structured output uses llama.cpp JSON-schema constrained generation followed by deterministic Pydantic semantic validation. If an otherwise structured response violates a semantic invariant that JSON Schema cannot express (for example, the same evidence ID appearing in both support and contradiction lists for one claim), the runtime performs exactly one fixed repair turn that may only correct the reported validation error and may not add facts or evidence IDs. A second failure aborts the run. The repair instruction hash and retry count are part of the Stage 1 experiment fingerprint.

### Single-inference review protocol

Each Critic makes its initial judgment in **one xhigh inference**, not a chain of separate self-refinement calls. Multiple sequential calls would introduce self-conditioning on earlier model-generated text, make the result depend on an additional call-count/order hyperparameter, and partially duplicate the role of the later Stage 2 inter-Critic debate. A single content-addressed `Forecast + EvidencePack + prompt -> CriticAssessment` mapping is therefore easier to reproduce and keeps Stage 1's initial judgment genuinely pre-debate.

Reasoning depth is encouraged with one fixed internal review order inside that single inference. Both critics must internally review: **(1) evidence boundary/sufficiency, (2) forecast-supporting evidence, (3) forecast-challenging evidence, (4) alternative explanations, (5) temporal consistency, (6) unsupported-specificity and claim-traceability audit, and (7) residual uncertainty followed by the final judgment.** The role-specific prompt changes the substance of these checks for Threat vs PMT, but not their order. Intermediate reasoning is not treated as evidence or stored as a research result; only the final structured `CriticAssessment` is frozen.

### Frozen seven-case main evaluation set

The main Stage 1 experiment does **not** run language-model critique over all 303 Threat-PMT relations. The 303 relations remain the quantitative forecast/gap population; Stage 1 performs detailed critique on a frozen seven-case set defined in `Stage1/cases.py` before any main Stage 1 outputs are generated.

Case selection uses Stage 0 information only. Stage 1 stance, confidence, disagreement, or downstream policy outputs are never selection variables. This prevents post-hoc case selection based on favorable Agent results.

Two cases are predeclared anchors and are retained regardless of evidence coverage. This is deliberate: dropping a difficult low-coverage case would bias evaluation toward evidence-rich examples and would avoid testing the `PARTIAL` / `INSUFFICIENT_EVIDENCE` path. Five additional cases cover distinct gap/slope regimes. A regime representative must have at least **5 of 6 semantic evidence slots** covered and at least **4 source families** under the fixed `2024-12-31` ex-ante cutoff. Once the v1 set is frozen, runtime code never substitutes another relation because of a later result.

| Selection role | Threat | PMT | Reason |
| --- | --- | --- | --- |
| Anchor | DDoS | NLP/LLM | Low-coverage stress case retained to test evidence-insufficiency handling instead of filtering it away |
| Anchor | Malware | Anomaly Detection | Evidence-rich stable reference case for comparison with the low-coverage anchor |
| Large gap + steep slope | Ransomware | Cryptography | Escalating high-imbalance regime |
| Large gap + moderate slope | Insider Threat | Cryptography | Persistent large imbalance without equally steep acceleration |
| Moderate gap + steep slope | Brute Force Attack | IDS/IPS | Emerging escalation before the gap is already extreme |
| Moderate gap + low slope | Targeted Attack | Access Control | Managed/stable mid-gap trajectory rather than another escalation-heavy case |
| Small gap + low/negative slope | Session Hijacking | HTTPS | Low-risk alignment control |

The frozen 303-relation thresholds are: mean gap `Q25=-0.499008`, `Q50=-0.098995`, `Q75=0.516488`; slope `Q25=-0.112731`, `Q50=0.059498`, `Q75=0.224801`. Regimes are fixed as: large+steep = `gap>=Q75, slope>=Q75`; large+moderate = `gap>=Q75, slope Q25..Q75`; moderate+steep = `gap Q25..Q75, slope>=Q75`; moderate+low = `gap Q25..Q75, slope Q25..Q50`; small+low/negative = `gap<=Q25, slope<Q50`. The exact pair identities, thresholds, coverage rule, and per-case rationale are versioned by `stage1-main-cases-v1`. Changing the main set requires a new case-set version rather than overwriting this definition.

## Background and Related Work

Cybersecurity strategy development often relies on siloed expert analysis, which may overlook complex interdependencies. Multi-agent systems enable collaborative decision-making, while Stage 0 supplies case-specific evidence from a frozen, provenance-tracked corpus before Agent reasoning begins.

The active runtime does not use the legacy mutable LightRAG store as an additional factual evidence channel. The Evidence Store is physically stored under `../Data/Evidence/snapshots/<snapshot_id>/`; the current frozen snapshot is `../Data/Evidence/snapshots/stage0-2026-09-12-v2/`, containing `records.jsonl`, `manifest.json`, and copied raw `artifacts/`. The fixed source registry is `../Data/Evidence/source_registry.json`.

Stage 0 retrieval is **temporally bounded deterministic BM25 evidence retrieval**, not a vector-similarity LightRAG call. It first filters frozen records by Threat/PMT tags and `available_at <= analysis_cutoff_date`, then maps candidates to six semantic evidence slots through source/claim contracts. Within each slot, a fixed non-LLM lexical query is generated from the Threat name, PMT name, forecast gap direction, and slot vocabulary; candidates are ranked with Okapi BM25 (`k1=1.2`, `b=0.75`). Source diversity is applied before filling the remaining positions, and at most four records are selected per slot. Both Stage 1 critics receive the same resulting `EvidencePack`.

For reproducibility, `retrieval_metadata` records the retriever version, exact query text per slot, BM25 parameters, candidate counts, and per-record BM25 scores. Query generation is deterministic and does not depend on Agent persona or model sampling.

The full retrieval audit metadata is retained in the frozen `EvidencePack` but is not injected wholesale into the LLM prompt. The Agent-facing payload receives the selected evidence records plus a compact retrieval summary (cutoff status, evidence sufficiency, slot coverage, source-family/chain counts, and violation counts). BM25 candidate score tables and query audit traces are provenance/audit data rather than factual evidence; excluding them from the prompt prevents audit bookkeeping from consuming the reasoning context while preserving exact reproducibility in the frozen artifact.

For comparison, the archived legacy implementation in `Lagacy/rag.py` used mutable `LightRAG` with Ollama embedding/completion functions and `QueryParam(mode="hybrid")`. Each Agent issued a different natural-language retrieval query during its own turn, and `add_conversation_data()` inserted generated Agent analyses back into the same RAG storage. That behavior is intentionally excluded from the redesigned runtime because it mixes generated reasoning with external evidence and makes Agent-to-Agent evidence coverage differ.

## System Overview

The Multi-Agent system is built on LangGraph. Stage 0 constructs one case-specific `EvidencePack` from the canonical 124-node forecast dataset and an immutable evidence snapshot. Stage 1 sends that same payload in parallel to an Attack Feasibility Critic and a Defense Robustness Critic, producing two independent structured pre-assessments.

### Key Features
- **Independent Forecast Critique**: Threat and PMT forecast components are evaluated independently before any debate
- **Frozen Evidence Input**: Case-specific retrieval is completed in Stage 0 with provenance and temporal-cutoff metadata
- **Structured Traceability**: Stage 1 returns stance, confidence, evidence sufficiency, claim/evidence links, unresolved questions, and unsupported-specificity flags
- **No Premature Anchoring**: Both critics see the same Forecast + EvidencePack and cannot see each other's initial output

## Methodology

### System Architecture

#### Core Components

1. **Pipeline Entry (`Pipeline/main.py`)**: Responsible for system initialization/execution and cross-stage logging/configuration.

2. **Pipeline Graph/State (`Pipeline/graph.py`, `Pipeline/state.py`)**: Defines the active Stage 0 → Stage 1 preparation → parallel critic workflow and the shared state contract. Stage 2+ is intentionally not active yet.

3. **Stage 0 (`Stage0/`)**: Canonical 124-node migration, immutable Evidence Store, source registry/ingestion, deterministic retrieval, Agent-input construction, and the Stage 0 load node.

4. **Stage 1 (`Stage1/`)**: Implements all active Stage 1-specific code:
   - **Attack Feasibility Critic**: Evaluates the Threat forecast against adversary capability and observed threat evidence
   - **Defense Robustness Critic**: Evaluates the PMT forecast against maturity, deployability, applicability, and implementation evidence
   - `Stage1/runtime.py`: pinned Qwen3.8/llama.cpp client and runtime verification
   - `Stage1/prompts.py`: only the implemented Stage 1 critic prompts
   - `Stage1/nodes.py`: artifact preparation, critic execution, validation and freeze
   - `Stage1/artifact.py`: immutable content-addressed Stage 1 artifact contract

5. **Legacy Archive (`Lagacy/`)**: Historical inputs/logs plus the retired LightRAG and transition-era node/prompt code. It is not part of the active import graph.

### Active Operation Process

#### Phase 1: Data Loading and Initialization
Load the canonical forecast dataset and frozen Stage 0 evidence snapshot, then configure environment settings.

#### Phase 2: Independent Forecast Critique
The runtime first checks for an exact frozen Stage 1 artifact. On a hit, its validated pre-assessments are reused with no model call. On a miss, the Attack Feasibility Critic and Defense Robustness Critic receive the same Stage 0 payload independently, each returns a validated `CriticAssessment`, and the joined pair is frozen before the active runtime ends. Debate and later lenses are implemented in subsequent stages, not silently delegated to the legacy chain.

## Data Flow

### Input Data
- **Prediction Data**: Cyber threat and mitigation technology trend predictions
- **Stage 0 Evidence Snapshot**: Frozen external cyber evidence with source/version/hash/cutoff metadata
- **Environment Configuration**: llama.cpp endpoint, fixed Qwen model ID, Stage 0 case settings, etc.

### Active Output Results
- **Log Files**: Records of Stage 1 execution (`../Data/Stage1/logs/`)
- **Stage 1 Assessments**: Structured Attack Feasibility and Defense Robustness pre-assessments
- **Frozen Stage 1 Artifact**: Content-addressed, hash-validated pre-assessments for exact reuse by later stages
- Agent reasoning remains in workflow/log state and is never written back into the Stage 0 Evidence Store.

## Technology Stack

- **Agent Framework**: LangChain, LangGraph
- **Language Model**: Qwen3.8-27B Q6_K_L via pinned llama.cpp CUDA runtime
- **Programming Language**: Python 3.10+
- **Asynchronous Processing**: asyncio
- **Environment Management**: python-dotenv

## Active Stage 1 Roles

### Attack Feasibility Critic
- **Responsibility**: Test whether the forecasted Threat trajectory is plausible given Stage 0 evidence
- **Does not do**: Attack-scenario or exploit-plan generation
- **Output**: `CriticAssessment(critic_type="attack_feasibility")`

### Defense Robustness Critic
- **Responsibility**: Test whether the forecasted PMT trajectory is plausible given maturity/deployment/applicability evidence
- **Does not do**: Reactive defense-plan, budget, ROI, or product recommendation generation
- **Output**: `CriticAssessment(critic_type="defense_robustness")`

## Run Stage 0 → Stage 1 end to end

Run from the `Multi-Agent/` directory. `Pipeline.main` now runs the frozen seven-case main set by default.

Terminal 1 — start the pinned Qwen3.8 runtime:

```powershell
cd Multi-Agent
.\Stage1\start_qwen38.ps1
```

Terminal 2 — execute the whole frozen main set:

```powershell
cd Multi-Agent
py -3 -m Pipeline.main
```

The batch is fail-fast but resumable. Exact final artifacts are reused automatically. In addition, each Critic output is hash-validated and checkpointed immediately. If a run stops after Attack Feasibility finishes but before Defense Robustness finishes, rerunning the same command reuses the exact Attack checkpoint and generates only the missing Defense assessment. Checkpoints are tied to the same exact input fingerprint as the final artifact, so changed Stage 0 data, prompts, model/runtime settings, schema, or seed never fall back to an older checkpoint.

Execution observability is intentionally verbose. Every case prints `[i/7]`, its selection reason, Stage 0 evidence coverage, artifact/checkpoint reuse status, and completion status. While Qwen is generating, the runtime polls llama.cpp `/slots` every 10 seconds by default and logs elapsed time plus available prompt/generated token counters. Final response usage is logged after each generation. The private reasoning text itself is **not** dumped to the run log; only progress/usage metadata is logged. Change only the logging cadence with `STAGE1_PROGRESS_INTERVAL_SECONDS`.

Logs are appended in real time under `../Data/Stage1/logs/`. Machine-readable batch progress is atomically updated after every case under `../Data/Stage1/runs/<case-set>__seed-<seed>__progress.json`, so an interrupted run shows exactly which case was running/completed/failed and which artifact was produced.

Inspect the frozen set without inference:

```powershell
py -3 -m Pipeline.main --list-cases
```

The old one-case environment-driven path remains available for debugging/smoke runs only:

```powershell
$env:STAGE0_SNAPSHOT_ID = "stage0-2026-09-12-v2"
$env:STAGE0_THREAT = "DDoS"
$env:STAGE0_PMT = "NLP/LLM"
$env:STAGE0_ANALYSIS_CUTOFF = "2024-12-31"
$env:STAGE0_EVALUATION_MODE = "ex_ante_replay"
$env:STAGE0_CASE_ID = "debug__ddos__nlp_llm"
py -3 -m Pipeline.main --single
```

## Conclusion

The implemented code currently stops after frozen Stage 1 independent critique. Stage 2 debate and all later decision lenses remain design contracts only until implemented and validated in their respective stage directories.
