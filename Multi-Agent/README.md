# Multi-Agent Cybersecurity Analysis System

> The implemented runtime is separated by responsibility. `Stage0/` owns canonical forecast/evidence preparation and deterministic retrieval, `Stage1/` owns the shared pinned Qwen/llama.cpp runtime plus independent critic generation/frozen Stage 1 artifacts, `Stage2/` owns structured debate/post-assessment contracts, `Stage3/` deterministically freezes the Common Decision Object, `Stage4/` runs the three information-isolated strategic feasibility lenses, `Stage5/` deterministically computes disagreement/evidence/context-sensitivity diagnostics, `Stage6/` deterministically freezes the policy recommendation and complete Stage 4 evidence trace before a constrained LLM writes the final evidence-grounded strategic intelligence report, `Stage7/` validates the frozen Agent-layer outputs without joining the decision graph, and `Pipeline/` contains cross-stage orchestration. `Lagacy/` is archival and is never imported by the active pipeline. No implemented downstream stage trains or reruns B-MTGNN.

`load_data_node` no longer reads B-MTGNN forecast TXT files directly. Agent runs consume an existing Stage 0 forecast dataset and frozen evidence snapshot. Configure the case with `STAGE0_SNAPSHOT_ID`, `STAGE0_THREAT`, `STAGE0_PMT`, and `STAGE0_ANALYSIS_CUTOFF`; optional `STAGE0_EVALUATION_MODE` and `STAGE0_CASE_ID` control case execution. `forecast_origin_date` is fixed by the canonical forecast manifest and cannot be overridden at runtime.

This repository contains the legacy multi-agent collaborative framework plus the redesigned Stage 0 data/evidence boundary, Stage 1 independent forecast critics, Stage 2 evidence-grounded structured debate, Stage 3 Common Decision Object builder, Stage 4 parallel strategic value-lens evaluation, Stage 5 deterministic disagreement/evidence diagnostics, Stage 6 disagreement-preserving synthesis, and the separate Stage 7 Agent-layer validation harness. The active graph stops after the frozen Stage 6 artifact; Stage 7 validates those frozen outputs rather than acting as another runtime decision Agent.

The active Multi-Agent environment is tested with **Python 3.11**. Install it with `pip install -r requirements.txt` from this directory; the requirements file includes the Stage 0 document-parsing dependencies and the regression-test runner.

### Stage 1 LLM runtime

The active Stage 1 runtime uses the local `Qwen3.8-27B-Q6_K_L` GGUF through a pinned llama.cpp CUDA server rather than Ollama. On an artifact miss, start the server with `Stage1/start_qwen38.ps1`; then run the implemented pipeline from the `Multi-Agent/` directory with `python -m Pipeline.main`. An exact frozen-artifact hit does not require the LLM server because no model call is made.

`Stage1/start_qwen38.ps1` no longer contains a machine-specific absolute path. It resolves the repository root from the script location, derives the default shared workspace root two levels above the repository (for the common `<workspace>/Papers/<repository>` + `<workspace>/LLMs/` layout), and then resolves llama.cpp/model files under `<workspace>/LLMs/`. Set `LLM_ROOT`, `LLAMA_CPP_RUNTIME`, or `QWEN38_MODEL_PATH` to override those defaults without editing the repository.

The main-experiment thinking profile is fixed to the Qwen3.8 recommended sampling values: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and `repeat_penalty=1.0`, with reasoning enabled at `xhigh` effort and preserved thinking. The llama.cpp server enables the model's embedded MTP path with `--spec-type draft-mtp`.

The active shared server uses `--parallel 2`. Stage 1 remains logically parallel and independent in LangGraph, and up to two information-isolated requests may execute concurrently without exposing either Critic's assessment to the other.

The pinned main-run artifact is `Qwen3.8-27B-Q6_K_L.gguf`, size `24,958,908,128` bytes, SHA-256 `e8750bb81ba49f90eb68df99776b250dbc14666043098952de6422cbecd77a21`, from `bartowski/Qwen3.8-27B-GGUF` revision `e4cc3ff3b37f5aabf253d8583b0b0247e8b25b70`. The runtime is llama.cpp build `b10919`, commit `d3146f2b5`. The active shared profile reserves **131,072 aggregate context tokens split across two 65,536-token inference slots**, keeps **all model layers on GPU**, uses Q8 target/draft KV caches, `batch=2048`, `ubatch=512`, MTP draft depth `4` with `p_min=0.05`, and Flash Attention.

The active contextual Stage 4 experiment uses the versioned **131,072-token total server context split across two 65,536-token slots**, `stage4-context-runtime-128k-total-2slot-v3`. A pre-optimization Financial/Adoption request contained **142,549 prompt tokens**, which exceeded 128K and triggered a temporary 256K test. The prompt projection was then compacted by removing duplicated evidence serialization, validator-only metadata, unrelated lens contracts, and pretty-print whitespace; representative payloads dropped by about **75%** without removing forecast, Stage 2 provenance, deployment context, or any selected evidence record. The active Stage 4 profile keeps the same 128K aggregate KV budget but allows two concurrent lens generations. Model weights/quantization, llama.cpp build, sampling, xhigh reasoning, MTP, Q8 target/draft KV, and batch/ubatch remain fixed.

Operationally, `Stage1/start_qwen38.ps1` is the **single shared llama-server launcher** and always starts the active two-slot runtime: `131072` aggregate context split into `2 × 65536-token` slots. Stage 1, Stage 2, contextual Stage 4, and Stage 6 all use client request concurrency `2`. Existing frozen Stage 1/2 artifacts created under the historical one-slot runtime are still reusable through an explicit concurrency-only legacy compatibility path; new inference uses the active two-slot profile.

### Pre-experiment runtime calibration

Inference acceleration is selected **before** main-case generation and then frozen as part of the experiment identity. Calibration is not allowed to adapt per Threat–PMT case. The procedure is: (1) verify the target 128K context and KV precision fit with full GPU residency, (2) measure a non-speculative baseline, (3) screen bounded MTP draft-depth / acceptance-threshold candidates under the same model, hardware, prompt, sampling profile and seed, (4) run a deterministic greedy probe against the non-speculative baseline as a correctness fingerprint gate, and (5) promote only a candidate that passes the correctness gate and then survives a main-like long-prompt recheck. A faster candidate is rejected if its deterministic output fingerprint differs from the non-speculative baseline.

On the current RTX 5000 Ada / llama.cpp build, the 128K full-GPU profile fit successfully. Screening measured approximately `20.37 tok/s` with MTP disabled, `29.09` at depth 1, `31.41` at depth 2, `31.18` at depth 3, and `32.52–32.55` at depth 4. In the deterministic correctness probe, the selected `depth=4, p_min=0.05` profile matched the non-speculative baseline fingerprint while tested depth-2/depth-3 candidates did not. A main-like Stage 1 request with a 15,747-token prompt, JSON-schema constraint and xhigh reasoning measured `38.47 tok/s` over a 2,048-token capped generation. These numbers are calibration measurements for this exact hardware/build, not universal model-speed claims. Calibration traces are stored under `Results/Stage1/runtime_bench/`.

### Frozen Stage 1 artifacts

Main-experiment Stage 1 output is generated once per exact experiment identity and frozen under `Results/Stage1/artifacts/<case_id>/<input_fingerprint>.json`. The fingerprint binds the Stage 0 `EvidencePack`, prompt-facing forecast payload, both Critic prompts, structured-output schema, model artifact/revision/hash, llama.cpp build, sampling/reasoning/MTP configuration, context profile, and seed. If the exact fingerprint already exists, both Critic LLM calls are skipped and the frozen pre-assessments are reused. A changed input, prompt, model, inference setting, schema, or seed creates a new fingerprint; the runtime never falls back to or overwrites another Stage 1 artifact.

Each frozen artifact carries its own SHA-256 plus a separate Stage 1 output SHA-256. Hash mismatch, unknown evidence references, wrong critic type, or an attempt to write different outputs under an already-frozen fingerprint is a hard error. Repeated-run robustness experiments therefore use different predeclared seeds/fingerprints and do not mutate the canonical seed-42 main artifact.

Structured output uses llama.cpp JSON-schema constrained generation followed by deterministic Pydantic semantic validation. The per-case generation schema binds `case_id`, `critic_type`, and every claim's `evidence_ids` to the exact Stage 0 values, so an unknown or prefix/digest-recombined evidence ID is rejected during constrained generation rather than being treated as a recoverable citation. Semantic validation remains as a backstop. If an otherwise structured response violates a semantic invariant, the runtime performs exactly one fixed repair turn. The repair must re-read the supplied Stage 0 records and choose only an exact evidence ID that supports the literal claim; digest/prefix/similarity based ID reconstruction is forbidden. If no supplied record supports the claim, the claim must be removed or rewritten. A second failure aborts the run. The constrained-schema hash, repair instruction hash, and retry count are part of the Stage 1 experiment fingerprint.

### Single-inference review protocol

Each Critic makes its initial judgment in **one xhigh inference**, not a chain of separate self-refinement calls. Multiple sequential calls would introduce self-conditioning on earlier model-generated text, make the result depend on an additional call-count/order hyperparameter, and partially duplicate the role of the later Stage 2 inter-Critic debate. A single content-addressed `Forecast + EvidencePack + prompt -> CriticAssessment` mapping is therefore easier to reproduce and keeps Stage 1's initial judgment genuinely pre-debate.

Reasoning depth is encouraged with one fixed internal review order inside that single inference. Both critics must internally review: **(1) every supplied evidence record and its boundary, (2) the forecast as separate components, (3) forecast-supporting evidence mapping, (4) forecast-challenging evidence mapping, (5) cross-source conflicts and alternative explanations, (6) modality and causal boundaries, (7) temporal consistency, (8) claim polarity / specificity / traceability against exact source wording, and (9) an explicit evidence-balance decision.** The role-specific prompt changes the substance of these checks for Threat vs PMT, but not their order. Intermediate reasoning is not treated as evidence or stored as a research result; only the final structured `CriticAssessment` is frozen.

### Stage 1 v4 decision contract

The active substantive contract is `stage1-artifact-v4` / `stage1-critic-checkpoint-v4`. Each final claim is an affirmatively evidence-grounded factual finding with:

- `evidence_ids`: exact Stage 0 record IDs supporting the literal statement
- `forecast_component`: `THREAT_TRAJECTORY`, `PMT_TRAJECTORY`, `GAP_DIRECTION`, `OPERATIONAL_INTERPRETATION`, or `CONTEXT`
- `forecast_relation`: `SUPPORTS_FORECAST`, `CHALLENGES_FORECAST`, or `NEUTRAL_CONTEXT`

`forecast_component` protects the measurement modality. `NoI` means incident count/frequency: attack bandwidth or magnitude, severity, tactic prevalence, malware-family share, or a percentage/share of intrusions cannot by themselves support a directional `NoI` trajectory claim. A direct `NoI` claim must cite a threat-tagged source whose contract allows `directional_threat_trend`, and both the cited record and final claim must explicitly report a temporal change in incident count/frequency. Likewise, evidence that a control is difficult to evade, widely recommended, or operationally effective is not automatically evidence that a `NoP` forecast rises or falls. Static bibliographic metadata (a title/DOI/publication record) also proves publication existence/topic, not the direction of publication activity. A directional `NoP` trajectory claim therefore requires evidence whose source contract explicitly permits `directional_publication_trend`. Other findings remain `OPERATIONAL_INTERPRETATION`/`CONTEXT` as appropriate. `GAP_DIRECTION` is not inferred from a one-sided operational weakness alone.

Stage 0 stores the signed yearly gap as `Threat state z - PMT state z`. Its direction label therefore uses `threat_minus_pmt_increasing`, `threat_minus_pmt_decreasing`, or `flat`; these labels describe signed movement and must not be read as absolute-distance "widening" or "narrowing".

The final stance is not a raw claim-count vote. `stance_basis` records the decisive claims and one of four decisions: `SUPPORT_DOMINATES`, `CHALLENGE_DOMINATES`, `BALANCED`, or `NO_DIRECTIONAL_EVIDENCE`. Attack Feasibility may use only `THREAT_TRAJECTORY` / `GAP_DIRECTION` claims as decisive basis; Defense Robustness may use only `PMT_TRAJECTORY` / `GAP_DIRECTION` claims. `stance=0` is therefore **not** a generic uncertainty fallback: it requires either genuinely balanced role-relevant directional evidence or no role-relevant directional evidence at all. Evidence sufficiency remains an independent field.

> **Mandatory paper-writing interpretation for the frozen seven-case Stage 1 result:** the current `stage1-artifact-v4` / `stage1-semantic-validation-v4` main run yields `0 / NO_DIRECTIONAL_EVIDENCE` for the Attack Feasibility Critic in all 7 cases. The paper must report this explicitly rather than hiding or post-hoc relaxing the modality contract. For the five selected Threats whose forecast state is `NoI`, the frozen EvidencePacks contain no eligible threat-specific temporal incident-count/frequency trend; attack magnitude, tactic prevalence, malware-family share, intrusion share, and similar operational signals remain context rather than direct `NoI` evidence. The two remaining Threat-side `NoP` cases likewise lack a Threat-tagged directional publication trend. This is therefore interpreted as **strict abstention plus Threat/PMT observability asymmetry**, not as absence of threat activity, predictive uncertainty, or symmetric critic performance. Reviewers are likely to ask why the Attack Critic adds value if all pre-stances are neutral; the required response is that Attack still contributes evidence-grounded operational/context claims for Stage 2 cross-review while refusing to convert modality-mismatched proxies into direct trajectory validation. Do not claim that Attack and Defense symmetrically validate both sides of the forecast.

Model-reported scalar confidence is not part of the contract. Without an external calibration set and correctness labels it is not treated as a probability of correctness; reliability is evaluated later with evidence diagnostics and empirical run/prompt stability.

### Frozen seven-case main evaluation set

The main Stage 1 experiment does **not** run language-model critique over all 303 Threat-PMT relations. The 303 relations remain the quantitative forecast/gap population; Stage 1 performs detailed critique on a frozen seven-case set defined in `Stage1/cases.py` before any main Stage 1 outputs are generated.

Case selection uses Stage 0 information only. Stage 1 stance, disagreement, or downstream policy outputs are never selection variables. This prevents post-hoc case selection based on favorable Agent results.

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

The active runtime does not use the legacy mutable LightRAG store as an additional factual evidence channel. The Evidence Store is physically stored under `Results/Stage0/Evidence/snapshots/<snapshot_id>/`, with `records.jsonl`, `manifest.json`, and copied raw `artifacts/`. The active Stage 0 contract is `stage0-registry-v3` / `stage0-source-manifest-v3` / `stage0-ingestion-v4`; pre-v3 registry snapshots are not runtime-compatible. The frozen main snapshot is `stage0-2026-09-13-v4`, generated from scratch by the Stage 0 E2E build command below. It contains 48 raw artifacts and 35,399 EvidenceRecords with a PASS coverage/integrity audit. Crossref source `33` queries all 98 PMTs and exposes 96 non-zero directional publication-trend records; the two zero-result queries (`BLACKLISTING`, `BLACKHOLING`) remain in the raw audit artifact but are not mislabeled as flat directional evidence. The fixed source registry is `Results/Stage0/Evidence/source_registry.json`.

Stage 0 retrieval is **temporally bounded deterministic BM25 evidence retrieval**, not a vector-similarity LightRAG call. It first filters frozen records by Threat/PMT tags and `available_at <= analysis_cutoff_date`, then maps candidates to six semantic evidence slots through source/claim contracts. Within each slot, a fixed non-LLM lexical query is generated from the Threat name, PMT name, forecast gap direction, and slot vocabulary; candidates are ranked with Okapi BM25 (`k1=1.2`, `b=0.75`). Source diversity is applied before filling the remaining positions, and at most four records are selected per slot. Both Stage 1 critics receive the same resulting `EvidencePack`.

Registry source `33` adds an external publication-activity validation axis for PMT `NoP` trajectories. It queries Crossref independently of the Scopus-derived B-MTGNN input, using `query.title=<PMT>` plus `query.bibliographic=security`, one `published` year facet over 2021–2024, and `until-created-date=2024-12-31`. The result is one deterministic `directional_publication_trend` record per PMT. Static Crossref title/DOI records from source `29` still cannot support a directional `NoP` claim; their `available_at` is also bounded by first Crossref deposit rather than being backdated to the publication date.

For reproducibility, `retrieval_metadata` records the retriever version, exact query text per slot, BM25 parameters, candidate counts, and per-record BM25 scores. Query generation is deterministic and does not depend on Agent persona or model sampling.

The full retrieval audit metadata is retained in the frozen `EvidencePack` but is not injected wholesale into the LLM prompt. The Agent-facing payload receives the selected evidence records plus a compact retrieval summary (cutoff status, evidence sufficiency, slot coverage, source-family/chain counts, and violation counts). BM25 candidate score tables and query audit traces are provenance/audit data rather than factual evidence; excluding them from the prompt prevents audit bookkeeping from consuming the reasoning context while preserving exact reproducibility in the frozen artifact.

For comparison, the archived legacy implementation in `Lagacy/rag.py` used mutable `LightRAG` with Ollama embedding/completion functions and `QueryParam(mode="hybrid")`. Each Agent issued a different natural-language retrieval query during its own turn, and `add_conversation_data()` inserted generated Agent analyses back into the same RAG storage. That behavior is intentionally excluded from the redesigned runtime because it mixes generated reasoning with external evidence and makes Agent-to-Agent evidence coverage differ.

## System Overview

The Multi-Agent system is built on LangGraph. Stage 0 constructs one case-specific `EvidencePack` from the canonical 124-node forecast dataset and an immutable evidence snapshot. Stage 1 sends that same payload in parallel to an Attack Feasibility Critic and a Defense Robustness Critic, producing two independent structured pre-assessments. Stage 2 freezes those pre-assessments, performs a bounded claim/evidence-level debate, and produces independent post-assessments. Stage 3 then deterministically freezes the common action/context/evidence evaluation unit, Stage 4 fans that same unit out to Technical, Institutional/Regional, and Financial/Adoption lenses without exposing sibling outputs, Stage 5 computes deterministic diagnostics without any additional LLM call, and Stage 6 applies a versioned deterministic decision policy to produce the scenario recommendation, preserves the full Stage 4 claim/evidence trace, and then uses a constrained LLM to produce the evidence-cited final strategic intelligence report.

### Why the critics/debate are not redundant with the three decision lenses

The active architecture has two intentionally different reasoning layers.

- **Stages 1–2: epistemic/adversarial validation.** Attack and Defense examine the Threat/PMT forecast against the frozen Stage 0 evidence and enforce modality, temporal, causal, and claim/evidence boundaries. Their job is to determine what can legitimately be said about the forecast and its external support, not whether an organization should adopt the PMT. The Mediator adjudicates conflicting interpretations inside the surfaced frozen evidence boundary, can revise unsupported claim polarity/scope, and preserves unresolved questions; it cannot make the downstream policy decision.
- **Stages 4–6: strategic decision evaluation.** Once the forecast interpretation and its unresolved limitations are frozen, the three Stage 4 lenses evaluate the already-selected action under Technical Feasibility, Institutional/Regional Feasibility, and Financial/Adoption Feasibility. These lenses answer whether the candidate is supportable on those decision dimensions; they do not re-estimate forecast truth. Stage 6 then applies the versioned deterministic policy to the three lens states and gives the final LLM explanatory, not decision-making, authority.
- **Stage 3 is the firewall/bridge.** It freezes the action plus Stage 1–2 reasoning provenance before Stage-4-specific cost, governance, regulation, procurement, staffing, and deployment evidence enters the decision layer. This prevents downstream feasibility evidence from retrospectively changing the upstream forecast critique while still forcing downstream lenses to inherit its contradictions, revisions, and unresolved gaps.

Accordingly, the paper should not describe Attack/Defense/Mediator as three more voters beside Technical/Institutional/Financial. The former are **forecast/evidence validators and adjudicators**; the latter are **functional decision evaluators**. Stage 2 can therefore add value even when the final pre/post critic stances do not flip: claim-level scope corrections, proxy-evidence downgrades, resolved/unresolved adjudications, and a cleaner frozen reasoning provenance are legitimate outputs. The final Stage 7 keeps grounding and Mediator-fidelity checks as internal integrity audits, while the current paper-facing experiments do not separately isolate Stage 2's independent necessity.

### Key Features
- **Independent Forecast Critique**: Threat and PMT forecast components are evaluated independently before any debate
- **Frozen Evidence Input**: Case-specific retrieval is completed in Stage 0 with provenance and temporal-cutoff metadata
- **Structured Traceability**: Stage 1 returns stance, evidence sufficiency, non-empty claim/evidence links, `forecast_component`, `forecast_relation`, `stance_basis`, unresolved questions, and unsupported-specificity flags. Final claim statements must be affirmatively grounded by exact cited evidence IDs. `forecast_component` prevents operational facts from being silently treated as direct evidence for a different forecast modality, while `stance_basis` may use only role-relevant trajectory/gap claims.
- **No Premature Anchoring**: Both critics see the same Forecast + EvidencePack and cannot see each other's initial output
- **Bounded Structured Debate**: Stage 2 runs one round by default; a second round runs only when the round-1 Mediator emits structured `CONTINUE` with exact claim-level focus, and the maximum is two rounds
- **Evidence-bounded Mediator Adjudication**: the Mediator compares only already-surfaced frozen evidence, can accept/reject/revise claim interpretations, mark disputes resolved/unresolved, and route one focused follow-up round; Python enforces the evidence/schema/hard-cap boundary while Stage 7 separately validates judgment stability and fidelity
- **Pre/Post Traceability**: Stage 1 pre-assessments remain immutable while Stage 2 records every flattened claim-level exchange, the independent post-assessments, and exact stance changes
- **Information-isolated Strategic Lenses**: Stage 4 gives all three value lenses the same frozen DecisionObject and exact Stage 0 evidence universe; sibling lens outputs are never included in another lens prompt
- **Deterministic Disagreement Diagnostics**: Stage 5 computes within-scenario `D_lens`, evaluability/joint-abstention, Stage 2 pre/post critique dispersion, system evidence coverage, and separate cross-jurisdiction stance-vs-evidence sensitivity. The three Stage 4 lenses are predeclared functional dimensions rather than a statistical sample, so `D_lens` is only descriptive within-set dispersion; it is never treated as Bayesian/predictive uncertainty, population uncertainty, correctness probability, or calibrated confidence.
- **Decision-bearing, Disagreement-preserving Strategic Intelligence**: Stage 6 does not majority-vote the three lenses, but it does produce a policy-conditional recommendation. Technical and Institutional/Regional assessments are predeclared critical feasibility gates; once both support the action, Financial/Adoption determines whether the outcome is `RECOMMEND`, `RECOMMEND_PILOT`, or `HOLD`. A challenge on either critical gate yields `DO_NOT_RECOMMEND`. This is an auditable result of `stage6-decision-policy-v1`, not a statistical claim that three agents establish a universally correct action. Qwen cannot choose or change the recommendation; it converts the frozen Stage 4 evidence trace into a human-facing, evidence-cited strategic intelligence report while preserving dissent and evidence limits.

## Methodology

### System Architecture

#### Core Components

1. **Pipeline Entry (`Pipeline/main.py`)**: Responsible for system initialization/execution and cross-stage logging/configuration.

2. **Pipeline Graph/State (`Pipeline/graph.py`, `Pipeline/state.py`)**: Defines the active Stage 0 → Stage 1 parallel critique → Stage 2 structured debate → Stage 3 deterministic Common Decision Object → Stage 4 parallel lens workflow → Stage 5 deterministic diagnostics → Stage 6 policy-conditional strategic-intelligence reporting workflow and the shared state contract.

3. **Stage 0 (`Stage0/`)**: Canonical 124-node migration, immutable Evidence Store, source registry/ingestion, deterministic retrieval, Agent-input construction, and the Stage 0 load node.

4. **Stage 1 (`Stage1/`)**: Implements all active Stage 1-specific code:
   - **Attack Feasibility Critic**: Evaluates the Threat forecast against adversary capability and observed threat evidence
   - **Defense Robustness Critic**: Evaluates the PMT forecast against maturity, deployability, applicability, and implementation evidence
   - `Stage1/runtime.py`: pinned Qwen3.8/llama.cpp client and runtime verification
   - `Stage1/prompts.py`: only the implemented Stage 1 critic prompts
   - `Stage1/nodes.py`: artifact preparation, critic execution, validation and freeze
   - `Stage1/artifact.py`: immutable content-addressed Stage 1 artifact contract

5. **Stage 2 (`Stage2/`)**: Implements bounded evidence-grounded deliberation:
   - `Stage2/schema.py`: claim-level response, Mediator, round, and final debate contracts plus exact-ID semantic validation
   - `Stage2/prompts.py`: critic cross-review, evidence-bounded Mediator adjudication, and post-assessment prompts
   - `Stage2/nodes.py`: one/two-round routing, exact-step resume, and post-assessment generation
   - `Stage2/artifact.py`: immutable Stage 2 artifact and per-step checkpoints bound to both experiment identity and exact upstream-step dependencies
   - Reuses `Stage1/runtime.py` unchanged for model/runtime/sampling/seed settings

6. **Stage 3 (`Stage3/`)**: Implements the deterministic Common Decision Object bridge and the active contextual v4 supplement:
   - `Stage3/schema.py`: frozen action-set, decision-object, deployment-context, and Stage 2 provenance contracts
   - `Stage3/builder.py`: pure programmatic transformation from the exact Stage 0 + Stage 2 result into the shared downstream evaluation unit
   - `Stage3/artifact.py`: immutable content-addressed `stage3-artifact-v2` bound to the exact Stage 2 artifact/output plus the Stage 3 decision context, selection rule, and output schema
   - `Stage3/nodes.py`: exact frozen Stage 1→2 provenance-chain validation, deterministic build/reuse, and pipeline handoff
   - `Stage3/context_schema.py`, `context_builder.py`, `context_artifact.py`, `context_nodes.py`: preserve the base v2 object as the context-free baseline, then freeze three scenario-conditioned objects (`KR__CIS_IG2`, `EU__CIS_IG2`, `US__CIS_IG2`) under `stage3-artifact-v4` without any additional LLM call
   - The contextual layer combines the original forecast EvidencePack, one hash-pinned CIS IG2 reference-enterprise record, and deterministic Stage-4-specific decision evidence. The base `stage0-2026-09-13-v4` snapshot remains unchanged; the active PMT-specific guidance snapshot is `stage3-decision-sources-2024-12-31-v3` (18 official documents / 2,187 records). Every main PMT has at least two publishing organizations: NIST + NCSC UK for NLP/LLM, and NIST + CISA/DHS for Anomaly Detection, IDS/IPS, Cryptography, Access Control, and HTTPS.
   - Decision evidence is intentionally added only after the candidate action and deployment context are frozen. Feeding implementation cost, staffing, regulatory burden, procurement difficulty, or successful-deployment evidence into Stage 1–2 would mix downstream adoption feasibility with upstream forecast critique and can create anchoring/selection bias. The active v5 contract is therefore **predeclared and result-blind**: the same source/action/PMT/cutoff rules are fixed before any new Stage 4 output, and all three lenses receive the same combined evidence universe.
   - `decision-evidence-retrieval-v5` splits decision evidence into distinct roles. `stage3-action-evidence-registry-v1` is a hash-pinned 14-record registry with one source-grounded action-effectiveness and one action-limitation finding for each of the seven frozen Threat×PMT actions; these are the only records allowed to carry Technical +/− direction on the main path. `stage3-main-guidance-selection-v1` is a hash-pinned exact evidence-ID map over the frozen Stage 0/decision-guidance stores. It predeclares **46 audited records** and their exact `deployment_maturity`, `adoption_benefit`, `adoption_burden`, `governance_enablement`, or `compliance_constraint` slots. Main Financial/Institutional/deployment guidance therefore bypasses BM25 selection entirely; BM25/semantic retrieval remains only for fallback/ablation use.
   - v5 does **not** require support/challenge counts to be equal. The earlier v4 quota of two documents/two publishers per direction exposed the model to pseudo-balanced lexical matches and the compact KR diagnostic collapsed to `[0,0,0]`; that run is a diagnostic, not a main result. The v5 main floor instead requires ≥6 generic records, ≥3 generic documents, ≥3 documents and ≥2 publishers per lens, plus at least one exact action-specific Technical support record and one exact action-specific Technical challenge record. If a valid Institutional or Financial challenge/support facet has no audited record, it remains absent rather than being filled to satisfy symmetry. Generic evidence IDs remain exact-invariant across KR/EU/US; only jurisdiction-specific `regulatory_applicability` may differ.
   - Active-v5 verification is result-blind: the 7×3 preflight passed under `de-83662ff07d`, and the real frozen Stage 0 + frozen Stage 2 → base Stage 3 → contextual Stage 3-v4 path rebuilt and validated all 21 objects without LLM inference. Observed Financial support:challenge counts across the seven KR objects are `2:2, 2:1, 2:1, 2:1, 3:1, 1:3, 2:1`, confirming that the input contract no longer manufactures 50:50 balance. The contextual constrained-output schema still requires all top-level array fields and `claims.minItems = 1`. `stage4-semantic-validation-v6` rejects both internally inconsistent neutral assessments and unexplained jurisdiction-label stance drift: when lens-relevant evidence is identical and no lens-specific jurisdiction evidence exists, KR/EU/US GENERAL stance must remain invariant. The current Stage 0–5 regression suite is **206/206 PASS** (Stage 0 55, Stage 1 39, Stage 2 47, Stage 3 10, Stage 4 50, Stage 5 5).
   - Active v1 uses `case-forecast-pmt-only-v1`: the PMT already present in the predeclared B-MTGNN Threat-PMT case is the sole main candidate. Other PMTs merely co-tagged by retrieved evidence are recorded as excluded rather than silently promoted; alternative action sets belong to explicit sensitivity analysis.
   - Every DecisionObject freezes `evaluation_evidence_ids` to the complete selected EvidencePack for that case. Supporting/challenging IDs are diagnostic subsets; later parallel lenses must start from the same frozen evidence universe.

7. **Stage 4 (`Stage4/`)**: Implements the information-isolated strategic value lenses. Context-free v1 remains reproducible; contextual v2/v3 are historical diagnostics, and the active graph uses contextual v4:
   - `Stage4/schema.py`: common `LensAssessment` contract with stance, evidence sufficiency, rationale, evidence-grounded claims, GENERAL/DEPLOYMENT_SPECIFIC scope, constraints, gaps, and conditional requirements
   - `Stage4/input.py`: materializes the exact same DecisionObject + full frozen evaluation evidence records for every lens
   - `Stage4/prompts.py`: Technical Feasibility, Institutional/Regional, and Financial/Adoption lens contracts with unsupported-specificity and missing-context boundaries
   - `Stage4/nodes.py`: LangGraph sibling fan-out/join, exact Stage 3 provenance revalidation, and per-lens resumable generation
   - `Stage4/artifact.py`: exact-fingerprint per-lens checkpoints plus immutable `stage4-artifact-v1`, bound to Stage 3, Stage 0 evidence, prompts, schemas, payload hashes, runtime, and seed
   - `Stage4/context_input.py`, `context_prompts.py`, `context_schema.py`, `context_artifact.py`, `context_nodes.py`: contextual `stage4-semantic-validation-v6` / `stage4-artifact-v4` path with three scenarios per case and scenario×lens `stage4-lens-checkpoint-v4` resumability. The artifact/checkpoint schemas remain v4; only the semantic-validation contract advanced to v6. `Stage4/schema.py`, `input.py`, `prompts.py`, `nodes.py`, and `artifact.py` remain the separate context-free v1 baseline implementation; their v1 constants are not stale contextual versions.
   - `Stage4/context_input.py` builds a compact prompt projection instead of dumping the full `ContextualDecisionObject`. Frozen provenance remains in Stage 3, while the LLM sees each selected evidence body once, only judgment-relevant evidence metadata, the assigned lens's grounding contract only, and compact JSON. It does **not** duplicate `evaluation_evidence_records` inside the decision object, duplicate `DecisionEvidencePack` records, or inject validator-only hashes/snapshot/allow-deny metadata. This changes serialization size, not the shared evidence universe. Measured on the available real frozen Stage 3 artifacts, serialized contextual payload size fell by about 75% (DDoS×NLP/LLM 201,561→~50.4k chars; Malware×Anomaly Detection 304,777→~76.1k chars).
- `Stage4/context_runtime.py` defines the active `stage4-context-runtime-128k-total-2slot-v3` contract: `131072` total server context, `2` inference slots, `65536` tokens per slot, and client request concurrency `2`. The live server is launched with the shared `Stage1/start_qwen38.ps1`; `context_artifact.py` hashes this Stage 4-specific runtime profile into the artifact/checkpoint identity, and `context_nodes.py` hard-validates the live two-slot llama.cpp server before a contextual lens call.
   - Contextual factual claims may cite any exact ID in the frozen `evaluation_evidence` union, but `CTX_*` records are restricted to reference-enterprise facts. The payload exposes a deterministic lens/scope/**direction** grounding contract. Technical support requires `technical_enablement` and challenge requires `technical_limitation`; Financial support requires `adoption_benefit` and challenge requires `adoption_burden`. `deployment_maturity` remains useful context but is not a direction carrier. Institutional support requires `governance_enablement`, challenge requires `compliance_constraint`, and a `DEPLOYMENT_SPECIFIC` legal direction additionally requires jurisdiction-specific `regulatory_applicability`; applicability alone is not interpreted as permission or prohibition. Wrong-polarity, scope-incomplete, or otherwise ineligible directional claims are downgraded to `NEUTRAL_CONTEXT`. Unsupported nonzero stances may only be weakened to 0; deterministic validation never promotes 0 to +1/-1. Residual deployment-detail gaps do not erase an independently grounded GENERAL direction, which is represented as `PARTIAL`.
   - Primary `D_lens` is computed only within one frozen scenario (`KR__CIS_IG2`, `EU__CIS_IG2`, or `US__CIS_IG2`). Cross-jurisdiction variation is a separate context-sensitivity diagnostic rather than another form of lens disagreement. Stage 7 tests this context contribution directly with the frozen context-free baseline and KR/EU/US contextual scenarios.
   - The active llama.cpp runtime exposes two 65,536-token inference slots. `contextual_priority_queue_lenses_node` uses a fixed two-worker queue, so two information-isolated contextual lens calls may execute concurrently without exposing sibling outputs to one another.

8. **Stage 5 (`Stage5/`)**: Deterministically computes disagreement/evidence diagnostics from the exact frozen upstream chain. It adds no LLM judgment, does not alter Stage 4 stances, separates within-scenario `D_lens` from jurisdiction stance/evidence sensitivity, and freezes `stage5-artifact-v1`.

9. **Stage 6 (`Stage6/`)**: Implements disagreement-preserving, policy-conditional strategic intelligence under `stage6-synthesis-v5` / `stage6-semantic-validation-v12`:
   - `builder.py` applies versioned `stage6-decision-policy-v1` before inference. Technical and Institutional/Regional are asymmetric critical feasibility gates; Financial/Adoption controls adoption scale once those two gates pass. The resulting `decision_recommendation` is one of `RECOMMEND`, `RECOMMEND_PILOT`, `HOLD`, or `DO_NOT_RECOMMEND`, with an auditable `decision_rule_id`.
   - The policy is deliberately not a 2-of-3 vote and does not use `D_lens` as a decision score. `D_lens`, supporting/dissenting/indeterminate lenses, gaps, conditions, and evidence sufficiency remain visible diagnostics. `PARTIAL` evidence qualifies the scope of the conclusion rather than acting as an additional negative vote because Stage 4 already maps genuinely insufficient evidence/context to stance `0`.
   - `schema.py` deliberately has no free-form LLM `final_stance`. The deterministic recommendation is fixed before inference under `recommendation_scope=STRATEGIC_CANDIDATE_ACTION`. The final deterministic output also retains the complete Stage 4 `lens_evidence_trace`. Qwen generates only the human-facing `strategic_intelligence_report`. The canonical inline citation form is `[EVIDENCE:<exact evidence_id>]`; Python also accepts a grouped explicit form such as `[EVIDENCE:E1, EVIDENCE:E2]` as the same two exact citations, but rejects malformed evidence brackets rather than inferring IDs from editorial text. Parsed IDs are accepted only from the frozen Stage 3 `evaluation_evidence_ids` universe, which is the exact forecast/context/decision-evidence union. Directional justification for Technical, Institutional/Regional, and Financial/Adoption stances is stricter: it must still cite the corresponding frozen Stage 4 claim evidence. The unique validated IDs are stored as `report_cited_evidence_ids`. The LLM does not generate a separate citation-array field.
   - `runtime.py` reuses the same `131072` aggregate / `2 × 65536` llama.cpp server and sets client concurrency to `2` under `stage6-synthesis-runtime-128k-total-2slot-v1`.
   - `nodes.py` schedules the three jurisdiction syntheses with exactly two workers, checkpoints each completed DecisionObject immediately, and never deduplicates the three main jurisdiction calls: the main experiment therefore has `7 × 3 = 21` Stage 6 synthesis calls.
   - `artifact.py` freezes `stage6-synthesis-checkpoint-v5` and immutable `stage6-artifact-v5`, binding exact Stage 3/4/5 artifact/output hashes, decision-policy version/rule-bearing frame hashes, prompt/schema hashes, and runtime identity under generation contract `policy-conditional-evidence-grounded-strategic-report-v5`.
   - The report fidelity check rejects recommendation drift, malformed/unknown inline evidence citations, missing Stage 4 claim-evidence coverage for evidence-bearing/decisive lens directions, unsupported numeric tokens, majority/final-stance/optimality meta language, unsupported production-deployment guarantees, and clear draft/debug/placeholder residue. A generated number is allowed only when its normalized value already occurs in the prompt-visible synthesis frame or in an exact frozen evidence record that the report actually cites inline; a quantity present only in uncited evidence is still rejected. Duplicate/grouped valid inline citations are harmless because Python materializes a deduplicated `report_cited_evidence_ids` set. `artifact.py` may reuse a frozen `stage6-semantic-validation-v11` artifact/checkpoint only by revalidating it under v12 and promoting it to the current fingerprint without generation; failure of the current validator makes that cache candidate incompatible. The contract does not prescribe prose wording or a case-specific answer template. Semantic claim↔source correctness remains a separate Stage 7 grounding/manual-audit question rather than a self-certified Stage 6 score.

10. **Stage 7 (`Stage7/`)**: Separate **agent-layer** validation harness over the frozen Stage 1-6 chain. The active decision graph still stops at Stage 6. The upstream B-MTGNN forecast/backbone is treated as a frozen input and is outside the Stage 7 contribution boundary. Final paper-facing validation has three axes only: **A Robustness**, **B Decision architecture and cost**, and **C Context contribution**. Grounding and Mediator-fidelity checks remain internal integrity audits. See `Stage7/README.md`.

11. **Legacy Archive (`Lagacy/`)**: Historical inputs/logs plus the retired LightRAG and transition-era node/prompt code. It is not part of the active import graph.

### Active Operation Process

#### Phase 1: Data Loading and Initialization
Load the canonical forecast dataset and frozen Stage 0 evidence snapshot, then configure environment settings.

#### Phase 2: Independent Forecast Critique
The runtime first checks for an exact frozen Stage 1 artifact. On a hit, its validated pre-assessments are reused with no model call. On a miss, the Attack Feasibility Critic and Defense Robustness Critic receive the same Stage 0 payload independently, each returns a validated `CriticAssessment`, and the joined pair is frozen.

#### Phase 3: Evidence-grounded Structured Debate
Stage 2 binds to that exact Stage 1 artifact, not merely to a case name. Both critics review every opponent claim against the same frozen EvidencePack using `AGREE`, `CHALLENGE`, `REVISE`, or `INSUFFICIENT_EVIDENCE`; this cross-review covers all opponent claims even though each critic's **final stance basis** remains restricted to its own Threat/PMT role. The active two-slot runtime allows the information-isolated Attack/Defense same-round turns to run concurrently; neither current-round output is included in the other's prompt.

The Mediator is an **evidence-bounded adjudicator**, not a passive organizer. It receives the frozen pre-assessments, current/previous debate trace, and the actual content of only those Stage 0 evidence records already surfaced by the critics. Within that boundary it may compare evidentiary strength, judge `RESOLVED`/`UNRESOLVED`, emit `ACCEPT_*`, `REJECT_*`, `REVISE_*`, `UNRESOLVED`, or `INSUFFICIENT_EVIDENCE` adjudications, identify conflicts/gaps that the critics missed, and decide whether a focused second round is useful. The explicit `REJECT_*` outcomes let it resolve one-sided challenges even when the challenger has no semantically matching frozen pre-claim. It may not retrieve unseen evidence, invent facts, emit residual-risk percentages, claim predictive/causal truth, or make policy recommendations. Python validates exact claim/evidence IDs, surfaced-evidence visibility, claim-local evidence use, and internal consistency; every non-gap adjudication must cite surfaced evidence for each frozen claim it actually judges. It also validates artifact integrity and the hard two-round limit without replacing the Mediator's substantive judgment with a hard-coded winner rule.

A second round runs only when the round-1 Mediator emits structured `CONTINUE` with exact claim-level `next_round_focus`; otherwise the debate stops after round 1. Round 2 re-reviews the same frozen Stage 1 claim IDs with the complete round-1 trace in context rather than creating new claim identities, and may cite only evidence already surfaced in the frozen pre-assessments or round 1 rather than introducing a new EvidencePack record. A focus identifies which frozen claim should receive another **opponent review**; it does not let a critic rewrite its own frozen Stage 1 claim inside the debate turn. Any owning-critic revision is made only in the final post-assessment. The final allowed round must `STOP`. After the final round, the Attack and Defense critics independently emit their post-debate `CriticAssessment` values under the same Stage 1 evidence/modality contract **concurrently on the two-slot runtime**; neither post output is included in the other's prompt. Post-assessment citations are limited to Stage 0 evidence already surfaced in either frozen pre-assessment or the completed debate trace, so a pre→post change cannot be caused by a previously unused EvidencePack record appearing only at the post step. `Stage2DebateResult.exchanges` is generated programmatically from the round turns so no critic response can disappear from the final trace, while top-level `final_adjudications` is an exact snapshot of the final Mediator round so one-sided judgments are not lost. Every LLM step is checkpointed with the exact Stage 2 fingerprint plus a hash of its upstream generated dependencies, and the final debate is frozen as a content-addressed `stage2-artifact-v9` artifact. Mediator reliability is intentionally not assumed: Stage 7 keeps independent grounding and Mediator-fidelity audits as internal integrity checks. The paper-facing B comparison holds the frozen Stage 1–3 chain fixed and changes only the Stage 4 lens-evaluation architecture, so it is not a Mediator/no-Mediator ablation.

#### Phase 4: Deterministic Common Decision Objects
Stage 3 first reuses/builds the base `stage3-artifact-v2`, then deterministically freezes `stage3-artifact-v4` with the three predeclared `KR__CIS_IG2`, `EU__CIS_IG2`, and `US__CIS_IG2` scenario objects. No LLM call is added here. Each contextual object contains the same frozen forecast/action, scenario provenance, original forecast evidence, CIS reference-enterprise context evidence, and one deterministic `decision-evidence-retrieval-v5` supplement. The identity binds the decision-source snapshot/hash, action-evidence registry/hash, exact main-guidance selection version/hash, contextual schema, and upstream Stage 3 provenance; older contextual progress/checkpoints cannot silently resume under v5.

Implementation discipline is contract-first: an upstream stage's output contract is frozen/versioned before downstream work may reuse it. Changing forecast/evidence/context/retrieval/prompt/schema/runtime identity creates a new compatible artifact identity; progress metadata never authorizes fallback to an older contract.

#### Phase 5: Contextual Parallel-Lens Evaluation
Stage 4 evaluates each of the three scenario objects with all three information-isolated lenses. A fixed two-worker priority queue shares the two llama.cpp inference slots while preserving information isolation. The main batch itself is **case-major**: a case completes its base Stage 3 → contextual Stage 3 → all nine Stage 4 scenario×lens assessments → immutable `stage4-artifact-v4` before `Pipeline.main` advances to the next case. Thus seeing contextual Stage 3 artifacts only for cases already reached by Stage 4 can be normal under case-major execution.

#### Phase 6: Deterministic Diagnostics and Evidence-grounded Strategic Intelligence
Stage 5 first computes and freezes the deterministic disagreement/evidence diagnostics. Stage 6 then builds one frozen synthesis frame for each KR/EU/US DecisionObject and applies `stage6-decision-policy-v1`. The recommendation is fixed before the model call: a Technical or Institutional/Regional challenge gives `DO_NOT_RECOMMEND`; an unresolved critical gate gives `HOLD`; if both critical gates support the action, Financial/Adoption maps `+1 → RECOMMEND`, `0 → RECOMMEND_PILOT`, `-1 → HOLD`. This recommendation is explicitly interpreted as policy-conditional decision support over three predeclared functional dimensions, not as a statistical conclusion from three sampled agents. The same frozen frame retains the complete Stage 4 lens rationales and claim→evidence links. Qwen then writes the final evidence-grounded `strategic_intelligence_report`, citing the exact upstream evidence IDs that support its substantive explanation, while the deterministic artifact separately preserves the full machine-readable evidence trace. The three scenario jobs are scheduled with two workers, and every completed synthesis is checkpointed before the final case-level `stage6-artifact-v5` is frozen.

### Stage 7 validation harness

`Stage7/` is an executable validation harness over the immutable Stage 1-6 chain; it is not another decision-making Agent. Its evaluation target is the **agent-layer transformation from frozen forecast/evidence inputs to strategic decisions**, not the upstream B-MTGNN forecast itself. `python -m Stage7` runs the deterministic validation core, while `python -m Stage7 --with-llm-audits` additionally runs the grounding and Mediator-fidelity integrity subsets against the same shared llama.cpp runtime with exactly **two concurrent workers / two inference slots**. Semantic verifier output is checkpointed per audit item and never replaces deterministic provenance checks or human/manual audit.

The active validation identity is `stage7-validation-harness-v10` / `stage7-experiment-manifest-v11` / `stage7-validation-artifact-v6`. Paper-facing axes are exactly: **A Robustness** (prompt paraphrase), **B Decision architecture and cost** (information-isolated contextual Stage 4 lenses vs one joint three-lens Stage 4 agent with the frozen Stage 1-3 chain and deterministic downstream policy held fixed, plus policy sensitivity and compute/deployment boundary), and **C Context contribution** (jurisdiction sensitivity, context-free vs contextualized Stage 3/4 decision-path comparison, scenario provenance). The C comparison is not interpreted as a single-factor causal estimate of deployment context because the contextual Stage 3/4 evidence and validation contract also changes. Decision-bearing variants are executed with `python -m Stage7 --run-decision-variants A,B,C`.

Stage 7 deliberately does **not** score external forecast convergence, historical forecast validity, 26-threat forecast generalization, or B-MTGNN backbone quality. Those are upstream forecasting questions rather than agent-layer contributions. `Results/Stage7/` therefore stores only the active three-axis manifests/artifacts, A/B/C variant checkpoints/results, and grounding/Mediator integrity-audit material.

## Data Flow

### Input Data
- **Prediction Data**: Cyber threat and mitigation technology trend predictions
- **Stage 0 Evidence Snapshot**: Frozen external cyber evidence with source/version/hash/cutoff metadata
- **Environment Configuration**: llama.cpp endpoint, fixed Qwen model ID, Stage 0 case settings, etc.

### Active Output Results
- **Log Files**: Stage 1 smoke logs under `Results/Stage1/logs/`; active main Stage 0→1→2→3→4→5→6 run logs are under `Results/Stage6/logs/`
- **Stage 1 Assessments**: Structured Attack Feasibility and Defense Robustness pre-assessments
- **Frozen Stage 1 Artifact**: Content-addressed, hash-validated pre-assessments for exact reuse by later stages
- **Stage 2 Debate Result**: Structured round exchanges, evidence-bounded Mediator adjudications/routing, post-assessments, and stance changes
- **Frozen Stage 2 Artifact**: Content-addressed, hash-validated debate result with exact per-step resumability during generation
- **Stage 3 Decision Bundle**: Deterministic base Common Decision Object plus three contextual DecisionObjects with frozen KR/EU/US CIS-IG2 scenarios and forecast/context/decision evidence partitions
- **Frozen Stage 3 Artifacts**: Baseline `stage3-artifact-v2` plus active contextual `stage3-artifact-v4`, each content-addressed and bound to exact upstream provenance
- **Stage 4 Lens Evaluations**: Structured Technical, Institutional/Regional, and Financial/Adoption assessments for each frozen contextual DecisionObject
- **Frozen Stage 4 Artifacts**: Baseline `stage4-artifact-v1`; active contextual v4 runs freeze `stage4-artifact-v4` with `stage4-lens-checkpoint-v4` exact checkpoint resumability. Older contextual versions are historical diagnostics, not active-contract artifacts.
- **Stage 5 Diagnostics**: Deterministic disagreement/evidence/context-sensitivity diagnostics, including separate jurisdiction stance and evidence sensitivity.
- **Frozen Stage 5 Artifact**: Content-addressed `stage5-artifact-v1` bound to the exact upstream Stage 0/2/contextual-Stage3/contextual-Stage4 chain.
- **Stage 6 Decision Syntheses**: Three disagreement-preserving `DecisionSynthesis` objects per case, one for each frozen KR/EU/US scenario. Each contains the deterministic policy recommendation, exact rule ID, complete Stage 4 lens evidence trace, retained disagreement/evidence limitations, one model-generated strategic intelligence report, and deterministic `report_cited_evidence_ids` extracted from its validated inline `[EVIDENCE:...]` citations.
- **Frozen Stage 6 Artifact**: Content-addressed `stage6-artifact-v5` with exact per-DecisionObject `stage6-synthesis-checkpoint-v5` resumability during the 2-worker synthesis run.
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

## Run Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 end to end

Run from the `Multi-Agent/` directory. `Pipeline.main` now runs the frozen seven-case main set by default.

Build Stage 0 from scratch first. This reruns the paper-native forecast migration and all 48 source-manifest acquisitions, including the Crossref publication-trend API, then freezes the immutable `stage0-2026-09-13-v4` snapshot under the v3 registry contract. It does not augment or reuse a v2 snapshot.

```powershell
cd Multi-Agent
py -3 Stage0/cli.py build-stage0 --snapshot-date 2026-09-13 --snapshot-id stage0-2026-09-13-v4
```

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

Before freezing a changed prompt/schema contract, use a **small random critic smoke sample** rather than repeatedly running all fourteen critic/case combinations:

```powershell
cd Multi-Agent
py -3 -m Pipeline.main --smoke-critics 3 --smoke-seed 202609131
```

`--smoke-seed` controls only selection of the critic/case subset; it does **not** change the frozen LLM inference seed. Smoke sampling is without replacement from the 14 critic/case pairs. Smoke results are stored under `Results/Stage1/smoke/`, do not create/reuse main artifacts or per-critic checkpoints, and are diagnostic development traces rather than main experiment results. The purpose is to catch schema/semantic failures on a small, independently sampled subset and then freeze the contract, not to tune until a preferred stance distribution appears.

The batch is fail-fast but resumable. Exact Stage 1 through Stage 6 final artifacts are reused automatically. Stage 1 checkpoints each critic immediately; Stage 2 checkpoints every debate/post-assessment LLM step immediately; Stage 4 checkpoints every completed DecisionObject/lens assessment immediately; Stage 6 checkpoints every completed DecisionObject synthesis immediately. Each content-addressed artifact/checkpoint is validated against its bound upstream inputs before reuse. Stage 3 and Stage 5 add no LLM checkpoint because both are deterministic. A rerun therefore resumes only compatible completed steps. Changed Stage 0 data, upstream outputs, prompts, model/runtime settings, schema, seed, Stage 3 context, Stage 4 lens contract, Stage 5 diagnostic contract, or Stage 6 synthesis contract never silently falls back to incompatible output.

Execution observability is intentionally verbose. Every case prints `[i/7]`, its selection reason, Stage 0 evidence coverage, artifact/checkpoint reuse status, and completion status. While Qwen is generating, the runtime polls llama.cpp `/slots` every 10 seconds by default and logs elapsed time plus available prompt/generated token counters. Final response usage is logged after each generation. The private reasoning text itself is **not** dumped to the run log; only progress/usage metadata is logged. Change only the logging cadence with `STAGE1_PROGRESS_INTERVAL_SECONDS`.

Main-run logs are appended in real time under `Results/Stage6/logs/`. Machine-readable batch progress is atomically updated after every case under `Results/Stage6/runs/`, so an interrupted run shows exactly which case was running/completed/failed and which Stage 1–6 artifact was produced. Stage 5 artifacts remain under `Results/Stage5/artifacts/`; Stage 6 artifacts are frozen under `Results/Stage6/artifacts/<case_id>/<input_fingerprint>.json`, with resumable per-DecisionObject checkpoints under `Results/Stage6/checkpoints/`.

Inspect the frozen set without inference:

```powershell
py -3 -m Pipeline.main --list-cases
```

The old one-case environment-driven path remains available for debugging/smoke runs only:

```powershell
$env:STAGE0_SNAPSHOT_ID = "stage0-2026-09-13-v4"
$env:STAGE0_THREAT = "DDoS"
$env:STAGE0_PMT = "NLP/LLM"
$env:STAGE0_ANALYSIS_CUTOFF = "2024-12-31"
$env:STAGE0_EVALUATION_MODE = "ex_ante_replay"
$env:STAGE0_CASE_ID = "debug__ddos__nlp_llm"
py -3 -m Pipeline.main --single
```

## Conclusion

The implemented runtime currently stops after immutable Stage 6 disagreement-preserving synthesis. Stage 7 remains a validation harness over the frozen upstream outputs rather than another decision-making Agent in the active runtime chain.
