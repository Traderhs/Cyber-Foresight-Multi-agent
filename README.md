# Context-aware Cyber Foresight with Multi-agent Critique and Strategic Uncertainty

> This repository is organized for reproducible research, with module-specific environments, frozen experiment artifacts, and explicit data/agent contracts.

![Framework Architecture](figure/framework.png)

## Repository Structure

- `Data_Preparation/` and `Dataset/`: source preparation and the integrated forecasting dataset.
- `B-MTGNN/`: frozen forecasting backbone and training/evaluation artifacts.
- `Comparative_Evaluation/`: forecasting baselines and comparative experiments.
- `Multi-Agent/Stage0/`: canonical 124-node forecast migration, immutable evidence snapshots, and deterministic evidence retrieval.
- `Multi-Agent/Stage1/`: independent Attack Feasibility and Defense Robustness critics.
- `Multi-Agent/Stage2/`: evidence-grounded structured debate with bounded Mediator adjudication.
- `Multi-Agent/Stage3/`: deterministic Common Decision Object construction and KR/EU/US contextualization.
- `Multi-Agent/Stage4/`: information-isolated Technical, Institutional/Regional, and Financial/Adoption feasibility lenses.
- `Multi-Agent/Stage5/`: deterministic disagreement, evidence, and context-sensitivity diagnostics.
- `Multi-Agent/Stage6/`: deterministic decision policy plus evidence-grounded strategic-intelligence synthesis.
- `Multi-Agent/Stage7/`: validation harness for robustness, decision architecture/cost, context contribution, and integrity audits; it is not another decision-making Agent.
- `Multi-Agent/Pipeline/`: active Stage 0→6 orchestration. `Multi-Agent/Lagacy/` is archival and is not imported by the active pipeline.

See `Multi-Agent/README.md` for the detailed Agent contracts, artifact versions, runtime profile, and Stage 7 validation design.

## Requirements

We provide two ways to set up the environment:

**1. Unified Installation (Recommended for exploring the whole project)**\
To install all required dependencies across all modules into a single environment, run:
```bash
pip install -r requirements.txt
```

**2. Module-Specific Installation (Recommended for strict reproducibility)**\
Each module has its own `requirements.txt` to prevent potential version mismatches. If you only want to run a specific component:
```bash
# For Data Preparation
pip install -r Data_Preparation/requirements.txt

# For Modeling (B-MTGNN)
pip install -r B-MTGNN/requirements.txt

# For Multi-Agent Framework
pip install -r Multi-Agent/requirements.txt
```

## Dataset

The complete dataset (including raw and normalized data) used in this paper is directly included in this supplementary material package.

- `Dataset/`: Contains the final integrated dataset (`CT-0711-0125.csv`).
- `Data_Preparation/`: Contains intermediate outputs and raw sources.
- `B-MTGNN/data/`: Contains graph adjacency matrices and inputs for the B-MTGNN model.

## Training

To train the core B-MTGNN model on the dataset, navigate to the `B-MTGNN` directory and run the training script:

```bash
cd B-MTGNN
python train.py --data ./data/sm_data.txt --save model/Bayesian/o_model.pt
```

*For hyper-parameter optimization using random search, run:*
```bash
python train_test.py
```

## Evaluation

To evaluate the trained model and compare it against the baseline models, navigate to the `Comparative_Evaluation` directory and execute the respective evaluation scripts.

**Evaluate B-MTGNN (Example with 30 iterations):**
```bash
cd Comparative_Evaluation/BMTGNN
python BMTGNN.py
```

**Evaluate Baseline Models (Example: ARIMA):**
```bash
cd Comparative_Evaluation/Baselines/ARIMA
python ARIMA.py
```

## Pre-trained Models

To facilitate reproducibility without requiring full training, we provide pre-trained model checkpoints:

- **B-MTGNN Checkpoints**: Located in `Comparative_Evaluation/BMTGNN/` (e.g., `modelb10.pt`, `modelb30.pt`).
- **MTGNN Checkpoint**: Located in `Comparative_Evaluation/MTGNN/modelb1.pt`.

## Results

The current paper-contract migration exposes 124 canonical forecast node states (26 Threat + 98 PMT) over the 36-month 2025–2027 horizon for downstream Agent analysis.

## Multi-Agent System

Building upon the existing B-MTGNN experiment artifacts, the redesigned **LangGraph** decision pipeline implements Stage 0 through constrained Stage 6 synthesis, with Stage 7 maintained separately as the validation harness over frozen upstream outputs. It does not retrain B-MTGNN or silently invoke the legacy Attacker→Defender→Mediator chain.

### Architectural separation: forecast validation is not decision evaluation

The pipeline deliberately separates two different questions that can otherwise be conflated in a multi-agent design.

**Stages 1–2 are an epistemic/adversarial validation layer.** The Attack Feasibility Critic and Defense Robustness Critic do not recommend actions. They test whether the frozen Threat/PMT forecast interpretation is actually supported by modality-matched evidence, surface contradictory or only-contextual evidence, and preserve unresolved questions. The Stage 2 Mediator is likewise an evidence-bounded adjudicator, not a policy judge: it resolves or revises claim/evidence interpretations within the already-surfaced frozen evidence boundary and cannot introduce a new action, cost assumption, legal conclusion, or policy recommendation. Its purpose is to improve the fidelity of the **state of knowledge** that is handed downstream: what the forecast says, what external evidence supports or challenges, what is merely context, and what remains unresolved.

**Stages 4–6 are a decision-evaluation layer.** After Stage 3 freezes the candidate action, validated forecast provenance, deployment context, and decision-evidence universe, the Technical, Institutional/Regional, and Financial/Adoption lenses ask a different question: given that frozen state of knowledge, is the candidate strategically feasible under each predeclared functional dimension? They do not re-decide whether the B-MTGNN forecast is true. Stage 6 then applies the predeclared deterministic decision policy to those feasibility assessments and uses the final LLM only to explain the already-fixed recommendation.

**Stage 3 is the deterministic information firewall between the two layers.** Forecast critique is completed and frozen before deployment-cost, staffing, procurement, regulatory-burden, or successful-deployment evidence is introduced for Stage 4. This ordering prevents downstream feasibility information from retrospectively anchoring or rewriting upstream forecast validation. Conversely, Stage 1–2 reasoning provenance is carried forward so Stage 4 cannot silently discard forecast contradictions or unresolved evidence gaps. The design therefore follows `forecast → adversarial validation → evidence adjudication → frozen decision object → feasibility evaluation → deterministic recommendation → evidence-grounded explanation`, rather than treating all six Agents as interchangeable voters.

This separation is also the intended interpretation in the paper: Stage 1–2 contribution is **claim/evidence calibration and forecast-interpretation validation**, whereas Stage 4–6 contribution is **policy-conditional strategic decision support**. A lack of Stage 1/2 stance reversal therefore does not make those stages redundant; their value can appear as scope correction, rejected proxy evidence, preserved unresolved questions, and cleaner downstream provenance. Stage 7 evaluates prompt robustness, compares the frozen information-isolated Stage 4 three-lens path with a joint three-lens Stage 4 single-agent baseline while holding Stage 1–3 and the deterministic downstream policy fixed, compares the context-free Stage 3/4 path with the bundled contextualized Stage 3/4 path, and keeps grounding/Mediator checks as internal integrity audits. It does not claim a separate no-Mediator/no-debate effect from these experiments.

**Stage 0 — Canonical Forecast + Immutable Evidence**
- Migrates the existing experiment outputs into the paper-defined 124-node contract (26 Threat + 98 PMT).
- Builds case-specific `EvidencePack` objects from a frozen Evidence Store under `Multi-Agent/Results/Stage0/Evidence/snapshots/`.
- Enforces `available_at <= analysis_cutoff_date` for temporal filtering.
- Uses deterministic Threat/PMT + claim-contract eligibility, followed by fixed-query Okapi BM25 ranking within six semantic evidence slots and bounded source-diverse top-k selection.
- Uses the strict `stage0-registry-v3` / `stage0-source-manifest-v3` / `stage0-ingestion-v4` contract. Registry source `33` adds a Crossref API publication-trend axis that is independent of the Scopus-derived B-MTGNN `NoP` input: all 98 PMTs are queried under the same cutoff-safe rule, while zero-result queries remain audit-only rather than being mislabeled as flat directional evidence. Static Crossref DOI/publication metadata cannot substitute for this directional evidence.
- Builds Stage 0 from scratch with `py -3 Stage0/cli.py build-stage0 --snapshot-date 2026-09-13 --snapshot-id stage0-2026-09-13-v4`; pre-v3 registry snapshots are intentionally incompatible rather than silently upgraded or augmented.
- Never inserts Agent reasoning back into the external Evidence Store.

**Stage 1 — Independent Forecast Critique**
- Runs an **Attack Feasibility Critic** and a **Defense Robustness Critic** independently on the same Forecast + EvidencePack.
- Uses one xhigh inference per Critic with a fixed nine-step internal review order: exhaustive evidence inventory → forecast decomposition → supporting-evidence mapping → challenging-evidence mapping → cross-source conflict/alternative explanations → modality/causal-boundary audit → temporal consistency → claim-polarity/specificity/traceability audit → explicit evidence-balance decision. Stage 1 does not chain multiple self-refinement calls before the later inter-Critic debate.
- Returns structured `CriticAssessment` objects with stance, evidence sufficiency, non-empty claim→evidence links, an explicit `forecast_component`, `forecast_relation`, a structured `stance_basis`, unresolved questions, and unsupported-specificity flags. Final claims are affirmatively grounded factual findings; operational interpretation is kept separate from direct evidence about the forecast-state modality. `NoI` is treated literally as incident count/frequency, so attack magnitude, severity, tactic prevalence, malware-family share, or intrusion share cannot substitute for a directional NoI observation. `NoP` direction likewise requires explicit publication-activity trend evidence. `stance=0` is not the default for incomplete evidence: it is reserved for genuinely balanced directional evidence or the absence of role-relevant directional evidence.
- Stage 0 gap labels describe the signed quantity `Threat state z - PMT state z`: `threat_minus_pmt_increasing`, `threat_minus_pmt_decreasing`, or `flat`. They are not absolute-distance widening/narrowing labels.
- Exact evidence IDs are never reconstructed from a digest/prefix/similarity heuristic. Unknown IDs and other cross-field semantic violations receive one fixed repair turn; the corrected output must select an exact supplied Stage 0 ID that supports the literal claim, otherwise the claim must be rewritten/removed. A second failure aborts the Critic.
- During prompt/schema development, `py -3 -m Pipeline.main --smoke-critics N --smoke-seed <selection-seed>` samples a small random subset of the 14 case/critic pairs without creating or reusing main artifacts. Smoke outputs are diagnostic only and are kept separate from the frozen main experiment to reduce case-specific tuning pressure.
- Freezes the paired pre-assessments as immutable content-addressed artifacts so an exact experiment identity is generated once and reused downstream.
- The main Agent experiment uses a frozen seven-case set selected before inference: two unconditional anchors plus one evidence-qualified representative from each of five gap/slope regimes. Regime cases require at least 5/6 evidence slots and four source families; the full 303 relations remain the quantitative forecast/gap population.
- `py -3 -m Pipeline.main` runs the seven cases as one resumable Stage 0→1→2→base-Stage3→contextual-Stage3-v4→contextual-Stage4-v4→Stage5→Stage6 batch. Exact Stage 1–6 artifacts and compatible Stage 1/2/Stage4-v4/Stage6 checkpoints are reused automatically; both Stage 3 layers and Stage 5 are deterministic and add no LLM call.
- Main orchestration is **case-major**, not stage-major: one case is carried end-to-end through base Stage 3, contextual Stage 3, and all `3 jurisdictions × 3 lenses = 9` contextual Stage 4 assessments before the next case begins. Stage 3 artifacts therefore appear incrementally as earlier cases finish Stage 4; the runtime does not materialize all seven Stage 3 cases behind a global barrier first.

**Stage 2 — Evidence-grounded Structured Debate**
- Reuses the exact frozen Stage 1 pre-assessments; Stage 1 is not regenerated inside Stage 2.
- Each critic reviews every opponent claim using only the same frozen Stage 0 evidence and one of `AGREE`, `CHALLENGE`, `REVISE`, or `INSUFFICIENT_EVIDENCE`.
- Uses an **evidence-bounded Mediator adjudicator** that compares the surfaced frozen evidence behind the critics' claims, can accept/reject/revise interpretations, mark disputes resolved/unresolved, and decide whether one focused second round is useful. The Mediator sees only Stage 0 evidence records already surfaced in the frozen pre-assessments or completed/current debate trace; it cannot retrieve new evidence, invent facts, emit residual-risk percentages, claim predictive/causal truth, or make policy recommendations.
- Runs one debate round by default; round 2 runs only when the round-1 Mediator emits structured `CONTINUE` with exact claim-level `next_round_focus`. Two rounds is the hard maximum and the final round must `STOP`.
- Python validates claim/evidence IDs, surfaced-evidence boundaries, Mediator internal consistency, exact routing schema, and the two-round hard cap, but does not replace the Mediator's substantive adjudication with a hard-coded winner rule. Stage 7 keeps grounding and Mediator-fidelity checks as independent integrity audits. Its B architecture comparison keeps Stage 1–3 fixed and changes the Stage 4 lens-evaluation architecture only, so it is not evidence for the independent necessity of the Mediator.
- Produces independent Attack/Defense post-assessments under the same Stage 1 `CriticAssessment` contract, preserving the original pre-assessments and recording stance changes.
- Uses the exact same pinned Qwen3.8/llama.cpp runtime, sampling parameters, seed, reasoning effort, and semantic-repair policy as Stage 1.
- Freezes content-addressed `stage2-artifact-v9` outputs and per-step checkpoints under `Multi-Agent/Results/Stage2/`; each step checkpoint binds both the Stage 2 experiment fingerprint and the exact upstream generated dependency/schema hash, so interrupted runs resume only compatible completed LLM steps.

**Stage 3 — Common Decision Object**
- Adds no LLM judgment. It verifies the exact frozen Stage 2 artifact and programmatically freezes the common evaluation unit later Technical / Institutional-Regional / Financial-Adoption lenses must share.
- Revalidates the frozen Stage 1→2 artifact chain before building Stage 3, rather than trusting a self-consistent file hash alone.
- Carries the forecast interpretation, Attack/Defense post-stances, final Mediator adjudications, matched resolved/unresolved claims, required revisions, both critics' post-debate unresolved questions, Mediator evidence gaps, predictive uncertainty, timing window, and explicitly unknown deployment-context fields. It also freezes `evaluation_evidence_ids` to the complete selected Stage 0 EvidencePack so later lenses start from the same evidence universe.
- Uses the fixed main selection rule `case-forecast-pmt-only-v1`: the current B-MTGNN case PMT is the sole main candidate action. Other PMTs co-tagged by Stage 0 evidence are recorded as excluded rather than being selected after observing downstream results; alternate action-set rules are reserved for explicit sensitivity analysis.
- Freezes a content-addressed `stage3-artifact-v2` under `Multi-Agent/Results/Stage3/`, bound to the exact Stage 2 artifact/output, frozen Stage 0 EvidencePack, selection rule/output schema, and Stage 3 decision context including deployment fields.
- The active contextual supplement freezes `stage3-artifact-v4`: three predeclared `KR__CIS_IG2`, `EU__CIS_IG2`, and `US__CIS_IG2` objects sharing the same forecast/action while adding a hash-pinned CIS IG2 reference-enterprise context and deterministic Stage-4 decision evidence. The original `stage3-artifact-v2` object remains the context-free baseline. Because the context-free and contextual paths use different Stage 3/4 input and validation contracts, Stage 7 treats their comparison as a contextualized-decision-path contribution comparison rather than a pure single-factor causal estimate of deployment context. Stage-4 decision evidence combines the unchanged `stage0-2026-09-13-v4` store with a separate cutoff-frozen PMT-specific decision-guidance snapshot (`stage3-decision-sources-2024-12-31-v3`) containing 18 official guidance documents / 2,187 records. Each main PMT has at least two independent publishing organizations (NIST + NCSC UK for NLP/LLM; NIST + CISA/DHS for the other main PMTs), so Stage 1/2 evidence is not mutated and Stage 4 is not sourced from one institution only.
- Stage-4-specific decision evidence is deliberately injected only after action/context freeze: exposing cost, staffing, regulatory burden, procurement difficulty, or successful-deployment evidence during Stage 1–2 could anchor the upstream forecast critique on downstream feasibility. The contract is frozen before Stage 4 results, is applied uniformly across all six unique main PMTs, and all three lenses receive the same combined evidence universe.
- The active decision contract is now **`decision-evidence-retrieval-v5`**. It separates (1) a hash-pinned 14-record `stage3-action-evidence-registry-v1` containing one source-grounded direct support and one direct limitation finding for each of the seven frozen Threat×PMT actions, from (2) a hash-pinned **exact evidence-ID main guidance selection** `stage3-main-guidance-selection-v1`. Technical +/− direction must come from the direct Threat×PMT action layer; generic PMT guidance cannot manufacture Technical direction. Main Financial/Institutional/deployment guidance is no longer chosen by BM25 at run time: 46 audited frozen evidence IDs are predeclared with exact slot assignments. BM25/semantic retrieval remains available only as fallback/ablation infrastructure.
- v5 deliberately **does not force equal positive/negative evidence counts**. The old v4 2-document/2-publisher-per-facet quota produced a numerically balanced but semantically noisy evidence universe and the compact KR diagnostic collapsed to `[0,0,0]`. That output is retained only as a pseudo-balance diagnostic, not a main result. v5 instead keeps the same selection rules across all cases while allowing the audited source distribution to be asymmetric (for example Access Control Financial support:challenge = 1:3). Main preflight still requires ≥6 generic records, ≥3 generic documents, and ≥3 documents from ≥2 publishers per lens; Technical additionally requires at least one exact action-specific support record and one exact action-specific challenge record. Generic evidence is identical across KR/EU/US; only `regulatory_applicability` may vary by jurisdiction.
- The active v5 no-inference audit passed all **7 cases × 3 scenarios = 21/21** units under contract tag `de-83662ff07d`, and the actual frozen Stage 0 + frozen Stage 2 → base Stage 3 → contextual Stage 3-v4 chain rebuilt and validated all 21 objects without LLM inference. Financial support:challenge counts on the seven KR objects are `2:2, 2:1, 2:1, 2:1, 3:1, 1:3, 2:1`; this asymmetry is an observed input property, not a target. Contextual Stage 4 now uses `stage4-semantic-validation-v6`: neutral PARTIAL/SUFFICIENT assessments must expose both grounded support and challenge, and identical lens-relevant evidence may not yield different GENERAL stances merely because the scenario label changes between KR/EU/US. A jurisdiction label is context, not evidence; a region-driven stance change requires exact lens-specific jurisdiction evidence. The current Stage 0–5 regression suite is **206/206 PASS** (Stage 0 55, Stage 1 39, Stage 2 47, Stage 3 10, Stage 4 50, Stage 5 5).

**Stage 4 — Parallel Strategic Value-Lens Evaluation**
- The active contextual v4 path fans each frozen scenario-conditioned DecisionObject and its exact same forecast/context/decision evidence union out to three information-isolated lenses: Technical Feasibility, Institutional/Regional Feasibility, and Financial/Adoption Feasibility. The earlier context-free v1 graph remains available as a baseline reproduction path. Contextual v2/v3 outputs are historical diagnostics only and are not part of the active experiment identity.
- Uses one shared structured `LensAssessment` contract with `stance {-1,0,+1}`, evidence sufficiency, rationale, evidence-grounded claims, `GENERAL` vs `DEPLOYMENT_SPECIFIC` scope, constraints, evidence gaps, and conditional requirements. Uncalibrated self-reported confidence is not used.
- Missing infrastructure/region/organization/budget context cannot be guessed into deployment-specific claims. Unsupported vendor/version/ROI/budget/timeline specificity is explicitly forbidden by the lens prompts and later grounding audit.
- Stage 2 adjudications remain reasoning provenance rather than external evidence; factual Stage 4 claims must cite exact IDs from the frozen `evaluation_evidence` union. `CTX_*` evidence can support only standardized reference-enterprise facts, not PMT effectiveness, cost, or legal applicability.
- The LLM-facing contextual Stage 4 payload is a **lossless judgment projection**, not a serialization of the entire frozen Stage 3 artifact. The full Stage 3 artifact still retains every evidence record and provenance field, but the prompt contains each selected evidence body exactly once, omits duplicated `DecisionEvidencePack` / `evaluation_evidence_records` copies and validator-only hashes/snapshot/claim-policy metadata, exposes only the assigned lens's grounding contract, and uses compact JSON. The shared evidence universe itself is unchanged. On the available frozen real artifacts this reduced serialized payload size by about **75%**: DDoS×NLP/LLM from 201,561 to about 50.4k characters and Malware×Anomaly Detection from 304,777 to about 76.1k characters, before chat-template tokenization.
- Every directional feasibility claim (`SUPPORTS_FEASIBILITY` / `CHALLENGES_FEASIBILITY`) must cite a lens-, scope-, **and direction-eligible** decision-evidence slot. Technical support requires `technical_enablement`, while challenges require `technical_limitation`; Financial support requires `adoption_benefit`, while challenges require `adoption_burden`. `deployment_maturity` is supplementary context and cannot create direction by itself. Institutional support requires `governance_enablement`, while challenges require `compliance_constraint`; a deployment-specific legal direction additionally requires jurisdiction-specific `regulatory_applicability`. Applicability alone is neither permission nor prohibition. Forecast/context evidence alone cannot manufacture +1/-1, and an opposite-polarity facet cannot be cited to reverse its grounded direction.
- Contextual v4 exposes this direction-aware contract in the frozen payload and enforces it conservatively. A directional claim backed only by forecast/context, an ineligible slot, or the wrong polarity is downgraded to `NEUTRAL_CONTEXT`, never supplied with a replacement citation. If a nonzero stance loses its directional basis, it may be reduced to 0; deterministic validation never raises 0 to +1/-1. If one-sided grounded claims survive while the model emits 0, the fixed one-turn repair must explicitly choose the matching direction or reclassify the claim as neutral. A grounded `GENERAL` direction with only narrower deployment-detail gaps uses `PARTIAL` rather than `INSUFFICIENT_CONTEXT`. The context-free v1 contract remains unchanged.
- The three lens nodes are sibling LangGraph branches. Contextual Stage 4 now allows **two concurrent llama.cpp inference slots**, while preserving information isolation because no sibling lens output is included in another lens prompt.
- Contextual Stage 4 now uses a dedicated **131,072-token aggregate runtime split into two 65,536-token slots**, `stage4-context-runtime-128k-total-2slot-v3`. A pre-optimization Financial/Adoption request had reached **142,549 prompt tokens**, but the subsequent lossless prompt projection removed redundant serialization and cut representative payload size by about **75%** while preserving the complete selected evidence universe, Stage 2 provenance, forecast, and deployment context. The active experiment keeps the same 128K aggregate KV budget rather than paying the 256K prefill/VRAM cost, while permitting two concurrent lens generations. The Stage 4 runtime profile/version, total context, per-slot context, and parallel-slot count are bound into the contextual Stage 4 artifact/checkpoint fingerprint.
- Context-free baseline generation uses `stage4-artifact-v1`. The active contextual revision uses scenario×lens `stage4-lens-checkpoint-v4` checkpoints and freezes a content-addressed `stage4-artifact-v4` after all nine assessments for the case validate. Earlier contextual v2/v3 contracts remain historical comparison points, not active results.
- Contextual v4 is a new explicit experiment identity rather than an in-place mutation. Decision-source snapshot/hash, retrieval version, prompt hashes, payload hashes, semantic-validation version, artifact/checkpoint schema versions, and the versioned batch progress path prevent any v2/v3 contextual checkpoint from being silently reused as v4.
- Contextual reporting keeps two effects separate: primary `D_lens` is computed only within one frozen jurisdiction scenario, while KR/EU/US variation is reported separately as jurisdiction sensitivity. Stage 7 compares the frozen context-free Stage 3/4 path with the KR/EU/US contextualized Stage 3/4 path and reports that comparison as a bundled contextualization-path contribution, not as a context-only causal effect; neither effect is collapsed into the primary disagreement metric.

**Stage 5 — Deterministic Disagreement and Evidence Diagnostics**
- Adds no LLM judgment and never changes a Stage 4 stance. It deterministically computes same-scenario `D_lens`, evaluability/joint-abstention, directional-lens counts, Stage 2 pre/post Critic dispersion, and system evidence diagnostics from the exact frozen upstream artifacts.
- Treats `D_lens` only as **decision-lens dispersion**, not Bayesian/predictive uncertainty, correctness probability, or calibrated confidence.
- Keeps jurisdiction effects separate from lens effects. `jurisdiction_stance_sensitivity` records whether a lens's `{-1,0,+1}` direction changes across KR/EU/US, while `jurisdiction_evidence_sensitivity` can detect unchanged stance with changed lens-relevant evidence, cited directional evidence, claim scope/signature, or Institutional regulatory-applicability grounding.
- Does not pretend exact-ID validation is a semantic grounding score. Exact claim→evidence-reference integrity is recorded deterministically, while claim↔evidence semantic-support auditing remains an independent Stage 7 integrity check.
- Freezes content-addressed `stage5-artifact-v1` outputs under `Multi-Agent/Results/Stage5/artifacts/`, bound to the exact Stage 0 EvidencePack and Stage 2/contextual-Stage3/contextual-Stage4 artifact/output hashes.

**Stage 6 — Disagreement-preserving Decision Synthesis**
- Does not use a final LLM judge or majority vote, but it **does produce a policy-conditional action recommendation**. The three Stage 4 lenses are not treated as a statistical sample, committee, or population of stakeholders; they are three predeclared functional decision dimensions. `stage6-decision-policy-v1` treats Technical and Institutional/Regional as critical feasibility gates and Financial/Adoption as the adoption/scale gate. The deterministic recommendation is one of `RECOMMEND`, `RECOMMEND_PILOT`, `HOLD`, or `DO_NOT_RECOMMEND`, with an exact `decision_rule_id` recorded in the artifact.
- A Technical or Institutional challenge yields `DO_NOT_RECOMMEND`; an unresolved critical gate yields `HOLD`; when both critical gates support the action, Financial/Adoption maps `+1 → RECOMMEND`, `0 → RECOMMEND_PILOT`, and `-1 → HOLD`. `D_lens` remains a diagnostic and is not used as a vote or score. `PARTIAL` evidence qualifies scope rather than acting as an additional negative vote.
- The artifact explicitly records `recommendation_interpretation=POLICY_CONDITIONAL_DECISION_SUPPORT`, `lens_set_role=PREDECLARED_FUNCTIONAL_DIMENSIONS_NOT_STATISTICAL_SAMPLE`, and `d_lens_interpretation=DESCRIPTIVE_WITHIN_SET_DISPERSION_NOT_POPULATION_UNCERTAINTY`. Thus `RECOMMEND` means “recommended under the frozen decision policy for the frozen candidate action/scenario,” not “three agents statistically established the universally correct action.”
- Stage 6 is the human-facing **evidence-grounded strategic intelligence report** layer, not a one-paragraph recommendation explainer. The deterministic output now preserves the complete Stage 4 `lens_evidence_trace` (stance, sufficiency, rationale, evidence-backed claims, constraints, gaps, and conditions). Qwen receives that frozen trace and writes only `strategic_intelligence_report`, citing substantive evidence inline as `[EVIDENCE:<exact evidence_id>]`. Python then extracts every inline citation and accepts it only if it belongs to the frozen Stage 3 `evaluation_evidence_ids` universe, which is the exact forecast/context/decision-evidence union. Lens-direction justification remains stricter and must be backed by the corresponding frozen Stage 4 claim evidence. Python deterministically stores the unique validated citation set as `report_cited_evidence_ids`. This restores the original cyber-paper goal of evidence-grounded user-level explanation and the inter-firm paper's final strategic-intelligence-report philosophy without giving the final LLM authority to change the decision.
- Qwen cannot change the policy outcome, replace the candidate action, turn `D_lens` into calibrated uncertainty, or turn the strategic recommendation into a guaranteed immediate production deployment claim. Active `stage6-semantic-validation-v12` separates citation existence from decision-lens grounding: any inline citation must belong to the frozen upstream `evaluation_evidence_ids` universe, while each lens's decisive support/challenge coverage must come from that lens's frozen Stage 4 claim evidence. Numeric claims are provenance-bounded: a number must already occur in the prompt-visible synthesis frame or in an exact evidence record cited inline by the report; numbers found only in uncited evidence do not become available. The canonical citation form is `[EVIDENCE:<exact evidence_id>]`; grouped explicit markers such as `[EVIDENCE:E1, EVIDENCE:E2]` are parsed as the same exact citations, while malformed evidence brackets are rejected. The v12 fidelity contract also preserves the asymmetric policy cause: a single challenged critical gate may be decisive under the frozen policy without being described as universally decisive outside that policy.
- Runs one synthesis per contextual DecisionObject, so the main experiment contains `7 × 3 = 21` Stage 6 calls rather than 63 lens-level calls. A fixed **two-worker** queue shares the same `131072` aggregate / `2 × 65536` llama.cpp runtime.
- Freezes resumable `stage6-synthesis-checkpoint-v5` outputs and immutable `stage6-artifact-v5` artifacts bound to the exact Stage 3/4/5 upstream hashes, decision-policy-bearing frame hashes, prompt/schema hashes, and `stage6-synthesis-runtime-128k-total-2slot-v1` runtime identity.
- Stage 6 fidelity is structural; semantic claim↔source correctness remains independently audited in Stage 7 rather than being self-certified by the synthesizer.

**Local Language Model Runtime**
- Model: **Qwen3.8-27B Q6_K_L** GGUF.
- Runtime is unified on the same pinned **llama.cpp** CUDA build and the single shared launcher `Multi-Agent/Stage1/start_qwen38.ps1`. The active server always uses **131,072 aggregate tokens split into 2 × 65,536-token slots**, and Stage 1, Stage 2, contextual Stage 4, and Stage 6 may all issue up to two independent requests concurrently. Already-frozen historical Stage 1/2 one-slot artifacts remain explicitly reusable through a narrow concurrency-only compatibility path, so the throughput migration does not force upstream re-inference. Full model GPU offload, Q8 target/draft KV, `batch=2048`, `ubatch=512`, and model-native thinking at `xhigh` reasoning effort remain shared.
- Speculative decoding is frozen only after a pre-experiment calibration that compares a non-speculative baseline with bounded MTP candidates and rejects candidates that fail a deterministic output-fingerprint correctness gate. The current selected profile is MTP depth 4 with `p_min=0.05`.
- Main-run sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 0.0, repetition penalty 1.0, seed 42.

The previous mutable **LightRAG + Ollama** implementation is retained only under `Multi-Agent/Lagacy/` for audit. In that legacy flow, each Agent issued its own hybrid RAG query and generated Agent analyses were inserted back into the same RAG storage; the active Stage 0/1/2/3/4/5/6 pipeline deliberately does not use that mechanism.

---

## Detailed Project Structure

For a deeper dive into the individual components of our framework, please refer to the documentation within each directory:

*   **[`PT_Extractor`](./PT_Extractor)**: Scripts for extracting Pertinent Technologies (PTs) using E-GPT and D-GPT, forming the Threats and Pertinent Technologies (TPT) graph.
*   **[`Data_Preparation`](./Data_Preparation)**: Scripts for extracting time-series features (NoI, A_NoM, PT_NoM, ACA, PH).
*   **[`B-MTGNN`](./B-MTGNN)**: The core implementation of the Bayesian Graph Neural Network, including data smoothing and future forecasting scripts (`forecast.py`, `pt_plots.py`).
*   **[`Comparative_Evaluation`](./Comparative_Evaluation)**: Extensive evaluation logic against baseline models.
*   **[`Multi-Agent`](./Multi-Agent)**: Stage-separated LangGraph implementation. `Stage0/` owns forecast/evidence preparation and retrieval, `Stage1/` owns the shared pinned Qwen runtime plus independent critique/frozen artifacts, `Stage2/` owns structured debate/post-assessment contracts, `Stage3/` owns deterministic Common Decision Object construction/freezing, `Stage4/` owns information-isolated strategic value-lens evaluation, `Stage5/` owns deterministic disagreement/evidence/context-sensitivity diagnostics, `Stage6/` owns policy-conditional decision synthesis and the final evidence-grounded strategic intelligence report, `Stage7/` owns the separate validation harness over frozen Agent-layer outputs, `Pipeline/` owns Stage 0→6 orchestration, and `Lagacy/` is archival only.
