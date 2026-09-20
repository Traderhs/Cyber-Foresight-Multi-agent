# Stage 7 — Agent-Layer Validation Harness

Stage 7 validates the agent layer over immutable Stage 1–6 outputs. The upstream B-MTGNN forecast/backbone is treated as a frozen input and is **outside the Stage 7 contribution boundary**. Stage 7 therefore does not score forecast accuracy, external threat convergence, temporal forecast validity, or backbone quality.

## Final paper-facing design

Stage 7 reports exactly three validation axes:

- **A — Robustness**: compare the frozen main result with two semantically equivalent prompt paraphrases on the fixed representative three-case subset.
- **B — Decision architecture and cost**: compare the active information-isolated contextual Stage 4 three-lens evaluation with one joint three-lens Stage 4 agent while holding the frozen Stage 1-3 chain and deterministic downstream policy fixed; report decision-policy dependence, available compute/runtime evidence, and the deployment-claim boundary separately.
- **C — Context contribution**: report observed jurisdiction sensitivity and compare the context-free Stage 3/4 path with the contextualized decision path while preserving scenario provenance. Because this comparison changes the contextual Stage 3/4 evidence and validation contract together with deployment context, it is reported as a contextualized-path contribution comparison rather than a single-factor causal estimate of deployment context.

Grounding and Mediator-fidelity semantic checks remain internal integrity audits. They verify agent-layer claim/evidence behavior and are not promoted to independent paper experiments.

### Statistical reporting

Paper-facing confidence intervals for the A prompt-paraphrase, B architecture, and C contextualized-path comparison means use a **case-cluster bootstrap** rather than resampling the 18/21 scenario-comparison rows as independent observations. The bootstrap resamples case_id clusters with replacement and carries all jurisdiction/variant rows from each sampled case together. This preserves the point estimates while accounting for the dependence among KR/EU/US scenarios derived from the same Threat-PMT case. The frozen statistical contract uses 2,000 repetitions, seed 20260916, and a 95% interval.

## Commands

Run from `Multi-Agent/`:

```powershell
py -3 -m Stage7
py -3 -m Stage7 --with-llm-audits
py -3 -m Stage7 --run-decision-variants A,B,C
```

The expensive variant families execute sequentially, while each family uses exactly two workers against the existing two-slot llama.cpp runtime. All Stage 7 outputs are written under `Results/Stage7/`; Stage 1–6 artifacts are never overwritten.

The existing `--run-decision-variants B` command also performs the architecture seed-robustness check. Seed 42 reuses the frozen main isolated output and frozen joint baseline. Only seeds 43 and 44 generate new inference, and both isolated and joint jobs share the same global two-request limit so the existing two-slot parallelism is preserved. Seed-specific outputs are stored under the separate `B_SEED_ROBUSTNESS` axis and never overwrite the original B baseline.

The completed A prompt-robustness artifacts retain their frozen `stage7-decision-variant-runner-v1` identity. The B joint-three-lens baseline and C contextualized-path comparison use `stage7-decision-variant-runner-v3`. The B single-agent prompt explicitly preserves Stage 4's GENERAL-versus-DEPLOYMENT_SPECIFIC grounding contract so a contextual scenario label does not by itself promote a general evidence claim to deployment-specific direction.

## Validation boundary

- The Stage 0 forecast and B-MTGNN backbone are frozen upstream inputs, not contributions evaluated by Stage 7.
- Stage 7 does not claim external forecast accuracy, model-unseen temporal forecasting validity, 26-threat forecast generalization, or backbone superiority.
- The evaluation target is the transformation from frozen forecast/evidence inputs into evidence-bounded, context-aware strategic decisions.
- `D_lens` remains descriptive within-scenario lens dispersion only; it is not predictive uncertainty.
- Grounding and Mediator semantic verifiers are auxiliary integrity checks, not human ground truth.
- The contextual experiment uses the frozen KR/EU/US scenarios under one CIS IG2 reference-enterprise profile; it does not establish worldwide or organization-level field validity.
- Stage 7 never collapses the three axes into a single validity score.

## Reproducibility

Variant artifacts are content-addressed and bound to exact source Stage 3/Stage 6 hashes, prompt hashes, structured-output schemas, runtime configuration, and runner identity. Completed LLM assessments are preserved byte-for-byte at the assessment level whenever only axis/path/statistical-reporting metadata is re-frozen; assessment hashes remain the integrity anchor.
