from __future__ import annotations

from typing import Any


STAGE7_HARNESS_VERSION = "stage7-validation-harness-v12"
STAGE7_ARTIFACT_SCHEMA_VERSION = "stage7-validation-artifact-v6"
STAGE7_EXPERIMENT_MANIFEST_VERSION = "stage7-experiment-manifest-v13"
STAGE7_GROUNDING_VERIFIER_VERSION = "stage7-grounding-verifier-v1"
STAGE7_MEDIATOR_VERIFIER_VERSION = "stage7-mediator-verifier-v1"
STAGE7_VARIANT_RECORD_VERSION = "stage7-variant-result-v1"
STAGE7_INPUT_CONTRACT_VERSION = "stage7-validation-inputs-v2"
STAGE7_STATISTICAL_CONTRACT_VERSION = "stage7-statistical-reporting-v2"
STAGE7_MANUAL_AUDIT_SAMPLE_VERSION = "stage7-manual-audit-sample-v1"
STAGE7_MEDIATOR_AUDIT_SAMPLE_VERSION = "stage7-mediator-audit-sample-v1"

EXPECTED_MAIN_CASE_COUNT = 7
EXPECTED_MAIN_SCENARIO_COUNT = 21
ARCHITECTURE_ROBUSTNESS_SEEDS = (42, 43, 44)

MANUAL_AUDIT_SELECTION_SEED = 20260916
MANUAL_AUDIT_PER_STRATUM = 2
MEDIATOR_AUDIT_SELECTION_SEED = 20260916
MEDIATOR_AUDIT_PER_CASE_OUTCOME_STRATUM = 1
MEDIATOR_AUDIT_INCLUDE_ALL_FOLLOWUP_ROUNDS = True

# Representative robustness subset frozen by design role, not by observed outcome.
SENSITIVITY_CASE_IDS = (
    "main__ddos__nlp_llm",
    "main__malware__anomaly_detection",
    "main__session_hijacking__https",
)

PROMPT_PARAPHRASE_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "P1_CANONICAL",
        "description": "Canonical main prompt wording used by the frozen Stage 1-6 run.",
    },
    {
        "variant_id": "P2_EVIDENCE_FIRST",
        "description": (
            "Semantically equivalent wording that asks for evidence inventory and boundaries "
            "before role judgment while preserving the same output contract and evidence access."
        ),
    },
    {
        "variant_id": "P3_DECISION_FIRST",
        "description": (
            "Semantically equivalent wording that states the role decision question first and "
            "then the same evidence, modality, temporal, and provenance constraints."
        ),
    },
)

DECISION_POLICY_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "policy_id": "MAIN_ROLE_BASED_GATE_V1",
        "description": (
            "Technical and Institutional are critical feasibility gates; Financial is the adoption gate."
        ),
    },
    {
        "policy_id": "ALT_STRICT_ADOPTION_V1",
        "description": (
            "Critical gates retain the main semantics, but unresolved Financial/Adoption yields HOLD "
            "instead of RECOMMEND_PILOT."
        ),
    },
    {
        "policy_id": "ALT_PERMISSIVE_PILOT_V1",
        "description": (
            "If both critical gates are supported, Financial/Adoption challenge or unresolved status "
            "is limited to RECOMMEND_PILOT rather than HOLD."
        ),
    },
)

DECISION_ARCHITECTURE_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "variant_id": "FULL_REDESIGNED_PIPELINE",
        "description": "Current contextual Stage 4 information-isolated three-lens evaluation with the frozen upstream Stage 1-3 chain and deterministic downstream policy held fixed.",
    },
    {
        "variant_id": "SINGLE_AGENT_BASELINE",
        "description": "Joint three-lens Stage 4 single-agent baseline over the same frozen contextual DecisionObjects/evidence and the same deterministic downstream policy.",
    },
)

# Final paper-facing scope: Stage 7 evaluates the agent layer only. The frozen
# forecast/backbone is an upstream input and is not re-validated here.
STAGE7_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"id": "A", "name": "Robustness", "requires": ["prompt_paraphrase_variant_artifacts"]},
    {
        "id": "B",
        "name": "Decision architecture and cost",
        "requires": ["single_agent_baseline", "main_artifacts", "run_logs"],
    },
    {
        "id": "C",
        "name": "Context contribution",
        "requires": ["main_artifacts", "context_free_variant_artifacts"],
    },
)


def experiment_manifest() -> dict[str, Any]:
    return {
        "manifest_version": STAGE7_EXPERIMENT_MANIFEST_VERSION,
        "harness_version": STAGE7_HARNESS_VERSION,
        "statistical_contract_version": STAGE7_STATISTICAL_CONTRACT_VERSION,
        "prompt_paraphrase_variants": list(PROMPT_PARAPHRASE_VARIANTS),
        "sensitivity_case_ids": list(SENSITIVITY_CASE_IDS),
        "decision_architecture_variants": list(DECISION_ARCHITECTURE_VARIANTS),
        "architecture_robustness_seeds": list(ARCHITECTURE_ROBUSTNESS_SEEDS),
        "decision_policy_variants": list(DECISION_POLICY_VARIANTS),
        "manual_audit_selection_seed": MANUAL_AUDIT_SELECTION_SEED,
        "manual_audit_per_stratum": MANUAL_AUDIT_PER_STRATUM,
        "mediator_audit_selection_seed": MEDIATOR_AUDIT_SELECTION_SEED,
        "mediator_audit_per_case_outcome_stratum": MEDIATOR_AUDIT_PER_CASE_OUTCOME_STRATUM,
        "mediator_audit_include_all_followup_rounds": MEDIATOR_AUDIT_INCLUDE_ALL_FOLLOWUP_ROUNDS,
        "experiments": list(STAGE7_EXPERIMENTS),
    }
