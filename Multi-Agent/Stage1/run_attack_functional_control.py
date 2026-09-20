from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MULTI_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(MULTI_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTI_AGENT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from Stage1.prompts import ATTACK_FEASIBILITY_SYSTEM_PROMPT, STAGE1_EVALUATION_USER_PROMPT
from Stage1.runtime import (
    LlamaCppChatClient,
    assert_llama_server_ready,
    get_experiment_runtime_profile,
    get_runtime_config,
)
from Stage1.schema import (
    STAGE1_SEMANTIC_VALIDATION_VERSION,
    CriticAssessment,
    CriticType,
    build_constrained_critic_response_schema,
    validate_critic_assessment,
)


CONTROL_VERSION = "attack-directionality-functional-control-v1"
CONTROL_CASE_ID = "functional_control__ddos_noi_directionality"
CONTROL_EVIDENCE_ID = "E_SYNTH_DDOS_NOI_TREND"
CONTROL_SEED = 42
CONTROL_REQUEST_CONCURRENCY = 1

EXPECTED_BY_DIRECTION = {
    "increasing": {
        "stance": 1,
        "decision": "SUPPORT_DOMINATES",
        "relation": "SUPPORTS_FORECAST",
    },
    "decreasing": {
        "stance": -1,
        "decision": "CHALLENGE_DOMINATES",
        "relation": "CHALLENGES_FORECAST",
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _synthetic_evidence_record() -> dict[str, Any]:
    return {
        "evidence_id": CONTROL_EVIDENCE_ID,
        "source_registry_id": "SYNTHETIC_FUNCTIONAL_CONTROL",
        "source_family": "synthetic_functional_control",
        "source": "Synthetic functional-control record; not empirical evidence",
        "publication_date": "2024-12-31",
        "evidence_date": "2024-12-31",
        "available_at": "2024-12-31",
        "source_document_id": "synthetic-ddos-noi-directional-trend-v1",
        "source_locator": "functional-control://attack-directionality",
        "source_snapshot_id": CONTROL_VERSION,
        "evidence_chain_id": "synthetic-ddos-noi-directional-trend-v1",
        "evidence_type": "synthetic_directional_threat_trend",
        "retrieval_role": "functional_control",
        "allowed_claim_types": ["directional_threat_trend"],
        "prohibited_claim_types": [
            "attack_magnitude",
            "attack_severity",
            "tactic_prevalence",
            "causal_attribution",
        ],
        "threat_ids": ["THREAT_DDOS"],
        "pmt_ids": [],
        "content": (
            "Under unchanged reporting coverage, the number of DDoS incidents increased "
            "from 100 in 2022 to 130 in 2023 and 160 in 2024. This record reports "
            "incident count/frequency only; it does not report attack size, bandwidth, "
            "severity, tactic prevalence, mitigation effectiveness, or causal attribution."
        ),
        "content_hash": "synthetic-functional-control-not-source-derived",
    }


def build_control_input(direction: str) -> dict[str, Any]:
    if direction not in EXPECTED_BY_DIRECTION:
        raise ValueError(f"Unsupported control direction: {direction!r}")

    evidence = _synthetic_evidence_record()
    evidence_pack = {
        "case_id": CONTROL_CASE_ID,
        "source_registry_version": CONTROL_VERSION,
        "source_snapshot_id": CONTROL_VERSION,
        "forecast_origin_date": "2025-01-01",
        "training_data_end": "2024-12-31",
        "analysis_cutoff_date": "2024-12-31",
        "evaluation_mode": "ex_ante_replay",
        "threat_id": "THREAT_DDOS",
        "pmt_id": "PMT_NLP_LLM",
        "forecast_summary": {
            "threat_state": {
                "state_modality": "NoI",
                "direction": direction,
            },
            "pmt_state": {
                "state_modality": "NoP",
                "direction": "indeterminate",
            },
            "predictive_uncertainty": {
                "status": "FUNCTIONAL_CONTROL_NOT_A_FORECAST_EVALUATION"
            },
        },
        "evidence": [evidence],
        "retrieval_metadata": {
            "cutoff_applied": True,
            "retrieved_count": 1,
            "source_family_count": 1,
            "independent_evidence_chain_count": 1,
            "required_evidence_slots": ["direct_threat_direction"],
            "covered_evidence_slots": ["direct_threat_direction"],
            "missing_evidence_slots": [],
            "evidence_sufficiency": "PARTIAL",
            "temporal_violation_count": 0,
            "source_contract_violation_count": 0,
        },
    }

    prompt_payload = {
        "case_id": evidence_pack["case_id"],
        "threat_id": evidence_pack["threat_id"],
        "pmt_id": evidence_pack["pmt_id"],
        "forecast_origin_date": evidence_pack["forecast_origin_date"],
        "training_data_end": evidence_pack["training_data_end"],
        "analysis_cutoff_date": evidence_pack["analysis_cutoff_date"],
        "evaluation_mode": evidence_pack["evaluation_mode"],
        "forecast_summary": copy.deepcopy(evidence_pack["forecast_summary"]),
        "evidence": [
            {
                "evidence_id": evidence["evidence_id"],
                "source": evidence["source"],
                "source_family": evidence["source_family"],
                "publication_date": evidence["publication_date"],
                "evidence_date": evidence["evidence_date"],
                "available_at": evidence["available_at"],
                "source_document_id": evidence["source_document_id"],
                "source_snapshot_id": evidence["source_snapshot_id"],
                "source_locator": evidence["source_locator"],
                "evidence_chain_id": evidence["evidence_chain_id"],
                "evidence_type": evidence["evidence_type"],
                "retrieval_role": evidence["retrieval_role"],
                "retrieval_slots": ["direct_threat_direction"],
                "allowed_claim_types": evidence["allowed_claim_types"],
                "prohibited_claim_types": evidence["prohibited_claim_types"],
                "content": evidence["content"],
            }
        ],
        "retrieval_summary": {
            key: value
            for key, value in evidence_pack["retrieval_metadata"].items()
            if key
            in {
                "cutoff_applied",
                "retrieved_count",
                "source_family_count",
                "independent_evidence_chain_count",
                "required_evidence_slots",
                "covered_evidence_slots",
                "missing_evidence_slots",
                "evidence_sufficiency",
                "temporal_violation_count",
                "source_contract_violation_count",
            }
        },
    }
    return {
        "direction": direction,
        "evidence_pack": evidence_pack,
        "forecast_data": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        "prompt_payload": prompt_payload,
    }


def _diff_paths(left: Any, right: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[tuple[str, ...]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                paths.append((*prefix, str(key)))
            else:
                paths.extend(_diff_paths(left[key], right[key], (*prefix, str(key))))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[tuple[str, ...]] = []
        for index, (l_item, r_item) in enumerate(zip(left, right, strict=True)):
            paths.extend(_diff_paths(l_item, r_item, (*prefix, str(index))))
        return paths
    return [] if left == right else [prefix]


def _ideal_assessment(direction: str) -> dict[str, Any]:
    expected = EXPECTED_BY_DIRECTION[direction]
    claim_id = "AF_CONTROL_C1"
    return {
        "case_id": CONTROL_CASE_ID,
        "critic_type": "attack_feasibility",
        "stance": expected["stance"],
        "evidence_sufficiency": "PARTIAL",
        "claims": [
            {
                "claim_id": claim_id,
                "statement": (
                    "The number of DDoS incidents increased from 100 in 2022 to 130 in 2023 "
                    "and 160 in 2024."
                ),
                "evidence_ids": [CONTROL_EVIDENCE_ID],
                "forecast_component": "THREAT_TRAJECTORY",
                "forecast_relation": expected["relation"],
            }
        ],
        "stance_basis": {
            "supporting_claim_ids": [claim_id] if expected["stance"] == 1 else [],
            "challenging_claim_ids": [claim_id] if expected["stance"] == -1 else [],
            "decision": expected["decision"],
            "rationale": "The direct NoI incident-count trend is decisive for this functional control.",
        },
        "unresolved_questions": [],
        "unsupported_specificity_detected": False,
    }


def validate_control_design() -> dict[str, Any]:
    increasing = build_control_input("increasing")
    decreasing = build_control_input("decreasing")
    diffs = _diff_paths(increasing["prompt_payload"], decreasing["prompt_payload"])
    expected_diff = [("forecast_summary", "threat_state", "direction")]
    if diffs != expected_diff:
        raise AssertionError(
            "Paired functional-control prompts must differ only in forecast_summary.threat_state.direction; "
            f"observed differences: {diffs}"
        )

    for control in (increasing, decreasing):
        validate_critic_assessment(
            _ideal_assessment(control["direction"]),
            evidence_pack=control["evidence_pack"],
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        )

    return {
        "control_version": CONTROL_VERSION,
        "paired_prompt_differences": [".".join(path) for path in diffs],
        "synthetic_evidence_id": CONTROL_EVIDENCE_ID,
        "synthetic_evidence_sha256": _sha256_json(_synthetic_evidence_record()),
        "design_validation": "PASS",
    }


def _run_one(client: LlamaCppChatClient, control: dict[str, Any]) -> dict[str, Any]:
    direction = control["direction"]
    evidence_pack = control["evidence_pack"]
    prompt = ATTACK_FEASIBILITY_SYSTEM_PROMPT.format(forecast_data=control["forecast_data"])
    structured_llm = client.with_structured_output(
        CriticAssessment,
        method="json_schema",
        progress_label=f"AttackFunctionalControl-{direction}",
        response_schema=build_constrained_critic_response_schema(
            evidence_pack=evidence_pack,
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        ),
        semantic_validator=lambda assessment: validate_critic_assessment(
            assessment,
            evidence_pack=evidence_pack,
            expected_critic_type=CriticType.ATTACK_FEASIBILITY,
        ),
    )
    assessment = structured_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=STAGE1_EVALUATION_USER_PROMPT),
        ]
    )
    parsed = validate_critic_assessment(
        assessment,
        evidence_pack=evidence_pack,
        expected_critic_type=CriticType.ATTACK_FEASIBILITY,
    )
    assessment_dict = parsed.model_dump(mode="json")

    expected = EXPECTED_BY_DIRECTION[direction]
    decisive_claims = [
        claim
        for claim in assessment_dict["claims"]
        if claim["forecast_component"] == "THREAT_TRAJECTORY"
        and claim["forecast_relation"] == expected["relation"]
        and CONTROL_EVIDENCE_ID in claim["evidence_ids"]
    ]
    passed = (
        assessment_dict["stance"] == expected["stance"]
        and assessment_dict["stance_basis"]["decision"] == expected["decision"]
        and bool(decisive_claims)
    )
    return {
        "direction": direction,
        "expected": expected,
        "passed": passed,
        "assessment": assessment_dict,
    }


def _write_result(payload: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"{CONTROL_VERSION}__seed-{CONTROL_SEED}__{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paired synthetic functional control for Attack Feasibility directionality. "
            "The two prompts are identical except for the forecasted threat direction."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the paired fixtures and semantic contract without contacting the LLM server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            MULTI_AGENT_ROOT
            / "Results"
            / "Stage1"
            / "functional_controls"
            / "attack_directionality"
        ),
        help="Directory for the isolated functional-control artifact.",
    )
    args = parser.parse_args()

    design = validate_control_design()
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return 0

    # Use the already-running canonical Stage 1 llama.cpp server. The server may
    # expose two slots, but this diagnostic process submits requests strictly
    # sequentially and permits only one in-flight request from this client.
    assert_llama_server_ready(profile_name="Attack functional control")
    config = replace(get_runtime_config(), seed=CONTROL_SEED)
    client = LlamaCppChatClient(
        config=config,
        request_concurrency=CONTROL_REQUEST_CONCURRENCY,
    )

    controls = [build_control_input("increasing"), build_control_input("decreasing")]
    results: list[dict[str, Any]] = []
    for control in controls:
        direction = control["direction"]
        print(f"\n=== Attack functional control: forecast={direction} ===", flush=True)
        try:
            results.append(_run_one(client, control))
        except Exception as exc:
            results.append(
                {
                    "direction": direction,
                    "expected": EXPECTED_BY_DIRECTION[direction],
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    runtime_profile = get_experiment_runtime_profile()
    runtime_profile["seed"] = CONTROL_SEED
    runtime_profile["client_request_concurrency"] = CONTROL_REQUEST_CONCURRENCY
    payload = {
        "schema_version": CONTROL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Synthetic functional control only. It tests whether the Attack Feasibility critic can emit "
            "opposite directional stances when supplied modality-matched directional evidence. It is not "
            "part of the seven empirical evaluation cases and must not be pooled with empirical results."
        ),
        "design": design,
        "case_id": CONTROL_CASE_ID,
        "synthetic_evidence": _synthetic_evidence_record(),
        "pair_invariant": (
            "The two agent-facing prompt payloads differ only at "
            "forecast_summary.threat_state.direction."
        ),
        "prompt_sha256": _sha256_text(ATTACK_FEASIBILITY_SYSTEM_PROMPT),
        "user_prompt_sha256": _sha256_text(STAGE1_EVALUATION_USER_PROMPT),
        "semantic_validation_version": STAGE1_SEMANTIC_VALIDATION_VERSION,
        "runtime": runtime_profile,
        "results": results,
        "pair_passed": len(results) == 2 and all(result.get("passed") is True for result in results),
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    output_path = _write_result(payload, args.output_dir.resolve())
    print(f"\nFunctional-control artifact: {output_path}")
    print(f"PAIR RESULT: {'PASS' if payload['pair_passed'] else 'FAIL'}")
    return 0 if payload["pair_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
