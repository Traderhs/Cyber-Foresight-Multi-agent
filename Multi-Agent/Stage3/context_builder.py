from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from Stage0.context_scenarios import (
    CONTEXT_SCENARIO_SET_VERSION,
    context_scenario_manifest,
    ensure_frozen_context_scenario_manifest,
    main_context_scenarios,
)
from Stage0.action_evidence import load_action_evidence_records
from Stage0.decision_evidence import (
    DECISION_EVIDENCE_RETRIEVAL_VERSION,
    MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET,
    MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET,
    MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS,
    MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS,
    MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS,
    MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS,
    build_decision_evidence_pack,
    open_snapshot_for_decision_evidence,
)
from Stage0.decision_sources import load_decision_source_snapshot
from Stage0.main_guidance_selection import load_main_guidance_selection
from Stage3.context_schema import (
    ContextProvenanceType,
    ContextScenario,
    ContextValue,
    ContextualDecisionObject,
    DecisionEvidencePack,
    Stage3ContextualDecisionBundle,
)
from Stage3.schema import DeploymentContext, Stage3DecisionBundle, Stage3ValidationError


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _short_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id:
            raise Stage3ValidationError("Contextual Stage 3 evidence record is missing evidence_id")
        existing = by_id.get(evidence_id)
        if existing is not None and existing != record:
            raise Stage3ValidationError(f"Conflicting evidence records share ID {evidence_id}")
        by_id[evidence_id] = record
    return [by_id[key] for key in sorted(by_id)]


def _scenario_model(raw: dict[str, Any]) -> ContextScenario:
    context_ids = list(raw["context_evidence_ids"])
    standard = ContextValue(
        value=str(raw["reference_enterprise_profile"]),
        provenance_type=ContextProvenanceType.STANDARD_DERIVED,
        source_evidence_ids=context_ids,
    )
    return ContextScenario(
        scenario_id=str(raw["scenario_id"]),
        scenario_set_version=str(raw["scenario_set_version"]),
        context_selection_rule=str(raw["context_selection_rule"]),
        jurisdiction_code=str(raw["jurisdiction_code"]),
        jurisdiction_source_registry_id=str(raw["jurisdiction_source_registry_id"]),
        region=ContextValue(
            value=str(raw["region"]),
            provenance_type=ContextProvenanceType.SCENARIO_PARAMETER,
        ),
        organization_type=ContextValue(
            value=str(raw["organization_type"]),
            provenance_type=ContextProvenanceType.STANDARD_DERIVED,
            source_evidence_ids=context_ids,
        ),
        infrastructure_context=ContextValue(
            value=str(raw["infrastructure_context"]),
            provenance_type=ContextProvenanceType.STANDARD_DERIVED,
            source_evidence_ids=context_ids,
        ),
        budget_context=ContextValue(
            value=str(raw["budget_context"]),
            provenance_type=ContextProvenanceType.SCENARIO_PARAMETER,
        ),
        reference_enterprise_profile=standard,
        assumptions=list(raw.get("assumptions") or []),
        non_assumptions=list(raw.get("non_assumptions") or []),
    )


def build_contextual_stage3_bundle(
    *,
    project_root: str | Path,
    evidence_pack: dict[str, Any],
    base_stage3_bundle: dict[str, Any],
    base_stage3_input_fingerprint: str,
) -> Stage3ContextualDecisionBundle:
    """Build the versioned contextual evaluation objects without any LLM call."""

    ensure_frozen_context_scenario_manifest(project_root=project_root)

    base_bundle = Stage3DecisionBundle.model_validate(base_stage3_bundle)
    if len(base_bundle.decision_objects) != 1:
        raise Stage3ValidationError("Contextual Stage 3 main path requires exactly one frozen base DecisionObject")
    base_object = base_bundle.decision_objects[0]
    case_id = str(evidence_pack.get("case_id") or "")
    if case_id != base_bundle.case_id or base_object.case_id != case_id:
        raise Stage3ValidationError("Stage 0 / base Stage 3 case mismatch during contextualization")
    snapshot_id = str(evidence_pack.get("source_snapshot_id") or "")
    cutoff = str(evidence_pack.get("analysis_cutoff_date") or "")
    if not snapshot_id or not cutoff:
        raise Stage3ValidationError("Contextual Stage 3 requires frozen snapshot_id and analysis cutoff")
    store = open_snapshot_for_decision_evidence(project_root=project_root, snapshot_id=snapshot_id)
    supplemental_records, supplemental_manifest = load_decision_source_snapshot(project_root=project_root)
    supplemental_records_tuple = tuple(supplemental_records)
    action_records, action_manifest = load_action_evidence_records()
    action_records_tuple = tuple(action_records)
    main_guidance_records, main_guidance_slot_coverage, main_guidance_manifest = (
        load_main_guidance_selection(
            pmt_id=base_object.pmt_or_mitigation_id,
            records=(*store.records, *supplemental_records_tuple),
        )
    )
    main_guidance_records_tuple = tuple(main_guidance_records)

    base_records = [dict(record) for record in evidence_pack.get("evidence", [])]
    if not base_records:
        raise Stage3ValidationError("Contextual Stage 3 requires the original Stage 0 EvidencePack records")
    base_record_ids = sorted(str(record["evidence_id"]) for record in base_records)
    if set(base_record_ids) != set(base_object.evaluation_evidence_ids):
        raise Stage3ValidationError("Base Stage 3 evidence universe does not match the frozen Stage 0 EvidencePack")

    contextual_objects: list[ContextualDecisionObject] = []
    scenario_manifest = context_scenario_manifest()
    for raw_scenario in main_context_scenarios():
        scenario = _scenario_model(raw_scenario)
        decision_pack_raw = build_decision_evidence_pack(
            store=store,
            cutoff_date=cutoff,
            threat_id=base_object.threat_id,
            pmt_id=base_object.pmt_or_mitigation_id,
            threat_name=base_object.threat,
            pmt_name=base_object.pmt_or_mitigation,
            scenario=raw_scenario,
            supplemental_records=supplemental_records_tuple,
            supplemental_manifest=supplemental_manifest,
            action_records=action_records_tuple,
            action_manifest=action_manifest,
            main_guidance_records=main_guidance_records_tuple,
            main_guidance_slot_coverage=main_guidance_slot_coverage,
            main_guidance_manifest=main_guidance_manifest,
            minimum_generic_records=MAIN_MIN_GENERIC_DECISION_EVIDENCE_RECORDS,
            minimum_generic_documents=MAIN_MIN_GENERIC_DECISION_EVIDENCE_DOCUMENTS,
            minimum_lens_documents=MAIN_MIN_LENS_DECISION_EVIDENCE_DOCUMENTS,
            minimum_balance_documents_per_facet=MAIN_MIN_BALANCE_DOCUMENTS_PER_FACET,
            minimum_balance_publishers_per_facet=MAIN_MIN_BALANCE_PUBLISHERS_PER_FACET,
            minimum_lens_publishers=MAIN_MIN_LENS_DECISION_EVIDENCE_PUBLISHERS,
        )
        decision_pack = DecisionEvidencePack.model_validate(decision_pack_raw)
        all_records = _dedupe_records(
            [
                *base_records,
                *decision_pack.context_evidence_records,
                *decision_pack.decision_evidence_records,
            ]
        )
        evaluation_ids = [str(record["evidence_id"]) for record in all_records]
        identity_payload = {
            "base_decision_object_id": base_object.decision_object_id,
            "scenario": scenario.model_dump(mode="json"),
            "decision_evidence_pack": decision_pack.model_dump(mode="json"),
            "evaluation_evidence_ids": evaluation_ids,
            "scenario_manifest_sha256": scenario_manifest["manifest_sha256"],
            "retrieval_version": DECISION_EVIDENCE_RETRIEVAL_VERSION,
        }
        decision_object_id = (
            f"contextual__{base_object.decision_object_id}__{scenario.scenario_id.lower()}__"
            f"{_short_hash(identity_payload)}"
        )
        base_dict = base_object.model_dump(mode="json")
        base_dict.update(
            {
                "decision_object_id": decision_object_id,
                "deployment_context": DeploymentContext(
                    region=scenario.region.value,
                    organization_type=scenario.organization_type.value,
                    infrastructure_context=scenario.infrastructure_context.value,
                    budget_context=scenario.budget_context.value,
                ).model_dump(mode="json"),
                "known_constraints": sorted(
                    set(base_dict.get("known_constraints", []))
                    | {
                        "Deployment context is a frozen experimental scenario rather than an observed real organization.",
                        "No numeric budget, price, ROI, TCO, staffing count, or vendor assumption is introduced by the scenario.",
                        "KR/EU/US main scenarios share the same CIS IG2 enterprise profile and differ only by jurisdiction parameter and jurisdiction-specific primary-law retrieval.",
                    }
                ),
                "unknown_context_fields": [],
                "evaluation_evidence_ids": evaluation_ids,
                "base_decision_object_id": base_object.decision_object_id,
                "scenario_id": scenario.scenario_id,
                "scenario_set_version": scenario.scenario_set_version,
                "context_scenario": scenario.model_dump(mode="json"),
                "forecast_evidence_ids": base_record_ids,
                "context_evidence_ids": list(decision_pack.context_evidence_ids),
                "decision_evidence_ids": list(decision_pack.decision_evidence_ids),
                "evaluation_evidence_records": all_records,
                "decision_evidence_pack": decision_pack.model_dump(mode="json"),
            }
        )
        contextual_objects.append(ContextualDecisionObject.model_validate(base_dict))

    if {item.scenario_id for item in contextual_objects} != {
        "KR__CIS_IG2",
        "EU__CIS_IG2",
        "US__CIS_IG2",
    }:
        raise Stage3ValidationError("Main contextual Stage 3 must freeze exactly KR/EU/US CIS-IG2 scenarios")

    # Generic decision evidence must be invariant across the jurisdiction sweep;
    # only regulatory evidence is allowed to differ.
    generic_sets: list[set[str]] = []
    for item in contextual_objects:
        regulatory_ids = {
            evidence_id
            for evidence_id, slots in item.decision_evidence_pack.slot_coverage.items()
            if "regulatory_applicability" in slots
        }
        generic_sets.append(set(item.decision_evidence_ids) - regulatory_ids)
    if any(item != generic_sets[0] for item in generic_sets[1:]):
        raise Stage3ValidationError("Generic decision evidence changed across the KR/EU/US jurisdiction sweep")

    return Stage3ContextualDecisionBundle(
        case_id=case_id,
        base_stage3_input_fingerprint=base_stage3_input_fingerprint,
        scenario_set_version=CONTEXT_SCENARIO_SET_VERSION,
        contextual_decision_objects=contextual_objects,
    )

