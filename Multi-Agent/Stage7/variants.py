from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Stage7.loaders import CaseBundle
from Stage7.metrics import jaccard
from Stage7.schema import VariantResultRecord


LENSES = ("technical_feasibility", "institutional_regional", "financial_adoption")


def baseline_record_map(bundles: list[CaseBundle]) -> dict[tuple[str, str], VariantResultRecord]:
    output: dict[tuple[str, str], VariantResultRecord] = {}
    for bundle in bundles:
        for synthesis in bundle.syntheses:
            traces = synthesis.get("lens_evidence_trace") or {}
            stances = {lens: int(traces[lens]["stance"]) for lens in LENSES}
            evidence_ids = {
                lens: list(
                    dict.fromkeys(
                        evidence_id
                        for claim in traces[lens].get("claims") or []
                        for evidence_id in claim.get("evidence_ids") or []
                    )
                )
                for lens in LENSES
            }
            directions = {
                lens: [str(claim.get("status") or "") for claim in traces[lens].get("claims") or []]
                for lens in LENSES
            }
            record = VariantResultRecord(
                axis="MAIN",
                variant_id="FULL_REDESIGNED_PIPELINE",
                case_id=bundle.case_id,
                scenario_id=str(synthesis["scenario_id"]),
                decision_object_id=str(synthesis["decision_object_id"]),
                stance_by_lens=stances,
                d_lens=float(synthesis["d_lens"]),
                evidence_ids_by_lens=evidence_ids,
                claim_directions_by_lens=directions,
                recommendation=str(synthesis["decision_recommendation"]),
                prompt_hashes=dict(bundle.stage6.get("prompt_hashes") or {}),
                runtime=dict(bundle.stage6.get("runtime") or {}),
                provenance={
                    "stage6_artifact_sha256": bundle.stage6["artifact_sha256"],
                    "stage6_input_fingerprint": bundle.stage6["input_fingerprint"],
                },
            )
            key = (record.case_id, record.scenario_id)
            if key in output:
                raise ValueError(f"Duplicate baseline Stage 7 record: {key}")
            output[key] = record
    return output


def load_variant_records(root: Path) -> list[VariantResultRecord]:
    folder = root / "Multi-Agent" / "Results" / "Stage7" / "variants"
    if not folder.exists():
        return []
    records: list[VariantResultRecord] = []
    for path in sorted(folder.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            payloads = raw
        elif isinstance(raw, dict) and isinstance(raw.get("records"), list):
            payloads = raw["records"]
        elif isinstance(raw, dict):
            payloads = [raw]
        else:
            raise ValueError(f"Unsupported Stage 7 variant file shape: {path}")
        for payload in payloads:
            records.append(VariantResultRecord.model_validate(payload))
    return records


def compare_variant_to_baseline(
    variant: VariantResultRecord,
    baseline: VariantResultRecord,
) -> dict[str, Any]:
    common_lenses = sorted(set(variant.stance_by_lens) & set(baseline.stance_by_lens))
    stance_matches = [
        int(variant.stance_by_lens[lens] == baseline.stance_by_lens[lens])
        for lens in common_lenses
    ]
    evidence_jaccards = [
        jaccard(
            variant.evidence_ids_by_lens.get(lens, []),
            baseline.evidence_ids_by_lens.get(lens, []),
        )
        for lens in common_lenses
    ]
    direction_jaccards = [
        jaccard(
            variant.claim_directions_by_lens.get(lens, []),
            baseline.claim_directions_by_lens.get(lens, []),
        )
        for lens in common_lenses
    ]
    return {
        "axis": variant.axis,
        "variant_id": variant.variant_id,
        "repeat_index": variant.repeat_index,
        "case_id": variant.case_id,
        "scenario_id": variant.scenario_id,
        "common_lens_count": len(common_lenses),
        "stance_agreement": (sum(stance_matches) / len(stance_matches)) if stance_matches else None,
        "d_lens_abs_delta": abs(variant.d_lens - baseline.d_lens),
        "evidence_overlap": (sum(evidence_jaccards) / len(evidence_jaccards)) if evidence_jaccards else None,
        "claim_direction_overlap": (
            sum(direction_jaccards) / len(direction_jaccards) if direction_jaccards else None
        ),
        "recommendation_consistent": variant.recommendation == baseline.recommendation,
    }
