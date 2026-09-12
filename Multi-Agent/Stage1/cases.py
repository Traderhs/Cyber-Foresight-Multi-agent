from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MAIN_CASE_SET_VERSION = "stage1-main-cases-v1"
MAIN_SNAPSHOT_ID = "stage0-2026-09-12-v2"
MAIN_ANALYSIS_CUTOFF = "2024-12-31"
MAIN_EVALUATION_MODE = "ex_ante_replay"

# These thresholds were computed once from the frozen 303-relation canonical
# Threat-PMT gap table. Runtime Stage 1 inference never re-selects cases from
# model outputs.
GAP_QUANTILES = {
    "q25": -0.49900797901303073,
    "q50": -0.09899536871089283,
    "q75": 0.5164884419030555,
}
SLOPE_QUANTILES = {
    "q25": -0.11273098490753453,
    "q50": 0.059498037847772245,
    "q75": 0.2248009525173117,
}

REGIME_DEFINITIONS = {
    "large_gap_steep_slope": "mean_gap >= gap_Q75 and slope >= slope_Q75",
    "large_gap_moderate_slope": "mean_gap >= gap_Q75 and slope_Q25 <= slope < slope_Q75",
    "moderate_gap_steep_slope": "gap_Q25 < mean_gap < gap_Q75 and slope >= slope_Q75",
    "moderate_gap_low_slope": "gap_Q25 < mean_gap < gap_Q75 and slope_Q25 <= slope < slope_Q50",
    "small_gap_low_or_negative_slope": "mean_gap <= gap_Q25 and slope < slope_Q50",
}

REGIME_EVIDENCE_REQUIREMENTS = {
    "min_covered_evidence_slots": 5,
    "total_evidence_slots": 6,
    "min_source_families": 4,
}


@dataclass(frozen=True)
class MainStage1Case:
    case_id: str
    threat: str
    pmt: str
    selection_class: str
    regime: str | None
    selection_reason: str
    coverage_rule: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Frozen before the main Stage 1 inference. The two anchors are deliberately
# retained even when the strict cutoff leaves partial evidence so the pipeline's
# evidence-insufficiency behavior is evaluated rather than hidden by filtering.
MAIN_STAGE1_CASES: tuple[MainStage1Case, ...] = (
    MainStage1Case(
        case_id="main__ddos__nlp_llm",
        threat="DDoS",
        pmt="NLP/LLM",
        selection_class="predeclared_anchor",
        regime=None,
        selection_reason=(
            "Predeclared reference case retained regardless of evidence coverage. "
            "It intentionally stress-tests the strict ex-ante evidence boundary and the "
            "PARTIAL/INSUFFICIENT_EVIDENCE path instead of dropping a difficult case post hoc."
        ),
        coverage_rule="anchor_exempt_from_regime_coverage_filter",
    ),
    MainStage1Case(
        case_id="main__malware__anomaly_detection",
        threat="Malware",
        pmt="ANOMALY DETECTION",
        selection_class="predeclared_anchor",
        regime=None,
        selection_reason=(
            "Predeclared reference case with full slot coverage. It provides a stable, evidence-rich "
            "counterpart to the low-coverage anchor and allows redesign comparisons without selecting "
            "cases after seeing Stage 1 outputs."
        ),
        coverage_rule="anchor_exempt_from_regime_coverage_filter",
    ),
    MainStage1Case(
        case_id="main__ransomware__cryptography",
        threat="Ransomware",
        pmt="CRYPTOGRAPHY",
        selection_class="regime_representative",
        regime="large_gap_steep_slope",
        selection_reason=(
            "Coverage-qualified representative of the large-gap/steep-slope regime. "
            "Selected before Stage 1 generation using only frozen Stage 0 forecast/evidence metadata."
        ),
        coverage_rule="at_least_5_of_6_slots_and_4_source_families",
    ),
    MainStage1Case(
        case_id="main__insider_threat__cryptography",
        threat="Insider Threat",
        pmt="CRYPTOGRAPHY",
        selection_class="regime_representative",
        regime="large_gap_moderate_slope",
        selection_reason=(
            "Coverage-qualified representative of the large-gap/moderate-slope regime, included to "
            "separate persistent structural imbalance from rapidly accelerating imbalance."
        ),
        coverage_rule="at_least_5_of_6_slots_and_4_source_families",
    ),
    MainStage1Case(
        case_id="main__brute_force__ids_ips",
        threat="Brute Force Attack",
        pmt="IDS/IPS",
        selection_class="regime_representative",
        regime="moderate_gap_steep_slope",
        selection_reason=(
            "Coverage-qualified representative of the moderate-gap/steep-slope regime, included to "
            "test whether agents distinguish emerging escalation from already-large gaps."
        ),
        coverage_rule="at_least_5_of_6_slots_and_4_source_families",
    ),
    MainStage1Case(
        case_id="main__targeted_attack__access_control",
        threat="Targeted Attack",
        pmt="ACCESS CONTROL",
        selection_class="regime_representative",
        regime="moderate_gap_low_slope",
        selection_reason=(
            "Coverage-qualified representative of the moderate-gap/low-slope regime, included as a "
            "managed/stable trajectory rather than another escalation-heavy example."
        ),
        coverage_rule="at_least_5_of_6_slots_and_4_source_families",
    ),
    MainStage1Case(
        case_id="main__session_hijacking__https",
        threat="Session Hijacking",
        pmt="HTTPS",
        selection_class="regime_representative",
        regime="small_gap_low_or_negative_slope",
        selection_reason=(
            "Coverage-qualified representative of the small-gap/low-or-negative-slope regime, included "
            "as a low-risk alignment control against high-gap and accelerating cases."
        ),
        coverage_rule="at_least_5_of_6_slots_and_4_source_families",
    ),
)


def main_case_set_manifest() -> dict[str, Any]:
    return {
        "case_set_version": MAIN_CASE_SET_VERSION,
        "snapshot_id": MAIN_SNAPSHOT_ID,
        "analysis_cutoff_date": MAIN_ANALYSIS_CUTOFF,
        "evaluation_mode": MAIN_EVALUATION_MODE,
        "population_relation_count": 303,
        "gap_quantiles": GAP_QUANTILES,
        "slope_quantiles": SLOPE_QUANTILES,
        "regime_definitions": REGIME_DEFINITIONS,
        "regime_evidence_requirements": REGIME_EVIDENCE_REQUIREMENTS,
        "selection_constraints": [
            "Stage 1 outputs are never used for case selection.",
            "Two predeclared anchors are always retained, including low-evidence cases.",
            "Five additional cases cover five distinct gap/slope regimes.",
            "Regime representatives must cover at least 5 of 6 evidence slots and at least 4 source families.",
            "The case set is frozen before main Stage 1 generation and is not re-selected from downstream results.",
        ],
        "cases": [case.to_dict() for case in MAIN_STAGE1_CASES],
    }


def validate_main_case_set() -> None:
    if len(MAIN_STAGE1_CASES) != 7:
        raise ValueError("Main Stage 1 case set must contain exactly seven cases")
    ids = [case.case_id for case in MAIN_STAGE1_CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("Main Stage 1 case_id values must be unique")
    pairs = [(case.threat, case.pmt) for case in MAIN_STAGE1_CASES]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Main Stage 1 Threat-PMT pairs must be unique")
    regimes = [case.regime for case in MAIN_STAGE1_CASES if case.selection_class == "regime_representative"]
    expected = {
        "large_gap_steep_slope",
        "large_gap_moderate_slope",
        "moderate_gap_steep_slope",
        "moderate_gap_low_slope",
        "small_gap_low_or_negative_slope",
    }
    if set(regimes) != expected:
        raise ValueError(f"Main Stage 1 regime coverage mismatch: {regimes}")
