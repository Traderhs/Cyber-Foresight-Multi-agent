from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .schema import Stage0ValidationError


HISTORICAL_MONTHS = 162
FORECAST_MONTHS = 36
TRAINING_DATA_END = "2024-12-01"
FORECAST_ORIGIN_DATE = "2024-12-31"
FORECAST_START_DATE = "2025-01-01"
FORECAST_END_DATE = "2027-12-01"
EXPECTED_THREAT_COUNT = 26
EXPECTED_PMT_COUNT = 98
EXPECTED_NODE_COUNT = 124
FEATURE_NAMES = ("NoI", "NoP", "ACA", "PH")
SOURCE_MATCH_TOLERANCE = 1e-5


def _norm(value: str) -> str:
    value = (
        value.lower()
        .replace("behaviour", "behavior")
        .replace("sanitisation", "sanitization")
        .replace("standardised", "standardized")
    )
    return re.sub(r"[^a-z0-9]", "", value)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper() or "UNKNOWN"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _jsonl_dump_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _month_range(start_year: int, start_month: int, count: int) -> list[str]:
    out: list[str] = []
    year, month = start_year, start_month
    for _ in range(count):
        out.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def _zscore_stats(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0:
        raise Stage0ValidationError("Cannot z-score a constant feature series")
    return mean, std


class PaperForecastMigrator:
    """Map legacy backing artifacts into the paper-defined 124-node x 4-feature contract."""

    def __init__(self, project_root: str | Path, output_dir: str | Path | None = None):
        self.root = Path(project_root).resolve()
        self.graph_path = self.root / "B-MTGNN/data/graph.csv"
        self.forecast_dir = self.root / "B-MTGNN/model/Bayesian/forecast/data"
        self.source_path = self.root / "Data_Preparation/Smoothed_CyberTrend_Forecasting_All.csv"
        self.stale_source_path = self.root / "B-MTGNN/data/data.csv"
        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else self.root / "Multi-Agent/Results/Stage0/Forecast"
        )
        self._files = list(self.forecast_dir.glob("*.txt"))
        self._cache: dict[str, dict[str, list[float]]] = {}

    def _parse_forecast(self, filename: str) -> dict[str, list[float]]:
        if filename in self._cache:
            return self._cache[filename]
        path = self.forecast_dir / filename
        parsed: dict[str, list[float]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if ": " not in line:
                continue
            key, raw = line.split(": ", 1)
            if key in {"Data", "Forecast", "95% Confidence", "Variance"}:
                parsed[key] = [float(v) for v in ast.literal_eval(raw)]
        if set(parsed) != {"Data", "Forecast", "95% Confidence", "Variance"}:
            raise Stage0ValidationError(f"Incomplete legacy forecast file: {path}")
        if len(parsed["Data"]) != HISTORICAL_MONTHS:
            raise Stage0ValidationError(f"{path}: historical length is not {HISTORICAL_MONTHS}")
        if any(len(parsed[key]) != FORECAST_MONTHS for key in ("Forecast", "95% Confidence", "Variance")):
            raise Stage0ValidationError(f"{path}: forecast length is not {FORECAST_MONTHS}")
        if not all(math.isfinite(v) for values in parsed.values() for v in values):
            raise Stage0ValidationError(f"{path}: non-finite value")
        self._cache[filename] = parsed
        return parsed

    def run(self) -> dict[str, Any]:
        threats, pmts, relations, state_sources = self._load_graph()
        header, source_rows = self._load_source()
        if len(threats) != EXPECTED_THREAT_COUNT or len(pmts) != EXPECTED_PMT_COUNT:
            raise Stage0ValidationError(
                f"Paper graph mismatch: threats={len(threats)}, pmts={len(pmts)}"
            )
        if source_rows[0][0] != "Jul-11" or source_rows[HISTORICAL_MONTHS - 1][0] != "Dec-24":
            raise Stage0ValidationError("Paper historical window must be Jul-11 through Dec-24")

        nodes = self._build_nodes(threats, pmts)
        contract = self._build_feature_contract(nodes, header)
        feature_rows = self._build_feature_rows(contract, header, source_rows)
        state_contract = self._build_state_contract(nodes, state_sources)
        source_audit, source_hashes = self._audit_selected_node_state_series(state_contract, header, source_rows)
        state_rows, state_audit = self._build_node_state_rows(state_contract)
        gap_rows, gap_audit = self._build_gap_rows(relations, state_rows)

        availability = {
            feature: sum(1 for slot in contract if slot["feature_name"] == feature and slot["available"])
            for feature in FEATURE_NAMES
        }
        if availability != {"NoI": 16, "NoP": 98, "ACA": 124, "PH": 124}:
            raise Stage0ValidationError(f"Unexpected paper feature availability: {availability}")

        stale_rows, stale_last_date = self._stale_source_info()
        manifest = {
            "schema_version": "stage0-paper-forecast-v3",
            "generated_at": datetime.now().astimezone().isoformat(),
            "derivation": {
                "mode": "existing_experiment_artifacts_only",
                "model_training_executed": False,
                "model_inference_executed": False,
                "upstream_artifacts_are_read_only": True,
            },
            "paper_contract": {
                "node_count": EXPECTED_NODE_COUNT,
                "threat_count": EXPECTED_THREAT_COUNT,
                "pmt_count": EXPECTED_PMT_COUNT,
                "feature_dim": 4,
                "feature_names": list(FEATURE_NAMES),
                "logical_historical_shape": [HISTORICAL_MONTHS, EXPECTED_NODE_COUNT, 4],
                "historical_start": "2011-07-01",
                "historical_end": TRAINING_DATA_END,
                "normalization": "z_score_per_available_node_feature_using_162_month_history",
                "structured_masking": "unavailable modality is available=false with null values",
                "feature_semantics": {
                    "NoI": "incident frequency / operational threat activity; sparse on threats and unavailable on PMTs",
                    "NoP": "publication-derived technology activity; available on PMTs and masked on Threat nodes",
                    "ACA": "global geopolitical/macro contextual stress broadcast to every node",
                    "PH": "global seasonality proxy broadcast to every node",
                },
            },
            "paper_forecast_output_contract": {
                "forecast_horizon_months": FORECAST_MONTHS,
                "forecast_start": FORECAST_START_DATE,
                "forecast_end": FORECAST_END_DATE,
                "expected_node_state_shape": [FORECAST_MONTHS, EXPECTED_NODE_COUNT],
                "paper_semantics": "Y(m) is one node-level latent intensity/maturity state per Threat/PMT node",
                "status": "PASS",
                "selection_rule": (
                    "Use the graph node's representative series: Threat NoI when graph.csv uses *-ALL, "
                    "Threat NoP when graph.csv uses Mentions-*, and PMT NoP for Solution_* nodes."
                ),
                "normalization": "z_score_per_node_using_162_month_history",
            },
            "forecast_provenance": {
                "training_data_end": TRAINING_DATA_END,
                "forecast_origin_date": FORECAST_ORIGIN_DATE,
                "forecast_horizon_start": FORECAST_START_DATE,
                "forecast_horizon_end": FORECAST_END_DATE,
                "runtime_override_allowed": False,
            },
            "feature_availability_node_counts": availability,
            "source_validation": {
                "node_state_series_count": source_audit["series_count"],
                "historical_points_checked_per_series": source_audit["historical_points_checked_per_series"],
                "stale_bmtgnn_data_rows": stale_rows,
                "stale_bmtgnn_data_last_date": stale_last_date,
            },
            "uncertainty_semantics": {
                "paper_status": "PAPER_124_NODE_Y",
                "confidence_95_half_width": (
                    "Selected node-state confidence uses the legacy 1.96 * sample_std(10 stochastic runs) / sqrt(10) "
                    "half-width, linearly transformed into each node's z-score scale."
                ),
                "legacy_variance": "legacy inverse scaling uses scale rather than scale^2; retained as raw provenance only",
            },
            "gap_semantics": {
                "years": [2025, 2026, 2027],
                "canonical_role": "yearly mean difference between paper-aligned 124-node z-scored Threat and PMT state forecasts",
                "formula": "Gap(t,p,y) = mean(Y_t_z over year y) - mean(Y_p_z over year y)",
            },
            "files": {
                "node_registry": "node_registry.json",
                "feature_contract": "feature_contract.json",
                "historical_node_features": "historical_node_features.jsonl",
                "node_state_contract": "node_state_contract.json",
                "node_state_series": "node_state_series.jsonl",
                "threat_pmt_gaps": "threat_pmt_gaps.jsonl",
                "migration_audit": "migration_audit.json",
            },
            "source_hashes": {
                "graph_csv": _sha256(self.graph_path),
                "smoothed_source": _sha256(self.source_path),
                "node_state_forecast_files": source_hashes,
            },
        }
        audit = {
            "status": "PASS",
            "paper_contract": {
                "node_count": len(nodes),
                "feature_dim": 4,
                "feature_availability_node_counts": availability,
                "historical_months": HISTORICAL_MONTHS,
                "forecast_months": FORECAST_MONTHS,
            },
            "source_match": source_audit,
            "node_state_selection": state_audit,
            "gap_generation": gap_audit,
            "known_legacy_issues": [
                {
                    "id": "LEGACY_STALE_BMTGNN_DATA",
                    "severity": "high",
                    "detail": f"B-MTGNN/data/data.csv has {stale_rows} months ending {stale_last_date}; study artifacts use 162 months.",
                },
                {
                    "id": "LEGACY_VARIANCE_RESCALE",
                    "severity": "high",
                    "detail": "Legacy variance inverse scaling uses scale rather than scale^2 and is not silently reinterpreted.",
                },
            ],
        }

        _json_dump_atomic(self.output_dir / "node_registry.json", nodes)
        _json_dump_atomic(self.output_dir / "feature_contract.json", contract)
        _jsonl_dump_atomic(self.output_dir / "historical_node_features.jsonl", feature_rows)
        _json_dump_atomic(self.output_dir / "node_state_contract.json", state_contract)
        _jsonl_dump_atomic(self.output_dir / "node_state_series.jsonl", state_rows)
        _jsonl_dump_atomic(self.output_dir / "threat_pmt_gaps.jsonl", gap_rows)
        _json_dump_atomic(self.output_dir / "manifest.json", manifest)
        _json_dump_atomic(self.output_dir / "migration_audit.json", audit)
        self.validate_output()
        return audit

    def validate_output(self) -> dict[str, Any]:
        required = {
            "node_registry.json",
            "feature_contract.json",
            "historical_node_features.jsonl",
            "node_state_contract.json",
            "node_state_series.jsonl",
            "threat_pmt_gaps.jsonl",
            "manifest.json",
            "migration_audit.json",
        }
        actual = {path.name for path in self.output_dir.iterdir()} if self.output_dir.exists() else set()
        missing = required - actual
        if missing:
            raise Stage0ValidationError(f"Paper forecast output missing: {sorted(missing)}")
        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        paper = manifest.get("paper_contract", {})
        if manifest.get("schema_version") != "stage0-paper-forecast-v3":
            raise Stage0ValidationError("Unsupported paper forecast schema")
        if paper.get("node_count") != 124 or paper.get("feature_dim") != 4:
            raise Stage0ValidationError(f"Invalid paper tensor contract: {paper}")
        if paper.get("feature_names") != list(FEATURE_NAMES):
            raise Stage0ValidationError(f"Invalid paper feature order: {paper.get('feature_names')}")
        provenance = manifest.get("forecast_provenance", {})
        expected_provenance = {
            "training_data_end": TRAINING_DATA_END,
            "forecast_origin_date": FORECAST_ORIGIN_DATE,
            "forecast_horizon_start": FORECAST_START_DATE,
            "forecast_horizon_end": FORECAST_END_DATE,
            "runtime_override_allowed": False,
        }
        if provenance != expected_provenance:
            raise Stage0ValidationError(f"Invalid forecast provenance contract: {provenance}")
        nodes = json.loads((self.output_dir / "node_registry.json").read_text(encoding="utf-8"))
        slots = json.loads((self.output_dir / "feature_contract.json").read_text(encoding="utf-8"))
        state_contract = json.loads((self.output_dir / "node_state_contract.json").read_text(encoding="utf-8"))
        if len(nodes) != 124 or len(slots) != 124 * 4 or len(state_contract) != 124:
            raise Stage0ValidationError(
                f"Canonical counts invalid: nodes={len(nodes)}, slots={len(slots)}, states={len(state_contract)}"
            )
        expected_rows = 124 * 4 * HISTORICAL_MONTHS
        row_count = 0
        with (self.output_dir / "historical_node_features.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row_count += 1
                if row["available"] and (row["value_raw"] is None or row["value_z"] is None):
                    raise Stage0ValidationError(f"Available feature row has null value: {row['field_id']}")
                if not row["available"] and (row["value_raw"] is not None or row["value_z"] is not None):
                    raise Stage0ValidationError(f"Masked feature row carries a value: {row['field_id']}")
        if row_count != expected_rows:
            raise Stage0ValidationError(f"Expected {expected_rows} node-feature rows, got {row_count}")
        state_rows = self._load_jsonl_file(self.output_dir / "node_state_series.jsonl")
        if len(state_rows) != 124 * (HISTORICAL_MONTHS + FORECAST_MONTHS):
            raise Stage0ValidationError(f"Invalid node-state row count: {len(state_rows)}")
        state_ids = {row["node_id"] for row in state_rows}
        if len(state_ids) != 124:
            raise Stage0ValidationError(f"Node-state coverage is not 124 nodes: {len(state_ids)}")
        return {
            "status": "PASS",
            "nodes": 124,
            "feature_dim": 4,
            "feature_slots": 496,
            "feature_rows": row_count,
            "state_series": 124,
            "state_rows": len(state_rows),
        }

    def _load_graph(self) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, tuple[str, str]]]:
        with self.graph_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        threats: list[str] = []
        pmts: list[str] = []
        relations: dict[str, list[str]] = {}
        state_sources: dict[str, tuple[str, str]] = {}
        for row in rows:
            raw_threat = row[0]
            threat = self._graph_threat_core(raw_threat)
            if raw_threat.startswith("Mentions-"):
                state_sources[threat] = ("paper_threat", threat)
            elif raw_threat.endswith("-ALL"):
                state_sources[threat] = ("incident", threat)
            else:
                raise Stage0ValidationError(f"Unsupported graph threat node encoding: {raw_threat}")
            related: list[str] = []
            for raw in row[1:]:
                if not raw:
                    continue
                pmt = self._graph_pmt_core(raw)
                related.append(pmt)
                if pmt not in pmts:
                    pmts.append(pmt)
            threats.append(threat)
            relations[threat] = related
        return threats, pmts, relations, state_sources

    def _load_source(self) -> tuple[list[str], list[list[str]]]:
        with self.source_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if len(rows) < HISTORICAL_MONTHS:
            raise Stage0ValidationError(f"Paper source has only {len(rows)} rows")
        return header, rows

    def _build_nodes(self, threats: list[str], pmts: list[str]) -> list[dict[str, str]]:
        return [
            *[
                {"node_id": f"THREAT_{_slug(name)}", "node_type": "threat", "canonical_name": name, "graph_name": name}
                for name in threats
            ],
            *[
                {"node_id": f"PMT_{_slug(name)}", "node_type": "pmt", "canonical_name": name, "graph_name": name}
                for name in pmts
            ],
        ]

    def _build_feature_contract(self, nodes: list[dict[str, str]], header: list[str]) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        for node in nodes:
            for feature in FEATURE_NAMES:
                source_type: str | None = None
                source_name: str | None = None
                if feature == "NoI" and node["node_type"] == "threat":
                    source_type, source_name = "incident", node["canonical_name"]
                elif feature == "NoP" and node["node_type"] == "pmt":
                    source_type, source_name = "solution", node["canonical_name"]
                elif feature == "ACA":
                    source_type, source_name = "context", "War_Conflict_All"
                elif feature == "PH":
                    source_type, source_name = "context", "Holidays"

                source_column = self._optional_source_column(header, source_type, source_name) if source_type else None
                available = source_column is not None
                slots.append(
                    {
                        "node_id": node["node_id"],
                        "node_type": node["node_type"],
                        "feature_name": feature,
                        "available": available,
                        "mask_value": 1 if available else 0,
                        "source_scope": "global" if feature in {"ACA", "PH"} else "node",
                        "source_column": source_column,
                    }
                )
        return slots

    def _build_feature_rows(
        self,
        contract: list[dict[str, Any]],
        header: list[str],
        source_rows: list[list[str]],
    ) -> list[dict[str, Any]]:
        history_dates = _month_range(2011, 7, HISTORICAL_MONTHS)
        rows: list[dict[str, Any]] = []

        for slot in contract:
            node_id = slot["node_id"]
            feature = slot["feature_name"]
            if not slot["available"]:
                for timestamp in history_dates:
                    rows.append(
                        {
                            "field_id": f"X_{node_id}_{feature.upper()}_{timestamp[:7]}",
                            "node_id": node_id,
                            "feature_name": feature,
                            "timestamp": timestamp,
                            "available": False,
                            "value_raw": None,
                            "value_z": None,
                        }
                    )
                continue

            source_column = slot["source_column"]
            assert source_column
            source_index = header.index(source_column)
            source_values = [float(row[source_index]) for row in source_rows[:HISTORICAL_MONTHS]]
            mean, std = _zscore_stats(source_values)
            for idx, value in enumerate(source_values):
                rows.append(
                    {
                        "field_id": f"X_{node_id}_{feature.upper()}_{history_dates[idx][:7]}",
                        "node_id": node_id,
                        "feature_name": feature,
                        "timestamp": history_dates[idx],
                        "available": True,
                        "value_raw": value,
                        "value_z": (value - mean) / std,
                    }
                )
        return rows

    def _build_state_contract(
        self,
        nodes: list[dict[str, str]],
        threat_sources: dict[str, tuple[str, str]],
    ) -> list[dict[str, Any]]:
        contract: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for node in nodes:
            if node["node_type"] == "threat":
                source_type, source_name = threat_sources[node["canonical_name"]]
                state_modality = "NoI" if source_type == "incident" else "NoP"
            else:
                source_type, source_name = "solution", node["canonical_name"]
                state_modality = "NoP"
            forecast_file = self._find_forecast_file(source_type, source_name)
            counts[state_modality] = counts.get(state_modality, 0) + 1
            contract.append(
                {
                    "node_id": node["node_id"],
                    "node_type": node["node_type"],
                    "canonical_name": node["canonical_name"],
                    "state_modality": state_modality,
                    "legacy_source_type": source_type,
                    "legacy_forecast_file": forecast_file.name,
                }
            )
        if counts != {"NoI": 16, "NoP": 108}:
            raise Stage0ValidationError(f"Unexpected 124-node state selection: {counts}")
        return contract

    def _build_node_state_rows(
        self,
        contract: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        history_dates = _month_range(2011, 7, HISTORICAL_MONTHS)
        forecast_dates = _month_range(2025, 1, FORECAST_MONTHS)
        rows: list[dict[str, Any]] = []
        selected_files: set[str] = set()
        by_modality: dict[str, int] = {}
        for item in contract:
            filename = item["legacy_forecast_file"]
            parsed = self._parse_forecast(filename)
            mean, std = _zscore_stats(parsed["Data"])
            selected_files.add(filename)
            by_modality[item["state_modality"]] = by_modality.get(item["state_modality"], 0) + 1
            for idx, value in enumerate(parsed["Data"]):
                rows.append(
                    {
                        "state_field_id": f"YH_{item['node_id']}_{history_dates[idx][:7]}",
                        "node_id": item["node_id"],
                        "timestamp": history_dates[idx],
                        "phase": "historical",
                        "state_modality": item["state_modality"],
                        "value_raw": value,
                        "value_z": (value - mean) / std,
                        "confidence_95_half_width_z": None,
                        "legacy_variance_statistic_raw": None,
                    }
                )
            for idx, value in enumerate(parsed["Forecast"]):
                rows.append(
                    {
                        "state_field_id": f"YF_{item['node_id']}_{forecast_dates[idx][:7]}",
                        "node_id": item["node_id"],
                        "timestamp": forecast_dates[idx],
                        "phase": "forecast",
                        "state_modality": item["state_modality"],
                        "value_raw": value,
                        "value_z": (value - mean) / std,
                        "confidence_95_half_width_z": parsed["95% Confidence"][idx] / std,
                        "legacy_variance_statistic_raw": parsed["Variance"][idx],
                    }
                )
        if len(selected_files) != 124:
            raise Stage0ValidationError(f"Expected 124 unique node-state series, found {len(selected_files)}")
        return rows, {
            "status": "PASS",
            "selected_series_count": len(selected_files),
            "selected_by_modality": dict(sorted(by_modality.items())),
            "selection_source": "B-MTGNN/data/graph.csv",
        }

    def _audit_selected_node_state_series(
        self,
        contract: list[dict[str, Any]],
        header: list[str],
        source_rows: list[list[str]],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        max_error = 0.0
        worst: dict[str, Any] | None = None
        hashes: dict[str, str] = {}
        checked = 0
        by_type: dict[str, int] = {}
        seen_files: set[str] = set()
        for item in contract:
            path = self.forecast_dir / item["legacy_forecast_file"]
            if path.name in seen_files:
                raise Stage0ValidationError(f"Duplicate paper node-state source: {path.name}")
            seen_files.add(path.name)
            source_type = item["legacy_source_type"]
            source_name = item["canonical_name"]
            source_column = self._source_column(header, source_type, source_name)
            source_index = header.index(source_column)
            source_values = [float(row[source_index]) for row in source_rows[:HISTORICAL_MONTHS]]
            parsed = self._parse_forecast(path.name)
            error = max(abs(a - b) for a, b in zip(source_values, parsed["Data"]))
            if error > SOURCE_MATCH_TOLERANCE:
                raise Stage0ValidationError(
                    f"Paper node-state source mismatch for {path.name}: {error} > {SOURCE_MATCH_TOLERANCE}"
                )
            checked += 1
            by_type[source_type] = by_type.get(source_type, 0) + 1
            hashes[path.name] = _sha256(path)
            if error > max_error:
                max_error = error
                worst = {"forecast_file": path.name, "source_column": source_column, "max_abs_error": error}
        if checked != EXPECTED_NODE_COUNT:
            raise Stage0ValidationError(f"Expected {EXPECTED_NODE_COUNT} paper node-state series, found {checked}")
        return (
            {
                "status": "PASS",
                "series_count": checked,
                "series_by_source_type": dict(sorted(by_type.items())),
                "historical_points_checked_per_series": HISTORICAL_MONTHS,
                "tolerance": SOURCE_MATCH_TOLERANCE,
                "max_abs_error": max_error,
                "worst_match": worst,
            },
            hashes,
        )

    def _build_gap_rows(
        self,
        relations: dict[str, list[str]],
        state_rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        forecast_by_node: dict[str, list[dict[str, Any]]] = {}
        for row in state_rows:
            if row["phase"] != "forecast":
                continue
            forecast_by_node.setdefault(row["node_id"], []).append(row)
        for values in forecast_by_node.values():
            values.sort(key=lambda row: row["timestamp"])

        rows: list[dict[str, Any]] = []
        threat_counts: dict[str, int] = {}
        for threat, pmts in relations.items():
            threat_id = f"THREAT_{_slug(threat)}"
            threat_forecast = forecast_by_node.get(threat_id)
            if threat_forecast is None or len(threat_forecast) != FORECAST_MONTHS:
                raise Stage0ValidationError(f"Missing 36-month node-state forecast for {threat}")
            count = 0
            for pmt in pmts:
                pmt_id = f"PMT_{_slug(pmt)}"
                pmt_forecast = forecast_by_node.get(pmt_id)
                if pmt_forecast is None or len(pmt_forecast) != FORECAST_MONTHS:
                    raise Stage0ValidationError(f"Missing 36-month node-state forecast for PMT {pmt}")
                for year_index, year in enumerate((2025, 2026, 2027)):
                    start = year_index * 12
                    end = start + 12
                    threat_mean = sum(row["value_z"] for row in threat_forecast[start:end]) / 12
                    pmt_mean = sum(row["value_z"] for row in pmt_forecast[start:end]) / 12
                    rows.append(
                        {
                            "gap_field_id": f"GAP_{threat_id}_{pmt_id}_{year}",
                            "threat_id": threat_id,
                            "pmt_id": pmt_id,
                            "year": year,
                            "gap": threat_mean - pmt_mean,
                            "threat_state_mean_z": threat_mean,
                            "pmt_state_mean_z": pmt_mean,
                            "value_semantics": "yearly_mean_zscore_node_state_gap",
                        }
                    )
                count += 1
            threat_counts[threat] = count
        return rows, {
            "status": "PASS",
            "threat_count": len(threat_counts),
            "gap_relation_count": sum(threat_counts.values()),
            "gap_row_count": len(rows),
            "per_threat_relation_count": threat_counts,
            "source": "canonical paper-aligned 124-node state forecast",
        }

    def _source_column(self, header: list[str], source_type: str | None, name: str | None) -> str:
        if source_type is None or name is None:
            raise Stage0ValidationError("Unavailable feature cannot request a source column")

        candidate = self._optional_source_column(header, source_type, name)
        if candidate is None:
            raise Stage0ValidationError(f"Source column mapping failed for {source_type}:{name}: []")
        return candidate

    def _optional_source_column(self, header: list[str], source_type: str, name: str) -> str | None:

        def parse(column: str) -> tuple[str | None, str]:
            if column.startswith("Papers_"):
                return "paper_threat", column[len("Papers_") :]
            if column.startswith("Solution_") and column.endswith("_Papers"):
                return "solution", column[len("Solution_") : -len("_Papers")]
            if column == "War_Conflict_All":
                return "context", "War_Conflict_All"
            if column == "Holidays":
                return "context", "Holidays"
            if column.endswith("-ALL"):
                return "incident", column[:-4]
            return None, column

        target = _norm(name)
        aliases = {target}
        if target.endswith("ibe"):
            aliases.add(target[:-3])
        candidates = [
            column
            for column in header
            if parse(column)[0] == source_type and _norm(parse(column)[1]) in aliases
        ]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise Stage0ValidationError(f"Source column mapping failed for {source_type}:{name}: {candidates}")
        return candidates[0]

    def _find_forecast_file(self, source_type: str, name: str) -> Path:
        matches: list[Path] = []
        target = _norm(name)
        aliases = {target}
        if target.endswith("ibe"):
            aliases.add(target[:-3])
        for path in self._files:
            file_type, file_name = self._classify_forecast_file(path)
            if file_type == source_type and _norm(file_name) in aliases:
                matches.append(path)
        if len(matches) != 1:
            raise Stage0ValidationError(
                f"Forecast file mapping failed for {source_type}:{name}: {[path.name for path in matches]}"
            )
        return matches[0]

    @staticmethod
    def _classify_forecast_file(path: Path) -> tuple[str, str]:
        stem = path.stem
        if stem.startswith("Solution_"):
            return "solution", stem[len("Solution_") :]
        if stem.endswith("-ALL"):
            return "incident", stem[:-4]
        if stem in {"War_Conflict_All", "Holidays"}:
            return "context", stem
        return "paper_threat", stem

    def _stale_source_info(self) -> tuple[int, str]:
        with self.stale_source_path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        return len(rows), rows[-1][0]

    @staticmethod
    def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _graph_threat_core(raw: str) -> str:
        if raw.startswith("Mentions-"):
            return raw[len("Mentions-") :]
        if raw.endswith("-ALL"):
            return raw[:-4]
        return raw

    @staticmethod
    def _graph_pmt_core(raw: str) -> str:
        if raw.startswith("Solution_") and raw.endswith("_Mentions"):
            return raw[len("Solution_") : -len("_Mentions")]
        return raw
