from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution despite the parent directory name containing a hyphen.
STAGE0_PARENT = Path(__file__).resolve().parents[1]
if str(STAGE0_PARENT) not in sys.path:
    sys.path.insert(0, str(STAGE0_PARENT))

from Stage0.evidence_store import EvidenceSnapshotWriter, EvidenceStore  # noqa: E402
from Stage0.ingest import build_evidence_snapshot  # noqa: E402
from Stage0.paper_migration import PaperForecastMigrator  # noqa: E402
from Stage0.registry import validate_source_registry  # noqa: E402


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0 paper-native forecast and evidence utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-registry")
    sub.add_parser("migrate-forecast")
    validate = sub.add_parser("validate-forecast")
    validate.add_argument("--project-root", default=str(_project_root()))
    empty = sub.add_parser("create-empty-snapshot")
    empty.add_argument("snapshot_id")
    empty.add_argument("--project-root", default=str(_project_root()))
    evidence = sub.add_parser("validate-evidence")
    evidence.add_argument("snapshot_id")
    evidence.add_argument("--project-root", default=str(_project_root()))
    ingest = sub.add_parser("ingest-evidence")
    ingest.add_argument("--snapshot-date", required=True, help="ISO YYYY-MM-DD")
    ingest.add_argument("--snapshot-id", default=None)
    ingest.add_argument("--project-root", default=str(_project_root()))
    build = sub.add_parser("build-stage0")
    build.add_argument("--snapshot-date", required=True, help="ISO YYYY-MM-DD")
    build.add_argument("--snapshot-id", default=None)
    build.add_argument("--project-root", default=str(_project_root()))
    args = parser.parse_args()

    if args.command == "validate-registry":
        print(json.dumps(validate_source_registry(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "migrate-forecast":
        audit = PaperForecastMigrator(_project_root()).run()
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-forecast":
        result = PaperForecastMigrator(args.project_root).validate_output()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "create-empty-snapshot":
        root = Path(args.project_root)
        writer = EvidenceSnapshotWriter(root / "Multi-Agent/Results/Stage0/Evidence", args.snapshot_id)
        path = writer.finalize()
        print(path)
        return 0
    if args.command == "validate-evidence":
        root = Path(args.project_root)
        store = EvidenceStore(root / "Multi-Agent/Results/Stage0/Evidence/snapshots" / args.snapshot_id)
        print(json.dumps({"status": "PASS", "snapshot_id": store.snapshot_id, "record_count": len(store.records)}, indent=2))
        return 0
    if args.command == "ingest-evidence":
        audit = build_evidence_snapshot(args.project_root, args.snapshot_date, args.snapshot_id)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0
    if args.command == "build-stage0":
        forecast_audit = PaperForecastMigrator(args.project_root).run()
        evidence_audit = build_evidence_snapshot(args.project_root, args.snapshot_date, args.snapshot_id)
        print(
            json.dumps(
                {"status": "PASS", "forecast": forecast_audit, "evidence": evidence_audit},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
