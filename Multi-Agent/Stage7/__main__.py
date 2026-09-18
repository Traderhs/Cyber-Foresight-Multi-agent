from __future__ import annotations

import asyncio
import argparse
from pathlib import Path

from Stage7.loaders import load_main_case_bundles
from Stage7.runner import run_stage7_validation
from Stage7.variant_generation import run_final_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 7 validation harness over frozen Stage 1-6 artifacts.")
    parser.add_argument(
        "--with-llm-audits",
        action="store_true",
        help="Run the internal grounding and mediator semantic integrity audits.",
    )
    parser.add_argument(
        "--run-decision-variants",
        default="",
        metavar="A,B,C",
        help=(
            "Run only the final paper-facing decision variants before scoring. "
            "A runs the prompt-paraphrase robustness variants, B runs the single-agent architecture baseline, "
            "and C runs the context-free baseline. Each family uses exactly two workers on the existing "
            "two-slot llama.cpp runtime."
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.run_decision_variants:
        requested = {part.strip().upper() for part in args.run_decision_variants.split(",") if part.strip()}
        unsupported = sorted(requested - {"A", "B", "C"})
        if unsupported:
            raise ValueError(f"Unsupported --run-decision-variants axis: {unsupported}")
        bundles = load_main_case_bundles(root)
        variant_paths = asyncio.run(
            run_final_variants(root, bundles, axes=requested)
        )
        print("Stage 7 final decision variants:")
        for variant_path in variant_paths:
            print(f"  {variant_path}")
    artifact, path = run_stage7_validation(root, with_llm_audits=args.with_llm_audits)
    statuses = {axis["experiment_id"]: axis["status"] for axis in artifact["axes"]}
    print(f"Stage 7 artifact: {path}")
    print(f"Artifact SHA-256: {artifact['artifact_sha256']}")
    print("Axis statuses:")
    for key, value in statuses.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
