# Forecast data

This directory is generated from the paper-defined Stage 0 forecast contract.

Stage 0 does **not** train or rerun B-MTGNN. The existing B-MTGNN forecast artifacts are treated as read-only upstream experiment outputs; this directory is a deterministic Agent-facing canonical view of those results.

The historical input schema is **124 graph nodes × 4 features** (`NoI`, `NoP`, `ACA`, `PH`) with structured masking and z-score normalization. Threat `NoP` and PMT `NoI` are masked by the paper role contract; Threat `NoI` remains sparse where incident data is unavailable.

The paper forecast output `Y` is a separate **124-node state**. `graph.csv` deterministically identifies the existing experiment output used for each paper node (16 Threat NoI + 10 Threat NoP + 98 PMT NoP). Stage 0 validates and exposes exactly those 124 node-state series and does not enumerate non-contract forecast series.

The selected 124 node states are z-scored using each node's 162-month history. Canonical Threat–PMT gaps are derived deterministically from those existing forecast outputs; no model inference is executed by Stage 0.

Generate the dataset with:

```powershell
py -3.13 Multi-Agent/Stage0/cli.py migrate-forecast
```

Expected generated files:

- `node_registry.json`
- `feature_contract.json`
- `historical_node_features.jsonl`
- `node_state_contract.json`
- `node_state_series.jsonl`
- `threat_pmt_gaps.jsonl`
- `manifest.json`
- `migration_audit.json`
