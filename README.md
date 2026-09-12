# Context-aware Cyber Foresight with Multi-agent Critique and Strategic Uncertainty

> This repository is organized for reproducible research, with module-specific environments, frozen experiment artifacts, and explicit data/agent contracts.

![Framework Architecture](figure/framework.png)

## Requirements

We provide two ways to set up the environment:

**1. Unified Installation (Recommended for exploring the whole project)**\
To install all required dependencies across all modules into a single environment, run:
```bash
pip install -r requirements.txt
```

**2. Module-Specific Installation (Recommended for strict reproducibility)**\
Each module has its own `requirements.txt` to prevent potential version mismatches. If you only want to run a specific component:
```bash
# For Data Preparation
pip install -r Data_Preparation/requirements.txt

# For Modeling (B-MTGNN)
pip install -r B-MTGNN/requirements.txt

# For Multi-Agent Framework
pip install -r Multi-Agent/requirements.txt
```

## Dataset

The complete dataset (including raw and normalized data) used in this paper is directly included in this supplementary material package.

- `Dataset/`: Contains the final integrated dataset (`CT-0711-0125.csv`).
- `Data_Preparation/`: Contains intermediate outputs and raw sources.
- `B-MTGNN/data/`: Contains graph adjacency matrices and inputs for the B-MTGNN model.

## Training

To train the core B-MTGNN model on the dataset, navigate to the `B-MTGNN` directory and run the training script:

```bash
cd B-MTGNN
python train.py --data ./data/sm_data.txt --save model/Bayesian/o_model.pt
```

*For hyper-parameter optimization using random search, run:*
```bash
python train_test.py
```

## Evaluation

To evaluate the trained model and compare it against the baseline models, navigate to the `Comparative_Evaluation` directory and execute the respective evaluation scripts.

**Evaluate B-MTGNN (Example with 30 iterations):**
```bash
cd Comparative_Evaluation/BMTGNN
python BMTGNN.py
```

**Evaluate Baseline Models (Example: ARIMA):**
```bash
cd Comparative_Evaluation/Baselines/ARIMA
python ARIMA.py
```

## Pre-trained Models

To facilitate reproducibility without requiring full training, we provide pre-trained model checkpoints:

- **B-MTGNN Checkpoints**: Located in `Comparative_Evaluation/BMTGNN/` (e.g., `modelb10.pt`, `modelb30.pt`).
- **MTGNN Checkpoint**: Located in `Comparative_Evaluation/MTGNN/modelb1.pt`.

## Results

The current paper-contract migration exposes 124 canonical forecast node states (26 Threat + 98 PMT) over the 36-month 2025–2027 horizon for downstream Agent analysis.

## Multi-Agent System

Building upon the existing B-MTGNN experiment artifacts, the redesigned **LangGraph** pipeline currently implements Stage 0 and Stage 1 only. It does not retrain B-MTGNN or silently invoke the legacy Attacker→Defender→Mediator chain.

**Stage 0 — Canonical Forecast + Immutable Evidence**
- Migrates the existing experiment outputs into the paper-defined 124-node contract (26 Threat + 98 PMT).
- Builds case-specific `EvidencePack` objects from a frozen Evidence Store under `Data/Evidence/snapshots/`.
- Enforces `available_at <= analysis_cutoff_date` for temporal filtering.
- Uses deterministic Threat/PMT + claim-contract eligibility, followed by fixed-query Okapi BM25 ranking within six semantic evidence slots and bounded source-diverse top-k selection.
- Never inserts Agent reasoning back into the external Evidence Store.

**Stage 1 — Independent Forecast Critique**
- Runs an **Attack Feasibility Critic** and a **Defense Robustness Critic** independently on the same Forecast + EvidencePack.
- Uses one xhigh inference per Critic with a fixed internal review order: evidence sufficiency → supporting evidence → challenging evidence → alternative explanations → temporal consistency → specificity/traceability audit → residual uncertainty/final judgment. Stage 1 does not chain multiple self-refinement calls before the later inter-Critic debate.
- Returns structured `CriticAssessment` objects with stance, confidence, evidence sufficiency, claim→evidence links, unresolved questions, and unsupported-specificity flags.
- Freezes the paired pre-assessments as immutable content-addressed artifacts so an exact experiment identity is generated once and reused downstream.
- The main Agent experiment uses a frozen seven-case set selected before inference: two unconditional anchors plus one evidence-qualified representative from each of five gap/slope regimes. Regime cases require at least 5/6 evidence slots and four source families; the full 303 relations remain the quantitative forecast/gap population.
- `py -3 -m Pipeline.main` runs the seven cases as one resumable batch. Exact final artifacts and exact per-critic checkpoints are reused automatically, with live case/token progress written to `Data/Stage1/logs/` and `Data/Stage1/runs/`.

**Local Language Model Runtime**
- Model: **Qwen3.8-27B Q6_K_L** GGUF.
- Runtime: pinned **llama.cpp** CUDA build with a **131,072-token context**, full-GPU residency on the calibrated RTX 5000 Ada profile, Q8 KV, `batch=2048`, `ubatch=512`, and model-native thinking at `xhigh` reasoning effort.
- Speculative decoding is frozen only after a pre-experiment calibration that compares a non-speculative baseline with bounded MTP candidates and rejects candidates that fail a deterministic output-fingerprint correctness gate. The current selected profile is MTP depth 4 with `p_min=0.05`.
- Main-run sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 0.0, repetition penalty 1.0, seed 42.

The previous mutable **LightRAG + Ollama** implementation is retained only under `Multi-Agent/Lagacy/` for audit. In that legacy flow, each Agent issued its own hybrid RAG query and generated Agent analyses were inserted back into the same RAG storage; the active Stage 0/1 pipeline deliberately does not use that mechanism.

---

## Detailed Project Structure

For a deeper dive into the individual components of our framework, please refer to the documentation within each directory:

*   **[`PT_Extractor`](./PT_Extractor)**: Scripts for extracting Pertinent Technologies (PTs) using E-GPT and D-GPT, forming the Threats and Pertinent Technologies (TPT) graph.
*   **[`Data_Preparation`](./Data_Preparation)**: Scripts for extracting time-series features (NoI, A_NoM, PT_NoM, ACA, PH).
*   **[`B-MTGNN`](./B-MTGNN)**: The core implementation of the Bayesian Graph Neural Network, including data smoothing and future forecasting scripts (`forecast.py`, `pt_plots.py`).
*   **[`Comparative_Evaluation`](./Comparative_Evaluation)**: Extensive evaluation logic against baseline models.
*   **[`Multi-Agent`](./Multi-Agent)**: Stage-separated LangGraph implementation. `Stage0/` owns forecast/evidence preparation and retrieval, `Stage1/` owns Qwen critique/frozen artifacts, `Pipeline/` owns cross-stage orchestration, and `Lagacy/` is archival only.
