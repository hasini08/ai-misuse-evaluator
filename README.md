# AI Misuse Evaluator

A modular Python framework for evaluating the **justifiability** of AI use cases using a multi-dimensional responsibility framework. The system integrates environmental impact, ethical risk, creative displacement, and purpose justification into a single **Justifiability Score (0–100)** with explainable component breakdown, robustness analysis, and quantitative evaluation outputs.

> COMP3931 Individual Project — University of Leeds — 2025/26

---

## Project Goal

To provide a **reproducible computational framework** for comparing AI deployments and identifying use cases that are high-cost or low-benefit relative to their justification. Unlike existing responsible AI frameworks that focus on a single dimension (e.g. fairness, or environmental impact), this framework integrates four dimensions into one unified, configurable score.

---

## How It Works

Each AI use case is evaluated across four normalised components (0–1 goodness scale):

| Component | Input field | Direction |
|---|---|---|
| Environmental impact | `emissions_kgco2e` | Lower = better |
| Ethical risk | `ethical_risk_score` (1–5) | Lower = better |
| Creative displacement | `creative_displacement_score` (1–5) | Lower = better |
| Purpose justification | `purpose_category` | essential > beneficial > low_benefit > harmful |

The **Justifiability Score** is computed as:

```
score = 100 × (w_em × emissions_norm + w_eth × ethics_norm + w_cr × creativity_norm + w_pur × purpose_norm)
              ─────────────────────────────────────────────────────────────────────────────────────────────
                                        w_em + w_eth + w_cr + w_pur
```

Default weights: `w_em=0.40, w_eth=0.25, w_cr=0.20, w_pur=0.15`

Scores are classified into three bands:

| Band | Score range | Interpretation |
|---|---|---|
| High | 71–100 | Justifiable deployment |
| Medium | 41–70 | Borderline — requires scrutiny |
| Low | 0–40 | Difficult to justify |

> **Guardrail:** Use cases with `purpose_category = harmful` are penalised regardless of other scores, reflecting the design principle that explicitly harmful intent is disqualifying.

---

## Results (40 Use Cases)

| Rank | Use case | Score | Band |
|---|---|---|---|
| 1 | Accessibility_Captioning | 100.00 | High |
| 2 | Medical_Triage_Assistant | 96.72 | High |
| 3 | Earthquake_Early_Warning | 93.44 | High |
| … | … | … | … |
| 38 | Deepfake_Voice_Tool | 10.01 | Low |
| 39 | Autonomous_Weapon_Targeting | 8.74 | Low |
| 40 | Nonconsensual_Deepfake_Generator | 0.00 | Low |

**Score distribution by purpose category:**

| Category | n | Mean | Min | Max |
|---|---|---|---|---|
| essential | 10 | 86.51 | 75.64 | 100.00 |
| beneficial | 13 | 72.74 | 55.61 | 87.47 |
| low_benefit | 10 | 41.61 | 23.50 | 62.41 |
| harmful | 7 | 11.27 | 0.00 | 16.22 |

**Correlation with Justifiability Score (Pearson):**
- Ethical risk: **−0.820** (strongest predictor)
- Creative displacement: **−0.575**
- Emissions: **−0.225**

---

## Validation

### Unit tests — 51 passing

```bash
pytest tests/ -v
# 51 passed in 11.61s
```

Tests cover: output structure, all four component behaviours, label thresholds, weight configuration, edge cases (single row, constant values, unknown categories, large datasets), guardrail penalties, and the `minmax` helper.

### Face validity — 87% accuracy

15 hand-labelled use cases with expected High/Medium/Low outcomes. The framework correctly classified 13/15, with two borderline beneficial cases (AI_Resume_Screener, AI_Legal_Document_Drafter) scoring just above the Medium/High boundary.

### Baseline comparison

The multi-dimensional score produces **meaningfully different rankings** from any single-component baseline:
- An emissions-only baseline incorrectly rewards low-emissions harmful use cases
- An ethics-only baseline misses creative displacement entirely
- Spearman correlation between default and emissions-only rankings: < 0.95

### Robustness

| Experiment | Finding |
|---|---|
| Sensitivity analysis (w_em 0.20→0.60) | Rank stability: 0.968–1.000 |
| Monte Carlo (200 Dirichlet samples) | Accessibility_Captioning top-ranked in 100% of runs |
| Baseline comparison | Rank correlation with equal weights: 0.988 |

---

## Repository Structure

```
ai-misuse-evaluator/
├── src/
│   ├── scoring_engine.py        # Core scoring logic — compute_scores()
│   ├── run_pipeline.py          # Pipeline orchestration and CLI
│   ├── visualise.py             # Figure generation
│   └── emissions_measurement.py # CodeCarbon integration
├── tests/
│   ├── test_scoring_engine.py   # Unit tests (structure, components, edge cases)
│   ├── test_face_validity.py    # Ground truth validation (15 labelled cases)
│   ├── test_baseline.py         # Baseline comparison tests
│   └── test_sanity.py           # Sanity checks
├── data/
│   ├── usecases.csv             # Input dataset (40 use cases)
│   ├── results_scored.csv       # Scored output
│   ├── analysis_summary.txt     # Human-readable analysis report
│   ├── sensitivity_results.csv  # Sensitivity experiment results
│   ├── pareto_frontier.csv      # Pareto-optimal use cases
│   └── montecarlo_stability.csv # Monte Carlo win rates
├── figures/                     # Generated visualisations
│   ├── score_bar.png
│   ├── score_distribution.png
│   ├── correlation_matrix.png
│   ├── sensitivity_weights.png
│   ├── montecarlo_top10.png
│   └── pareto_emissions_vs_ethics.png
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/hasini08/ai-misuse-evaluator.git
cd ai-misuse-evaluator
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python src/run_pipeline.py --analyze --sensitivity --pareto --montecarlo
```

### Flags

| Flag | Description |
|---|---|
| `--analyze` | Generate human-readable analysis summary |
| `--sensitivity` | Run sensitivity analysis (vary emissions weight) |
| `--pareto` | Compute Pareto-optimal frontier |
| `--montecarlo` | Run Monte Carlo robustness experiment (200 iterations) |
| `--mc_iters N` | Set number of Monte Carlo iterations (default: 200) |
| `--input PATH` | Custom input CSV path (default: data/usecases.csv) |
| `--output PATH` | Custom output CSV path (default: data/results_scored.csv) |

### Run tests

```bash
pytest tests/ -v
```

### Custom weights

Weights are configurable in code via `compute_scores()`:

```python
from src.scoring_engine import compute_scores
import pandas as pd

df = pd.read_csv("data/usecases.csv")
scored = compute_scores(df, w_emissions=0.30, w_ethics=0.40, w_creativity=0.20, w_purpose=0.10)
print(scored[["use_case_name", "justifiability_score", "label"]])
```

---

## Input Format

The pipeline expects `data/usecases.csv` with these columns:

| Column | Type | Description |
|---|---|---|
| `use_case_name` | string | Name of the AI use case |
| `emissions_kgco2e` | float | Estimated CO₂ emissions in kg |
| `purpose_category` | string | One of: `essential`, `beneficial`, `low_benefit`, `harmful` |
| `ethical_risk_score` | int 1–5 | 1 = lowest risk, 5 = highest risk |
| `creative_displacement_score` | int 1–5 | 1 = minimal displacement, 5 = high displacement |
| `model_size` | int | (optional) Model parameter count |

---

## Design Decisions

**Why weighted sum?** The weighted sum model (WSM) was chosen for interpretability — each component's contribution is directly proportional to its weight, making scores auditable. Sensitivity analysis confirms rankings are robust to weight variation (Spearman stability ≥ 0.968).

**Why w_em = 0.40?** Environmental impact is the most objectively measurable dimension (quantified in physical units via CodeCarbon), making it the most reliable anchor for the framework.

**Why a guardrail for harmful use cases?** Initial validation found that low-emissions harmful use cases (e.g. Targeted_Harassment_Tool at 0.008 kgCO₂e) could score above the Low/Medium boundary purely from favourable emissions scores. A penalty was introduced reflecting the principle that explicitly harmful intent is disqualifying regardless of other properties.

---

## Requirements

```
pandas
numpy
matplotlib
pytest
codecarbon
```

---

## Author

Hasini — University of Leeds, COMP3931 Individual Project, 2024/25

GitHub: [hasini08/ai-misuse-evaluator](https://github.com/hasini08/ai-misuse-evaluator)