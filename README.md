# AI Misuse Evaluator

A modular Python framework for evaluating the **justifiability** of AI use cases using a five-dimensional responsibility framework. The system integrates environmental impact, ethical risk, creative displacement, purpose justification, and transparency/explainability into a single **Justifiability Score (0–100)** with explainable component breakdown, robustness analysis, and quantitative evaluation outputs.

> COMP3931 Individual Project — University of Leeds — 2025/26

🌐 **[Live web demo →](https://hasini08.github.io/ai-misuse-evaluator)**

---

## Project Goal

To provide a **reproducible computational framework** for comparing AI deployments and identifying use cases that are high-cost or low-benefit relative to their justification. Unlike existing responsible AI frameworks that focus on a single dimension (e.g. fairness, or environmental impact alone), this framework integrates **five dimensions** into one unified, configurable score with full component transparency.

---

## How It Works

Each AI use case is evaluated across five normalised components (0–1 goodness scale):

| Component | Input field | Direction |
|---|---|---|
| Environmental impact | `emissions_kgco2e` | Lower = better |
| Ethical risk | `ethical_risk_score` (1–5) | Lower = better |
| Creative displacement | `creative_displacement_score` (1–5) | Lower = better |
| Purpose justification | `purpose_category` | essential > beneficial > low_benefit > harmful |
| Transparency / explainability | `transparency_score` (1–5) | Higher = better |

The **Justifiability Score** is computed as:

```
score = 100 × (w_em × E + w_eth × H + w_cr × C + w_pur × P + w_tr × T)
              ────────────────────────────────────────────────────────────
                           w_em + w_eth + w_cr + w_pur + w_tr
```

Default weights: `w_em=0.35, w_eth=0.25, w_cr=0.15, w_pur=0.10, w_tr=0.15`

Scores are classified into three bands:

| Band | Score range | Interpretation |
|---|---|---|
| High | 71–100 | Justifiable deployment |
| Medium | 41–70 | Borderline — requires scrutiny |
| Low | 0–40 | Difficult to justify |

> **Non-compensatory guardrail:** Use cases with `purpose_category = harmful` receive a 0.60× penalty. Those that also have `ethical_risk_score ≥ 4` receive a further 0.70× penalty (combined multiplier = 0.42), reflecting the design principle that explicitly harmful intent is disqualifying regardless of other properties.

---

## Results (40 Use Cases)

| Rank | Use case | Score | Band |
|---|---|---|---|
| 1 | Accessibility_Captioning | 96.25 | High |
| 2 | Earthquake_Early_Warning | 94.26 | High |
| 3 | Wildfire_Spread_Predictor | 87.64 | High |
| … | … | … | … |
| 38 | Deepfake_Voice_Tool | 8.23 | Low |
| 39 | Autonomous_Weapon_Targeting | 6.60 | Low |
| 40 | Nonconsensual_Deepfake_Generator | 0.00 | Low |

**Score distribution by purpose category:**

| Category | n | Mean | Min | Max |
|---|---|---|---|---|
| essential | 10 | 81.57 | 67.27 | 96.25 |
| beneficial | 13 | 67.15 | 43.66 | 85.13 |
| low_benefit | 10 | 42.94 | 27.75 | 62.16 |
| harmful | 7 | 9.14 | 0.00 | 13.40 |

**Correlation with Justifiability Score (Pearson r, two-tailed):**

| Dimension | r | p-value |
|---|---|---|
| Ethical risk | −0.881 | < 0.001 *** |
| Transparency | +0.717 | < 0.001 *** |
| Creative displacement | −0.488 | < 0.001 *** |
| Environmental emissions | −0.192 | 0.229 |

---

## Validation

### Unit tests — 66 passing

```bash
pytest tests/ -v
# 66 passed
```

| Test file | Tests | Coverage |
|---|---|---|
| `test_scoring_engine.py` | 38 | Output structure, component behaviour, label thresholds, weight config, edge cases, guardrail |
| `test_transparency.py` | 15 | Transparency dimension: bounds, backward compatibility, contribution, integration |
| `test_face_validity.py` | 5 | Ground truth validation (15 hand-labelled cases) |
| `test_baseline.py` | 4 | Baseline comparison (ethics-only, emissions-only, equal weights) |
| `test_sanity.py` | 4 | Sanity checks (monotonicity, harmful penalty) |

### Face validity — 87% accuracy

15 hand-labelled use cases with expected High/Medium/Low outcomes. The framework correctly classified 13/15. The two misclassified cases were near the Medium/High boundary, reflecting genuine ambiguity in the scoring of beneficial-purpose tools with moderate ethical risk. For context, a majority-class baseline achieves 47% and a random classifier achieves 33%.

### Inter-rater reliability

A two-rater inter-rater reliability study was conducted across 12 use cases using a bespoke HTML scoring instrument:

| Dimension | Cohen's weighted κ | Interpretation |
|---|---|---|
| Ethical risk | 0.955 | Almost perfect |
| Transparency | 0.720 | Substantial |

### Robustness

| Experiment | Finding |
|---|---|
| Sensitivity analysis (w_em 0.20→0.60, 5D) | Rank stability: 0.970–0.999 (Spearman ρ) |
| Monte Carlo (200 Dirichlet samples, 5D) | Earthquake_Early_Warning wins 56%, Accessibility_Captioning 44% |
| Ablation (leave-one-dimension-out) | Emissions removal causes most band changes (22.5%); transparency 17.5% |
| Threshold sensitivity (4 configurations) | Max 22.5% of cases change band under extreme threshold variation |
| Baseline comparison (equal-weight 5D) | Spearman ρ = 0.989 vs default |

---

## Repository Structure

```
ai-misuse-evaluator/
├── src/
│   ├── scoring_engine.py          # Core 5D scoring logic — compute_scores()
│   ├── run_pipeline.py            # Pipeline orchestration and CLI
│   ├── visualise.py               # Figure generation (6 figures)
│   └── emissions_measurement.py   # CodeCarbon integration
├── tests/
│   ├── test_scoring_engine.py     # 38 unit tests (structure, components, edge cases)
│   ├── test_transparency.py       # 15 transparency dimension tests
│   ├── test_face_validity.py      # Ground truth validation (15 labelled cases)
│   ├── test_baseline.py           # Baseline comparison tests
│   └── test_sanity.py             # Sanity checks
├── data/
│   ├── usecases.csv               # Input dataset (40 use cases, 5 dimensions)
│   ├── results_scored.csv         # Scored output with component breakdown
│   ├── analysis_summary.txt       # Human-readable analysis report
│   ├── sensitivity_results.csv    # Sensitivity experiment results
│   ├── pareto_frontier.csv        # Pareto-optimal use cases
│   ├── montecarlo_stability.csv   # Monte Carlo win rates
│   ├── ablation_results.csv       # Leave-one-dimension-out results
│   └── rater_1.csv                # Inter-rater reliability data
├── figures/                       # Generated visualisations
│   ├── score_top_bottom.png
│   ├── score_distribution.png
│   ├── correlation_matrix.png
│   ├── sensitivity_weights.png
│   ├── montecarlo_top10.png
│   └── pareto_emissions_vs_ethics.png
├── inter_rater_scoring_sheet.html # HTML scoring instrument for inter-rater study
├── inter_rater_analysis.py        # Cohen's kappa computation
├── index.html                     # Interactive web interface
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
python src/run_pipeline.py --analyze --sensitivity --pareto --montecarlo --ablation --threshold
```

### Flags

| Flag | Description |
|---|---|
| `--analyze` | Generate human-readable analysis summary |
| `--sensitivity` | Run 5D sensitivity analysis (vary emissions weight) |
| `--pareto` | Compute Pareto-optimal frontier (5 dimensions) |
| `--montecarlo` | Run 5D Monte Carlo robustness experiment (200 iterations) |
| `--ablation` | Run leave-one-dimension-out ablation study |
| `--threshold` | Run band threshold sensitivity analysis |
| `--mc_iters N` | Set number of Monte Carlo iterations (default: 200) |
| `--input PATH` | Custom input CSV path (default: data/usecases.csv) |
| `--output PATH` | Custom output CSV path (default: data/results_scored.csv) |

### Run tests

```bash
pytest tests/ -v
```

### Custom weights

```python
from src.scoring_engine import compute_scores
import pandas as pd

df = pd.read_csv("data/usecases.csv")
scored = compute_scores(df, w_emissions=0.30, w_ethics=0.40,
                        w_creativity=0.15, w_purpose=0.10, w_transparency=0.05)
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
| `transparency_score` | int 1–5 | 1 = fully opaque, 5 = fully explainable/auditable |
| `model_size` | int | (optional) Model parameter count |

> **Backward compatibility:** Datasets without a `transparency_score` column are handled gracefully — the dimension defaults to a neutral value of 0.5 and the framework continues to operate on the remaining four dimensions.

---

## Design Decisions

**Why weighted sum?** The weighted sum model (WSM) was chosen for interpretability — each component's contribution is directly proportional to its weight, making scores auditable. Sensitivity analysis confirms rankings are robust to weight variation (Spearman ρ ≥ 0.970).

**Why five dimensions?** The v2 transparency extension addresses a gap in the v1 framework: two use cases with identical emissions, ethics, creativity, and purpose scores could have very different societal justifiability depending on whether their decision-making is auditable. Ablation analysis confirms transparency is load-bearing (17.5% band changes when removed).

**Why w_em = 0.35?** Environmental impact is the most objectively measurable dimension (quantified in physical units via CodeCarbon), making it the most reliable anchor. Emissions weight is highest, but reduced from v1 (0.40) to accommodate transparency.

**Why a non-compensatory guardrail?** Initial validation found that low-emissions harmful use cases could score above the Low/Medium boundary purely from favourable emissions scores. The guardrail reflects the principle that explicitly harmful intent is disqualifying regardless of other properties.

**Why log-clip normalisation for emissions?** A single outlier (High_Emission_Novelty_Chatbot, 0.500 kgCO₂e vs median ~0.008) would compress 95% of cases into under 5% of the normalised scale using linear normalisation. The 95th-percentile clip + log1p transform resolves this without discarding the outlier.

---

## Requirements

See `requirements.txt`. Key dependencies:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.0
scikit-learn>=1.3
codecarbon>=2.3
```

---

## Author

Hasini Yahampath — University of Leeds, COMP3931 Individual Project, 2025/26

GitHub: [hasini08/ai-misuse-evaluator](https://github.com/hasini08/ai-misuse-evaluator)