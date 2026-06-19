# ⚽ Player Selection MCDA — Evidence-Based Multi-Criteria Decision Support System for European Football

> A transparent, reproducible **Multi-Criteria Decision Analysis (MCDA)** pipeline that combines **objective weighting** (Random Forest, Entropy, CRITIC) with **TOPSIS** ranking to produce position-specific player shortlists across Europe's Top-5 Leagues (2024/25 season). Built as a human-centred **Decision Support System (DSS)** to augment — not replace — professional scouting.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 📌 Project Overview

European football has become a **multi-billion-pound industry** where player recruitment decisions carry significant sporting, financial, and reputational risk. Despite the rise of analytics, transfer decisions are still dominated by **subjective scouting heuristics** and noisy single-metric heuristics (e.g. ranking strikers by goals alone).

This project delivers a **fully reproducible, evidence-based DSS** that:

- Ingests season-level performance data for **2,854 player-seasons × 267 variables** from the Top-5 European Leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).
- Applies **position-aware feature engineering** (FW, MF, DF, GK).
- Computes **three complementary objective weighting schemes** (Random Forest feature importance, Shannon Entropy, CRITIC).
- Aggregates criteria via the **TOPSIS** algorithm to generate position-specific rankings.
- Validates rankings using **NDCG**, **Kendall's τ**, **Spearman's ρ**, and **Jaccard@k** sensitivity analyses.
- Exposes everything through an **interactive Streamlit DSS** with exportable shortlists and visualisations.

---

## 🎯 Business Problem & Research Objectives

### Business Problem
Football clubs invest hundreds of millions annually in transfers under **incomplete information**, **Financial Fair Play constraints**, and **post-Bosman market volatility**. Traditional scouting is subjective, single-metric rankings are reductive, and existing ML approaches often function as black boxes — unsuitable for high-stakes managerial decisions.

### Research Questions
| ID  | Question |
|-----|----------|
| **RQ0** | Can an MCDM framework combining objective weighting + TOPSIS produce transparent, position-specific rankings that complement human scouting? |
| **RQ1** | Which weighting method (Random Forest / Entropy / CRITIC) yields the most stable and interpretable rankings? |
| **RQ2** | How does ranking quality (NDCG, Kendall's τ) vary across weighting schemes, positions, and top-k thresholds? |
| **RQ3** | How sensitive are outputs to design choices (criteria, benefit/cost assignment, normalisation)? |
| **RQ4** | Do shortlists align with external benchmarks and expert assessments? |
| **RQ5** | What is the impact of alternative criteria settings on shortlist composition? |
| **RQ6** | How should the DSS be designed for maximum transparency, reproducibility, and usability? |

---

## 🧪 Methodology

The pipeline follows a five-stage architecture: **Data → Feature Engineering → Weighting (ML/Statistical) → MCDM Aggregation (TOPSIS) → Robustness Evaluation**.

### 1. Position-Aware Feature Engineering
Distinct criteria sets are mapped to each positional group to reflect tactical responsibilities:

| Position | Benefit Criteria (↑) | Cost Criteria (↓) |
|----------|----------------------|-------------------|
| **Forwards (FW)** | Gls, xG, Ast, xAG, SoT, PrgC, KP | CrdY, CrdR |
| **Midfielders (MF)** | PrgP, KP, xAG, Ast, SCA, Cmp%, Int | CrdY, CrdR |
| **Defenders (DF)** | Tkl, Int, Clr, Blocks, PrgP, Recov | Err, CrdR |
| **Goalkeepers (GK)** | Save%, CS%, PKsv | GA90, Err |

### 2. Objective Weighting Schemes
Three orthogonal techniques are computed per position to triangulate criterion importance:

- **🌲 Random Forest Feature Importance** — A `RandomForestRegressor` is trained to predict goal output (proxy for sporting contribution); Gini-based importances yield supervised weights.
- **📐 Shannon Entropy Method** — Quantifies information dispersion of each criterion across players; high-entropy (informative) criteria receive higher weights.
- **⚖️ CRITIC (CRiteria Importance Through Intercriteria Correlation)** — Combines **contrast intensity** (standard deviation) with **conflict** (inter-criterion correlation) to weight criteria that are both discriminative and non-redundant.

### 3. TOPSIS Aggregation
The **Technique for Order of Preference by Similarity to Ideal Solution** is applied to the weighted normalised decision matrix:

1. **Vector normalisation** of the criteria matrix.
2. **Weighting** using one of {RF, Entropy, CRITIC}.
3. Construction of the **Positive Ideal Solution (PIS)** and **Negative Ideal Solution (NIS)** respecting benefit/cost orientations.
4. Computation of **Euclidean distances** of each player to PIS and NIS.
5. Calculation of the **Closeness Coefficient** *Cᵢ ∈ [0,1]* — the final TOPSIS score driving the ranking.

### 4. Ranking Evaluation & Sensitivity Analysis
| Metric | Purpose |
|--------|---------|
| **NDCG@k** | Quality of top-k ranking against graded relevance proxies |
| **Kendall's τ** | Pairwise concordance between weighting schemes |
| **Spearman's ρ** | Monotonic agreement with single-metric baselines |
| **Jaccard@k** | Shortlist overlap stability under perturbations |
| **Sanity checks** | Validate benefit/cost monotonicity |

Target thresholds: **NDCG ≥ 0.70**, **Kendall's τ ≥ 0.60**, **top-k variation ≤ 15%** under weight perturbation.

---

## 📊 Data Sources & Processing

### Primary Dataset
- **Source:** Kaggle — *Top-5 European Leagues 2024/25 Season Player Statistics*
- **Coverage:** Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- **Volume:** 2,854 player-seasons × 267 variables (232 numeric)
- **Categories:** Identification, Playing Time, Attacking, Passing & Creativity, Defensive, Goalkeeping, Possession, Discipline

### Data Preparation Pipeline
1. **Deduplication** and harmonisation of club/league naming conventions.
2. **Position normalisation** into {FW, MF, DF, GK} preserving multi-role players via lenient substring matching.
3. **Missing-data handling** — median imputation for numerical fields, mode for categorical, and **k-NN imputation** for context-dependent variables; structural missingness (e.g. GK stats for outfielders) preserved as informative.
4. **Per-90 normalisation** to ensure cross-player comparability irrespective of minutes played.
5. **Minimum exposure filter** (≥ 500 minutes) to mitigate small-sample noise.
6. **Outlier treatment** via winsorisation for heavy-tailed distributions.
7. **Multicollinearity reduction** — removal of redundant pairs (e.g. Gls vs G-PK, Tkl vs TklW).

---

## 🛠️ Technical Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10 |
| **Data Wrangling** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` (RandomForestRegressor, KNNImputer, preprocessing) |
| **MCDM Algorithms** | Custom NumPy implementations of TOPSIS, Entropy & CRITIC weighting |
| **Visualisation** | `plotly` (interactive scatter plots, bar charts, weight comparisons) |
| **DSS Interface** | `Streamlit` (filters: league / club / position / min-minutes; live ranking & export) |
| **Reproducibility** | Virtual env, pinned `requirements.txt`, fixed random seeds |
| **Version Control** | Git / GitHub |

---

## 📈 Key Results & Findings

- ✅ **Transparent position-specific shortlists** generated across all four positional groups under three weighting paradigms — fully auditable from raw data to final rank.
- ✅ **High inter-method agreement** between Random Forest, Entropy, and CRITIC schemes (Kendall's τ consistently above target threshold for most positions), confirming **ranking robustness**.
- ✅ **MCDA outperforms naive single-metric rankings** (e.g. ranking strikers by Gls alone) by surfacing **well-rounded players** — e.g. forwards with strong xG + xAG + progressive carries who would be missed by goal-count baselines.
- ✅ **Sensitivity analysis confirms stability**: top-k shortlists vary by less than the 15% target threshold under controlled weight perturbations and alternative normalisation strategies.
- ✅ **Interactive DSS** delivers exportable rankings, weight visualisations, and scatter-plot diagnostics (e.g. *Goals vs Assists sized by TOPSIS score*) suitable for direct integration into scouting workflows.
- ⚠️ **Acknowledged limitation:** absence of market-value variable in the dataset prevents direct economic optimisation — flagged as primary direction for future work.

---

## 🖼️ Visualisations & DSS Interface

The Streamlit application delivers:

- 🔍 **Filter panel** — league, club, position, minimum minutes, weighting scheme.
- 🏆 **Ranked shortlist table** — TOPSIS score, rank, and key positional metrics.
- 📊 **Weight inspection** — bar charts comparing RF / Entropy / CRITIC weight vectors per position.
- 🔘 **Performance scatter plots** — e.g. xG vs xAG with point size proportional to TOPSIS score.
- 💾 **Export** — full rankings to CSV for downstream analysis or report generation.

*Example use case demonstrated in the dissertation: ranking AC Milan strikers in Serie A 2024/25 — full ranking, weight breakdown, and Goals × Assists scatter delivered in seconds.*

---

## 🗂️ Repository Structure

```
- `mcdm_tool_2024_2025.py`: Main Python script containing the full analytical pipeline (Data loading, Weighting with Random Forest/Entropy, TOPSIS implementation, and Results export).
- `requirements.txt`: List of Python libraries required to run the project.
- `README.md`: Project documentation and overview.
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<username>/player-selection-mcda.git
cd player-selection-mcda

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch the Streamlit DSS
streamlit run app/mcdm_tool_2024_2025.py
```

---

## 🔭 Future Work

- Integration of **market-value data** for cost-adjusted scoring and ROI-aware shortlisting.
- Extension to **binary integer programming** for full XI selection under formation, budget, and home-grown constraints.
- **One-season-ahead forecasting** using gradient boosting and neural architectures.
- **Human-in-the-loop feedback** capture (accept / modify / reject decisions) for active-learning refinement of weights.
- **Multi-season longitudinal analysis** to capture player development trajectories.

---

## 👤 About

**Author:** Johann Darío Oviedo Silva
**Programme:** MSc Business Analytics and Decision Sciences (2024–2025)
**Institution:** [Leeds University Business School](https://business.leeds.ac.uk/) — **University of Leeds**, United Kingdom
**Module:** LUBS5579M — Dissertation
**Supervisor:** Sarah Fores
**Dissertation Title:** *Evidence-Based Multi-Criteria Decision Support System for Player Selection in European Football*

This repository hosts the full implementation, codebase, and Streamlit DSS developed as part of the MSc dissertation submitted to the University of Leeds in 2025. The project bridges **decision science**, **machine learning**, and **sports analytics**, demonstrating end-to-end competence in data engineering, ML-driven feature weighting, multi-criteria decision modelling, and human-centred decision-support system design.

---

## 📜 License & Citation

This work is released for **academic and research purposes**. If you use or build upon this methodology, please cite:

> Oviedo Silva, J.D. (2025). *Evidence-Based Multi-Criteria Decision Support System for Player Selection in European Football.* MSc Dissertation, Leeds University Business School, University of Leeds.

---

> 💬 *"The DSS is intended to supplement — not replace — the expertise of scouts and managers. It provides interpretable, transparently weighted, position-specific recommendations to inform, not override, professional judgement."*
