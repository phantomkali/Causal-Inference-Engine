# ⚗️ Causal Inference Engine

> **Estimate whether a treatment *causes* an outcome — not just correlates with it.**
>
> A production-ready Python toolkit for causal effect estimation from observational data,
> powered by DoWhy, EconML, and Streamlit.

---

## 📖 Table of Contents

1. [What is Causal Inference?](#what-is-causal-inference)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Features](#features)
5. [How it Works](#how-it-works)
6. [Example Outputs](#example-outputs)
7. [API Reference](#api-reference)
8. [Extending the Engine](#extending-the-engine)

---

## 🧠 What is Causal Inference?

Most data science tools measure **correlation** — whether A and B move together.
Causal inference asks a harder, more important question:

> **Does A actually *cause* B?**

### The Confounding Problem

Imagine we observe that people who take a supplement (treatment) have better health outcomes.
But suppose older, wealthier people both take the supplement *and* have better healthcare.
The naive correlation is driven by **confounders** (age, wealth), not the supplement.

```
    Age, Income
    ↙         ↘
Treatment  →  Outcome
```

Causal inference **adjusts for confounders** to isolate the true treatment effect.

### The Fundamental Problem

For any individual, we can only ever observe *one* outcome:
- `Y(1)` — outcome *if treated*
- `Y(0)` — outcome *if not treated*

One of these is always **counterfactual** (unobserved). Causal inference
imputes the missing potential outcome using assumptions + statistical methods.

### The Four Steps (DoWhy Framework)

| Step | Description |
|------|-------------|
| **1. Model** | Encode assumptions as a Causal DAG (Directed Acyclic Graph) |
| **2. Identify** | Derive a statistical formula for the causal effect from the DAG |
| **3. Estimate** | Compute the estimate from observed data |
| **4. Refute** | Stress-test the estimate with adversarial checks |

---

## 📁 Project Structure

```
summerproject/
├── app.py                        ← Streamlit web UI
├── data_loader.py                ← Load, validate, clean data
├── causal_model.py               ← DoWhy CausalModel + DAG builder
├── estimator.py                  ← OLS, PSM/IPW, LinearDML (CATE)
├── refutation.py                 ← Robustness / stress tests
├── simulator.py                  ← Counterfactual simulation (T-Learner)
├── visualization.py              ← All matplotlib/seaborn plots
├── __init__.py                   ← Package exports
├── eda.ipynb                     ← Optional notebook
├── Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
└── README.md                     ← You are here
```

---

## 🚀 Quick Start

### 1. Clone / download

```bash
git clone https://github.com/yourorg/causal-engine
cd causal-engine
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# OR
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 4. Default dataset behavior

- The app now loads the **Kevin Hillstrom dataset** by default.
- On load, it auto-creates a binary `treatment` column from `segment`:
  - `No E-Mail` → `0`
  - `Mens E-Mail` / `Womens E-Mail` → `1`
- It auto-prioritizes:
  - Treatment: `treatment`
  - Outcome: `spend`

### 5. Decision-support inputs

In the sidebar, set:
- **Outcome value multiplier** (business value per unit of outcome)
- **Cost per treated user**
- Optional **campaign budget cap**

The app then computes policy-level:
- Expected Outcome
- Expected Revenue
- Expected Cost
- Expected Net Value
- Recommended Policy

### 6. Run the notebook (optional)

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## ✨ Features

| Feature | Description | Module |
|---------|-------------|--------|
| **Data loading** | CSV upload, missing value handling, validation, Hillstrom prep helper | `data_loader.py` |
| **Causal DAG** | Graphical model with networkx + matplotlib | `causal_model.py` |
| **OLS ATE** | Regression-adjusted average treatment effect | `estimator.py` |
| **Propensity matching** | PSM / IPW via DoWhy | `estimator.py` |
| **CATE (DML)** | Heterogeneous effects via EconML LinearDML | `estimator.py` |
| **Refutation** | Random common cause, placebo, data subset | `refutation.py` |
| **Simulation** | T-Learner counterfactuals, policy comparison, budget-aware targeting | `simulator.py` |
| **Visualisations** | Outcome + net-value policy comparison charts | `visualization.py` |
| **Streamlit UI** | Interactive web app with recommendation card + summary export | `app.py` |
| **Decision Support** | Cost/budget-aware policy ranking and best-action output | `app.py`, `simulator.py` |
| **Notebook** | End-to-end EDA + analysis walkthrough | `notebooks/eda.ipynb` |

---

## 🔬 How it Works

### Dataset used by default (Kevin Hillstrom)

The default dataset is the MineThatData e-mail marketing challenge dataset.
In this app:

```
customer history/features
          ↙            ↘
email treatment      spend / conversion
```

- **Confounders** influence both treatment assignment and outcomes
- Treatment is modeled as binary (`treatment` column auto-generated from `segment`)
- Outcome is typically `spend` (continuous), but users can choose another column

### Estimation Methods

#### 1. OLS (Ordinary Least Squares)
```
Y = β₀ + τ·T + γ₁·X₁ + γ₂·X₂ + ... + ε
ATE = τ
```
Adjusts for confounders by including them as covariates.
Fast and interpretable but assumes linear relationships.

#### 2. Propensity Score Matching (PSM)
1. Estimate P(T=1|X) using logistic regression
2. Match each treated unit to a similar control unit
3. Compute ATE on matched pairs

Requires good **overlap** (common support) between treated and control groups.

#### 3. Inverse Probability Weighting (IPW)
```
ATE = E[Y·T/e(X)] − E[Y·(1−T)/(1−e(X))]
```
Reweights observations so treated and control have the same covariate distribution.

#### 4. Double Machine Learning — LinearDML (EconML)
```
Step 1: Ỹ = Y − E[Y|X]     (remove outcome's dependence on X)
Step 2: T̃ = T − E[T|X]     (remove treatment's dependence on X)
Step 3: τ(X) from Ỹ = τ(X)·T̃ + ε
```
- Uses cross-fitting to avoid overfitting bias
- Estimates **individual** treatment effects τ(Xᵢ) — who benefits most?
- Robust to high-dimensional, non-linear confounding
- Automatically configures treatment model type:
  - classifier for binary treatment
  - regressor for continuous treatment

### Refutation Tests

| Test | What it does | What to expect |
|------|-------------|----------------|
| Random Common Cause | Adds a random confounder | ATE should NOT change (< 10 %) |
| Placebo Treatment | Shuffles treatment randomly | ATE should collapse to ≈ 0 |
| Data Subset | Re-estimates on 80 % sub-samples | ATE should be stable |

All three passing = high confidence in the estimate.

---

## 📊 Example Outputs

### ATE Comparison (illustrative)

```
Method                           ATE
─────────────────────────────── ──────
Naive Diff-in-Means (biased)    +7.837   ← Confounded!
OLS Linear Regression           +4.891   ← Adjusted ✓
Propensity Score Matching       +4.923   ← Adjusted ✓
Mean CATE (LinearDML)           +4.867   ← Adjusted ✓
```

### Refutation Results

```
═══════════════════════════════════════════════════════
  CAUSAL ROBUSTNESS REPORT
═══════════════════════════════════════════════════════
  Tests passed : 3 / 3
  Tests failed : 0 / 3
─────────────────────────────────────────────────────
  ✅  Random Common Cause
  ✅  Placebo Treatment
  ✅  Data Subset Refuter
─────────────────────────────────────────────────────
  VERDICT: The estimate is ROBUST. It passed all refutation
  tests. You can have reasonable confidence in the ATE.
═══════════════════════════════════════════════════════
```

### Policy Simulation (Decision Support)

```
Policy                         ExpectedOutcome  ExpectedRevenue  ExpectedCost  ExpectedNetValue
─────────────────────────────── ─────────────── ─────────────── ────────────  ────────────────
Status Quo (observed)           1.243           1.243            0.210         1.033
Treat Everyone (T=1)            3.789           3.789            1.000         2.789
Treat No-one (T=0)             -1.102          -1.102            0.000        -1.102
Target Top 50% by ITE           4.512           4.512            0.500         4.012   ← Best policy
Budget-Aware Top-30% by ITE     4.031           4.031            0.300         3.731
```

---

## 🔧 API Reference

### `data_loader.py`
```python
load_data(source)                                     → pd.DataFrame
prepare_kevin_hillstrom(df)                          → pd.DataFrame
validate_columns(df, treatment, outcome, confounders) → None (raises on error)
handle_missing_values(df, strategy='median')          → pd.DataFrame
describe_data(df, treatment, outcome, confounders)    → dict
```

### `causal_model.py`
```python
build_causal_model(df, treatment, outcome, confounders) → CausalInferenceModel
    .build()                                            → self
    .identify()                                         → identified_estimand
    .estimate(method_name)                              → estimate
    .refute(estimate, method_name)                      → refutation
```

### `estimator.py`
```python
estimate_ate_linear(df, treatment, outcome, confounders)            → dict
estimate_ate_propensity(ci_model)                                   → dict
estimate_heterogeneous_effects(df, treatment, outcome, confounders) → dict
compute_propensity_scores(df, treatment, confounders)               → np.ndarray
```

### `refutation.py`
```python
run_random_common_cause(ci_model, estimate)    → dict
run_placebo_test(ci_model, estimate)           → dict
run_data_subset_refuter(ci_model, estimate)    → dict
interpret_refutation(list_of_dicts)            → str
```

### `simulator.py`
```python
simulate_counterfactuals(df, treatment, outcome, confounders) → dict
compare_two_treatments(df, treatment_a, treatment_b, outcome, confounders) → dict
```

---

## 🛠 Extending the Engine

### Add a new estimator

1. Add your function to `src/estimator.py`
2. Return a dict with at minimum `{"ate": float, "method": str}`
3. Call `summarise_estimates()` to include it in the comparison table

### Add a new refutation test

1. Add your function to `src/refutation.py`
2. Return a dict with `{"refutation_type", "passed", "interpretation"}`
3. Pass to `interpret_refutation()`

### Use your own data

Your CSV needs:
- A **binary treatment** column (values 0 and 1)
- A **continuous outcome** column (numeric)
- One or more **confounder** columns (numeric or will be label-encoded)

Notes from current app behavior:
- In the UI, **Treatment dropdown only shows binary columns** (0/1-like).
- Non-numeric confounders are label-encoded for modeling.
- Confounder-balance table in Data Overview is shown for numeric confounders.

Upload via the Streamlit UI or call `load_data(path)` directly.

---

## 🗣 Plain-English Summary in UI

After running analysis, the ATE tab includes a **Simple English Summary** section that explains:
- what was analyzed (treatment vs outcome)
- whether treatment increases/decreases the outcome
- confidence range in simple language
- whether results are likely real or uncertain
- whether findings are stable across checks
- the recommended policy and expected net value

This section is designed for non-technical readers.

---

## 📚 Further Reading

- [DoWhy Documentation](https://py-why.github.io/dowhy/)
- [EconML Documentation](https://econml.azurewebsites.net/)
- [The Book of Why — Judea Pearl](https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097609/)
- [Causal Inference: The Mixtape — Scott Cunningham](https://mixtape.scunning.com/)
- [Introduction to Double/Debiased ML — Chernozhukov et al.](https://arxiv.org/abs/1608.00060)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Python, DoWhy, EconML, and Streamlit.*
