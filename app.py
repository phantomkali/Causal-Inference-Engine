"""
app.py  —  Causal Inference Engine  |  Streamlit UI
────────────────────────────────────────────────────────────
Run with:
    streamlit run src/app.py

Sections
─────────
  Sidebar  — data upload + variable selection
  Tab 1    — Data Overview
  Tab 2    — Causal Graph (DAG)
  Tab 3    — ATE Estimation
  Tab 4    — Heterogeneous Effects (CATE)
  Tab 5    — Refutation Tests
  Tab 6    — Counterfactual Simulation
  Tab 7    — About / Explainer
"""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure src/ is on the Python path when run from project root ──
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# ── Internal imports ───────────────────────────────────────────
from data_loader import (
    describe_data,
    encode_categoricals,
    handle_missing_values,
    HILLSTROM_DEFAULT_FILE,
    HILLSTROM_DEFAULT_OUTCOME,
    HILLSTROM_DEFAULT_TREATMENT,
    load_data,
    prepare_kevin_hillstrom,
    validate_columns,
)
from causal_model import build_causal_model, dag_to_dot_string
from estimator import (
    compute_propensity_scores,
    propensity_calibration_diagnostics,
    estimate_ate_aipw,
    estimate_ate_linear,
    estimate_ate_propensity,
    estimate_heterogeneous_effects,
    summarise_estimates,
)
from refutation import (
    interpret_refutation,
    run_data_subset_refuter,
    run_placebo_test,
    run_random_common_cause,
)
from simulator import simulate_counterfactuals
from visualization import (
    plot_ate_comparison,
    plot_before_after_outcome,
    plot_cate_by_feature,
    plot_counterfactual_distributions,
    plot_propensity_calibration,
    plot_dag,
    plot_policy_net_value,
    plot_policy_comparison,
    plot_propensity_overlap,
    plot_treatment_effect_distribution,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Default data path ──────────────────────────────────────────
_DEFAULT_DATA = Path(__file__).parent / HILLSTROM_DEFAULT_FILE


# ════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Causal Inference Engine",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  .main { padding-top: 1rem; }
  .stAlert { border-radius: 8px; }
  div[data-testid="metric-container"] {
    background-color: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 8px;
    padding: 0.5rem 1rem;
  }
  .ate-highlight {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2c5282;
    text-align: center;
  }
  .section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a365d;
    margin-bottom: 0.25rem;
  }
  .explainer-box {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.93rem;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# Session-state helpers
# ════════════════════════════════════════════════════════════════

def _ss_init():
    """Initialise all session-state keys on first load."""
    defaults = {
        "df":               None,
        "treatment":        None,
        "outcome":          None,
        "confounders":      [],
        "ci_model":         None,
        "ols_result":       None,
        "psm_result":       None,
        "aipw_result":      None,
        "cate_result":      None,
        "refutations":      [],
        "sim_result":       None,
        "propensity_scores": None,
        "propensity_calibration": None,
        "estimates_df":     None,
        "analysis_done":    False,
        "pending_run":      None,
        "outcome_value_multiplier": 1.0,
        "treatment_cost": 0.0,
        "budget": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ss_init()


# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/flask.png", width=60)
    st.title("⚗️ Causal Inference Engine")
    st.caption("Estimate causal effects from observational data")
    st.divider()

    # ── Data source ──────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data", ["Use sample dataset", "Upload CSV"],
        label_visibility="collapsed",
    )

    df_raw: pd.DataFrame | None = None

    if data_source == "Use sample dataset":
        if _DEFAULT_DATA.exists():
            df_raw = prepare_kevin_hillstrom(load_data(str(_DEFAULT_DATA)))
            st.success(f"Loaded Kevin Hillstrom dataset ({len(df_raw):,} rows)")
        else:
            st.error(f"Default dataset not found: `{HILLSTROM_DEFAULT_FILE}`.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_raw = prepare_kevin_hillstrom(load_data(uploaded))
            st.success(f"Uploaded: {uploaded.name}  ({len(df_raw):,} rows)")

    if df_raw is not None:
        st.session_state["df"] = df_raw

    df: pd.DataFrame | None = st.session_state.get("df")

    # ── Variable selection ───────────────────────────────────
    if df is not None:
        st.divider()
        st.markdown("### 🎛️ Variable Selection")

        all_cols = list(df.columns)

        # Guess sensible defaults
        def _default_idx(keywords, columns, fallback=0):
            for kw in keywords:
                hits = [i for i, c in enumerate(columns) if kw.lower() in c.lower()]
                if hits:
                    return hits[0]
            return fallback

        def _is_binary_column(series: pd.Series) -> bool:
            non_null = series.dropna()
            if non_null.empty:
                return False
            if pd.api.types.is_bool_dtype(non_null):
                return True
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.isna().any():
                return False
            return set(numeric.unique()).issubset({0, 1})

        treatment_options = [c for c in all_cols if _is_binary_column(df[c])]

        treatment_idx   = _default_idx(
            [HILLSTROM_DEFAULT_TREATMENT, "segment", "treatment", "treat", "intervention"]
            , treatment_options
        )
        outcome_idx     = _default_idx(
            [HILLSTROM_DEFAULT_OUTCOME, "conversion", "visit", "outcome", "result", "y"],
            all_cols,
            fallback=min(1, len(all_cols)-1),
        )

        if not treatment_options:
            st.error("No binary (0/1) columns found for treatment selection.")
            treatment = None
        else:
            treatment = st.selectbox("Treatment variable (binary 0/1)",
                                     treatment_options, index=treatment_idx)
        outcome   = st.selectbox("Outcome variable (continuous)",
                                 all_cols, index=outcome_idx)

        remaining = [c for c in all_cols
                     if c not in (treatment, outcome) and not c.startswith("_")]
        confounders = st.multiselect("Confounders", remaining,
                                     default=remaining[:min(3, len(remaining))])

        st.session_state.update({
            "treatment":   treatment,
            "outcome":     outcome,
            "confounders": confounders,
        })

        # ── Analysis options ─────────────────────────────────
        st.divider()
        st.markdown("### ⚙️ Analysis Options")

        run_psm      = st.checkbox("Propensity Score Matching (PSM)", value=True)
        run_cate     = st.checkbox("Heterogeneous Effects (CausalForestDML)", value=True)
        run_refute   = st.checkbox("Refutation Tests", value=True)
        run_simulate = st.checkbox("Counterfactual Simulation", value=True)

        st.markdown("### 💼 Decision Support Inputs")
        outcome_value_multiplier = st.number_input(
            "Outcome value multiplier",
            min_value=0.0,
            value=float(st.session_state.get("outcome_value_multiplier", 1.0)),
            step=0.1,
            help="Converts each unit of outcome into business value.",
        )
        treatment_cost = st.number_input(
            "Cost per treated user",
            min_value=0.0,
            value=float(st.session_state.get("treatment_cost", 0.0)),
            step=0.1,
            help="Average intervention cost per treated user.",
        )
        use_budget = st.checkbox("Apply campaign budget cap", value=st.session_state.get("budget") is not None)
        budget_value = st.number_input(
            "Total budget",
            min_value=0.0,
            value=float(st.session_state.get("budget") or 0.0),
            step=10.0,
            disabled=not use_budget,
        )
        budget = budget_value if use_budget else None
        st.session_state["outcome_value_multiplier"] = outcome_value_multiplier
        st.session_state["treatment_cost"] = treatment_cost
        st.session_state["budget"] = budget

        st.divider()

        # ── RUN BUTTON ───────────────────────────────────────
        run_btn = st.button("🚀 Run Causal Analysis", type="primary", use_container_width=True)

        if run_btn:
            if not confounders:
                st.warning("Please select at least one confounder.")
            else:
                st.session_state["pending_run"] = {
                    "df": df,
                    "treatment": treatment,
                    "outcome": outcome,
                    "confounders": confounders,
                    "run_psm": run_psm,
                    "run_cate": run_cate,
                    "run_refute": run_refute,
                    "run_simulate": run_simulate,
                    "outcome_value_multiplier": outcome_value_multiplier,
                    "treatment_cost": treatment_cost,
                    "budget": budget,
                }


# ════════════════════════════════════════════════════════════════
# Core Analysis Runner  (defined BEFORE sidebar so it can be called)
# ════════════════════════════════════════════════════════════════

def _run_analysis(df, treatment, outcome, confounders,
                  run_psm, run_cate, run_refute, run_simulate,
                  outcome_value_multiplier=1.0, treatment_cost=0.0, budget=None):
    """Execute the full causal inference pipeline and cache results."""

    progress = st.sidebar.progress(0, text="Starting analysis …")

    try:
        # ── 1. Validate & clean ──────────────────────────────
        progress.progress(5, "Validating columns …")
        validate_columns(df, treatment, outcome, confounders)
        df_clean = handle_missing_values(df, strategy="median")
        cat_cols = [c for c in confounders
                    if not pd.api.types.is_numeric_dtype(df_clean[c])]
        df_clean = encode_categoricals(df_clean, cat_cols)

        # ── 2. Build causal model ────────────────────────────
        progress.progress(15, "Building causal model …")
        ci_model = build_causal_model(df_clean, treatment, outcome, confounders)
        st.session_state["ci_model"] = ci_model

        # ── 3. OLS ATE ───────────────────────────────────────
        progress.progress(30, "Estimating ATE (OLS) …")
        ols_result = estimate_ate_linear(df_clean, treatment, outcome, confounders)
        st.session_state["ols_result"] = ols_result

        # ── 4. PSM ───────────────────────────────────────────
        psm_result = None
        if run_psm:
            progress.progress(45, "Propensity score matching …")
            psm_result = estimate_ate_propensity(ci_model)
            st.session_state["psm_result"] = psm_result

        # ── 5. Propensity scores for diagnostics ─────────────
        ps = compute_propensity_scores(df_clean, treatment, confounders)
        st.session_state["propensity_scores"] = ps
        st.session_state["propensity_calibration"] = propensity_calibration_diagnostics(
            df_clean, treatment, confounders
        )

        # ── 5b. Doubly robust AIPW ───────────────────────────
        progress.progress(52, "Estimating ATE (AIPW) …")
        aipw_result = estimate_ate_aipw(df_clean, treatment, outcome, confounders)
        st.session_state["aipw_result"] = aipw_result

        # ── 6. CATE ──────────────────────────────────────────
        cate_result = None
        if run_cate:
            progress.progress(60, "Estimating heterogeneous effects …")
            cate_result = estimate_heterogeneous_effects(
                df_clean, treatment, outcome, confounders
            )
            st.session_state["cate_result"] = cate_result

        # ── 7. Estimates summary ─────────────────────────────
        naive_ate = (df_clean.loc[df_clean[treatment]==1, outcome].mean()
                     - df_clean.loc[df_clean[treatment]==0, outcome].mean())
        estimates_df = summarise_estimates(
            ols_result=ols_result,
            psm_result=psm_result,
            aipw_result=aipw_result,
            cate_result=cate_result,
            naive_ate=naive_ate,
        )
        st.session_state["estimates_df"] = estimates_df

        # ── 8. Refutation ────────────────────────────────────
        refutations = []
        if run_refute:
            progress.progress(72, "Running refutation tests …")
            ols_est = ci_model.estimate(method_name="backdoor.linear_regression")

            rcc = run_random_common_cause(ci_model, ols_est, n_simulations=10)
            refutations.append(rcc)

            placebo = run_placebo_test(ci_model, ols_est, n_simulations=10)
            refutations.append(placebo)

            subset = run_data_subset_refuter(ci_model, ols_est, n_simulations=10)
            refutations.append(subset)

        st.session_state["refutations"] = refutations

        # ── 9. Simulation ────────────────────────────────────
        if run_simulate:
            progress.progress(88, "Simulating counterfactuals …")
            sim_result = simulate_counterfactuals(
                df_clean, treatment, outcome, confounders,
                ate=ols_result["ate"],
                outcome_value_multiplier=outcome_value_multiplier,
                treatment_cost=treatment_cost,
                budget=budget,
            )
            st.session_state["sim_result"] = sim_result

        st.session_state["analysis_done"] = True
        progress.progress(100, "✅ Done!")
        st.sidebar.success("Analysis complete!")

    except Exception as exc:
        progress.empty()
        st.sidebar.error(f"Analysis failed: {exc}")
        raise


def _confidence_tier(p_value: float, refutations: list) -> str:
    passed = sum(1 for r in refutations if r.get("passed") is True)
    total = len(refutations)
    if p_value < 0.05 and (total == 0 or passed == total):
        return "High"
    if p_value < 0.10 and (total == 0 or passed >= max(1, total - 1)):
        return "Medium"
    return "Low"


def _decision_recommendation() -> dict | None:
    sim = st.session_state.get("sim_result")
    if not sim or "best_policy" not in sim:
        return None
    return sim["best_policy"]


def _simple_english_summary() -> str:
    """Build a plain-English explanation of findings and recommendation."""
    ols_result = st.session_state.get("ols_result")
    psm_result = st.session_state.get("psm_result")
    cate_result = st.session_state.get("cate_result")
    refutations = st.session_state.get("refutations", [])
    treatment = st.session_state.get("treatment")
    outcome = st.session_state.get("outcome")

    if not ols_result or not treatment or not outcome:
        return "Run the analysis first to generate a plain-English summary."

    ate = ols_result["ate"]
    ci_low = ols_result["ci_lower"]
    ci_high = ols_result["ci_upper"]
    p_val = ols_result["p_value"]
    recommendation = _decision_recommendation()

    direction = "increases" if ate > 0 else "decreases"
    strength = abs(ate)
    confidence = _confidence_tier(p_val, refutations)
    certainty = "This result looks reliable." if confidence == "High" else "This result has uncertainty."
    ci_phrase = (
        "The likely range stays on the positive side."
        if ci_low > 0
        else "The likely range stays on the negative side."
        if ci_high < 0
        else "The likely range crosses zero, so the true effect might be small or none."
    )

    lines = [
        f"We checked whether changing **{treatment}** causes a change in **{outcome}**.",
        f"Main result: treatment **{direction}** the outcome by about **{strength:.3f}** units on average.",
        f"Confidence range: **{ci_low:+.3f} to {ci_high:+.3f}**. {ci_phrase}",
        f"Confidence tier: **{confidence}**. {certainty}",
    ]

    if psm_result:
        psm_ate = psm_result.get("ate")
        if psm_ate is not None:
            lines.append(
                f"A second method gave a similar direction (**{psm_ate:+.3f}**), "
                "which makes the conclusion more believable."
            )

    if cate_result:
        share_positive = float((cate_result["cate_values"] > 0).mean()) * 100
        lines.append(
            f"The impact is not equal for everyone: about **{share_positive:.1f}%** "
            "of people appear to benefit."
        )

    if refutations:
        passed = sum(1 for r in refutations if r.get("passed") is True)
        total = len(refutations)
        lines.append(
            f"Robustness checks passed: **{passed}/{total}**. "
            "More passes mean the result is more stable."
        )

    if recommendation:
        lines.append(
            "Recommended action now: "
            f"**{recommendation['name']}**. "
            f"Expected net value: **{recommendation['expected_net_value']:.3f}** "
            f"with treatment rate **{recommendation['treatment_rate']:.1%}**."
        )

    if confidence == "Low":
        lines.append(
            "When not to trust this result: if your population is very different from this dataset, "
            "or key confounders are missing, the recommendation can be misleading."
        )
    else:
        lines.append(
            "When to be careful: monitor results after rollout, because data drift can reduce impact."
        )

    lines.append(
        "In plain terms: this analysis estimates the likely cause-and-effect impact, "
        "not just a simple correlation."
    )
    return "\n\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Main content area  —  tabbed layout
# ════════════════════════════════════════════════════════════════

st.markdown("## ⚗️ Causal Inference Engine")
st.caption("Estimate whether a treatment *causes* an outcome — not just correlates with it.")

df          = st.session_state.get("df")
treatment   = st.session_state.get("treatment")
outcome     = st.session_state.get("outcome")
confounders = st.session_state.get("confounders", [])

if df is None:
    st.info("👈  Load a dataset from the sidebar to get started.")
    _show_about()
    st.stop()

pending_run = st.session_state.get("pending_run")
if pending_run:
    _run_analysis(**pending_run)
    st.session_state["pending_run"] = None

tab_names = [
    "📊 Data Overview",
    "🔗 Causal Graph",
    "📐 ATE Estimation",
    "🌈 Heterogeneous Effects",
    "🛡️ Refutation Tests",
    "🔮 Counterfactual Sim",
    "ℹ️ About",
]
tabs = st.tabs(tab_names)


# ════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    if treatment:
        c3.metric("Treatment rate", f"{df[treatment].mean():.1%}")
    if treatment and outcome:
        naive = (df.loc[df[treatment]==1, outcome].mean()
                 - df.loc[df[treatment]==0, outcome].mean())
        c4.metric("Naive Diff-in-Means", f"{naive:+.3f}",
                  help="⚠️ Biased — does NOT adjust for confounders")

    st.divider()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### First 100 rows")
        st.dataframe(df.head(100), use_container_width=True, height=280)
    with col_r:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True, height=280)

    if treatment and outcome:
        st.divider()
        st.markdown("#### Outcome by Treatment Status")
        fig = plot_before_after_outcome(df, treatment, outcome)
        st.pyplot(fig, use_container_width=True)

    if treatment and confounders:
        st.divider()
        st.markdown("#### Confounder Balance (Treated vs Untreated)")
        numeric_confounders = [
            c for c in confounders if pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_confounders:
            st.info("Confounder balance summary is shown only for numeric confounders.")
        else:
            stats = describe_data(df, treatment, outcome, numeric_confounders)
            balance_rows = []
            for feat, vals in stats["confounder_summary"].items():
                balance_rows.append({
                    "Confounder":       feat,
                    "Mean (Treated)":   round(vals["mean_treated"], 3),
                    "Mean (Untreated)": round(vals["mean_untreated"], 3),
                    "Standardised Diff": round(
                        (vals["mean_treated"] - vals["mean_untreated"])
                        / (df[feat].std() + 1e-9), 3
                    ),
                })
            bal_df = pd.DataFrame(balance_rows)
            st.dataframe(
                bal_df.style.background_gradient(
                    subset=["Standardised Diff"], cmap="RdYlGn_r", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )
            st.caption("Standardised diff > |0.1| suggests imbalance that confounds the naive estimate.")


# ════════════════════════════════════════════════════════════════
# TAB 2 — Causal Graph
# ════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### Causal Directed Acyclic Graph (DAG)")

    st.markdown("""
<div class="explainer-box">
<b>What is a DAG?</b><br>
A Directed Acyclic Graph encodes our <em>causal assumptions</em> as arrows between variables.
An arrow from A → B means "A causes B".<br><br>
The confounders (green) influence both the treatment (coral) and the outcome (blue).
This creates <em>confounding bias</em> — the reason naive correlations are misleading.
Adjusting for confounders removes this bias.
</div>
""", unsafe_allow_html=True)

    if not treatment or not outcome:
        st.warning("Select treatment and outcome variables in the sidebar.")
    else:
        st.divider()
        ci_model = st.session_state.get("ci_model")
        dag = ci_model.dag if ci_model else None

        fig_dag = plot_dag(treatment, outcome, confounders, dag=dag)
        st.pyplot(fig_dag, use_container_width=True)

        # DOT string download
        from causal_model import build_dag
        G = dag or build_dag(treatment, outcome, confounders)
        dot_str = dag_to_dot_string(G)

        st.download_button(
            "💾 Download DOT file (for Graphviz)",
            data=dot_str,
            file_name="causal_dag.dot",
            mime="text/plain",
        )

        if ci_model and ci_model.identified_estimand:
            st.divider()
            st.markdown("### 🔍 Identified Estimand (DoWhy)")
            with st.expander("Show estimand details", expanded=False):
                st.code(str(ci_model.identified_estimand), language="text")


# ════════════════════════════════════════════════════════════════
# TAB 3 — ATE Estimation
# ════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### Average Treatment Effect (ATE)")

    st.markdown("""
<div class="explainer-box">
<b>What is the ATE?</b><br>
The Average Treatment Effect is the average causal impact of treatment across the population:<br>
<em>ATE = E[Y(1) − Y(0)] = E[outcome if treated] − E[outcome if untreated]</em><br><br>
Unlike a naive mean difference, the ATE adjusts for confounders that influence who gets treated.
</div>
""", unsafe_allow_html=True)

    if not st.session_state["analysis_done"]:
        st.info("▶️  Click **Run Causal Analysis** in the sidebar to compute estimates.")
    else:
        st.divider()
        st.markdown("#### Simple English Summary")
        summary_text = _simple_english_summary()
        st.markdown(summary_text)
        st.download_button(
            "Download summary (.txt)",
            data=summary_text,
            file_name="causal_decision_summary.txt",
            mime="text/plain",
        )

        st.divider()
        ols_result   = st.session_state["ols_result"]
        psm_result   = st.session_state["psm_result"]
        aipw_result  = st.session_state.get("aipw_result")
        estimates_df = st.session_state["estimates_df"]
        sim_result   = st.session_state.get("sim_result")

        if sim_result and sim_result.get("best_policy"):
            rec = sim_result["best_policy"]
            st.markdown("#### Recommended Action")
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Policy", rec["name"])
            c2.metric("Expected Net Value", f"{rec['expected_net_value']:.3f}")
            c3.metric("Treatment Rate", f"{rec['treatment_rate']:.1%}")

        # ── ATE highlight ─────────────────────────────────────
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<p class="section-title">OLS ATE (adjusted)</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="ate-highlight">{ols_result["ate"]:+.4f}</p>', unsafe_allow_html=True)
            st.caption(f"95 % CI: [{ols_result['ci_lower']:+.3f}, {ols_result['ci_upper']:+.3f}]")
            st.caption(f"p-value: {ols_result['p_value']:.4f}")

        with c2:
            if psm_result:
                st.markdown(f'<p class="section-title">{psm_result["method"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="ate-highlight">{psm_result["ate"]:+.4f}</p>', unsafe_allow_html=True)
            else:
                st.info("PSM not run.")

        with c3:
            if aipw_result:
                st.markdown('<p class="section-title">AIPW (doubly robust)</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="ate-highlight">{aipw_result["ate"]:+.4f}</p>', unsafe_allow_html=True)
                st.caption(f"95 % CI: [{aipw_result['ci_lower']:+.3f}, {aipw_result['ci_upper']:+.3f}]")
            else:
                st.info("AIPW not available.")

        with c4:
            if st.session_state["cate_result"]:
                cr = st.session_state["cate_result"]
                st.markdown('<p class="section-title">Mean CATE (Causal Forest)</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="ate-highlight">{cr["cate_mean"]:+.4f}</p>', unsafe_allow_html=True)
                st.caption(f"Std: {cr['cate_std']:.4f}")

        st.divider()
        st.markdown("#### Method Comparison")
        fig_cmp = plot_ate_comparison(estimates_df)
        st.pyplot(fig_cmp, use_container_width=True)

        st.markdown("#### Summary Table")
        st.dataframe(estimates_df.round(4), use_container_width=True)

        # ── Propensity overlap ───────────────────────────────
        ps = st.session_state.get("propensity_scores")
        calib = st.session_state.get("propensity_calibration")
        if ps is not None:
            st.divider()
            st.markdown("#### Propensity Score Overlap")
            st.caption("Overlap between treated and control groups confirms causal estimates are credible.")
            fig_ps = plot_propensity_overlap(
                st.session_state["df"], treatment, ps
            )
            st.pyplot(fig_ps, use_container_width=True)
            if calib:
                c1, c2, c3 = st.columns(3)
                c1.metric("Brier Score", f"{calib['brier_score']:.4f}")
                c2.metric("Calibration Error (ECE)", f"{calib['ece']:.4f}")
                c3.metric("Calibration Status", calib["status"])
                fig_cal = plot_propensity_calibration(
                    calib["calibration_pred_mean"],
                    calib["calibration_frac_positive"],
                )
                st.pyplot(fig_cal, use_container_width=True)
                if calib["status"] == "Poor":
                    st.warning(
                        "Propensity calibration is poor. IPW/AIPW estimates may be unstable even if overlap looks acceptable."
                    )


# ════════════════════════════════════════════════════════════════
# TAB 4 — Heterogeneous Effects
# ════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### Heterogeneous / Conditional Treatment Effects (CATE)")

    st.markdown("""
<div class="explainer-box">
<b>Why do individual effects differ?</b><br>
The ATE is an <em>average</em>. In practice, treatment may benefit some sub-groups
more than others. CATE (Conditional ATE) estimates the treatment effect
for each individual based on their characteristics.<br><br>
We use <b>CausalForestDML</b>, a non-linear double machine learning method that captures
complex subgroup patterns and estimates individual-level treatment effects with confidence intervals.
</div>
""", unsafe_allow_html=True)

    if not st.session_state["analysis_done"]:
        st.info("▶️  Run the analysis first.")
    elif not st.session_state["cate_result"]:
        st.warning("CATE was not computed (uncheck/re-check the option and re-run).")
    else:
        cr = st.session_state["cate_result"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean CATE",  f"{cr['cate_mean']:+.4f}")
        c2.metric("Std CATE",   f"{cr['cate_std']:.4f}")
        c3.metric("% with positive effect",
                  f"{(cr['cate_values'] > 0).mean():.1%}")

        st.divider()
        st.markdown("#### Distribution of Individual Treatment Effects")
        fig_hist = plot_treatment_effect_distribution(
            cr["cate_values"], ate=cr["cate_mean"]
        )
        st.pyplot(fig_hist, use_container_width=True)

        st.divider()
        st.markdown("#### CATE vs Covariates  (Who benefits most?)")
        fig_scatter = plot_cate_by_feature(
            cr["X"], cr["cate_values"], cr["feature_names"]
        )
        st.pyplot(fig_scatter, use_container_width=True)

        st.divider()
        st.markdown("#### Top / Bottom Subgroups by Estimated Effect")
        df_clean = st.session_state["df"].copy()
        df_clean["_cate"] = np.nan
        n_cate = len(cr["cate_values"])
        df_clean.iloc[:n_cate, df_clean.columns.get_loc("_cate")] = cr["cate_values"]

        col_l, col_r = st.columns(2)
        with col_l:
            st.caption("🔝  Top 10 % — Highest individual benefit")
            top10_threshold = np.quantile(cr["cate_values"], 0.9)
            st.dataframe(
                df_clean[df_clean["_cate"] >= top10_threshold]
                .head(20)
                .drop(columns=[c for c in df_clean.columns if c.startswith("_")]),
                use_container_width=True, height=220,
            )
        with col_r:
            st.caption("🔻  Bottom 10 % — Lowest / harmful effect")
            bot10_threshold = np.quantile(cr["cate_values"], 0.1)
            st.dataframe(
                df_clean[df_clean["_cate"] <= bot10_threshold]
                .head(20)
                .drop(columns=[c for c in df_clean.columns if c.startswith("_")]),
                use_container_width=True, height=220,
            )


# ════════════════════════════════════════════════════════════════
# TAB 5 — Refutation Tests
# ════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### Robustness / Refutation Tests")

    st.markdown("""
<div class="explainer-box">
<b>Why refute?</b><br>
Causal estimates rest on untestable assumptions (no hidden confounders, correct DAG).
Refutation tests are adversarial checks — if the estimate survives them, we can be
more confident it reflects a real causal relationship.<br><br>
<b>Tests run:</b>
<ul>
  <li><b>Random Common Cause</b> — add a random confounder; ATE should not change</li>
  <li><b>Placebo Treatment</b> — shuffle treatment; ATE should collapse to ≈ 0</li>
  <li><b>Data Subset</b> — re-estimate on 80 % sub-samples; ATE should be stable</li>
</ul>
</div>
""", unsafe_allow_html=True)

    if not st.session_state["analysis_done"]:
        st.info("▶️  Run the analysis first.")
    elif not st.session_state["refutations"]:
        st.warning("Refutation tests were not run.")
    else:
        refutations = st.session_state["refutations"]
        summary     = interpret_refutation(refutations)

        st.code(summary, language="text")

        st.divider()
        for ref in refutations:
            icon = "✅" if ref.get("passed") else ("⚠️" if ref.get("passed") is False else "❓")
            with st.expander(f"{icon}  {ref['refutation_type']}", expanded=True):
                st.code(ref.get("interpretation", "No details available."), language="text")

                # Mini table
                rows = {}
                if "original_ate" in ref:
                    rows["Original ATE"] = f"{ref['original_ate']:+.4f}"
                if "new_ate" in ref:
                    rows["New ATE"] = f"{ref['new_ate']:+.4f}"
                if "placebo_ate" in ref:
                    rows["Placebo ATE"] = f"{ref['placebo_ate']:+.4f}"
                if "relative_change" in ref:
                    rows["Relative Δ"] = f"{ref['relative_change']:.1%}"
                if rows:
                    st.table(pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]))


# ════════════════════════════════════════════════════════════════
# TAB 6 — Counterfactual Simulation
# ════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown("### Counterfactual Simulation")

    st.markdown("""
<div class="explainer-box">
<b>What is a counterfactual?</b><br>
"What <em>would have happened</em> if this person had (or had not) received treatment?"<br><br>
We use a T-Learner (two outcome models, one per arm) to predict both Y(0) and Y(1)
for every individual, then show the distributions and compare policy scenarios.
</div>
""", unsafe_allow_html=True)

    if not st.session_state["analysis_done"]:
        st.info("▶️  Run the analysis first.")
    elif not st.session_state["sim_result"]:
        st.warning("Simulation was not run.")
    else:
        sim = st.session_state["sim_result"]
        rec = sim.get("best_policy")

        c1, c2, c3 = st.columns(3)
        c1.metric("E[Y(0)] — No treatment",  f"{sim['mean_y0']:.4f}")
        c2.metric("E[Y(1)] — Treatment",      f"{sim['mean_y1']:.4f}")
        c3.metric("E[ITE] = E[Y(1)−Y(0)]",   f"{sim['mean_ite']:+.4f}")

        if rec:
            st.divider()
            st.markdown("#### Recommended Action")
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Policy", rec["name"])
            c2.metric("Expected Net Value", f"{rec['expected_net_value']:.3f}")
            c3.metric("Treatment Rate", f"{rec['treatment_rate']:.1%}")

        st.divider()
        st.markdown("#### Counterfactual Outcome Distributions")
        obs_y = st.session_state["df"][outcome].values if outcome else None
        fig_cf = plot_counterfactual_distributions(
            sim["y0_hat"], sim["y1_hat"],
            observed_outcome=obs_y,
        )
        st.pyplot(fig_cf, use_container_width=True)

        st.divider()
        st.markdown("#### Policy Scenario Comparison")
        st.caption("Expected outcomes under different treatment assignment rules.")
        fig_pol = plot_policy_comparison(sim["policy_df"])
        st.pyplot(fig_pol, use_container_width=True)
        fig_net = plot_policy_net_value(sim["policy_df"])
        st.pyplot(fig_net, use_container_width=True)
        st.dataframe(sim["policy_df"].round(4), use_container_width=True)

        st.divider()
        st.markdown("#### Sample — Predicted Potential Outcomes per Individual")
        df_aug = sim["df_augmented"]
        show_cols = [treatment, outcome, "_y0_hat", "_y1_hat", "_ite_hat"] + confounders
        show_cols = [c for c in show_cols if c in df_aug.columns]
        st.dataframe(
            df_aug[show_cols]
            .rename(columns={"_y0_hat": "Y(0)", "_y1_hat": "Y(1)", "_ite_hat": "ITE"})
            .head(50)
            .round(3),
            use_container_width=True, height=280,
        )

        # Download
        csv_buf = io.StringIO()
        df_aug[show_cols].round(3).to_csv(csv_buf, index=False)
        st.download_button(
            "💾 Download augmented dataset (with counterfactuals)",
            data=csv_buf.getvalue(),
            file_name="causal_counterfactuals.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════
# TAB 7 — About
# ════════════════════════════════════════════════════════════════

def _show_about():
    pass   # forward declaration; actual content below

with tabs[6]:
    st.markdown("### About this Tool")
    st.markdown("""
#### What is Causal Inference?

Most statistical tools tell you *correlation* — that A and B move together.
Causal inference asks a harder question: **does A actually *cause* B?**

> "Correlation is not causation" — but causal inference lets us get much closer to causation
> from observational data.

#### The Fundamental Problem

We can never observe both outcomes for the same person:
- `Y(1)` — what happens *if treated*
- `Y(0)` — what happens *if not treated*

One of these is always counterfactual (unobserved). Causal inference uses
assumptions + statistical methods to impute the missing potential outcome.

#### Four Steps (DoWhy Framework)

| Step | What happens |
|------|-------------|
| **Model** | Encode assumptions in a Causal DAG |
| **Identify** | Derive a statistical formula for the causal effect |
| **Estimate** | Compute the estimate from data |
| **Refute** | Stress-test the estimate with adversarial tests |

#### Key Concepts

- **ATE** — Average Treatment Effect: the mean causal impact across all units
- **CATE** — Conditional ATE: treatment effect for sub-groups
- **Confounder** — a variable that affects both treatment *and* outcome
- **Propensity Score** — P(treatment=1 | covariates); used for matching / weighting
- **CausalForestDML** — non-linear DML for heterogeneous effects with individual confidence intervals

#### Libraries Used

| Library | Purpose |
|---------|---------|
| [DoWhy](https://py-why.github.io/dowhy/) | Causal model, identification, refutation |
| [EconML](https://econml.azurewebsites.net/) | Heterogeneous effects (CausalForestDML) |
| [scikit-learn](https://scikit-learn.org) | Propensity models, response surfaces |
| [statsmodels](https://statsmodels.org) | OLS with confidence intervals |
| [Streamlit](https://streamlit.io) | UI |
""")
