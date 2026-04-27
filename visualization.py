"""
visualization.py
────────────────────────────────────────────────────────────
All plotting functions for the Causal Inference Engine.

Each function returns a matplotlib Figure so it can be
displayed in Streamlit with st.pyplot(fig) or saved to disk.

PLOTS
──────
  1. plot_dag                       — causal directed graph
  2. plot_before_after_outcome      — outcome distributions T=0 vs T=1
  3. plot_treatment_effect_distribution — histogram of CATE
  4. plot_cate_by_feature           — CATE vs each confounder (scatter)
  5. plot_propensity_overlap        — propensity score distributions
  6. plot_counterfactual_distributions — Y(0) vs Y(1) densities
  7. plot_ate_comparison            — bar chart of ATE methods
  8. plot_policy_comparison         — bar chart of policy outcomes
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")   # non-interactive backend; safe for Streamlit

logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────
_PALETTE = {
    "treated":       "#E05252",   # red
    "control":       "#5B8FDE",   # blue
    "cate":          "#6A4ADB",   # purple
    "treatment_node":"#FA8072",   # salmon
    "outcome_node":  "#4682B4",   # steelblue
    "confounder_node":"#3CB371",  # mediumseagreen
    "policy":        "#2CA02C",
    "neutral":       "#777777",
}

sns.set_theme(style="whitegrid", palette="muted")


# ────────────────────────────────────────────────────────────
# 1. DAG Visualisation
# ────────────────────────────────────────────────────────────

def plot_dag(
    treatment: str,
    outcome: str,
    confounders: List[str],
    dag: Optional[nx.DiGraph] = None,
    title: str = "Causal DAG",
) -> plt.Figure:
    """
    Draw the causal directed acyclic graph.

    Colours:
      • Coral    → treatment
      • Steelblue → outcome
      • Green    → confounders
    """
    if dag is None:
        try:
            from .causal_model import build_dag
        except ImportError:
            from causal_model import build_dag
        dag = build_dag(treatment, outcome, confounders)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Build layout — treatment left, outcome right, confounders top
    pos = {}
    n_conf = len(confounders)
    for i, c in enumerate(confounders):
        x = 0.1 + i * (0.8 / max(n_conf - 1, 1))
        pos[c] = (x, 0.85)
    pos[treatment] = (0.15, 0.2)
    pos[outcome]   = (0.85, 0.2)

    node_colors = []
    for node in dag.nodes():
        if node == treatment:
            node_colors.append(_PALETTE["treatment_node"])
        elif node == outcome:
            node_colors.append(_PALETTE["outcome_node"])
        else:
            node_colors.append(_PALETTE["confounder_node"])

    labels = {n: n.replace("_", "\n") for n in dag.nodes()}

    nx.draw_networkx(
        dag, pos=pos, labels=labels,
        node_color=node_colors, node_size=3500,
        font_color="white", font_weight="bold", font_size=9,
        arrows=True, arrowsize=20,
        edge_color="#444444", width=2.0, ax=ax,
        connectionstyle="arc3,rad=0.05",
    )

    # Legend
    patches = [
        mpatches.Patch(color=_PALETTE["treatment_node"],  label="Treatment"),
        mpatches.Patch(color=_PALETTE["outcome_node"],    label="Outcome"),
        mpatches.Patch(color=_PALETTE["confounder_node"], label="Confounder"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=3,
              framealpha=0.9, fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 2. Before vs After — Outcome by Treatment Status
# ────────────────────────────────────────────────────────────

def plot_before_after_outcome(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    title: str = "Outcome Distribution by Treatment Status",
) -> plt.Figure:
    """
    Overlapping KDE + histogram of outcome for T=0 and T=1.
    Shows the raw (unadjusted) difference in means.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    t0 = df.loc[df[treatment] == 0, outcome].dropna()
    t1 = df.loc[df[treatment] == 1, outcome].dropna()

    # ── Left: overlapping KDE ───────────────────────────────
    ax = axes[0]
    sns.kdeplot(t0, ax=ax, color=_PALETTE["control"],  fill=True, alpha=0.35, label="Control (T=0)")
    sns.kdeplot(t1, ax=ax, color=_PALETTE["treated"],  fill=True, alpha=0.35, label="Treated (T=1)")

    for val, col, lbl in [
        (t0.mean(), _PALETTE["control"], f"μ₀={t0.mean():.2f}"),
        (t1.mean(), _PALETTE["treated"], f"μ₁={t1.mean():.2f}"),
    ]:
        ax.axvline(val, color=col, linewidth=2, linestyle="--", label=lbl)

    ax.set_xlabel(outcome, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("KDE — Outcome Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right: box plot ──────────────────────────────────────
    ax2 = axes[1]
    plot_df = df[[treatment, outcome]].copy().dropna()
    plot_df["Group"] = plot_df[treatment].map({0: "Control (T=0)", 1: "Treated (T=1)"})

    sns.boxplot(
        data=plot_df, x="Group", y=outcome, ax=ax2,
        palette={"Control (T=0)": _PALETTE["control"],
                 "Treated (T=1)": _PALETTE["treated"]},
        width=0.45, linewidth=1.5,
    )
    sns.stripplot(
        data=plot_df.sample(min(400, len(plot_df)), random_state=42),
        x="Group", y=outcome, ax=ax2,
        palette={"Control (T=0)": _PALETTE["control"],
                 "Treated (T=1)": _PALETTE["treated"]},
        alpha=0.25, size=3, jitter=True,
    )

    naive_ate = t1.mean() - t0.mean()
    ax2.set_title(f"Boxplot  |  Naive Diff = {naive_ate:+.3f}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("", fontsize=11)
    ax2.set_ylabel(outcome, fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 3. Treatment Effect Distribution (CATE histogram)
# ────────────────────────────────────────────────────────────

def plot_treatment_effect_distribution(
    cate_values: np.ndarray,
    ate: Optional[float] = None,
    title: str = "Distribution of Individual Treatment Effects (CATE)",
) -> plt.Figure:
    """
    Histogram + KDE of estimated CATE values.
    Vertical line at ATE (mean) and zero.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    sns.histplot(cate_values, bins=40, kde=True, ax=ax,
                 color=_PALETTE["cate"], alpha=0.65, edgecolor="white", linewidth=0.4)

    ate_val = ate if ate is not None else float(np.mean(cate_values))
    ax.axvline(0,       color="black",           linewidth=1.5, linestyle=":",  label="Zero effect")
    ax.axvline(ate_val, color=_PALETTE["treated"], linewidth=2.0, linestyle="--",
               label=f"ATE = {ate_val:+.3f}")

    # Shade positive / negative regions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axvspan(0, xlim[1], alpha=0.04, color=_PALETTE["treated"], label="Benefit")
    ax.axvspan(xlim[0], 0, alpha=0.04, color=_PALETTE["control"], label="Harm")

    ax.set_xlabel("Individual Treatment Effect", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    pct_positive = 100 * (cate_values > 0).mean()
    ax.text(0.02, 0.96,
            f"Benefit (ITE > 0): {pct_positive:.1f} %\n"
            f"Harm   (ITE < 0): {100-pct_positive:.1f} %",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 4. CATE vs Feature (Scatter)
# ────────────────────────────────────────────────────────────

def plot_cate_by_feature(
    X_df: pd.DataFrame,
    cate_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_features: int = 4,
    title: str = "Heterogeneous Treatment Effects by Covariate",
) -> plt.Figure:
    """
    Scatter + LOWESS smoother of CATE vs each feature.
    Reveals which subgroups benefit most / least.
    """
    features = feature_names or list(X_df.columns)[:max_features]
    features  = features[:max_features]
    n_cols    = min(len(features), 2)
    n_rows    = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    for i, feat in enumerate(features):
        ax = axes[i // n_cols][i % n_cols]
        x_vals = X_df[feat].values if feat in X_df.columns else np.zeros(len(cate_values))

        ax.scatter(x_vals, cate_values, alpha=0.3, s=10,
                   color=_PALETTE["cate"], rasterized=True)

        # LOWESS smoother
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(cate_values, x_vals, frac=0.3, it=1)
            ax.plot(smooth[:, 0], smooth[:, 1],
                    color=_PALETTE["treated"], linewidth=2.5, label="LOWESS")
        except Exception:
            # Fallback: polynomial fit
            poly  = np.polyfit(x_vals, cate_values, deg=2)
            xs    = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(xs, np.polyval(poly, xs), color=_PALETTE["treated"],
                    linewidth=2.5, label="Poly fit")

        ax.axhline(0, color="black", linewidth=1, linestyle=":")
        ax.set_xlabel(feat.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("CATE", fontsize=11)
        ax.set_title(f"CATE vs {feat}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(features), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 5. Propensity Score Overlap
# ────────────────────────────────────────────────────────────

def plot_propensity_overlap(
    df: pd.DataFrame,
    treatment: str,
    propensity_scores: np.ndarray,
    title: str = "Propensity Score Overlap (Common Support)",
) -> plt.Figure:
    """
    Compare propensity score distributions for T=0 and T=1.
    Good overlap → causal estimates are more credible.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    t0_ps = propensity_scores[df[treatment] == 0]
    t1_ps = propensity_scores[df[treatment] == 1]

    sns.kdeplot(t0_ps, ax=ax, fill=True, alpha=0.4, color=_PALETTE["control"], label="Control (T=0)")
    sns.kdeplot(t1_ps, ax=ax, fill=True, alpha=0.4, color=_PALETTE["treated"], label="Treated (T=1)")

    ax.axvline(0.1, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.axvline(0.9, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(0.105, ax.get_ylim()[1] * 0.95, "trim=0.1", fontsize=8, color="gray")
    ax.text(0.905, ax.get_ylim()[1] * 0.95, "trim=0.9", fontsize=8, color="gray")

    ax.set_xlabel("Propensity Score P(T=1|X)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)

    # Overlap quality note
    overlap = min(t0_ps.max(), t1_ps.max()) - max(t0_ps.min(), t1_ps.min())
    quality = "Good" if overlap > 0.4 else ("Moderate" if overlap > 0.1 else "Poor")
    ax.text(0.5, 0.92, f"Overlap quality: {quality}",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    return fig


def plot_propensity_calibration(
    calibration_pred_mean: np.ndarray,
    calibration_frac_positive: np.ndarray,
    title: str = "Propensity Calibration Curve",
) -> plt.Figure:
    """
    Reliability curve for propensity predictions.
    Perfect calibration lies on the 45-degree diagonal.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration_pred_mean,
        calibration_frac_positive,
        marker="o",
        linewidth=2,
        color=_PALETTE["cate"],
        label="Model calibration",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted treatment probability")
    ax.set_ylabel("Observed treatment frequency")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 6. Counterfactual Distributions
# ────────────────────────────────────────────────────────────

def plot_counterfactual_distributions(
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    observed_outcome: Optional[np.ndarray] = None,
    title: str = "Counterfactual Outcome Distributions",
) -> plt.Figure:
    """
    Plot the estimated potential outcome distributions Y(0) and Y(1).
    The horizontal gap between their means represents the ATE.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    sns.kdeplot(y0_hat, ax=ax, fill=True, alpha=0.45,
                color=_PALETTE["control"],  label=f"Y(0) — No treatment  μ={y0_hat.mean():.2f}")
    sns.kdeplot(y1_hat, ax=ax, fill=True, alpha=0.45,
                color=_PALETTE["treated"],  label=f"Y(1) — Treatment      μ={y1_hat.mean():.2f}")

    if observed_outcome is not None:
        sns.kdeplot(observed_outcome, ax=ax, linestyle="--", color="gray",
                    alpha=0.7, label=f"Observed outcome   μ={observed_outcome.mean():.2f}")

    ax.axvline(y0_hat.mean(), color=_PALETTE["control"], linewidth=2, linestyle="--", alpha=0.8)
    ax.axvline(y1_hat.mean(), color=_PALETTE["treated"], linewidth=2, linestyle="--", alpha=0.8)

    # Bracket showing ATE
    ymax = ax.get_ylim()[1]
    ax.annotate("", xy=(y1_hat.mean(), ymax * 0.75),
                xytext=(y0_hat.mean(), ymax * 0.75),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    ate = y1_hat.mean() - y0_hat.mean()
    ax.text((y0_hat.mean() + y1_hat.mean()) / 2, ymax * 0.77,
            f"ATE = {ate:+.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Outcome", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 7. ATE Method Comparison
# ────────────────────────────────────────────────────────────

def plot_ate_comparison(
    estimates_df: pd.DataFrame,
    title: str = "ATE Estimates — Method Comparison",
) -> plt.Figure:
    """
    Bar chart comparing ATE estimates from different methods.
    estimates_df must have columns: Method, ATE, CI Lower (opt), CI Upper (opt)
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    methods = estimates_df["Method"].tolist()
    ates    = estimates_df["ATE"].tolist()

    colors = [_PALETTE["neutral"]] + [_PALETTE["cate"]] * (len(methods) - 1)
    if "Naive" in methods[0]:
        colors[0] = "#CCCCCC"

    bars = ax.barh(methods, ates, color=colors, edgecolor="white", height=0.55)

    # Confidence interval whiskers
    if "CI Lower" in estimates_df.columns:
        for i, row in estimates_df.iterrows():
            if pd.notna(row.get("CI Lower")) and pd.notna(row.get("CI Upper")):
                ax.plot([row["CI Lower"], row["CI Upper"]], [i, i],
                        "k-", linewidth=2.5, solid_capstyle="round")
                ax.plot([row["CI Lower"]], [i], "k|", markersize=8)
                ax.plot([row["CI Upper"]], [i], "k|", markersize=8)

    # Value labels
    for bar, val in zip(bars, ates):
        ax.text(val + abs(val) * 0.02 + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=10)

    ax.axvline(0, color="black", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Average Treatment Effect (ATE)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# 8. Policy Comparison
# ────────────────────────────────────────────────────────────

def plot_policy_comparison(
    policy_df: pd.DataFrame,
    title: str = "Expected Outcomes under Different Treatment Policies",
) -> plt.Figure:
    """
    Bar chart of expected outcomes under various treatment policies.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    colors = [_PALETTE["neutral"] if "Quo" in p else _PALETTE["policy"]
              for p in policy_df["Policy"]]

    bars = ax.barh(
        policy_df["Policy"],
        policy_df["Expected Outcome"],
        color=colors, edgecolor="white", height=0.55,
    )

    for bar, val in zip(bars, policy_df["Expected Outcome"]):
        ax.text(val + abs(val) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Expected Outcome", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_policy_net_value(
    policy_df: pd.DataFrame,
    title: str = "Expected Net Value by Policy",
) -> plt.Figure:
    """
    Bar chart of expected net value under each policy.
    Requires column: Expected Net Value.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = policy_df["Expected Net Value"]
    colors = [_PALETTE["policy"] if v >= 0 else _PALETTE["treated"] for v in values]

    bars = ax.barh(policy_df["Policy"], values, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="black", linewidth=1.2, linestyle=":")

    for bar, val in zip(bars, values):
        offset = max(abs(values).max(), 1.0) * 0.01
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    ax.set_xlabel("Expected Net Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
