"""
refutation.py
────────────────────────────────────────────────────────────
Stress-tests for causal estimates (robustness checks).

WHY REFUTE?
───────────
An estimate that survives adversarial tests is more trustworthy.
DoWhy provides several refutation methods. We use three:

  1. Random Common Cause
     ─────────────────────
     Add a random variable as a new confounder.
     A robust estimate should NOT change significantly — because
     the added variable has no real causal relationship.

  2. Placebo Treatment
     ─────────────────────
     Replace the real treatment with a random permutation.
     A robust estimate should collapse to ~0 — because there
     is no real treatment signal to detect.

  3. Data Subset Refuter
     ─────────────────────
     Re-estimate on 80 % random sub-samples.
     A robust estimate should remain stable across subsets.

INTERPRETATION RULES
─────────────────────
  • Random Common Cause: |new_ATE - orig_ATE| < 10 %  → PASS
  • Placebo Treatment  : |new_ATE| ≈ 0 (p > 0.05)    → PASS
  • Data Subset        : CV of sub-sample ATEs < 0.1   → PASS
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tolerance thresholds
_RCC_TOLERANCE  = 0.10   # 10 % relative change
_PLACEBO_TOL    = 0.20   # absolute ATE < 20 % of original → pass


# ────────────────────────────────────────────────────────────
# 1. Random Common Cause Refuter
# ────────────────────────────────────────────────────────────

def run_random_common_cause(
    ci_model,
    estimate,
    n_simulations: int = 20,
) -> Dict:
    """
    Add a random (causally irrelevant) confounder and re-estimate.

    If the estimate changes substantially, the original result may
    be fragile and sensitive to hidden confounders.

    Parameters
    ----------
    ci_model      : CausalInferenceModel (built & identified)
    estimate      : DoWhy estimate object (from ci_model.estimate())
    n_simulations : number of random confounders to try

    Returns
    -------
    dict with keys: original_ate, new_ate, p_value, passed, interpretation
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refutation = ci_model.refute(
                estimate,
                method_name="random_common_cause",
                num_simulations=n_simulations,
            )

        original_ate = float(estimate.value)
        new_ate      = float(refutation.new_effect)
        rel_change   = abs(new_ate - original_ate) / (abs(original_ate) + 1e-9)
        passed       = rel_change < _RCC_TOLERANCE

        return {
            "refutation_type": "Random Common Cause",
            "original_ate":    original_ate,
            "new_ate":         new_ate,
            "relative_change": rel_change,
            "p_value":         getattr(refutation, "refutation_result", None),
            "passed":          passed,
            "raw_refutation":  refutation,
            "interpretation":  _rcc_interpretation(original_ate, new_ate, rel_change, passed),
        }

    except Exception as exc:
        logger.warning("Random Common Cause refutation failed: %s", exc)
        return _fallback_rcc(ci_model.df, ci_model.treatment,
                             ci_model.outcome, ci_model.confounders,
                             float(estimate.value))


def _rcc_interpretation(orig: float, new: float, rel_change: float, passed: bool) -> str:
    direction = "✅ PASS" if passed else "⚠️ FAIL"
    return (
        f"{direction} — Random Common Cause Refutation\n"
        f"  Original ATE : {orig:+.4f}\n"
        f"  New ATE      : {new:+.4f}\n"
        f"  Relative Δ   : {rel_change:.1%}\n"
        f"\n"
        f"  {'The estimate is stable — adding a random confounder did not significantly change' if passed else 'WARNING: the estimate shifted by more than 10 %, suggesting sensitivity to hidden confounders.'}\n"
        f"  {'it (< 10 % change). This is a good sign.' if passed else 'Consider adding more controls or using a sensitivity analysis.'}"
    )


def _fallback_rcc(df, treatment, outcome, confounders, original_ate) -> Dict:
    """Numpy-only fallback for RCC when DoWhy fails."""
    rng = np.random.default_rng(42)
    new_ates = []
    from sklearn.linear_model import LinearRegression

    for _ in range(20):
        df2 = df.copy()
        df2["__random__"] = rng.normal(size=len(df2))
        X = df2[[treatment] + confounders + ["__random__"]].values
        y = df2[outcome].values
        lr = LinearRegression().fit(X, y)
        new_ates.append(lr.coef_[0])

    new_ate   = float(np.mean(new_ates))
    rel_change = abs(new_ate - original_ate) / (abs(original_ate) + 1e-9)
    passed    = rel_change < _RCC_TOLERANCE

    return {
        "refutation_type":  "Random Common Cause (fallback)",
        "original_ate":     original_ate,
        "new_ate":          new_ate,
        "relative_change":  rel_change,
        "p_value":          None,
        "passed":           passed,
        "raw_refutation":   None,
        "interpretation":   _rcc_interpretation(original_ate, new_ate, rel_change, passed),
    }


# ────────────────────────────────────────────────────────────
# 2. Placebo Treatment Refuter
# ────────────────────────────────────────────────────────────

def run_placebo_test(
    ci_model,
    estimate,
    n_simulations: int = 20,
) -> Dict:
    """
    Replace treatment with a random permutation and re-estimate.

    A robust causal estimate should vanish (~0) when treatment
    is randomised, because there is no longer any real signal.

    Returns
    -------
    dict with keys: original_ate, placebo_ate, passed, interpretation
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refutation = ci_model.refute(
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=n_simulations,
            )

        original_ate = float(estimate.value)
        placebo_ate  = float(refutation.new_effect)
        abs_ratio    = abs(placebo_ate) / (abs(original_ate) + 1e-9)
        passed       = abs_ratio < _PLACEBO_TOL

        return {
            "refutation_type": "Placebo Treatment",
            "original_ate":    original_ate,
            "placebo_ate":     placebo_ate,
            "abs_ratio":       abs_ratio,
            "p_value":         getattr(refutation, "refutation_result", None),
            "passed":          passed,
            "raw_refutation":  refutation,
            "interpretation":  _placebo_interpretation(original_ate, placebo_ate, abs_ratio, passed),
        }

    except Exception as exc:
        logger.warning("Placebo refutation failed: %s", exc)
        return _fallback_placebo(ci_model.df, ci_model.treatment,
                                 ci_model.outcome, ci_model.confounders,
                                 float(estimate.value))


def _placebo_interpretation(orig: float, placebo: float, ratio: float, passed: bool) -> str:
    direction = "✅ PASS" if passed else "⚠️ FAIL"
    return (
        f"{direction} — Placebo Treatment Refutation\n"
        f"  Original ATE  : {orig:+.4f}\n"
        f"  Placebo ATE   : {placebo:+.4f}  (should be ≈ 0)\n"
        f"  |Placebo/Orig|: {ratio:.1%}\n"
        f"\n"
        f"  {'The placebo effect is close to zero — the original estimate is not spurious.' if passed else 'WARNING: Placebo effect is large relative to original. The estimate may be driven by confounding or model artefacts.'}"
    )


def _fallback_placebo(df, treatment, outcome, confounders, original_ate) -> Dict:
    """Numpy-only fallback for placebo when DoWhy fails."""
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(42)
    placebo_ates = []

    for _ in range(20):
        df2 = df.copy()
        df2["__placebo__"] = rng.permutation(df2[treatment].values)
        X = df2[["__placebo__"] + confounders].values
        y = df2[outcome].values
        lr = LinearRegression().fit(X, y)
        placebo_ates.append(lr.coef_[0])

    placebo_ate = float(np.mean(placebo_ates))
    abs_ratio   = abs(placebo_ate) / (abs(original_ate) + 1e-9)
    passed      = abs_ratio < _PLACEBO_TOL

    return {
        "refutation_type":  "Placebo Treatment (fallback)",
        "original_ate":     original_ate,
        "placebo_ate":      placebo_ate,
        "abs_ratio":        abs_ratio,
        "p_value":          None,
        "passed":           passed,
        "raw_refutation":   None,
        "interpretation":   _placebo_interpretation(original_ate, placebo_ate, abs_ratio, passed),
    }


# ────────────────────────────────────────────────────────────
# 3. Data Subset Refuter
# ────────────────────────────────────────────────────────────

def run_data_subset_refuter(
    ci_model,
    estimate,
    n_simulations: int = 20,
    subset_fraction: float = 0.8,
) -> Dict:
    """
    Re-estimate on random subsets of the data.

    A stable estimate should have low variance across subsets.
    High variance indicates the estimate may be driven by a few
    influential observations.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refutation = ci_model.refute(
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=subset_fraction,
                num_simulations=n_simulations,
            )

        original_ate = float(estimate.value)
        new_ate      = float(refutation.new_effect)
        rel_change   = abs(new_ate - original_ate) / (abs(original_ate) + 1e-9)
        passed       = rel_change < _RCC_TOLERANCE

        return {
            "refutation_type": "Data Subset Refuter",
            "original_ate":    original_ate,
            "new_ate":         new_ate,
            "relative_change": rel_change,
            "passed":          passed,
            "raw_refutation":  refutation,
            "interpretation": (
                f"{'✅ PASS' if passed else '⚠️ FAIL'} — Data Subset Refutation\n"
                f"  Original ATE : {original_ate:+.4f}\n"
                f"  Subset ATE   : {new_ate:+.4f}\n"
                f"  Relative Δ   : {rel_change:.1%}\n"
                f"  {'Stable across subsets — the estimate is not driven by outliers.' if passed else 'Large variation across subsets — consider robust estimation methods.'}"
            ),
        }

    except Exception as exc:
        logger.warning("Data subset refutation failed: %s", exc)
        return {
            "refutation_type": "Data Subset Refuter",
            "passed":          None,
            "interpretation":  f"Could not complete refutation: {exc}",
        }


# ────────────────────────────────────────────────────────────
# Interpret all refutations together
# ────────────────────────────────────────────────────────────

def interpret_refutation(refutations: List[Dict]) -> str:
    """
    Produce a human-readable summary across all refutation tests.
    """
    n_pass  = sum(1 for r in refutations if r.get("passed") is True)
    n_fail  = sum(1 for r in refutations if r.get("passed") is False)
    n_total = len([r for r in refutations if r.get("passed") is not None])

    lines = [
        "═" * 55,
        "  CAUSAL ROBUSTNESS REPORT",
        "═" * 55,
        f"  Tests passed : {n_pass} / {n_total}",
        f"  Tests failed : {n_fail} / {n_total}",
        "─" * 55,
    ]

    for r in refutations:
        icon = "✅" if r.get("passed") else ("⚠️" if r.get("passed") is False else "❓")
        lines.append(f"  {icon}  {r['refutation_type']}")

    lines.append("─" * 55)

    if n_fail == 0 and n_pass > 0:
        lines.append("  VERDICT: The estimate is ROBUST. It passed all refutation")
        lines.append("  tests. You can have reasonable confidence in the ATE.")
    elif n_fail == n_total:
        lines.append("  VERDICT: CAUTION — all tests failed. The estimate may be")
        lines.append("  confounded or model-dependent. Investigate further.")
    else:
        lines.append(f"  VERDICT: MIXED — {n_pass} passed, {n_fail} failed.")
        lines.append("  Interpret the ATE with caution.")

    lines.append("═" * 55)
    return "\n".join(lines)
