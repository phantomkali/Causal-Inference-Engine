"""
simulator.py
────────────────────────────────────────────────────────────
Counterfactual simulation engine.

WHAT IS A COUNTERFACTUAL?
──────────────────────────
"What *would* have happened if this individual had received
a different treatment?"

For individual i:
  Y_i(1) = potential outcome under treatment
  Y_i(0) = potential outcome under control

We observe only ONE of these (Fundamental Problem of Causal
Inference). The simulator imputes the unobserved potential
outcome using a fitted response-surface model.

APPROACHES IMPLEMENTED
────────────────────────
  1. Response Surface Model (T-learner)
     ─────────────────────────────────
     Fit two separate outcome models:
       μ₀(X) = E[Y | T=0, X]
       μ₁(X) = E[Y | T=1, X]
     Counterfactual: predict Y(1) for control units and Y(0)
     for treated units.

  2. Simple Counterfactual (shift-based)
     ────────────────────────────────────
     Apply the estimated ATE to construct counterfactual
     distributions. Less flexible but fast.

  3. Policy Comparison
     ────────────────────────────────────
     Simulate "what if we treated everyone / no-one / X% of
     the population?" and report expected outcomes.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 1. Response-Surface (T-Learner) Counterfactuals
# ────────────────────────────────────────────────────────────

class TLearner:
    """
    Fit two separate outcome models (one per treatment arm)
    and impute counterfactual outcomes for each unit.
    """

    def __init__(self, base_model=None):
        self._model_0  = base_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self._model_1  = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> "TLearner":
        """Fit μ₀ and μ₁ on the respective sub-populations."""
        X_all = self._scaler.fit_transform(df[confounders].fillna(0))
        Y     = df[outcome].values
        T     = df[treatment].values

        idx_0 = T == 0
        idx_1 = T == 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_0.fit(X_all[idx_0], Y[idx_0])
            self._model_1.fit(X_all[idx_1], Y[idx_1])

        self._fitted = True
        return self

    def predict_potential_outcomes(
        self,
        df: pd.DataFrame,
        confounders: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        y0_hat : predicted Y(0) for all units
        y1_hat : predicted Y(1) for all units
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X = self._scaler.transform(df[confounders].fillna(0))
        y0_hat = self._model_0.predict(X)
        y1_hat = self._model_1.predict(X)
        return y0_hat, y1_hat

    def individual_treatment_effects(
        self,
        df: pd.DataFrame,
        confounders: List[str],
    ) -> np.ndarray:
        """ITE = Y(1) - Y(0)  for each unit."""
        y0, y1 = self.predict_potential_outcomes(df, confounders)
        return y1 - y0


# ────────────────────────────────────────────────────────────
# 2. Main simulation function
# ────────────────────────────────────────────────────────────

def simulate_counterfactuals(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    ate: Optional[float] = None,
    outcome_value_multiplier: float = 1.0,
    treatment_cost: float = 0.0,
    budget: Optional[float] = None,
) -> Dict:
    """
    Simulate counterfactual outcomes for the entire dataset.

    For each unit we compute:
      Y(0) — outcome if NOT treated
      Y(1) — outcome if treated
      ITE  — individual treatment effect = Y(1) − Y(0)

    Parameters
    ----------
    df          : full dataset
    treatment   : binary treatment column
    outcome     : continuous outcome column
    confounders : list of confounders used for response-surface model
    ate         : pre-computed ATE (used for simple-shift backup)

    Returns
    -------
    dict with keys:
        y0_hat, y1_hat, ite_hat,
        mean_y0, mean_y1, mean_ite,
        policy_df,
        df_augmented,
        t_learner
    """
    # ── Fit T-Learner ─────────────────────────────────────────
    tl = TLearner()
    tl.fit(df, treatment, outcome, confounders)

    y0_hat, y1_hat = tl.predict_potential_outcomes(df, confounders)
    ite_hat        = y1_hat - y0_hat

    # ── Augmented dataset ─────────────────────────────────────
    df_aug = df.copy()
    df_aug["_y0_hat"]  = y0_hat
    df_aug["_y1_hat"]  = y1_hat
    df_aug["_ite_hat"] = ite_hat

    # ── Policy-level simulations ──────────────────────────────
    policy_df = _policy_comparison(
        df_aug=df_aug,
        treatment=treatment,
        outcome=outcome,
        outcome_value_multiplier=outcome_value_multiplier,
        treatment_cost=treatment_cost,
        budget=budget,
    )
    best_row = policy_df.sort_values("Expected Net Value", ascending=False).iloc[0].to_dict()

    result = {
        "y0_hat":      y0_hat,
        "y1_hat":      y1_hat,
        "ite_hat":     ite_hat,
        "mean_y0":     float(y0_hat.mean()),
        "mean_y1":     float(y1_hat.mean()),
        "mean_ite":    float(ite_hat.mean()),
        "df_augmented": df_aug,
        "policy_df":   policy_df,
        "best_policy": {
            "name": best_row["Policy"],
            "expected_outcome": float(best_row["Expected Outcome"]),
            "treatment_rate": float(best_row["Treatment Rate"]),
            "expected_revenue": float(best_row["Expected Revenue"]),
            "expected_cost": float(best_row["Expected Cost"]),
            "expected_net_value": float(best_row["Expected Net Value"]),
            "budget_applied": budget is not None,
        },
        "t_learner":   tl,
    }

    logger.info(
        "Counterfactuals: E[Y(0)]=%.3f  E[Y(1)]=%.3f  E[ITE]=%.3f",
        result["mean_y0"], result["mean_y1"], result["mean_ite"]
    )
    return result


# ────────────────────────────────────────────────────────────
# 3. Policy comparison
# ────────────────────────────────────────────────────────────

def _policy_comparison(
    df_aug: pd.DataFrame,
    treatment: str,
    outcome: str,
    outcome_value_multiplier: float = 1.0,
    treatment_cost: float = 0.0,
    budget: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compare expected outcomes under different treatment policies:
      - Status Quo       : observed treatment assignment
      - Treat Everyone   : T = 1 for all
      - Treat No-one     : T = 0 for all
      - Treat Top 50 %   : treat units with highest predicted benefit
      - Treat Bottom 50 %: treat units with lowest predicted benefit
    """
    n = len(df_aug)
    treatment_rate_budget = None
    if budget is not None and n > 0:
        max_treated = int(np.floor(max(budget, 0.0) / max(treatment_cost, 1e-12)))
        max_treated = min(max_treated, n)
        treatment_rate_budget = max_treated / n

    def _evaluate_policy(policy_name: str, treat_mask: np.ndarray, observed: bool = False) -> Dict:
        treated_frac = float(np.mean(treat_mask))
        if observed:
            expected_outcome = float(df_aug[outcome].mean())
        else:
            expected_outcome = float(
                np.mean(np.where(treat_mask, df_aug["_y1_hat"].values, df_aug["_y0_hat"].values))
            )
        expected_revenue = expected_outcome * float(outcome_value_multiplier)
        expected_cost = treated_frac * float(treatment_cost)
        expected_net_value = expected_revenue - expected_cost
        return {
            "Policy": policy_name,
            "Expected Outcome": expected_outcome,
            "Treatment Rate": treated_frac,
            "Expected Revenue": expected_revenue,
            "Expected Cost": expected_cost,
            "Expected Net Value": expected_net_value,
        }

    observed_mask = (
        df_aug[treatment].astype(float).values >= 0.5
        if treatment in df_aug.columns
        else np.zeros(n, dtype=bool)
    )
    top50_mask = df_aug["_ite_hat"].values >= np.median(df_aug["_ite_hat"].values)
    bottom50_mask = ~top50_mask

    rows = [
        _evaluate_policy("Status Quo (observed)", observed_mask, observed=True),
        _evaluate_policy("Treat Everyone (T=1)", np.ones(n, dtype=bool)),
        _evaluate_policy("Treat No-one (T=0)", np.zeros(n, dtype=bool)),
        _evaluate_policy("Target Top 50% by ITE", top50_mask),
        _evaluate_policy("Target Bottom 50% by ITE", bottom50_mask),
    ]

    if treatment_rate_budget is not None:
        k = int(np.floor(treatment_rate_budget * n))
        sorted_idx = np.argsort(-df_aug["_ite_hat"].values)
        budget_top_mask = np.zeros(n, dtype=bool)
        budget_top_mask[sorted_idx[:k]] = True
        rows.append(
            _evaluate_policy(
                f"Budget-Aware Top-{int(treatment_rate_budget * 100)}% by ITE",
                budget_top_mask,
            )
        )

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────
# 4. Compare two treatments
# ────────────────────────────────────────────────────────────

def compare_two_treatments(
    df: pd.DataFrame,
    treatment_a: str,
    treatment_b: str,
    outcome: str,
    confounders: List[str],
) -> Dict:
    """
    Compare the effects of two binary treatments on the same outcome.

    For each treatment T_a and T_b:
      • Estimate ATE relative to no treatment (T=0)
      • Compute relative advantage:  ATE_A − ATE_B

    Returns
    -------
    dict with per-treatment results and a comparison summary
    """
    try:
        from .estimator import estimate_ate_linear
    except ImportError:
        from estimator import estimate_ate_linear

    result_a = estimate_ate_linear(df, treatment_a, outcome, confounders + [treatment_b])
    result_b = estimate_ate_linear(df, treatment_b, outcome, confounders + [treatment_a])

    comparison = {
        "treatment_a":        treatment_a,
        "treatment_b":        treatment_b,
        "ate_a":              result_a["ate"],
        "ate_b":              result_b["ate"],
        "relative_advantage": result_a["ate"] - result_b["ate"],
        "winner":             treatment_a if result_a["ate"] > result_b["ate"] else treatment_b,
        "result_a":           result_a,
        "result_b":           result_b,
    }

    logger.info(
        "Treatment comparison: %s ATE=%.3f vs %s ATE=%.3f  Δ=%.3f",
        treatment_a, result_a["ate"],
        treatment_b, result_b["ate"],
        comparison["relative_advantage"],
    )
    return comparison
