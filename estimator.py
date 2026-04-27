"""
estimator.py
────────────────────────────────────────────────────────────
Causal effect estimators:

  ATE  (Average Treatment Effect)
  ─────
  1. Linear Regression (OLS with confounder controls)
  2. Propensity Score Matching (PSM via DoWhy)
  3. Inverse Probability Weighting (IPW)
  4. Doubly Robust AIPW

  CATE (Conditional / Heterogeneous Treatment Effects)
  ──────
  4. EconML — LinearDML (Double Machine Learning)

QUICK PRIMER
────────────
  Naive ATE = E[Y|T=1] - E[Y|T=0]   ← biased due to confounding
  Adjusted ATE controls for the confounders so the estimate
  reflects only the causal effect of treatment, not selection bias.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 1. Linear Regression ATE (OLS)
# ────────────────────────────────────────────────────────────

def estimate_ate_linear(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Dict:
    """
    Estimate ATE via Ordinary Least Squares with confounder controls.

    Model:  Y = β₀ + τ·T + γ·X + ε
    ATE    = τ  (coefficient on treatment)

    Returns
    -------
    dict with keys:
        ate, std_error, t_stat, p_value, ci_lower, ci_upper, method
    """
    import statsmodels.api as sm

    feature_cols = [treatment] + confounders
    X = df[feature_cols].copy()
    y = df[outcome].copy()

    # Drop any rows with NaN in these columns
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    X_const = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(y, X_const).fit()

    idx = X_const.columns.get_loc(treatment)

    result = {
        "ate":      float(ols.params[treatment]),
        "std_error": float(ols.bse[treatment]),
        "t_stat":   float(ols.tvalues[treatment]),
        "p_value":  float(ols.pvalues[treatment]),
        "ci_lower": float(ols.conf_int().loc[treatment, 0]),
        "ci_upper": float(ols.conf_int().loc[treatment, 1]),
        "r_squared": float(ols.rsquared),
        "n_obs":    int(ols.nobs),
        "method":   "OLS Linear Regression",
    }

    logger.info("OLS ATE = %.4f (95 CI: [%.4f, %.4f])",
                result["ate"], result["ci_lower"], result["ci_upper"])
    return result


# ────────────────────────────────────────────────────────────
# 2. Propensity Score Matching (PSM)
# ────────────────────────────────────────────────────────────

def estimate_ate_propensity(
    ci_model,          # CausalInferenceModel from causal_model.py
    method: str = "propensity_score_matching",
) -> Dict:
    """
    Estimate ATE via propensity-score-based methods using DoWhy.

    Parameters
    ----------
    ci_model : CausalInferenceModel (already built & identified)
    method   : 'propensity_score_matching' | 'propensity_score_weighting'

    Returns
    -------
    dict with keys: ate, method, raw_estimate
    """
    dowhy_method = f"backdoor.{method}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate = ci_model.estimate(method_name=dowhy_method)

        ate = float(estimate.value)
        return {
            "ate":          ate,
            "method":       method.replace("_", " ").title(),
            "raw_estimate": estimate,
        }

    except Exception as exc:
        logger.warning("PSM failed (%s) — falling back to IPW. Error: %s", method, exc)
        return _ipw_ate(
            df=ci_model.df,
            treatment=ci_model.treatment,
            outcome=ci_model.outcome,
            confounders=ci_model.confounders,
        )


def _ipw_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Dict:
    """
    Inverse Probability Weighting (IPW) fallback ATE estimator.

    Horvitz-Thompson estimator:
      ATE = E[ Y·T/e(X) ] - E[ Y·(1-T)/(1-e(X)) ]
    where e(X) = P(T=1|X) is the propensity score.
    """
    X = df[confounders].fillna(df[confounders].median())
    T = df[treatment].values
    Y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, T)
    propensity = ps_model.predict_proba(X_scaled)[:, 1]

    # Trim extreme propensity scores to reduce variance
    propensity = np.clip(propensity, 0.05, 0.95)

    ate = (Y * T / propensity).mean() - (Y * (1 - T) / (1 - propensity)).mean()

    return {
        "ate":    float(ate),
        "method": "Inverse Probability Weighting (IPW)",
        "raw_estimate": None,
    }


def estimate_ate_aipw(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> Dict:
    """
    Augmented Inverse Probability Weighting (AIPW) ATE estimator.

    Doubly robust: consistent if either the propensity model or the
    outcome regression model is correctly specified.
    """
    if not confounders:
        raise ValueError("AIPW requires at least one confounder.")

    cols = [treatment, outcome] + confounders
    sub = df[cols].dropna().copy()
    if sub.empty:
        raise ValueError("No complete rows available for AIPW.")

    X = sub[confounders]
    T = sub[treatment].astype(float).values
    Y = sub[outcome].astype(float).values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Propensity model e(X)
    ps_model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    ps_model.fit(X_sc, T)
    e_hat = np.clip(ps_model.predict_proba(X_sc)[:, 1], 0.01, 0.99)

    # Outcome models m1(X), m0(X)
    m1 = RandomForestRegressor(n_estimators=80, min_samples_leaf=10, random_state=42)
    m0 = RandomForestRegressor(n_estimators=80, min_samples_leaf=10, random_state=42)
    m1.fit(X_sc[T == 1], Y[T == 1])
    m0.fit(X_sc[T == 0], Y[T == 0])
    mu1 = m1.predict(X_sc)
    mu0 = m0.predict(X_sc)

    # AIPW score
    dr_scores = (mu1 - mu0) + T * (Y - mu1) / e_hat - (1 - T) * (Y - mu0) / (1 - e_hat)
    ate = float(np.mean(dr_scores))
    se = float(np.std(dr_scores, ddof=1) / np.sqrt(len(dr_scores)))

    return {
        "ate": ate,
        "std_error": se,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "n_obs": int(len(dr_scores)),
        "method": "Doubly Robust AIPW",
        "raw_estimate": None,
    }


# ────────────────────────────────────────────────────────────
# 3. Propensity Scores (for diagnostics / overlap check)
# ────────────────────────────────────────────────────────────

def compute_propensity_scores(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
) -> np.ndarray:
    """
    Estimate P(T=1|X) using logistic regression.
    Used for overlap / common-support diagnostics.
    """
    X = df[confounders].fillna(df[confounders].median())
    T = df[treatment].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    model.fit(X_scaled, T)

    return model.predict_proba(X_scaled)[:, 1]


def propensity_calibration_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    confounders: List[str],
    n_bins: int = 10,
) -> Dict:
    """
    Evaluate propensity model calibration quality.

    Returns Brier score, log-loss, and calibration-curve points.
    Lower Brier/log-loss indicates better calibration.
    """
    X = df[confounders].fillna(df[confounders].median())
    T = df[treatment].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    model.fit(X_scaled, T)
    p_hat = np.clip(model.predict_proba(X_scaled)[:, 1], 1e-6, 1 - 1e-6)

    frac_pos, mean_pred = calibration_curve(T, p_hat, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(T, p_hat))
    ll = float(log_loss(T, p_hat))
    ece = float(np.mean(np.abs(frac_pos - mean_pred))) if len(frac_pos) else float("nan")
    status = "Good" if brier < 0.20 else ("Moderate" if brier < 0.25 else "Poor")

    return {
        "brier_score": brier,
        "log_loss": ll,
        "ece": ece,
        "calibration_pred_mean": mean_pred,
        "calibration_frac_positive": frac_pos,
        "status": status,
    }


# ────────────────────────────────────────────────────────────
# 4. Heterogeneous Effects via EconML (CausalForestDML)
# ────────────────────────────────────────────────────────────

def estimate_heterogeneous_effects(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    heterogeneity_features: Optional[List[str]] = None,
    n_folds: int = 5,
) -> Dict:
    """
    Estimate Conditional Average Treatment Effects (CATE) using
    CausalForestDML from EconML.

    DML works by:
      1. Residualise Y ~ f(X)  →  Ỹ
      2. Residualise T ~ g(X)  →  T̃
      3. Estimate τ(X) from Ỹ = τ(X)·T̃ + ε

    This removes the influence of X on both Y and T, yielding
    an unbiased estimate of the causal effect.

    Parameters
    ----------
    df                     : full dataset
    treatment              : treatment column name
    outcome                : outcome column name
    confounders            : list of confounder column names
    heterogeneity_features : features to stratify CATE on
                             (defaults to confounders)
    n_folds                : cross-fitting folds

    Returns
    -------
    dict with keys:
        cate_values, cate_mean, cate_std, ci_lower, ci_upper,
        feature_names, model, method
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError("EconML is required: pip install econml") from exc

    het_features = heterogeneity_features or confounders

    # ── Prepare arrays ───────────────────────────────────────
    mask = (
        df[[treatment, outcome] + confounders + het_features]
        .notna().all(axis=1)
    )
    sub = df[mask].copy()

    Y = sub[outcome].values
    T = sub[treatment].values.astype(float)
    X = sub[het_features].values                  # heterogeneity covariates
    W = sub[confounders].values if confounders else None   # nuisance controls

    unique_t = set(np.unique(T))
    is_binary_treatment = unique_t.issubset({0.0, 1.0})

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    W_sc   = scaler.fit_transform(W) if W is not None else None

    # ── Fit CausalForestDML ──────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dml = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=300, min_samples_leaf=10, random_state=42
            ),
            model_t=(
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=10, random_state=42
                )
                if is_binary_treatment
                else RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=10, random_state=42
                )
            ),
            discrete_treatment=is_binary_treatment,
            cv=n_folds,
            n_estimators=400,
            min_samples_leaf=5,
            random_state=42,
        )
        dml.fit(Y, T, X=X_sc, W=W_sc)

    # ── CATE estimates ───────────────────────────────────────
    cate_values = dml.effect(X_sc).flatten()

    # Bootstrap-based confidence intervals (fast: 100 samples)
    try:
        inf = dml.effect_interval(X_sc, alpha=0.05)
        ci_lower = inf[0].flatten()
        ci_upper = inf[1].flatten()
    except Exception:
        se         = cate_values.std() / np.sqrt(len(cate_values))
        ci_lower   = cate_values - 1.96 * se
        ci_upper   = cate_values + 1.96 * se

    result = {
        "cate_values":    cate_values,
        "cate_mean":      float(cate_values.mean()),
        "cate_std":       float(cate_values.std()),
        "ci_lower":       ci_lower,
        "ci_upper":       ci_upper,
        "feature_names":  het_features,
        "X":              sub[het_features],     # original (un-scaled) for plotting
        "model":          dml,
        "method":         "Causal Forest DML (EconML)",
        "n_obs":          int(mask.sum()),
    }

    logger.info(
        "CATE estimated: mean=%.4f, std=%.4f (n=%d)",
        result["cate_mean"], result["cate_std"], result["n_obs"]
    )
    return result


# ────────────────────────────────────────────────────────────
# Summary helper
# ────────────────────────────────────────────────────────────

def summarise_estimates(
    ols_result: Dict,
    psm_result: Optional[Dict] = None,
    aipw_result: Optional[Dict] = None,
    cate_result: Optional[Dict] = None,
    naive_ate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compile all ATE estimates into a comparison DataFrame.
    """
    rows = []

    if naive_ate is not None:
        rows.append({"Method": "Naive Diff-in-Means (biased)",
                     "ATE": naive_ate, "CI Lower": None, "CI Upper": None})

    rows.append({
        "Method":   ols_result["method"],
        "ATE":      ols_result["ate"],
        "CI Lower": ols_result.get("ci_lower"),
        "CI Upper": ols_result.get("ci_upper"),
    })

    if psm_result:
        rows.append({
            "Method":   psm_result["method"],
            "ATE":      psm_result["ate"],
            "CI Lower": None,
            "CI Upper": None,
        })

    if aipw_result:
        rows.append({
            "Method":   aipw_result["method"],
            "ATE":      aipw_result["ate"],
            "CI Lower": aipw_result.get("ci_lower"),
            "CI Upper": aipw_result.get("ci_upper"),
        })

    if cate_result:
        rows.append({
            "Method":   cate_result["method"] + " (mean CATE)",
            "ATE":      cate_result["cate_mean"],
            "CI Lower": None,
            "CI Upper": None,
        })

    return pd.DataFrame(rows)
