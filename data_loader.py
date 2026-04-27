"""
data_loader.py
────────────────────────────────────────────────────────────
Responsible for:
  • Loading CSV / DataFrame inputs
  • Column validation
  • Missing-value imputation / removal
  • Basic descriptive statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Type alias ───────────────────────────────────────────────
FileLike = Union[str, Path, IO[bytes]]

HILLSTROM_DEFAULT_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
HILLSTROM_DEFAULT_OUTCOME = "spend"
HILLSTROM_DEFAULT_TREATMENT = "treatment"


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def load_data(source: FileLike, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a dataset from a CSV file path or file-like object.

    Parameters
    ----------
    source          : path string, Path object, or file-like (e.g. Streamlit UploadedFile)
    read_csv_kwargs : forwarded to pandas.read_csv

    Returns
    -------
    pd.DataFrame — raw DataFrame

    Raises
    ------
    FileNotFoundError : if a path string/Path points to a non-existent file
    ValueError        : if the file is empty
    """
    try:
        df = pd.read_csv(source, **read_csv_kwargs)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {source}") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df


def prepare_kevin_hillstrom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Kevin Hillstrom dataset for causal analysis.

    Adds a binary `treatment` column from `segment` where:
      - 1 => received any email (Mens or Womens E-Mail)
      - 0 => No E-Mail
    Also coerces known numeric outcome/feature columns to numeric dtype.
    """
    out = df.copy()

    if "segment" in out.columns and HILLSTROM_DEFAULT_TREATMENT not in out.columns:
        no_email = out["segment"].astype(str).str.strip().str.lower() == "no e-mail"
        out[HILLSTROM_DEFAULT_TREATMENT] = (~no_email).astype(int)

    numeric_candidates = [
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "visit",
        "conversion",
        "spend",
        HILLSTROM_DEFAULT_TREATMENT,
    ]
    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def validate_columns(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> None:
    """
    Validate that all required columns exist and have the correct dtypes.

    Raises
    ------
    KeyError   : missing column(s)
    ValueError : treatment is not binary, or outcome is not numeric
    """
    required = {treatment, outcome, *confounders}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in dataset: {sorted(missing)}")

    # Treatment must be binary (0/1)
    unique_vals = set(df[treatment].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Treatment column '{treatment}' must be binary (0/1). "
            f"Found values: {unique_vals}"
        )

    # Outcome must be numeric
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(
            f"Outcome column '{outcome}' must be numeric. "
            f"Got dtype: {df[outcome].dtype}"
        )

    # Confounders must exist and be numeric (for modelling)
    for col in confounders:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                "Confounder '%s' is not numeric (dtype=%s). "
                "It will be label-encoded automatically.",
                col, df[col].dtype
            )

    logger.info("Column validation passed.")


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Impute or drop columns / rows with missing data.

    Parameters
    ----------
    df             : input DataFrame
    strategy       : 'median' | 'mean' | 'drop_rows'
    drop_threshold : columns with > this fraction missing are dropped entirely

    Returns
    -------
    pd.DataFrame — cleaned copy
    """
    df = df.copy()

    # ── Drop columns that are mostly empty ──────────────────
    frac_missing = df.isnull().mean()
    cols_to_drop = frac_missing[frac_missing > drop_threshold].index.tolist()
    if cols_to_drop:
        logger.warning("Dropping high-missingness columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # ── Handle remaining missingness ─────────────────────────
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        return df

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with missing values.", before - len(df))

    elif strategy in ("median", "mean"):
        numeric_cols    = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(exclude="number").columns

        fill_fn = df[numeric_cols].median if strategy == "median" else df[numeric_cols].mean
        df[numeric_cols] = df[numeric_cols].fillna(fill_fn())

        # Categorical: fill with mode
        for col in categorical_cols:
            mode_val = df[col].mode()
            if len(mode_val):
                df[col] = df[col].fillna(mode_val[0])

        logger.info("Imputed %d missing values with strategy='%s'.", total_missing, strategy)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'median', 'mean', or 'drop_rows'.")

    return df


def encode_categoricals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Label-encode categorical columns so they are numeric.

    Parameters
    ----------
    df      : input DataFrame
    columns : list of column names to encode

    Returns
    -------
    pd.DataFrame — copy with encoded columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.Categorical(df[col]).codes.astype(float)
            logger.info("Label-encoded column: '%s'", col)
    return df


def describe_data(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> dict:
    """
    Return a dict of descriptive statistics useful for the UI summary panel.
    """
    treated   = df[df[treatment] == 1]
    untreated = df[df[treatment] == 0]

    stats = {
        "n_total":           len(df),
        "n_treated":         int(df[treatment].sum()),
        "n_untreated":       int((df[treatment] == 0).sum()),
        "treatment_rate":    float(df[treatment].mean()),
        "outcome_mean_treated":   float(treated[outcome].mean()),
        "outcome_mean_untreated": float(untreated[outcome].mean()),
        "naive_ate":         float(treated[outcome].mean() - untreated[outcome].mean()),
        "missing_pct":       float(df.isnull().mean().mean()),
        "confounder_summary": {
            col: {
                "mean_treated":   float(treated[col].mean()),
                "mean_untreated": float(untreated[col].mean()),
            }
            for col in confounders if col in df.columns
        },
    }
    return stats
