"""
causal_model.py
────────────────────────────────────────────────────────────
Builds and manages the DoWhy causal model.

Responsibilities
────────────────
  • Construct the CausalModel from data + variable specification
  • Generate a readable DAG (graphviz / networkx)
  • Identify the causal estimand
  • Expose utilities for re-use in estimation and refutation

WHY DOWHY?
──────────
DoWhy enforces the four steps of causal inference:
  1. Model   — encode assumptions as a causal graph
  2. Identify — derive the statistical estimand from the graph
  3. Estimate — compute the estimand from data
  4. Refute   — stress-test the estimate
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ── Optional DoWhy import (graceful degradation) ────────────
try:
    from dowhy import CausalModel as _DoWhyCausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not installed — some features will be unavailable.")


# ────────────────────────────────────────────────────────────
# DAG helpers
# ────────────────────────────────────────────────────────────

def build_dag(
    treatment: str,
    outcome: str,
    confounders: List[str],
    additional_edges: Optional[List[tuple]] = None,
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph representing the causal DAG.

    Edges:
        confounder → treatment   (each confounder affects treatment)
        confounder → outcome     (each confounder affects outcome)
        treatment  → outcome     (the causal effect we want to estimate)

    Parameters
    ----------
    treatment        : name of the treatment node
    outcome          : name of the outcome node
    confounders      : list of common-cause node names
    additional_edges : optional list of (src, dst) tuples for extra edges

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    G.add_node(treatment, node_type="treatment")
    G.add_node(outcome,   node_type="outcome")

    for conf in confounders:
        G.add_node(conf, node_type="confounder")
        G.add_edge(conf, treatment)
        G.add_edge(conf, outcome)

    G.add_edge(treatment, outcome)

    if additional_edges:
        for src, dst in additional_edges:
            G.add_edge(src, dst)

    return G


def dag_to_dot_string(G: nx.DiGraph) -> str:
    """
    Convert a NetworkX DiGraph to a DOT-format string for graphviz rendering.
    Node colours: treatment=coral, outcome=steelblue, confounders=lightgreen.
    """
    lines = ["digraph CausalDAG {",
             "  rankdir=LR;",
             "  node [fontname=\"Helvetica\" fontsize=12];"]

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "unknown")
        colour = {
            "treatment":  "#FA8072",   # salmon/coral
            "outcome":    "#4682B4",   # steelblue
            "confounder": "#90EE90",   # lightgreen
        }.get(node_type, "#DDDDDD")

        label = node.replace("_", " ").title()
        lines.append(
            f'  "{node}" [label="{label}" style=filled fillcolor="{colour}" '
            f'fontcolor="white" shape=ellipse];'
        )

    for src, dst in G.edges():
        lines.append(f'  "{src}" -> "{dst}";')

    lines.append("}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# CausalModel wrapper
# ────────────────────────────────────────────────────────────

class CausalInferenceModel:
    """
    High-level wrapper around DoWhy's CausalModel.

    Usage
    -----
    >>> model = CausalInferenceModel(df, treatment="T", outcome="Y",
    ...                              confounders=["age", "income"])
    >>> model.build()
    >>> estimand = model.identify()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ):
        self.df          = df
        self.treatment   = treatment
        self.outcome     = outcome
        self.confounders = confounders

        self._dag: Optional[nx.DiGraph]   = None
        self._model                        = None   # DoWhy CausalModel
        self._identified_estimand          = None

    # ── Build ────────────────────────────────────────────────

    def build(self) -> "CausalInferenceModel":
        """
        Construct the DAG and (if DoWhy is available) the DoWhy CausalModel.
        Must be called before identify() or estimate().
        """
        self._dag = build_dag(self.treatment, self.outcome, self.confounders)

        if not DOWHY_AVAILABLE:
            raise ImportError("DoWhy is required. Run: pip install dowhy")

        # DoWhy accepts the graph as a GML string or nx.DiGraph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = _DoWhyCausalModel(
                data=self.df,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self.confounders,
                graph=self._dag,           # pass networkx graph directly
            )

        logger.info("CausalModel built (treatment=%s, outcome=%s, confounders=%s)",
                    self.treatment, self.outcome, self.confounders)
        return self

    # ── Identify ─────────────────────────────────────────────

    def identify(self, proceed_when_unidentifiable: bool = True):
        """
        Run DoWhy's identification step.
        Returns the identified causal estimand.
        """
        if self._model is None:
            raise RuntimeError("Call .build() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._identified_estimand = self._model.identify_effect(
                proceed_when_unidentifiable=proceed_when_unidentifiable
            )

        logger.info("Causal effect identified.")
        return self._identified_estimand

    # ── Estimate (delegates to DoWhy) ────────────────────────

    def estimate(self, method_name: str = "backdoor.linear_regression", **kwargs):
        """
        Estimate the causal effect using the specified method.

        Common method names:
          backdoor.linear_regression
          backdoor.propensity_score_matching
          backdoor.propensity_score_weighting
        """
        if self._identified_estimand is None:
            raise RuntimeError("Call .identify() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate = self._model.estimate_effect(
                self._identified_estimand,
                method_name=method_name,
                **kwargs,
            )

        return estimate

    # ── Refute ────────────────────────────────────────────────

    def refute(self, estimate, method_name: str, **kwargs):
        """Pass-through to DoWhy's refute_estimate."""
        if self._model is None:
            raise RuntimeError("Call .build() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refutation = self._model.refute_estimate(
                self._identified_estimand,
                estimate,
                method_name=method_name,
                **kwargs,
            )
        return refutation

    # ── Accessors ────────────────────────────────────────────

    @property
    def dag(self) -> Optional[nx.DiGraph]:
        return self._dag

    @property
    def model(self):
        return self._model

    @property
    def identified_estimand(self):
        return self._identified_estimand


# ────────────────────────────────────────────────────────────
# Convenience factory
# ────────────────────────────────────────────────────────────

def build_causal_model(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
) -> CausalInferenceModel:
    """
    Factory: build + identify in one call.

    Returns a ready-to-use CausalInferenceModel with:
      .dag, .model, .identified_estimand all populated.
    """
    ci_model = CausalInferenceModel(df, treatment, outcome, confounders)
    ci_model.build()
    ci_model.identify()
    return ci_model


def identify_effect(ci_model: CausalInferenceModel):
    """
    Convenience wrapper — returns the already-identified estimand.
    Calls .identify() if not yet done.
    """
    if ci_model.identified_estimand is None:
        return ci_model.identify()
    return ci_model.identified_estimand
