# causal_engine/src — public API
from .data_loader   import (
    HILLSTROM_DEFAULT_FILE,
    HILLSTROM_DEFAULT_OUTCOME,
    HILLSTROM_DEFAULT_TREATMENT,
    handle_missing_values,
    load_data,
    prepare_kevin_hillstrom,
    validate_columns,
)
from .causal_model  import build_causal_model, identify_effect
from .estimator     import estimate_ate_aipw, estimate_ate_linear, estimate_ate_propensity, estimate_heterogeneous_effects, propensity_calibration_diagnostics
from .refutation    import run_random_common_cause, run_placebo_test, interpret_refutation
from .simulator     import simulate_counterfactuals
from .visualization import (
    plot_treatment_effect_distribution,
    plot_before_after_outcome,
    plot_dag,
    plot_cate_by_feature,
    plot_propensity_overlap,
    plot_counterfactual_distributions,
)

__all__ = [
    "HILLSTROM_DEFAULT_FILE", "HILLSTROM_DEFAULT_OUTCOME", "HILLSTROM_DEFAULT_TREATMENT",
    "load_data", "validate_columns", "handle_missing_values", "prepare_kevin_hillstrom",
    "build_causal_model", "identify_effect",
    "estimate_ate_aipw", "estimate_ate_linear", "estimate_ate_propensity", "estimate_heterogeneous_effects",
    "propensity_calibration_diagnostics",
    "run_random_common_cause", "run_placebo_test", "interpret_refutation",
    "simulate_counterfactuals",
    "plot_treatment_effect_distribution", "plot_before_after_outcome",
    "plot_dag", "plot_cate_by_feature", "plot_propensity_overlap",
    "plot_counterfactual_distributions",
]
