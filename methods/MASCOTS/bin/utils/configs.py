from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

from mascots.explainer.pipeline import get_default_borf_args

default_borf_explainer_init_args: dict[str, Any] = {
    "borf_config": "config/126_borf_full.json",
    "borf_args": get_default_borf_args(),
}

default_borf_explainer_build_args: dict[str, Any] = {
    "on_top_model": RandomForestClassifier(),
    "attribution_name": "shap",
    "attribution_args": {"mode": "deep"},
}

default_borf_explainer_counterfactual_args: dict[str, Any] = {
    "swap_method": "scalar",
    "max_borf_changes": 20,
    # "min_word_len_change": 16,
    # "max_word_len_change": 64,
    # "min_symbol_size": 8,
    # "max_symbol_size": -1,
    "select_top_k": 5,
    "C": 0.5,
}

alternative_borf_explainer_init_args: dict[str, list[Any]] = {
    "borf_config": ["config/126_borf_simple.json"],
}

alternative_borf_explainer_build_args: dict[str, list[Any]] = {
    "attribution_args": [{"scope": "global"}],
}

alternative_borf_explainer_counterfactual_args: dict[str, list[Any]] = {
    "swap_method": ["gaussian"],
    "min_word_len_change": [8, 32],
    "min_symbol_size": [4, 16],
    "select_top_k": [4],
    "C": [0.0, 2.0],
}
