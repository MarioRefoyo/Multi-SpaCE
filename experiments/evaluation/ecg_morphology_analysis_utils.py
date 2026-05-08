"""Utilities for Multi-SpaCE ECG morphology analysis.

The analysis mirrors the notebook section "Study about ecg changes":
select one counterfactual per explained instance, delineate the original
and NUN on Lead II only, propagate the resulting reference windows to all
channels, and measure original-to-counterfactual morphology changes.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    import neurokit2 as nk

    NEUROKIT_AVAILABLE = True
    NEUROKIT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local notebook env.
    nk = None
    NEUROKIT_AVAILABLE = False
    NEUROKIT_IMPORT_ERROR = exc


ORIG_KEYS = ["x_orig", "x_original", "original", "orig", "x", "query", "instance", "test_instance"]
CF_KEYS = ["x_cf", "cf", "counterfactual", "counterfactuals", "cfs", "x_counterfactual"]
NUN_KEYS = ["x_nun", "nun", "nearest_unlike_neighbor", "nearest_unlike", "x_target"]
MASK_KEYS = ["mask", "change_mask", "orig_change_mask", "cf_mask", "m"]
PARETO_KEYS = ["pareto", "pareto_front", "front", "population", "solutions", "candidates", "explanations"]
Y_TRUE_KEYS = ["y_true", "true_label", "label", "x_orig_true_label", "orig_true_label"]
PRED_ORIG_KEYS = ["predicted_orig", "pred_orig", "y_pred_orig", "original_prediction", "x_orig_pred_label", "orig_pred_label"]
PRED_CF_KEYS = ["predicted_cf", "pred_cf", "y_pred_cf", "cf_prediction", "cf_pred_labels", "cf_pred_label"]
TARGET_KEYS = ["target_class", "desired_class", "nun_class", "y_target"]
VALID_KEYS = ["validity", "valid", "is_valid"]

METRIC_ALIASES = {
    "adv": ["target_prob", "target_probability", "desired_class_prob", "prob", "prediction_prob", "proba", "adv", "adversarial"],
    "sparsity": ["sparsity", "nchanges", "num_changes", "number_changes", "L0"],
    "subseq": ["subsequences", "subsequences %", "num_subsequences", "nos", "NoS", "contiguity"],
    "plausibility": ["plausibility", "outlier_score", "increase_outlier_score", "ios", "IoS", "AE_IOS", "AE_OS", "LOF_IOS", "IF_IOS"],
}
UTILITY_DIRECTIONS = {"adv": "max", "sparsity": "min", "subseq": "min", "plausibility": "min"}

PTBXL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLINICAL_LEAD_CHANNELS = list(range(12))
AUXILIARY_CHANNELS = [12, 13, 14]
V2_V5_CHANNELS = [7, 8, 9, 10]
LIMB_LEAD_CHANNELS = [0, 1, 2, 3, 4, 5]
STANDARD_LIMB_LEAD_CHANNELS = [0, 1, 2]
AUGMENTED_LIMB_LEAD_CHANNELS = [3, 4, 5]
PRECORDIAL_CHANNELS = [6, 7, 8, 9, 10, 11]
ANALYSIS_DIRECTIONS = {"NORM_TO_MI", "MI_TO_NORM", "BOTH", "NONE"}


@dataclass
class AnalysisConfig:
    delineation_channel_idx: int = 1
    sampling_rate: int = 100
    use_neurokit_delineation: bool = True
    strict_delineation: bool = False
    strict_extraction: bool = False
    fixed_r_index: int | None = None
    neurokit_method: str = "dwt"
    reference_window_mode: str = "union"
    force_transpose: bool = False
    mask_tolerance: float = 1e-6
    analysis_direction: str = "NORM_TO_MI"
    apply_norm_mi_filter: bool = True
    strict_norm_mi_filter: bool = False
    class_names: Any = field(default_factory=lambda: {0: "NORM", 1: "MI"})
    norm_label: str = "NORM"
    mi_label: str = "MI"
    utility_weights: dict[str, float] = field(
        default_factory=lambda: {"adv": 0.1, "sparsity": 0.3, "subseq": 0.4, "plausibility": 0.2}
    )


@dataclass
class AdaptedPairs:
    x_orig_all: np.ndarray
    x_cf_all: np.ndarray
    masks_all: np.ndarray
    metadata_df: pd.DataFrame
    x_nun_all: np.ndarray | None = None
    instance_ids: np.ndarray | None = None
    candidate_ids: np.ndarray | None = None
    skipped_rows: list[tuple[int, str]] = field(default_factory=list)
    filter_status: dict[str, int] = field(default_factory=dict)


def shape_of(value: Any) -> tuple[int, ...] | None:
    try:
        return tuple(np.asarray(value).shape)
    except Exception:
        return None


def summarize_object(value: Any, max_keys: int = 20) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"type": type(value).__name__, "keys": list(value.keys())[:max_keys]}
    if isinstance(value, (list, tuple)):
        first = value[0] if value else None
        return {
            "type": type(value).__name__,
            "len": len(value),
            "first": summarize_object(first, max_keys=8) if first is not None else None,
        }
    return {"type": type(value).__name__, "shape": shape_of(value)}


def find_value(mapping: Any, keys: list[str], default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    lowered = {str(k).lower(): k for k in mapping.keys()}
    for key in keys:
        if key in mapping:
            return mapping[key]
        real_key = lowered.get(str(key).lower())
        if real_key is not None:
            return mapping[real_key]
    return default


def scalarize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.size == 1 and arr.dtype != object:
        return arr.reshape(-1)[0].item()
    if arr.ndim == 1 and arr.size <= 20 and arr.dtype != object:
        return arr.tolist()
    return value


def standardize_signal(value: Any, reference_shape: tuple[int, int] | None = None, name: str = "signal", config: AnalysisConfig | None = None) -> np.ndarray:
    config = config or AnalysisConfig()
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError(f"{name} is scalar; cannot convert to (time, channels)")
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        singleton_axes = [axis for axis, size in enumerate(arr.shape) if size == 1]
        for axis in reversed(singleton_axes):
            arr = np.squeeze(arr, axis=axis)
        if arr.ndim > 2:
            raise ValueError(f"{name} has unsupported shape {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"{name} has unsupported shape {arr.shape}")
    if reference_shape is not None:
        if arr.shape == tuple(reference_shape):
            return arr
        if arr.T.shape == tuple(reference_shape):
            return arr.T
    if config.force_transpose:
        return arr.T
    dim0, dim1 = arr.shape
    if dim0 < dim1 and dim0 <= 32 and dim1 >= dim0 * 2:
        return arr.T
    if dim1 < dim0 and dim1 <= 64:
        return arr
    warnings.warn(f"Ambiguous orientation for {name} shape {arr.shape}. Keeping as-is.")
    return arr


def ensure_time_channel(x: Any, config: AnalysisConfig | None = None) -> np.ndarray:
    return standardize_signal(x, name="signal", config=config)


def ensure_batch_time_channel(x: Any, config: AnalysisConfig | None = None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2:
        return ensure_time_channel(arr, config=config)[None, ...]
    if arr.ndim == 3:
        converted = [ensure_time_channel(arr[i], config=config) for i in range(arr.shape[0])]
        return np.stack(converted, axis=0)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return ensure_batch_time_channel(arr[:, 0], config=config)
    raise ValueError(f"Expected batch convertible to (n, time, channels), got shape {arr.shape}")


def infer_mask(x_orig: np.ndarray, x_cf: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    return np.abs(x_cf - x_orig) > tolerance


def count_subsequences(mask: np.ndarray) -> int:
    mask = np.asarray(mask).astype(bool)
    return int(np.count_nonzero(np.diff(mask.astype(int), prepend=0, axis=0) == 1))


def label_name(value: Any, class_names: Any = None) -> Any:
    value = scalarize(value)
    if value is None:
        return None
    if isinstance(value, list):
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size > 1 and np.issubdtype(arr.dtype, np.number):
            value = int(np.argmax(arr))
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size > 1 and np.issubdtype(value.dtype, np.number):
            value = int(np.argmax(value))
        elif value.size == 1:
            value = scalarize(value)
    if class_names is None:
        return value
    if isinstance(class_names, Mapping):
        return class_names.get(value, class_names.get(str(value), value))
    if isinstance(class_names, (list, tuple)):
        try:
            return class_names[int(value)]
        except Exception:
            return value
    return value


def channel_label(ch: int) -> str:
    if 0 <= int(ch) < len(PTBXL_LEADS):
        return PTBXL_LEADS[int(ch)]
    return f"aux_{int(ch)}"


def normalize_direction(direction: str | None) -> str:
    direction = str(direction or "NONE").upper()
    if direction not in ANALYSIS_DIRECTIONS:
        raise ValueError(f"analysis_direction={direction!r}; expected one of {sorted(ANALYSIS_DIRECTIONS)}")
    return direction


def get_explanation_direction(meta: Mapping[str, Any], config: AnalysisConfig) -> str | None:
    orig = label_name(meta.get("predicted_orig"), config.class_names)
    if orig is None:
        orig = label_name(meta.get("y_true"), config.class_names)
    target = label_name(meta.get("predicted_cf"), config.class_names)
    if target is None:
        target = label_name(meta.get("target_class"), config.class_names)
    validity = meta.get("validity")
    if validity is not None and not bool(validity):
        return None
    if orig is None or target is None:
        return None
    orig = str(orig).upper()
    target = str(target).upper()
    norm = str(config.norm_label).upper()
    mi = str(config.mi_label).upper()
    if orig == norm and target == mi:
        return "NORM_TO_MI"
    if orig == mi and target == norm:
        return "MI_TO_NORM"
    return "OTHER"


def is_norm_to_mi(meta: dict[str, Any], config: AnalysisConfig) -> bool | None:
    direction = get_explanation_direction(meta, config)
    if direction is None:
        return None
    return direction == "NORM_TO_MI"


def metric_from_candidate(candidate: Any, aliases: list[str]) -> float:
    containers = [candidate]
    if isinstance(candidate, Mapping):
        for nested_key in ["objectives", "metrics", "scores", "fitness", "fitness_values"]:
            nested = candidate.get(nested_key)
            if isinstance(nested, Mapping):
                containers.append(nested)
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        value = scalarize(find_value(container, aliases))
        if isinstance(value, (int, float, np.number, bool)) and not pd.isna(value):
            return float(value)
    return np.nan


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    vmin, vmax = np.nanmin(values[valid]), np.nanmax(values[valid])
    out[valid] = 0.5 if np.isclose(vmin, vmax) else (values[valid] - vmin) / (vmax - vmin)
    return out


def utility_scores(metric_df: pd.DataFrame, config: AnalysisConfig) -> tuple[np.ndarray | None, list[tuple[str, str]]]:
    score = np.zeros(len(metric_df), dtype=float)
    used = []
    for family, aliases in METRIC_ALIASES.items():
        col = next((alias for alias in aliases if alias in metric_df.columns and metric_df[alias].notna().any()), None)
        if col is None:
            continue
        scaled = minmax(metric_df[col].to_numpy(dtype=float))
        if UTILITY_DIRECTIONS[family] == "min":
            scaled = 1.0 - scaled
        score += config.utility_weights.get(family, 0.0) * np.nan_to_num(scaled, nan=0.0)
        used.append((family, col))
    return (score if used else None), used


def get_instances(obj: Any) -> list[Any]:
    if isinstance(obj, Mapping):
        direct_candidates = find_value(obj, PARETO_KEYS + CF_KEYS)
        if direct_candidates is not None:
            return [obj]
        list_values = [(k, v) for k, v in obj.items() if isinstance(v, (list, tuple))]
        if list_values:
            _, values = max(list_values, key=lambda item: len(item[1]))
            return list(values)
        return list(obj.values())
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise TypeError(f"Unsupported explanation object type: {type(obj).__name__}")


def get_candidates(instance: Any) -> list[Any]:
    if isinstance(instance, Mapping):
        pareto = find_value(instance, PARETO_KEYS)
        if isinstance(pareto, (list, tuple)):
            return list(pareto)
        cfs = find_value(instance, CF_KEYS)
        if cfs is not None:
            arr = np.asarray(cfs)
            if arr.ndim >= 3:
                return [{"x_cf": arr[i], "candidate_id": i} for i in range(arr.shape[0])]
            return [{"x_cf": arr, "candidate_id": 0}]
    return [{"x_cf": instance, "candidate_id": 0}]


def get_instance_id(instance: Any, pos: int, context: dict[str, Any]) -> Any:
    if isinstance(instance, Mapping):
        value = find_value(instance, ["instance_id", "original_index", "idx", "index", "ii"])
        if value is not None:
            return scalarize(value)
    subset_idx = context.get("subset_idx")
    if subset_idx is not None and pos < len(subset_idx):
        return int(subset_idx[pos])
    return pos


def get_original_signal(instance: Any, pos: int, context: dict[str, Any], config: AnalysisConfig) -> np.ndarray | None:
    value = find_value(instance, ORIG_KEYS) if isinstance(instance, Mapping) else None
    x_test = context.get("X_test")
    if value is None and x_test is not None and pos < len(x_test):
        value = x_test[pos]
    return standardize_signal(value, name=f"x_orig[{pos}]", config=config) if value is not None else None


def get_nun_signal(instance: Any, pos: int, reference_shape: tuple[int, int], context: dict[str, Any], config: AnalysisConfig) -> np.ndarray | None:
    value = find_value(instance, NUN_KEYS) if isinstance(instance, Mapping) else None
    nuns = context.get("nuns")
    if value is None and nuns is not None and pos < len(nuns):
        value = nuns[pos]
    return standardize_signal(value, reference_shape=reference_shape, name=f"x_nun[{pos}]", config=config) if value is not None else None


def get_candidate_cf(candidate: Any, reference_shape: tuple[int, int], config: AnalysisConfig) -> np.ndarray:
    value = find_value(candidate, CF_KEYS) if isinstance(candidate, Mapping) else candidate
    return standardize_signal(value, reference_shape=reference_shape, name="x_cf", config=config)


def compute_candidate_metric_table(
    candidates: list[Any],
    x_orig: np.ndarray,
    x_nun: np.ndarray | None,
    instance: Any,
    pos: int,
    context: dict[str, Any],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    rows, cfs = [], []
    instance_cf_predictions = find_value(instance, PRED_CF_KEYS) if isinstance(instance, Mapping) else None
    if instance_cf_predictions is not None:
        instance_cf_predictions = np.asarray(instance_cf_predictions).reshape(-1)

    for i, candidate in enumerate(candidates):
        x_cf = get_candidate_cf(candidate, reference_shape=x_orig.shape, config=config)
        cfs.append(x_cf)
        loaded_mask = find_value(candidate, MASK_KEYS) if isinstance(candidate, Mapping) else None
        if loaded_mask is not None:
            mask = standardize_signal(loaded_mask, reference_shape=x_orig.shape, name=f"mask[{i}]", config=config).astype(bool)
            mask_source = "loaded"
        else:
            mask = infer_mask(x_orig, x_cf, config.mask_tolerance)
            mask_source = "inferred"
        nchanges = int(mask.sum())
        row = {
            "candidate_id": scalarize(find_value(candidate, ["candidate_id", "id", "index"], i)) if isinstance(candidate, Mapping) else i,
            "validity": scalarize(find_value(candidate, VALID_KEYS)) if isinstance(candidate, Mapping) else None,
            "predicted_cf": scalarize(find_value(candidate, PRED_CF_KEYS)) if isinstance(candidate, Mapping) else None,
            "target_class": scalarize(find_value(candidate, TARGET_KEYS)) if isinstance(candidate, Mapping) else None,
            "nchanges": nchanges,
            "sparsity": nchanges / mask.size if mask.size else np.nan,
            "subsequences": count_subsequences(mask),
            "mask_source": mask_source,
        }
        if row["predicted_cf"] is None and instance_cf_predictions is not None and i < len(instance_cf_predictions):
            row["predicted_cf"] = scalarize(instance_cf_predictions[i])
        for _, aliases in METRIC_ALIASES.items():
            value = metric_from_candidate(candidate, aliases) if isinstance(candidate, Mapping) else np.nan
            if np.isfinite(value):
                row[aliases[0]] = value
        rows.append(row)

    cf_stack = np.stack(cfs)
    target_class = scalarize(find_value(instance, TARGET_KEYS)) if isinstance(instance, Mapping) else None
    model_wrapper = context.get("model_wrapper")
    if target_class is None and x_nun is not None and model_wrapper is not None:
        try:
            target_class = int(np.argmax(model_wrapper.predict(np.expand_dims(x_nun, axis=0)), axis=1)[0])
        except Exception as exc:
            print(f"Could not infer target class from NUN for instance {pos}: {exc}")
    if model_wrapper is not None:
        try:
            probs = model_wrapper.predict(cf_stack)
            pred_classes = np.argmax(probs, axis=1)
            for i, row in enumerate(rows):
                row["predicted_cf"] = int(pred_classes[i])
                if target_class is not None:
                    row["target_class"] = int(target_class)
                    row["target_prob"] = float(probs[i, int(target_class)])
        except Exception as exc:
            print(f"Could not compute model probabilities for instance {pos}: {exc}")
    ae_outlier_calculator = context.get("ae_outlier_calculator")
    if ae_outlier_calculator is not None:
        try:
            ae_os = ae_outlier_calculator.get_outlier_scores(cf_stack)
            ae_orig = ae_outlier_calculator.get_outlier_scores(np.expand_dims(x_orig, axis=0))[0]
            ae_ios = ae_os - ae_orig
            ae_ios[ae_ios < 0] = 0
            for i, row in enumerate(rows):
                row["AE_OS"] = float(ae_os[i])
                row["AE_IOS"] = float(ae_ios[i])
        except Exception as exc:
            print(f"Could not compute AE scores for instance {pos}: {exc}")
    return pd.DataFrame(rows), cfs


def select_candidate(metric_df: pd.DataFrame, config: AnalysisConfig) -> tuple[int | None, float, list[tuple[str, str]]]:
    if metric_df.empty:
        return None, np.nan, []
    analysis_direction = normalize_direction(config.analysis_direction)
    valid_df = metric_df.copy()
    if "validity" in valid_df and valid_df["validity"].notna().any():
        valid_mask = valid_df["validity"].astype(bool)
        if valid_mask.any():
            valid_df = valid_df[valid_mask]
    if config.apply_norm_mi_filter and analysis_direction in {"NORM_TO_MI", "MI_TO_NORM"} and config.class_names is not None:
        target_label = config.mi_label if analysis_direction == "NORM_TO_MI" else config.norm_label
        label_col = "predicted_cf" if "predicted_cf" in valid_df else "target_class" if "target_class" in valid_df else None
        if label_col is not None:
            target_mask = valid_df[label_col].map(
                lambda value: str(label_name(value, config.class_names)).upper() == str(target_label).upper()
            )
            if target_mask.any():
                valid_df = valid_df[target_mask]
    scores, used = utility_scores(valid_df, config)
    if scores is None:
        return int(valid_df.index[0]), np.nan, []
    selected_local = int(np.nanargmax(scores))
    selected_index = int(valid_df.index[selected_local])
    return selected_index, float(scores[selected_local]), used


def adapt_explanation_object(explanation_obj: Any, context: dict[str, Any] | None = None, source_object: str = "explanation_obj", config: AnalysisConfig | None = None) -> AdaptedPairs:
    config = config or AnalysisConfig()
    context = context or {}
    analysis_direction = normalize_direction(config.analysis_direction)
    instances = get_instances(explanation_obj)
    selected_orig, selected_cf, selected_nun, selected_masks, metadata_rows, skipped_rows = [], [], [], [], [], []
    filter_status = {
        "kept_norm_mi": 0,
        "kept_mi_norm": 0,
        "kept_unfiltered": 0,
        "filtered_out": 0,
        "unknown_labels": 0,
        "other_direction": 0,
    }

    filtered_preview = []

    for pos, instance in enumerate(instances):
        try:
            x_orig = get_original_signal(instance, pos, context, config)
            if x_orig is None:
                if config.strict_extraction:
                    raise ValueError("missing x_orig")
                skipped_rows.append((pos, "missing x_orig"))
                continue
            x_nun = get_nun_signal(instance, pos, x_orig.shape, context, config)
            candidates = get_candidates(instance)
            if not candidates:
                if config.strict_extraction:
                    raise ValueError("no candidates")
                skipped_rows.append((pos, "no candidates"))
                continue
            metric_df, cfs = compute_candidate_metric_table(candidates, x_orig, x_nun, instance, pos, context, config)
            selected_idx, utility_score, used_metrics = select_candidate(metric_df, config)
            if selected_idx is None:
                if config.strict_extraction:
                    raise ValueError("no selectable candidate")
                skipped_rows.append((pos, "no selectable candidate"))
                continue

            candidate = candidates[selected_idx]
            x_cf = cfs[selected_idx]
            explicit_mask = find_value(candidate, MASK_KEYS) if isinstance(candidate, Mapping) else None
            if explicit_mask is not None:
                mask = standardize_signal(explicit_mask, reference_shape=x_orig.shape, name=f"selected_mask[{pos}]", config=config).astype(bool)
                mask_source = "loaded"
            else:
                mask = infer_mask(x_orig, x_cf, config.mask_tolerance)
                mask_source = "inferred"

            selected_row = metric_df.loc[selected_idx].to_dict()
            meta = {
                "selected_row": len(selected_orig),
                "instance_position": pos,
                "instance_id": get_instance_id(instance, pos, context),
                "candidate_id": selected_row.get("candidate_id", selected_idx),
                "utility_score": utility_score,
                "utility_metrics_used": used_metrics,
                "mask_source": mask_source,
                "source_object": source_object,
                "x_orig_shape": x_orig.shape,
                "x_cf_shape": x_cf.shape,
            }
            if isinstance(instance, Mapping):
                meta.update({
                    "y_true": scalarize(find_value(instance, Y_TRUE_KEYS)),
                    "predicted_orig": scalarize(find_value(instance, PRED_ORIG_KEYS)),
                    "target_class": scalarize(find_value(instance, TARGET_KEYS)),
                    "nun_idx": scalarize(find_value(instance, ["nun_idx", "nun_index", "nun_id"])),
                })
            y_pred_test, y_test = context.get("y_pred_test"), context.get("y_test")
            if meta.get("predicted_orig") is None and y_pred_test is not None and pos < len(y_pred_test):
                meta["predicted_orig"] = scalarize(y_pred_test[pos])
            if meta.get("y_true") is None and y_test is not None and pos < len(y_test):
                meta["y_true"] = scalarize(y_test[pos])
            for key in ["predicted_cf", "target_class", "validity", "target_prob", "sparsity", "nchanges", "subsequences", "AE_OS", "AE_IOS"]:
                if key in selected_row and selected_row[key] is not None and not pd.isna(selected_row[key]):
                    meta[key] = selected_row[key]

            direction = get_explanation_direction(meta, config)
            meta["direction"] = direction
            if config.apply_norm_mi_filter and config.class_names is not None:
                keep = (
                    analysis_direction == "NONE"
                    or (analysis_direction == "BOTH" and direction in {"NORM_TO_MI", "MI_TO_NORM"})
                    or direction == analysis_direction
                )
                if keep:
                    if direction == "NORM_TO_MI":
                        filter_status["kept_norm_mi"] += 1
                    elif direction == "MI_TO_NORM":
                        filter_status["kept_mi_norm"] += 1
                    else:
                        filter_status["kept_unfiltered"] += 1
                else:
                    if direction is None:
                        filter_status["unknown_labels"] += 1
                    elif direction == "OTHER":
                        filter_status["other_direction"] += 1
                    filter_status["filtered_out"] += 1
                    if len(filtered_preview) < 10:
                        filtered_preview.append({
                            "pos": pos,
                            "instance_id": meta.get("instance_id"),
                            "y_true": meta.get("y_true"),
                            "predicted_orig": meta.get("predicted_orig"),
                            "predicted_cf": meta.get("predicted_cf"),
                            "target_class": meta.get("target_class"),
                            "validity": meta.get("validity"),
                            "direction": direction,
                            "analysis_direction": analysis_direction,
                        })
                    if config.strict_norm_mi_filter or analysis_direction != "NONE":
                        continue
            elif config.apply_norm_mi_filter and config.class_names is None:
                filter_status["unknown_labels"] += 1
                if config.strict_norm_mi_filter:
                    continue
                filter_status["kept_unfiltered"] += 1
            else:
                filter_status["kept_unfiltered"] += 1

            selected_orig.append(x_orig)
            selected_cf.append(x_cf)
            if x_nun is not None:
                selected_nun.append(x_nun)
            selected_masks.append(mask)
            metadata_rows.append(meta)
        except Exception as exc:
            if config.strict_extraction:
                raise RuntimeError(f"Failed to adapt explanation instance at position {pos}") from exc
            skipped_rows.append((pos, repr(exc)))

    if not selected_orig:
        raise RuntimeError(
            "No selected pairs were extracted. "
            f"filter_status={filter_status}. "
            f"first_filtered_rows={filtered_preview}. "
            f"first_skipped_rows={skipped_rows[:10]}"
        )

    metadata_df = pd.DataFrame(metadata_rows)
    return AdaptedPairs(
        x_orig_all=np.stack(selected_orig),
        x_cf_all=np.stack(selected_cf),
        masks_all=np.stack(selected_masks).astype(bool),
        x_nun_all=np.stack(selected_nun) if len(selected_nun) == len(selected_orig) else None,
        metadata_df=metadata_df,
        instance_ids=metadata_df["instance_id"].to_numpy() if "instance_id" in metadata_df else np.arange(len(metadata_df)),
        candidate_ids=metadata_df["candidate_id"].to_numpy() if "candidate_id" in metadata_df else np.arange(len(metadata_df)),
        skipped_rows=skipped_rows,
        filter_status=filter_status,
    )


def clip_window(window: tuple[float, float] | None, n_time: int) -> tuple[int, int] | None:
    if window is None:
        return None
    a, b = window
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    a = max(0, int(round(a)))
    b = min(n_time, int(round(b)))
    return (a, b) if a < b else None


def resolve_r_index(n_time: int, r_index: int | None = None, config: AnalysisConfig | None = None) -> int:
    config = config or AnalysisConfig()
    if r_index is not None:
        return int(r_index)
    if config.fixed_r_index is not None:
        return int(config.fixed_r_index)
    return n_time // 2


def get_centered_windows(
    n_time: int,
    r_index: int | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, tuple[int, int] | None]:
    config = config or AnalysisConfig()
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    raw = {
        "baseline": (0, r_index - 55),
        "p": (r_index - 55, r_index - 30),
        "q": (r_index - 30, r_index - 8),
        "r": (r_index - 8, r_index + 10),
        "st": (r_index + 10, r_index + 35),
        "t": (r_index + 35, n_time - 5),
    }
    return {name: clip_window(window, n_time) for name, window in raw.items()}


def estimate_qrs_reference_index(
    x: Any,
    search_center: int | None = None,
    search_radius: int = 35,
    clinical_channels_only: bool = True,
) -> int:
    x = ensure_time_channel(x)
    n_time, n_channels = x.shape
    if search_center is None:
        search_center = n_time // 2
    channels = slice(0, min(12, n_channels)) if clinical_channels_only else slice(None)
    baseline = np.median(np.r_[x[:15, channels], x[-15:, channels]], axis=0)
    lo = max(0, search_center - search_radius)
    hi = min(n_time, search_center + search_radius + 1)
    centered = x[lo:hi, channels] - baseline
    energy = np.sum(centered ** 2, axis=1)
    return int(lo + np.argmax(energy))


def nearest_valid_index(values: Any, target: int) -> int | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return int(arr[np.argmin(np.abs(arr - target))])


def neurokit_reference_windows_1d(signal: np.ndarray, config: AnalysisConfig, r_index: int | None = None) -> tuple[dict[str, tuple[int, int] | None], dict[str, Any]]:
    if not (config.use_neurokit_delineation and NEUROKIT_AVAILABLE):
        raise RuntimeError("NeuroKit delineation is disabled or unavailable")
    signal = np.asarray(signal, dtype=float).reshape(-1)
    n_time = len(signal)
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    cleaned = nk.ecg_clean(signal, sampling_rate=config.sampling_rate, method="neurokit")
    _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=config.sampling_rate)
    rpeaks = np.asarray(peaks_info.get("ECG_R_Peaks", []), dtype=int)
    if rpeaks.size == 0:
        rpeaks = np.asarray([r_index], dtype=int)
    rpeak = int(rpeaks[np.argmin(np.abs(rpeaks - r_index))])
    _, waves = nk.ecg_delineate(cleaned, rpeaks=np.asarray([rpeak], dtype=int), sampling_rate=config.sampling_rate, method=config.neurokit_method, show=False)
    landmark_keys = {
        "p_on": "ECG_P_Onsets", "p_peak": "ECG_P_Peaks", "p_off": "ECG_P_Offsets",
        "q_peak": "ECG_Q_Peaks", "r_on": "ECG_R_Onsets", "r_off": "ECG_R_Offsets",
        "s_peak": "ECG_S_Peaks", "t_on": "ECG_T_Onsets", "t_peak": "ECG_T_Peaks", "t_off": "ECG_T_Offsets",
    }
    lm = {name: nearest_valid_index(waves.get(key), rpeak) for name, key in landmark_keys.items()}
    lm["r_peak"] = rpeak
    sr = config.sampling_rate
    windows = {
        "baseline": clip_window(((lm["p_on"] - int(0.20 * sr)) if lm.get("p_on") is not None else rpeak - int(0.65 * sr), (lm["p_on"] - int(0.04 * sr)) if lm.get("p_on") is not None else rpeak - int(0.45 * sr)), n_time),
        "p": clip_window((lm.get("p_on"), lm.get("p_off")), n_time),
        "q": clip_window((lm.get("r_on") if lm.get("r_on") is not None else (lm.get("q_peak") - int(0.04 * sr) if lm.get("q_peak") is not None else rpeak - int(0.08 * sr)), lm.get("r_peak") if lm.get("r_peak") is not None else rpeak), n_time),
        "r": clip_window((lm.get("r_on") if lm.get("r_on") is not None else rpeak - int(0.04 * sr), lm.get("r_off") if lm.get("r_off") is not None else rpeak + int(0.04 * sr)), n_time),
        "st": clip_window((lm.get("r_off") if lm.get("r_off") is not None else rpeak + int(0.06 * sr), lm.get("t_on") if lm.get("t_on") is not None else rpeak + int(0.35 * sr)), n_time),
        "t": clip_window((lm.get("t_on") if lm.get("t_on") is not None else (lm.get("t_peak") - int(0.12 * sr) if lm.get("t_peak") is not None else rpeak + int(0.35 * sr)), lm.get("t_off") if lm.get("t_off") is not None else (lm.get("t_peak") + int(0.16 * sr) if lm.get("t_peak") is not None else rpeak + int(0.70 * sr))), n_time),
    }
    return windows, {"source": "neurokit", "r_index": rpeak, **lm}


def combine_two_windows(w1: tuple[int, int] | None, w2: tuple[int, int] | None, n_time: int, mode: str) -> tuple[int, int] | None:
    w1 = clip_window(w1, n_time)
    w2 = clip_window(w2, n_time)
    if mode == "orig":
        return w1
    if mode == "nun":
        return w2
    if w1 is None:
        return w2
    if w2 is None:
        return w1
    if mode == "intersection":
        return clip_window((max(w1[0], w2[0]), min(w1[1], w2[1])), n_time)
    return clip_window((min(w1[0], w2[0]), max(w1[1], w2[1])), n_time)


def reference_windows_from_orig_nun(x_orig: Any, x_nun: Any = None, r_index: int | None = None, config: AnalysisConfig | None = None) -> tuple[dict[str, tuple[int, int] | None], dict[str, Any]]:
    config = config or AnalysisConfig()
    x_orig = ensure_time_channel(x_orig, config=config)
    n_time, n_channels = x_orig.shape
    if config.delineation_channel_idx >= n_channels:
        raise ValueError(f"delineation_channel_idx={config.delineation_channel_idx}, but only {n_channels} channels are available")
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    if not config.use_neurokit_delineation:
        windows = get_centered_windows(n_time, r_index=r_index, config=config)
        return windows, {
            "delineation_channel": config.delineation_channel_idx,
            "reference_window_mode": "centered",
            "orig_source": "centered_windows",
            "nun_source": "not_used",
            "combined_source": "centered_windows",
            "orig_r_index": r_index,
            "nun_r_index": None,
        }

    if config.strict_delineation:
        orig_windows, orig_meta = neurokit_reference_windows_1d(
            x_orig[:, config.delineation_channel_idx],
            config=config,
            r_index=r_index,
        )
        if x_nun is not None:
            x_nun = ensure_time_channel(x_nun, config=config)
            nun_windows, nun_meta = neurokit_reference_windows_1d(
                x_nun[:, config.delineation_channel_idx],
                config=config,
                r_index=orig_meta.get("r_index") or r_index,
            )
            nun_source = nun_meta.get("source")
            nun_r_index = nun_meta.get("r_index")
        else:
            nun_windows = orig_windows
            nun_source = "missing"
            nun_r_index = None
        combined = {
            name: combine_two_windows(orig_windows.get(name), nun_windows.get(name), n_time, mode=config.reference_window_mode)
            for name in sorted(set(orig_windows) | set(nun_windows))
        }
        return combined, {
            "delineation_channel": config.delineation_channel_idx,
            "reference_window_mode": config.reference_window_mode,
            "orig_source": orig_meta.get("source"),
            "nun_source": nun_source,
            "combined_source": f"{config.reference_window_mode}_orig_nun_reference",
            "orig_r_index": orig_meta.get("r_index"),
            "nun_r_index": nun_r_index,
        }

    fallback = get_centered_windows(n_time, r_index=r_index, config=config)
    meta = {
        "delineation_channel": config.delineation_channel_idx,
        "reference_window_mode": config.reference_window_mode,
        "orig_source": "fallback_centered",
        "nun_source": "missing",
        "combined_source": "fallback_centered",
        "orig_r_index": r_index,
        "nun_r_index": None,
    }
    try:
        orig_windows, orig_meta = neurokit_reference_windows_1d(x_orig[:, config.delineation_channel_idx], config=config, r_index=r_index)
        meta.update({"orig_source": orig_meta.get("source"), "orig_r_index": orig_meta.get("r_index")})
    except Exception as exc:
        raise RuntimeError(f"Could not compute reference windows using NeuroKit for original signal: {exc}")
    if x_nun is not None:
        x_nun = ensure_time_channel(x_nun, config=config)
        try:
            nun_windows, nun_meta = neurokit_reference_windows_1d(x_nun[:, config.delineation_channel_idx], config=config, r_index=meta.get("orig_r_index") or r_index)
            meta.update({"nun_source": nun_meta.get("source"), "nun_r_index": nun_meta.get("r_index")})
        except Exception as exc:
            raise RuntimeError(f"Could not compute reference windows using NeuroKit for NUN signal: {exc}")
    else:
        nun_windows = orig_windows
    combined = {
        name: combine_two_windows(orig_windows.get(name), nun_windows.get(name), n_time, mode=config.reference_window_mode)
        for name in sorted(set(orig_windows) | set(nun_windows))
    }
    if meta["orig_source"] == "neurokit" or meta["nun_source"] == "neurokit":
        meta["combined_source"] = f"{config.reference_window_mode}_orig_nun_reference"
    return combined, meta


def safe_segment(signal: np.ndarray, window: tuple[int, int] | None) -> np.ndarray:
    if window is None:
        return np.array([], dtype=float)
    a, b = window
    return np.asarray(signal[a:b], dtype=float)


def safe_median(x: np.ndarray) -> float:
    return np.nan if np.asarray(x).size == 0 else np.nanmedian(x)


def safe_mean(x: np.ndarray) -> float:
    return np.nan if np.asarray(x).size == 0 else np.nanmean(x)


def safe_min(x: np.ndarray) -> float:
    return np.nan if np.asarray(x).size == 0 else np.nanmin(x)


def safe_max(x: np.ndarray) -> float:
    return np.nan if np.asarray(x).size == 0 else np.nanmax(x)


def local_lag_between_segments(orig: np.ndarray, cf: np.ndarray, max_lag: int = 8) -> tuple[int, float]:
    orig = np.asarray(orig, dtype=float).reshape(-1)
    cf = np.asarray(cf, dtype=float).reshape(-1)
    if orig.size < 3 or cf.size < 3:
        return 0, np.nan
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = orig[-lag:]
            b = cf[: cf.size + lag]
        elif lag > 0:
            a = orig[: orig.size - lag]
            b = cf[lag:]
        else:
            a = orig
            b = cf
        n = min(a.size, b.size)
        if n < 3:
            continue
        a = a[:n] - np.nanmean(a[:n])
        b = b[:n] - np.nanmean(b[:n])
        denom = float(np.sqrt(np.nansum(a * a) * np.nansum(b * b)))
        if denom <= 0 or not np.isfinite(denom):
            continue
        score = float(np.nansum(a * b) / denom)
        if score > best_score:
            best_lag, best_score = lag, score
    return (0, np.nan) if not np.isfinite(best_score) else (int(best_lag), float(best_score))


def compute_v2_v5_progression_features(
    x: np.ndarray,
    windows: dict[str, tuple[int, int] | None],
    channels: list[int] = V2_V5_CHANNELS,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        return {
            "v2_v5_r_amp_mean": np.nan,
            "v2_v5_r_amp_slope": np.nan,
            "v2_v5_r_amp_min": np.nan,
            "v2_v5_r_amp_max": np.nan,
            "v2_v5_r_amp_range": np.nan,
        }
    r_amps = []
    for ch in channels:
        if ch >= x.shape[1]:
            continue
        baseline = safe_median(safe_segment(x[:, ch], windows.get("baseline")))
        r_centered = safe_segment(x[:, ch], windows.get("r")) - baseline
        r_amps.append(safe_max(r_centered))
    vals = np.asarray(r_amps, dtype=float)
    valid = np.isfinite(vals)
    if not valid.any():
        slope = np.nan
    elif valid.sum() >= 2:
        slope = float(np.polyfit(np.arange(vals.size)[valid], vals[valid], 1)[0])
    else:
        slope = np.nan
    return {
        "v2_v5_r_amp_mean": float(np.nanmean(vals)) if valid.any() else np.nan,
        "v2_v5_r_amp_slope": slope,
        "v2_v5_r_amp_min": float(np.nanmin(vals)) if valid.any() else np.nan,
        "v2_v5_r_amp_max": float(np.nanmax(vals)) if valid.any() else np.nan,
        "v2_v5_r_amp_range": float(np.nanmax(vals) - np.nanmin(vals)) if valid.any() else np.nan,
    }


def morphology_features_1d(signal: np.ndarray, windows: dict[str, tuple[int, int] | None] | None = None, r_index: int | None = None, config: AnalysisConfig | None = None) -> dict[str, float]:
    config = config or AnalysisConfig()
    signal = np.asarray(signal, dtype=float)
    r_index = resolve_r_index(len(signal), r_index=r_index, config=config)
    if windows is None:
        windows = get_centered_windows(len(signal), r_index=r_index, config=config)
    baseline = safe_median(safe_segment(signal, windows.get("baseline")))
    q_centered = safe_segment(signal, windows.get("q")) - baseline
    r_centered = safe_segment(signal, windows.get("r")) - baseline
    st_centered = safe_segment(signal, windows.get("st")) - baseline
    t_centered = safe_segment(signal, windows.get("t")) - baseline
    post_qrs_parts = [part for part in [st_centered, t_centered] if np.asarray(part).size > 0]
    post_qrs = np.concatenate(post_qrs_parts) if post_qrs_parts else np.array([], dtype=float)
    if t_centered.size >= 2:
        late_t = t_centered[len(t_centered) // 2 :]
    else:
        late_t = np.array([], dtype=float)
    q_amp = safe_min(q_centered)
    st_level = safe_mean(st_centered)
    return {
        "baseline": baseline,
        "q_amp": q_amp,
        "q_extreme_abs": safe_max(np.abs(q_centered)),
        "st_level": st_level,
        "st_abs": safe_mean(np.abs(st_centered)),
        "st_q_contrast": st_level - q_amp,
        "t_peak_abs": safe_max(np.abs(t_centered)),
        "t_area_abs": safe_mean(np.abs(t_centered)),
        "qrs_peak_to_peak": safe_max(r_centered) - safe_min(r_centered) if r_centered.size else np.nan,
        "qrs_energy": safe_mean(np.abs(r_centered)),
        "r_max": safe_max(r_centered),
        "r_min": safe_min(r_centered),
        "post_qrs_energy": safe_mean(np.abs(post_qrs)),
        "late_t_energy": safe_mean(np.abs(late_t)),
        "late_t_min": safe_min(late_t),
        "late_t_max": safe_max(late_t),
        "t_prominence": safe_max(np.abs(t_centered)),
        "t_signed_area": safe_mean(t_centered),
    }


def analyze_cf_pair_global(x_orig: Any, x_cf: Any, x_nun: Any = None, instance_id: Any = None, r_index: int | None = None, config: AnalysisConfig | None = None) -> list[dict[str, Any]]:
    config = config or AnalysisConfig()
    x_orig = ensure_time_channel(x_orig, config=config)
    x_cf = ensure_time_channel(x_cf, config=config)
    x_nun = ensure_time_channel(x_nun, config=config) if x_nun is not None else None
    if x_orig.shape != x_cf.shape:
        raise ValueError(f"x_orig and x_cf shapes differ: {x_orig.shape} vs {x_cf.shape}")
    n_time, n_channels = x_orig.shape
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    windows, delineation_meta = reference_windows_from_orig_nun(x_orig, x_nun=x_nun, r_index=r_index, config=config)
    prog_orig = compute_v2_v5_progression_features(x_orig, windows)
    prog_cf = compute_v2_v5_progression_features(x_cf, windows)
    rows = []
    for ch in range(n_channels):
        f_orig = morphology_features_1d(x_orig[:, ch], windows=windows, r_index=r_index, config=config)
        f_cf = morphology_features_1d(x_cf[:, ch], windows=windows, r_index=r_index, config=config)
        qrs_best_lag, qrs_lag_score = local_lag_between_segments(
            safe_segment(x_orig[:, ch], windows.get("r")),
            safe_segment(x_cf[:, ch], windows.get("r")),
        )
        row = {
            "instance_id": instance_id,
            "channel": ch,
            "channel_label": channel_label(ch),
            "r_index": delineation_meta.get("orig_r_index", r_index),
            "delineation_channel": delineation_meta.get("delineation_channel"),
            "delineation_source": delineation_meta.get("combined_source"),
            "orig_delineation_source": delineation_meta.get("orig_source"),
            "nun_delineation_source": delineation_meta.get("nun_source"),
            "reference_window_mode": delineation_meta.get("reference_window_mode"),
            "qrs_best_lag": qrs_best_lag,
            "qrs_best_lag_abs": abs(qrs_best_lag),
            "qrs_lag_score": qrs_lag_score,
        }
        for region_name, window in windows.items():
            if window is not None:
                row[f"{region_name}_window_start"] = window[0]
                row[f"{region_name}_window_end"] = window[1]
        for key in f_orig:
            row[f"{key}_orig"] = f_orig[key]
            row[f"{key}_cf"] = f_cf[key]
            row[f"delta_{key}"] = f_cf[key] - f_orig[key]
        for key, value in prog_orig.items():
            row[f"{key}_orig"] = value
            row[f"{key}_cf"] = prog_cf[key]
            row[f"delta_{key}"] = prog_cf[key] - value
        rows.append(row)
    return rows


def analyze_all_pairs_global(x_orig_all: Any, x_cf_all: Any, x_nun_all: Any = None, instance_ids: Any = None, r_index: int | None = None, config: AnalysisConfig | None = None) -> pd.DataFrame:
    config = config or AnalysisConfig()
    x_orig_all = ensure_batch_time_channel(x_orig_all, config=config)
    x_cf_all = ensure_batch_time_channel(x_cf_all, config=config)
    x_nun_all = ensure_batch_time_channel(x_nun_all, config=config) if x_nun_all is not None else None
    if x_orig_all.shape != x_cf_all.shape:
        raise ValueError(f"Batch shapes differ: {x_orig_all.shape} vs {x_cf_all.shape}")
    if x_nun_all is not None and x_nun_all.shape != x_orig_all.shape:
        raise ValueError(f"NUN shape differs: {x_nun_all.shape} vs {x_orig_all.shape}")
    instance_ids = list(range(x_orig_all.shape[0])) if instance_ids is None else instance_ids
    all_rows = []
    for i in range(x_orig_all.shape[0]):
        all_rows.extend(analyze_cf_pair_global(x_orig_all[i], x_cf_all[i], x_nun=x_nun_all[i] if x_nun_all is not None else None, instance_id=instance_ids[i], r_index=r_index, config=config))
    return pd.DataFrame(all_rows)


def directional_metric_specs(direction: str | None = "NORM_TO_MI") -> dict[str, str]:
    direction = str(direction or "NORM_TO_MI").upper()
    if direction == "MI_TO_NORM":
        return {
            "delta_q_amp": "descriptive",
            "delta_q_extreme_abs": "descriptive",
            "delta_st_level": "descriptive",
            "delta_st_abs": "descriptive",
            "delta_st_q_contrast": "less",
            "delta_t_peak_abs": "greater",
            "delta_t_area_abs": "greater",
            "delta_qrs_peak_to_peak": "descriptive",
            "delta_qrs_energy": "descriptive",
            "delta_r_max": "descriptive",
            "delta_r_min": "descriptive",
            "delta_post_qrs_energy": "descriptive",
            "delta_late_t_energy": "greater",
            "delta_t_prominence": "greater",
            "delta_t_signed_area": "descriptive",
            "qrs_best_lag_abs": "descriptive",
        }
    return {
        "delta_q_amp": "less",
        "delta_q_extreme_abs": "greater",
        "delta_st_level": "greater",
        "delta_st_abs": "greater",
        "delta_st_q_contrast": "greater",
        "delta_t_peak_abs": "less",
        "delta_t_area_abs": "less",
        "delta_qrs_peak_to_peak": "descriptive",
        "delta_qrs_energy": "descriptive",
        "delta_r_max": "descriptive",
        "delta_r_min": "descriptive",
        "delta_post_qrs_energy": "descriptive",
        "delta_late_t_energy": "less",
        "delta_t_prominence": "less",
        "delta_t_signed_area": "descriptive",
        "qrs_best_lag_abs": "descriptive",
    }


def summarize_directional_morphology(df: pd.DataFrame, direction: str = "NORM_TO_MI") -> pd.DataFrame:
    if "direction" in df.columns and direction not in {"ALL", "NONE"}:
        df = df[df["direction"] == direction]
    metric_specs = directional_metric_specs(direction)
    rows = []
    for metric, expected_direction in metric_specs.items():
        if metric not in df.columns:
            continue
        vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(vals) == 0:
            continue
        pct_expected = np.nan
        if expected_direction == "less":
            pct_expected = np.mean(vals < 0) * 100
        elif expected_direction == "greater":
            pct_expected = np.mean(vals > 0) * 100
        try:
            alternative = "two-sided" if expected_direction == "descriptive" else expected_direction
            stat, p_value = wilcoxon(vals, alternative=alternative) if np.any(vals != 0) else (np.nan, np.nan)
        except Exception:
            stat, p_value = np.nan, np.nan
        rows.append({
            "direction": direction,
            "metric": metric,
            "expected_direction": expected_direction,
            "n_instance_channel_pairs": len(vals),
            "median_delta": np.median(vals),
            "q1": np.percentile(vals, 25),
            "q3": np.percentile(vals, 75),
            "mean_delta": np.mean(vals),
            "std_delta": np.std(vals),
            "pct_in_expected_direction": pct_expected,
            "pct_positive": np.mean(vals > 0) * 100,
            "pct_negative": np.mean(vals < 0) * 100,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p_value,
        })
    return pd.DataFrame(rows)


def summarize_global_morphology(df: pd.DataFrame) -> pd.DataFrame:
    if "direction" in df.columns and df["direction"].notna().any():
        summaries = [
            summarize_directional_morphology(df, direction)
            for direction in ["NORM_TO_MI", "MI_TO_NORM"]
            if (df["direction"] == direction).any()
        ]
        if summaries:
            return pd.concat(summaries, ignore_index=True)
    return summarize_directional_morphology(df, "NORM_TO_MI")


def summarize_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["direction", "channel"] if "direction" in df.columns else ["channel"]
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        ch = int(row["channel"])
        row.update({"channel_label": channel_label(ch), "n": len(g)})
        for metric in [
            "delta_q_amp", "delta_q_extreme_abs", "delta_st_level", "delta_st_abs",
            "delta_st_q_contrast", "delta_t_peak_abs", "delta_t_area_abs",
            "delta_qrs_energy", "delta_qrs_peak_to_peak", "delta_late_t_energy",
            "delta_post_qrs_energy", "qrs_best_lag_abs",
        ]:
            if metric not in g.columns:
                continue
            vals = g[metric].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(vals) == 0:
                continue
            row[f"{metric}_median"] = np.median(vals)
            row[f"{metric}_q1"] = np.percentile(vals, 25)
            row[f"{metric}_q3"] = np.percentile(vals, 75)
            row[f"{metric}_pct_positive"] = np.mean(vals > 0) * 100
            row[f"{metric}_pct_negative"] = np.mean(vals < 0) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def infer_mask_from_difference(x_orig: Any, x_cf: Any, tolerance: float = 1e-6, config: AnalysisConfig | None = None) -> np.ndarray:
    x_orig = ensure_time_channel(x_orig, config=config)
    x_cf = ensure_time_channel(x_cf, config=config)
    if x_orig.shape != x_cf.shape:
        raise ValueError(f"x_orig and x_cf shapes differ: {x_orig.shape} vs {x_cf.shape}")
    return np.abs(x_cf - x_orig) > tolerance


def mask_overlap_one(mask: Any, windows: dict[str, tuple[int, int] | None] | None = None, r_index: int | None = None, config: AnalysisConfig | None = None) -> dict[str, Any]:
    mask = ensure_time_channel(mask, config=config).astype(bool)
    n_time, n_channels = mask.shape
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    if windows is None:
        windows = get_centered_windows(n_time, r_index=r_index, config=config)
    total_changed = int(mask.sum())
    out = {"total_changed_points": total_changed}
    covered = np.zeros_like(mask, dtype=bool)
    for name, window in windows.items():
        if window is None:
            out[f"{name}_changed_points"] = 0
            out[f"{name}_changed_pct"] = 0.0
            continue
        a, b = window
        region_mask = np.zeros_like(mask, dtype=bool)
        region_mask[a:b, :] = True
        count = int(np.logical_and(mask, region_mask).sum())
        out[f"{name}_changed_points"] = count
        out[f"{name}_changed_pct"] = 100 * count / total_changed if total_changed > 0 else 0.0
        covered |= region_mask
    other_count = int(np.logical_and(mask, ~covered).sum())
    out["other_changed_points"] = other_count
    out["other_changed_pct"] = 100 * other_count / total_changed if total_changed > 0 else 0.0
    channel_groups = {
        "clinical": [ch for ch in CLINICAL_LEAD_CHANNELS if ch < n_channels],
        "v2_v5": [ch for ch in V2_V5_CHANNELS if ch < n_channels],
        "auxiliary": [ch for ch in AUXILIARY_CHANNELS if ch < n_channels],
    }
    for group_name, channels in channel_groups.items():
        if channels:
            count = int(mask[:, channels].sum())
        else:
            count = 0
        out[f"{group_name}_changed_points"] = count
        out[f"{group_name}_changed_pct_total"] = 100 * count / total_changed if total_changed > 0 else 0.0
    r_window = windows.get("r")
    if r_window is not None and channel_groups["v2_v5"]:
        a, b = r_window
        joint_count = int(mask[a:b, channel_groups["v2_v5"]].sum())
    else:
        joint_count = 0
    out["v2_v5_r_changed_points"] = joint_count
    out["v2_v5_r_changed_pct_total"] = 100 * joint_count / total_changed if total_changed > 0 else 0.0
    return out


def mask_overlap_all(
    x_orig_all: Any,
    x_cf_all: Any,
    x_nun_all: Any = None,
    masks_all: Any = None,
    tolerance: float = 1e-6,
    r_index: int | None = None,
    instance_ids: Any = None,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    config = config or AnalysisConfig()
    x_orig_all = ensure_batch_time_channel(x_orig_all, config=config)
    x_cf_all = ensure_batch_time_channel(x_cf_all, config=config)
    x_nun_all = ensure_batch_time_channel(x_nun_all, config=config) if x_nun_all is not None else None
    masks_all = ensure_batch_time_channel(masks_all, config=config) if masks_all is not None else None
    instance_ids = list(range(x_orig_all.shape[0])) if instance_ids is None else list(instance_ids)
    rows = []
    for i in range(x_orig_all.shape[0]):
        mask = masks_all[i] if masks_all is not None else infer_mask_from_difference(x_orig_all[i], x_cf_all[i], tolerance=tolerance, config=config)
        windows, delineation_meta = reference_windows_from_orig_nun(x_orig_all[i], x_nun=x_nun_all[i] if x_nun_all is not None else None, r_index=r_index, config=config)
        row = mask_overlap_one(mask, windows=windows, r_index=delineation_meta.get("orig_r_index", r_index), config=config)
        row["instance_id"] = instance_ids[i]
        row["delineation_channel"] = delineation_meta.get("delineation_channel")
        row["delineation_source"] = delineation_meta.get("combined_source")
        row["orig_delineation_source"] = delineation_meta.get("orig_source")
        row["nun_delineation_source"] = delineation_meta.get("nun_source")
        row["reference_window_mode"] = delineation_meta.get("reference_window_mode")
        rows.append(row)
    return pd.DataFrame(rows)


def mask_region_heatmap_table(
    masks_all: Any,
    x_orig_all: Any | None = None,
    x_nun_all: Any | None = None,
    instance_ids: Any = None,
    r_index: int | None = None,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    config = config or AnalysisConfig()
    masks_all = ensure_batch_time_channel(masks_all, config=config).astype(bool)
    x_orig_all = ensure_batch_time_channel(x_orig_all, config=config) if x_orig_all is not None else None
    x_nun_all = ensure_batch_time_channel(x_nun_all, config=config) if x_nun_all is not None else None
    instance_ids = list(range(masks_all.shape[0])) if instance_ids is None else list(instance_ids)
    rows = []
    for i, mask in enumerate(masks_all):
        n_time, n_channels = mask.shape
        if x_orig_all is not None:
            windows, _ = reference_windows_from_orig_nun(
                x_orig_all[i],
                x_nun=x_nun_all[i] if x_nun_all is not None else None,
                r_index=r_index,
                config=config,
            )
        else:
            windows = get_centered_windows(n_time, r_index=r_index, config=config)
        total_changed = int(mask.sum())
        channel_groups = {
            "Limb leads": [ch for ch in STANDARD_LIMB_LEAD_CHANNELS if ch < n_channels],
            "Augmented limb leads": [ch for ch in AUGMENTED_LIMB_LEAD_CHANNELS if ch < n_channels],
            "Precordial leads": [ch for ch in PRECORDIAL_CHANNELS if ch < n_channels],
            "Frank leads": [ch for ch in AUXILIARY_CHANNELS if ch < n_channels],
        }
        for region, window in windows.items():
            if window is None:
                continue
            a, b = window
            for ch in range(n_channels):
                count = int(mask[a:b, ch].sum())
                rows.append({
                    "instance_id": instance_ids[i],
                    "granularity": "channel",
                    "channel": ch,
                    "label": channel_label(ch),
                    "region": region,
                    "changed_points": count,
                    "total_changed_points": total_changed,
                    "changed_pct_total": 100 * count / total_changed if total_changed > 0 else 0.0,
                })
            for group_name, channels in channel_groups.items():
                count = int(mask[a:b, channels].sum()) if channels else 0
                rows.append({
                    "instance_id": instance_ids[i],
                    "granularity": "channel_group",
                    "channel": np.nan,
                    "label": group_name,
                    "region": region,
                    "changed_points": count,
                    "total_changed_points": total_changed,
                    "changed_pct_total": 100 * count / total_changed if total_changed > 0 else 0.0,
                })
    return pd.DataFrame(rows)


def summarize_mask_overlap(mask_overlap_df: pd.DataFrame) -> pd.DataFrame:
    pct_cols = [c for c in mask_overlap_df.columns if c.endswith("_changed_pct") or c.endswith("_changed_pct_total")]
    rows = []
    for col in pct_cols:
        vals = mask_overlap_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(vals) == 0:
            continue
        rows.append({
            "region": col.replace("_changed_pct_total", "").replace("_changed_pct", ""),
            "median_changed_pct": np.median(vals),
            "q1": np.percentile(vals, 25),
            "q3": np.percentile(vals, 75),
            "mean_changed_pct": np.mean(vals),
            "std_changed_pct": np.std(vals),
        })
    return pd.DataFrame(rows).sort_values("median_changed_pct", ascending=False)


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "< 1e-4" if p < 1e-4 else f"{p:.4f}"


def _median_col(df: pd.DataFrame, col: str, channels: list[int] | None = None) -> float:
    if col not in df.columns:
        return np.nan
    data = df
    if channels is not None and "channel" in data.columns:
        data = data[data["channel"].isin(channels)]
    vals = data[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else np.nan


def _pct_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if len(vals) else np.nan


def compute_narrative_indicators(
    morphology_df: pd.DataFrame,
    mask_overlap_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    direction_values = []
    if "direction" in morphology_df.columns:
        direction_values.extend([d for d in morphology_df["direction"].dropna().unique() if d in {"NORM_TO_MI", "MI_TO_NORM"}])
    if "direction" in mask_overlap_df.columns:
        direction_values.extend([d for d in mask_overlap_df["direction"].dropna().unique() if d in {"NORM_TO_MI", "MI_TO_NORM"}])
    directions = sorted(set(direction_values))
    groups = [("ALL", morphology_df, mask_overlap_df)]
    groups.extend(
        (
            direction,
            morphology_df[morphology_df["direction"] == direction] if "direction" in morphology_df.columns else morphology_df.iloc[0:0],
            mask_overlap_df[mask_overlap_df["direction"] == direction] if "direction" in mask_overlap_df.columns else mask_overlap_df.iloc[0:0],
        )
        for direction in directions
    )
    rows = []
    for direction, morph, mask in groups:
        if morph.empty and mask.empty:
            continue
        instance_ids = set()
        if "instance_id" in morph.columns:
            instance_ids.update(morph["instance_id"].dropna().tolist())
        if "instance_id" in mask.columns:
            instance_ids.update(mask["instance_id"].dropna().tolist())
        row = {
            "direction": direction,
            "n_instances": len(instance_ids),
            "n_instance_channel_pairs": len(morph),
            "pct_changed_in_clinical_leads": _pct_col(mask, "clinical_changed_pct_total"),
            "pct_changed_in_auxiliary_channels": _pct_col(mask, "auxiliary_changed_pct_total"),
            "pct_changed_in_v2_v5": _pct_col(mask, "v2_v5_changed_pct_total"),
            "pct_changed_in_qrs_window": _pct_col(mask, "r_changed_pct"),
            "pct_changed_in_r_window": _pct_col(mask, "r_changed_pct"),
            "pct_changed_in_st_window": _pct_col(mask, "st_changed_pct"),
            "pct_changed_in_t_window": _pct_col(mask, "t_changed_pct"),
            "pct_changed_joint_v2_v5_and_r_window": _pct_col(mask, "v2_v5_r_changed_pct_total"),
            "median_delta_qrs_energy_all_channels": _median_col(morph, "delta_qrs_energy"),
            "median_delta_qrs_energy_clinical_leads": _median_col(morph, "delta_qrs_energy", CLINICAL_LEAD_CHANNELS),
            "median_delta_qrs_energy_v2_v5": _median_col(morph, "delta_qrs_energy", V2_V5_CHANNELS),
            "median_delta_qrs_peak_to_peak_v2_v5": _median_col(morph, "delta_qrs_peak_to_peak", V2_V5_CHANNELS),
            "median_abs_qrs_lag_all_channels": _median_col(morph, "qrs_best_lag_abs"),
            "median_abs_qrs_lag_v2_v5": _median_col(morph, "qrs_best_lag_abs", V2_V5_CHANNELS),
            "pct_qrs_lag_abs_gt_2_samples_v2_v5": np.nan,
            "median_delta_t_area_abs_all_channels": _median_col(morph, "delta_t_area_abs"),
            "median_delta_t_area_abs_clinical_leads": _median_col(morph, "delta_t_area_abs", CLINICAL_LEAD_CHANNELS),
            "median_delta_late_t_energy_all_channels": _median_col(morph, "delta_late_t_energy"),
            "median_delta_late_t_energy_clinical_leads": _median_col(morph, "delta_late_t_energy", CLINICAL_LEAD_CHANNELS),
            "median_delta_st_q_contrast_clinical_leads": _median_col(morph, "delta_st_q_contrast", CLINICAL_LEAD_CHANNELS),
            "median_delta_v2_v5_r_amp_mean": np.nan,
            "median_delta_v2_v5_r_amp_slope": np.nan,
        }
        if "qrs_best_lag_abs" in morph.columns:
            v2v5 = morph[morph["channel"].isin(V2_V5_CHANNELS)] if "channel" in morph.columns else morph
            vals = v2v5["qrs_best_lag_abs"].replace([np.inf, -np.inf], np.nan).dropna()
            row["pct_qrs_lag_abs_gt_2_samples_v2_v5"] = float((vals > 2).mean() * 100) if len(vals) else np.nan
        if "instance_id" in morph.columns:
            instance_morph = morph.drop_duplicates("instance_id")
        else:
            instance_morph = morph
        row["median_delta_v2_v5_r_amp_mean"] = _median_col(instance_morph, "delta_v2_v5_r_amp_mean")
        row["median_delta_v2_v5_r_amp_slope"] = _median_col(instance_morph, "delta_v2_v5_r_amp_slope")
        rows.append(row)
    return pd.DataFrame(rows)


def make_result_sentences(global_summary: pd.DataFrame) -> list[str]:
    label_map = {
        "delta_q_amp": "Q-window amplitude decreased",
        "delta_q_extreme_abs": "Q-window absolute deviation increased",
        "delta_st_level": "ST-window level increased",
        "delta_st_abs": "ST-window absolute deviation increased",
        "delta_st_q_contrast": "ST-Q contrast increased",
        "delta_t_peak_abs": "T-window peak absolute amplitude decreased",
        "delta_t_area_abs": "T-window mean absolute amplitude decreased",
    }
    sentences = []
    for _, row in global_summary.iterrows():
        label = label_map.get(row["metric"], row["metric"])
        sentences.append(
            f"{label}: median delta = {row['median_delta']:.4g} "
            f"[IQR {row['q1']:.4g}, {row['q3']:.4g}], "
            f"{row['pct_in_expected_direction']:.1f}% in the expected direction, "
            f"one-sided Wilcoxon p = {format_p(row['wilcoxon_p'])}."
        )
    return sentences


def _indicator_value(indicators: pd.DataFrame | None, direction: str, col: str) -> float:
    if indicators is None or indicators.empty or col not in indicators.columns:
        return np.nan
    row = indicators[indicators["direction"] == direction]
    if row.empty:
        row = indicators[indicators["direction"] == "ALL"]
    if row.empty:
        return np.nan
    return float(row.iloc[0][col])


def make_directional_result_sentences(
    summary_df: pd.DataFrame,
    direction: str,
    narrative_indicators: pd.DataFrame | None = None,
) -> list[str]:
    direction = str(direction).upper()
    rows = summary_df[summary_df["direction"] == direction] if "direction" in summary_df.columns else summary_df
    sentences = []
    if direction == "NORM_TO_MI":
        st_pct = _indicator_value(narrative_indicators, direction, "pct_changed_in_st_window")
        t_pct = _indicator_value(narrative_indicators, direction, "pct_changed_in_t_window")
        late_t = _indicator_value(narrative_indicators, direction, "median_delta_late_t_energy_clinical_leads")
        t_area = _indicator_value(narrative_indicators, direction, "median_delta_t_area_abs_clinical_leads")
        sentences.append(
            f"For NORM->MI explanations, the median changed-point share was {st_pct:.1f}% in the ST window "
            f"and {t_pct:.1f}% in the T window, supporting inspection of localized post-QRS morphology."
        )
        sentences.append(
            f"The selected counterfactuals showed median delta late-T energy {late_t:.4g} and median delta "
            f"T-window absolute area {t_area:.4g} in clinical leads, compatible with the observed T/post-QRS modification tendency."
        )
        sentences.append(
            "These indicators describe model-relevant, clinically inspectable waveform changes without asserting diagnostic MI evidence or infarct localization."
        )
    elif direction == "MI_TO_NORM":
        v2v5_pct = _indicator_value(narrative_indicators, direction, "pct_changed_in_v2_v5")
        joint_pct = _indicator_value(narrative_indicators, direction, "pct_changed_joint_v2_v5_and_r_window")
        r_pct = _indicator_value(narrative_indicators, direction, "pct_changed_in_r_window")
        lag = _indicator_value(narrative_indicators, direction, "median_abs_qrs_lag_v2_v5")
        qrs_energy = _indicator_value(narrative_indicators, direction, "median_delta_qrs_energy_v2_v5")
        sentences.append(
            f"For MI->NORM explanations, the median changed-point share was {r_pct:.1f}% in the R-centered/QRS window "
            f"and {v2v5_pct:.1f}% in channels 7-10 (V2-V5)."
        )
        sentences.append(
            f"{joint_pct:.1f}% of changed points were jointly located in V2-V5 and the R-centered/QRS window; "
            f"median delta QRS energy in V2-V5 was {qrs_energy:.4g}."
        )
        sentences.append(
            f"The median absolute QRS lag in V2-V5 was {lag:.4g} samples, suggesting some edits involve local morphology displacement in addition to amplitude changes."
        )
        sentences.append(
            "These indicators suggest the classifier uses anterior/precordial QRS morphology as model-relevant evidence, without claiming automatic diagnosis or infarct localization."
        )
    else:
        sentences.append("Directional result sentences are intended for NORM_TO_MI or MI_TO_NORM subsets.")
    if not rows.empty:
        descriptive = rows[rows["expected_direction"] == "descriptive"] if "expected_direction" in rows else pd.DataFrame()
        if not descriptive.empty:
            metric_list = ", ".join(descriptive["metric"].head(4).astype(str))
            sentences.append(f"Descriptive morphology metrics summarized for this direction include {metric_list}.")
    return sentences


def _save_or_show(fig: plt.Figure, output_path: Path | None) -> None:
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_global_delta_distributions(df: pd.DataFrame, output_path: Path | None = None) -> None:
    metrics = [
        "delta_q_amp", "delta_q_extreme_abs", "delta_st_level", "delta_st_abs",
        "delta_st_q_contrast", "delta_t_peak_abs", "delta_t_area_abs",
        "delta_qrs_energy", "delta_qrs_peak_to_peak", "delta_late_t_energy",
    ]
    available = [m for m in metrics if m in df.columns]
    data = [df[m].replace([np.inf, -np.inf], np.nan).dropna().values for m in available]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=available, showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Global morphology changes: counterfactual minus original")
    ax.set_ylabel("Delta value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    _save_or_show(fig, output_path)


def plot_per_channel_medians(df: pd.DataFrame, metric: str, output_path: Path | None = None) -> None:
    med = df.groupby("channel")[metric].median().reset_index().sort_values("channel")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([channel_label(ch) for ch in med["channel"]], med[metric])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(f"Median {metric} by channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel(f"Median {metric}")
    ax.grid(True, axis="y", alpha=0.25)
    _save_or_show(fig, output_path)


def plot_changed_point_concentration(mask_overlap_summary: pd.DataFrame, output_path: Path | None = None) -> None:
    wanted = ["clinical", "v2_v5", "auxiliary", "v2_v5_r"]
    df = mask_overlap_summary[mask_overlap_summary["region"].isin(wanted)].copy()
    if df.empty:
        return
    labels = {
        "clinical": "Clinical leads",
        "v2_v5": "V2-V5",
        "auxiliary": "Auxiliary",
        "v2_v5_r": "V2-V5 + QRS",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels.get(x, x) for x in df["region"]], df["median_changed_pct"])
    ax.set_title("Changed-point concentration by channel group")
    ax.set_ylabel("Median share of changed points (%)")
    ax.grid(True, axis="y", alpha=0.25)
    _save_or_show(fig, output_path)


def plot_mask_overlap_summary(mask_overlap_summary: pd.DataFrame, output_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(mask_overlap_summary["region"], mask_overlap_summary["median_changed_pct"])
    ax.set_title("Median percentage of changed points by reference region")
    ax.set_ylabel("Median changed points (%)")
    ax.set_xlabel("Region")
    ax.grid(True, axis="y", alpha=0.25)
    _save_or_show(fig, output_path)


def plot_mask_region_heatmaps(mask_region_df: pd.DataFrame, output_path: Path | None = None) -> None:
    if mask_region_df.empty:
        return
    regions = [r for r in ["baseline", "p", "q", "r", "st", "t"] if r in set(mask_region_df["region"])]
    panels = [
        ("channel", "Changed-point concentration by channel and region"),
        ("channel_group", "Changed-point concentration by channel group and region"),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 6.8),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 0.04], "height_ratios": [1.35, 1.0]},
    )
    panel_axes = [axes[0, 0], axes[1, 0]]
    cbar_axes = [axes[0, 1], axes[1, 1]]
    for ax, cax, (granularity, title) in zip(panel_axes, cbar_axes, panels):
        df = mask_region_df[mask_region_df["granularity"] == granularity]
        if df.empty:
            ax.axis("off")
            cax.axis("off")
            continue
        if granularity == "channel":
            order_df = df[["label", "channel"]].drop_duplicates().sort_values("channel")
            labels = order_df["label"].tolist()
        else:
            preferred = ["Limb leads", "Augmented limb leads", "Precordial leads", "Frank leads"]
            labels = [label for label in preferred if label in set(df["label"])]
        grouped = df.groupby(["label", "region"], as_index=False)["changed_points"].sum()
        total_changed = df[["instance_id", "total_changed_points"]].drop_duplicates()["total_changed_points"].sum()
        grouped["changed_pct_total"] = 100 * grouped["changed_points"] / total_changed if total_changed > 0 else 0.0
        pivot = grouped.pivot(index="label", columns="region", values="changed_pct_total").reindex(index=labels, columns=regions).fillna(0.0)
        vmax = float(np.nanmax(pivot.to_numpy())) if pivot.size else 0.0
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-9))
        ax.set_title(title)
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Fixed R-centered region")
        ax.set_ylabel("Channel" if granularity == "channel" else "Channel group")
        text_threshold = max(vmax * 0.02, 1e-9)
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                value = pivot.iat[y, x]
                if value > 0:
                    color = "white" if value > vmax * 0.55 else "black"
                    label = f"{value:.2f}" if value < 1 else f"{value:.1f}"
                    if value >= text_threshold:
                        ax.text(x, y, label, ha="center", va="center", color=color, fontsize=7)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Pooled share of changed points (%)")
    _save_or_show(fig, output_path)


def plot_average_waveforms(
    x_orig_all: Any,
    x_cf_all: Any,
    channels: list[int] | None = None,
    r_index: int | None = None,
    output_path: Path | None = None,
    config: AnalysisConfig | None = None,
    first_label: str = "Original",
    second_label: str = "Counterfactual",
) -> None:
    config = config or AnalysisConfig()
    x_orig_all = ensure_batch_time_channel(x_orig_all, config=config)
    x_cf_all = ensure_batch_time_channel(x_cf_all, config=config)
    _, n_time, _ = x_orig_all.shape
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    if channels is None:
        orig_mean = x_orig_all.mean(axis=(0, 2))
        cf_mean = x_cf_all.mean(axis=(0, 2))
        title_suffix = "averaged across all channels"
    else:
        orig_mean = x_orig_all[:, :, channels].mean(axis=(0, 2))
        cf_mean = x_cf_all[:, :, channels].mean(axis=(0, 2))
        title_suffix = f"averaged across channels {channels}"
    windows = get_centered_windows(n_time, r_index=r_index, config=config)
    region_colors = {
        "baseline": "#8c8c8c",
        "p": "#4c78a8",
        "q": "#f58518",
        "r": "#e45756",
        "st": "#54a24b",
        "t": "#b279a2",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(orig_mean, label=f"{first_label} mean", linewidth=2)
    ax.plot(cf_mean, label=f"{second_label} mean", linewidth=2)
    ax.axvline(r_index, linestyle="--", linewidth=1, label=f"reference index={r_index}")
    for name, window in windows.items():
        if window is not None:
            ax.axvspan(
                window[0],
                window[1],
                alpha=0.14,
                color=region_colors.get(name, "#cccccc"),
                label=name,
            )
    ax.set_title(f"Average {first_label} vs {second_label} waveform ({title_suffix}, reference index={r_index})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncols=2)
    _save_or_show(fig, output_path)


def plot_one_pair_with_windows(x_orig: Any, x_cf: Any, x_nun: Any = None, channel: int = 0, r_index: int | None = None, title: str | None = None, output_path: Path | None = None, config: AnalysisConfig | None = None) -> None:
    config = config or AnalysisConfig()
    x_orig = ensure_time_channel(x_orig, config=config)
    x_cf = ensure_time_channel(x_cf, config=config)
    x_nun = ensure_time_channel(x_nun, config=config) if x_nun is not None else None
    n_time, n_channels = x_orig.shape
    r_index = resolve_r_index(n_time, r_index=r_index, config=config)
    if channel >= n_channels:
        raise ValueError(f"channel={channel} but only {n_channels} channels available")
    windows, delineation_meta = reference_windows_from_orig_nun(x_orig, x_nun=x_nun, r_index=r_index, config=config)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_orig[:, channel], label="Original", linewidth=2)
    ax.plot(x_cf[:, channel], label="Counterfactual", linewidth=2)
    if x_nun is not None:
        ax.plot(x_nun[:, channel], label="NUN", linewidth=1.2, linestyle="--")
    reference_index = delineation_meta.get("orig_r_index", r_index)
    ax.axvline(reference_index, linestyle="--", linewidth=1, label=f"reference index={reference_index}")
    region_colors = {
        "baseline": "#8c8c8c",
        "p": "#4c78a8",
        "q": "#f58518",
        "r": "#e45756",
        "st": "#54a24b",
        "t": "#b279a2",
    }
    for name, window in windows.items():
        if window is not None:
            ax.axvspan(window[0], window[1], alpha=0.14, color=region_colors.get(name, "#cccccc"), label=name)
    ax.set_title((title or f"Original vs counterfactual, channel {channel}") + f" ({delineation_meta.get('combined_source')})")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncols=2)
    _save_or_show(fig, output_path)


def plot_delineation_examples(
    x_orig_all: Any,
    x_nun_all: Any = None,
    instance_ids: Any = None,
    example_indices: list[int] | None = None,
    max_examples: int = 6,
    output_path: Path | None = None,
    config: AnalysisConfig | None = None,
) -> None:
    """
    Plot a few Lead-II delineation examples.

    Delineation is computed only from x_orig and x_nun on config.delineation_channel_idx.
    The figure shows the actual reference channel used for delineation plus shaded windows.
    """
    config = config or AnalysisConfig()
    x_orig_all = ensure_batch_time_channel(x_orig_all, config=config)
    x_nun_all = ensure_batch_time_channel(x_nun_all, config=config) if x_nun_all is not None else None

    n_instances, _, n_channels = x_orig_all.shape
    if config.delineation_channel_idx >= n_channels:
        raise ValueError(
            f"delineation_channel_idx={config.delineation_channel_idx}, "
            f"but only {n_channels} channels are available"
        )
    if example_indices is None:
        example_indices = list(range(min(max_examples, n_instances)))
    else:
        example_indices = [idx for idx in example_indices if 0 <= idx < n_instances][:max_examples]
    if not example_indices:
        raise ValueError("No valid example indices to plot")

    fig, axes = plt.subplots(len(example_indices), 1, figsize=(13, 3.2 * len(example_indices)), squeeze=False)
    axes = axes[:, 0]
    channel = config.delineation_channel_idx
    region_colors = {
        "baseline": "#8c8c8c",
        "p": "#4c78a8",
        "q": "#f58518",
        "r": "#e45756",
        "st": "#54a24b",
        "t": "#b279a2",
    }

    for ax, idx in zip(axes, example_indices):
        x_orig = x_orig_all[idx]
        x_nun = x_nun_all[idx] if x_nun_all is not None else None
        windows, meta = reference_windows_from_orig_nun(x_orig, x_nun=x_nun, config=config)
        source_label = meta.get("combined_source")
        if source_label == "centered_windows":
            source_label = "fixed R-centered windows"

        ax.plot(x_orig[:, channel], label="Original", linewidth=1.6)
        if x_nun is not None:
            ax.plot(x_nun[:, channel], label="NUN", linewidth=1.2, linestyle="--")
        reference_label = "reference index" if meta.get("combined_source") == "centered_windows" else "Original R"
        ax.axvline(meta.get("orig_r_index"), linestyle="--", linewidth=1, color="black", label=reference_label)
        if meta.get("nun_r_index") is not None:
            ax.axvline(meta.get("nun_r_index"), linestyle=":", linewidth=1, color="black", label="NUN R")

        for name, window in windows.items():
            if window is None:
                continue
            color = region_colors.get(name)
            ax.axvspan(window[0], window[1], alpha=0.14, color=color, label=name)

        instance_label = instance_ids[idx] if instance_ids is not None and idx < len(instance_ids) else idx
        ax.set_title(
            f"Delineation example {idx} | instance {instance_label} | "
            f"channel {channel} | {source_label}"
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), ncols=4, fontsize=8)

    _save_or_show(fig, output_path)
