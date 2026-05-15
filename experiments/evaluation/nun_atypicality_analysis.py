"""Analyze whether atypical NUNs propagate atypicality to counterfactuals.

This script is read-only with respect to existing Multi-SpaCE experiments. It
loads saved ``counterfactuals.pickle`` results, reconstructs the original
instances and NUNs when needed, scores train/original/NUN/CF samples with the
same AE raw reconstruction error used by the repository plausibility metric,
and writes analysis tables/figures under ``evaluations/nun_atypicality_analysis``.

Usage:
    Edit the configuration block below if needed, then run from the project root:
    python experiments/evaluation/nun_atypicality_analysis.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "experiments/evaluation/.matplotlib_cache")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from experiments.data_utils import label_encoder, local_data_loader
from experiments.evaluation.evaluation_utils import calculate_method_metrics
from experiments.experiment_utils import load_model
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder, SecondBestGlobalNUNFinder
from methods.outlier_calculators import AEOutlierCalculator


DEFAULT_OUTPUT_ROOT = Path("experiments/evaluation/nun_atypicality_analysis")
DEFAULT_RESULTS_ROOT = Path("experiments/results")
DEFAULT_MODEL_TO_EXPLAIN = "inceptiontime_noscaling"
DEFAULT_AE_EXPERIMENT = "ae_basic_train"
DEFAULT_PERCENTILES = (90.0, 95.0, 99.0)
# DEFAULT_MO_WEIGHTS = (0.1, 0.6*0.7, 0.4*0.7, 0.2)
DEFAULT_MO_WEIGHTS = (0.1, 0.1, 0.1, 0.7)
PLOT_GROUP_ORDER = [
    "Train data",
    "Original instances",
    "Selected NUNs",
    "Generated CFs",
]
PLOT_PALETTE = {
    "Train data": "#4C78A8",
    "Original instances": "#72B7B2",
    "Selected NUNs": "#F58518",
    "Generated CFs": "#E45756",
}
ANALYSIS_SCORE_NAME = "AE outlier score"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# If RESULT_PATHS has an entry for a dataset, that path is used directly.
# Otherwise the script discovers/builds result folders from
# RESULTS_ROOT / dataset / MODEL_TO_EXPLAIN / EXPERIMENT_FAMILY.
DATASETS = [
    "BasicMotions", "NATOPS", "UWaveGestureLibrary",
    "ArticularyWordRecognition", 
    'Cricket',
    'Epilepsy',# 'PenDigits',
    'RacketSports',
    'SelfRegulationSCP1',
    'PEMS-SF',
]
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
MODEL_TO_EXPLAIN = DEFAULT_MODEL_TO_EXPLAIN
EXPERIMENT_FAMILY = "multisubspace_v2_final"
EXPERIMENT_NAME = "v2_a4a30af26a7d15f8fc03f472ccb70b70c21d172d"
ALL_MATCHING = False
RESULT_PATHS: dict[str, Path] = {}

OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
AE_EXPERIMENT = DEFAULT_AE_EXPERIMENT
PERCENTILES = list(DEFAULT_PERCENTILES)
MO_WEIGHTS = DEFAULT_MO_WEIGHTS
SAVE_PLOTS = True
SAVE_PDF = False
PLOT_DPI = 300
HISTOGRAM_BINS = 22
UTILITY_NAME: str | None = None


@dataclass
class ReferenceContext:
    dataset: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    model_wrapper: Any
    ae_calculator: AEOutlierCalculator
    params: dict[str, Any]
    model_to_explain: str
    ae_model_path: Path


@dataclass
class CounterfactualBundle:
    instance_ids: np.ndarray
    x_orig: np.ndarray
    x_nun: np.ndarray
    x_cf: np.ndarray
    rows: list[dict[str, Any]]
    metrics_df: pd.DataFrame
    warnings: list[str]


@dataclass
class AnalysisResult:
    dataset: str
    output_dir: Path
    scores_df: pd.DataFrame
    train_scores: np.ndarray
    threshold_summary: pd.DataFrame
    correlation_summary: pd.DataFrame
    stratified_quality: pd.DataFrame


def get_configured_mo_weights(weights: tuple[float, ...] | list[float] | np.ndarray | None) -> np.ndarray | None:
    if weights is None:
        return None
    return np.asarray(weights, dtype=float)


def make_utility_name(weights: np.ndarray | None, utility_name: str | None = None) -> str:
    if utility_name:
        return utility_name
    if weights is None:
        return "utility_no_weights"
    weight_values = [str(int(round(float(value) * 100))) for value in weights]
    return "utility_weights_" + "_".join(weight_values)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as fp:
        return json.load(fp)


def infer_dataset_from_result_path(result_path: Path) -> str:
    parts = result_path.resolve().parts
    try:
        idx = parts.index("results")
        return parts[idx + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not infer dataset from result path: {result_path}") from exc


def infer_model_from_result_path(result_path: Path, default: str = DEFAULT_MODEL_TO_EXPLAIN) -> str:
    parts = result_path.resolve().parts
    try:
        idx = parts.index("results")
        return parts[idx + 2]
    except (ValueError, IndexError):
        return default


def find_result_dirs(
    dataset: str,
    results_root: Path,
    model_to_explain: str,
    experiment_family: str | None,
    experiment_name: str | None,
    all_matching: bool,
) -> list[Path]:
    base = results_root / dataset / model_to_explain
    if experiment_family:
        base = base / experiment_family
    if experiment_name:
        candidates = [base / experiment_name]
    else:
        candidates = [
            path
            for path in sorted(base.rglob("*") if base.is_dir() else [])
            if path.is_dir() and (path / "counterfactuals.pickle").is_file()
        ]
    candidates = [path for path in candidates if (path / "counterfactuals.pickle").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No counterfactual result folders found under {base}")
    if all_matching:
        return candidates
    return [max(candidates, key=lambda path: path.stat().st_mtime)]


def configured_result_paths_for_dataset(dataset: str) -> list[Path]:
    if dataset in RESULT_PATHS:
        return [RESULT_PATHS[dataset]]

    if EXPERIMENT_NAME is not None:
        path = RESULTS_ROOT / dataset / MODEL_TO_EXPLAIN
        if EXPERIMENT_FAMILY is not None:
            path = path / EXPERIMENT_FAMILY
        return [path / EXPERIMENT_NAME]

    return find_result_dirs(
        dataset=dataset,
        results_root=RESULTS_ROOT,
        model_to_explain=MODEL_TO_EXPLAIN,
        experiment_family=EXPERIMENT_FAMILY,
        experiment_name=EXPERIMENT_NAME,
        all_matching=ALL_MATCHING,
    )


def load_reference_data_and_autoencoder(
    dataset: str,
    result_path: Path,
    model_to_explain: str | None = None,
    ae_experiment: str = DEFAULT_AE_EXPERIMENT,
) -> ReferenceContext:
    params_path = result_path / "params.json"
    params = read_json(params_path) if params_path.is_file() else {}
    scaling = params.get("scaling", "none")
    model_to_explain = model_to_explain or infer_model_from_result_path(result_path)

    X_train, y_train, X_test, y_test = local_data_loader(
        str(dataset), scaling=scaling, backend="tf", data_path="./experiments/data"
    )
    y_train, y_test = label_encoder(y_train, y_test)
    n_classes = len(np.unique(y_train))
    model_folder = Path("experiments/models") / dataset / model_to_explain
    model_wrapper = load_model(str(model_folder), dataset, X_train.shape[2], X_train.shape[1], n_classes)
    y_pred_train = np.argmax(model_wrapper.predict(X_train, batch_size=16), axis=1)
    y_pred_test = np.argmax(model_wrapper.predict(X_test, batch_size=16), axis=1)

    ae_model_path = Path("experiments/models") / dataset / ae_experiment / "model.hdf5"
    if not ae_model_path.is_file():
        raise FileNotFoundError(f"Missing AE model for {dataset}: {ae_model_path}")
    ae_model = tf.keras.models.load_model(ae_model_path, compile=False)
    ae_calculator = AEOutlierCalculator(ae_model, X_train)

    return ReferenceContext(
        dataset=dataset,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        model_wrapper=model_wrapper,
        ae_calculator=ae_calculator,
        params=params,
        model_to_explain=model_to_explain,
        ae_model_path=ae_model_path,
    )


def result_get(result: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(result, dict):
        for key in keys:
            if key in result:
                return result[key]
    return default


def ensure_sample_array(value: Any, length: int, n_channels: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(length, n_channels)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected {name} shape: {arr.shape}")
    return arr.reshape(length, n_channels)


def ensure_cf_batch(value: Any, length: int, n_channels: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        return arr.reshape(1, length, n_channels)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], length, n_channels)
    if arr.ndim == 1:
        return arr.reshape(1, length, n_channels)
    raise ValueError(f"Unexpected CF shape: {arr.shape}")


def get_test_indexes(params: dict[str, Any], n_results: int, n_test: int) -> np.ndarray:
    for key in ("X_test_indexes", "test_indexes", "subset_idx", "test_idx", "indices"):
        if key in params:
            indexes = np.asarray(params[key], dtype=int)
            if len(indexes) >= n_results:
                return indexes[:n_results]
    return np.arange(min(n_results, n_test), dtype=int)


def build_nuns_for_indexes(context: ReferenceContext, indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = context.params
    strategy = params.get("nun_strategy")
    if strategy is None:
        strategy = "independent" if params.get("independent_channels_nun", False) else "global"

    if strategy == "global":
        finder = GlobalNUNFinder(
            context.X_train, context.y_train, context.y_pred_train,
            distance="euclidean", from_true_labels=False, backend="tf"
        )
    elif strategy == "second_best":
        finder = SecondBestGlobalNUNFinder(
            context.X_train, context.y_train, context.y_pred_train,
            distance="euclidean", from_true_labels=False, backend="tf", model=context.model_wrapper
        )
    elif strategy == "independent":
        finder = IndependentNUNFinder(
            context.X_train, context.y_train, context.y_pred_train,
            distance="euclidean", n_neighbors=int(params.get("n_neighbors", 1)),
            from_true_labels=False, backend="tf", model=context.model_wrapper
        )
    else:
        warnings.warn(f"Unsupported nun_strategy={strategy!r}; falling back to global NUN reconstruction.")
        finder = GlobalNUNFinder(
            context.X_train, context.y_train, context.y_pred_train,
            distance="euclidean", from_true_labels=False, backend="tf"
        )

    nuns, desired_classes, distances = finder.retrieve_nuns(
        context.X_test[indexes], context.y_pred_test[indexes]
    )
    nuns = np.asarray(nuns)[:, 0, :, :]
    return nuns, np.asarray(desired_classes), np.asarray(distances).reshape(len(indexes), -1)[:, 0]


def select_cf_indexes_with_existing_metrics(
    context: ReferenceContext,
    raw_results: list[Any],
    indexes: np.ndarray,
    nuns: np.ndarray,
    mo_weights: np.ndarray | None,
) -> pd.DataFrame:
    if not raw_results or not isinstance(raw_results[0], dict):
        return pd.DataFrame()
    if "cfs" in raw_results[0] and mo_weights is None:
        warnings.warn("Multiple CFs found but no objective weights were provided; selected CF index falls back to 0.")
        return pd.DataFrame()
    try:
        return calculate_method_metrics(
            context.model_wrapper,
            {"AE": context.ae_calculator},
            context.X_test[indexes],
            nuns,
            raw_results,
            context.y_pred_test[indexes],
            "Multi-SpaCE",
            mo_weights=mo_weights,
        )
    except Exception as exc:
        warnings.warn(f"Could not compute existing CF selection metrics; falling back to stored/first CF: {exc}")
        return pd.DataFrame()


def load_counterfactual_results(
    result_path: Path,
    context: ReferenceContext,
    mo_weights: np.ndarray | None,
) -> CounterfactualBundle:
    with (result_path / "counterfactuals.pickle").open("rb") as fp:
        raw_results = pickle.load(fp)
    if isinstance(raw_results, dict):
        for key in ("results", "counterfactuals", "cfs"):
            if key in raw_results:
                raw_results = raw_results[key]
                break
        else:
            raw_results = list(raw_results.values())
    if not isinstance(raw_results, (list, tuple)):
        raise ValueError(f"Unsupported counterfactual result format in {result_path / 'counterfactuals.pickle'}")
    raw_results = list(raw_results)

    indexes = get_test_indexes(context.params, len(raw_results), len(context.X_test))
    length, n_channels = context.X_train.shape[1], context.X_train.shape[2]
    reconstructed_nuns, desired_classes, nun_distances = build_nuns_for_indexes(context, indexes)
    metrics_df = select_cf_indexes_with_existing_metrics(context, raw_results, indexes, reconstructed_nuns, mo_weights)

    rows: list[dict[str, Any]] = []
    x_orig_list, x_nun_list, x_cf_list = [], [], []
    skipped: list[str] = []

    for local_i, result in enumerate(raw_results[: len(indexes)]):
        test_index = int(indexes[local_i])
        try:
            x_orig_raw = result_get(result, ("x_orig", "x", "original", "original_instance"))
            x_nun_raw = result_get(result, ("x_nun", "nun", "nun_example", "nearest_unlike_neighbor"))
            x_cf_raw = result_get(result, ("selected_cf", "x_cf", "cf", "counterfactual"))
            if x_cf_raw is None:
                x_cf_raw = result_get(result, ("cfs", "counterfactuals"))

            x_orig = (
                ensure_sample_array(x_orig_raw, length, n_channels, "x_orig")
                if x_orig_raw is not None
                else context.X_test[test_index]
            )
            x_nun = (
                ensure_sample_array(x_nun_raw, length, n_channels, "x_nun")
                if x_nun_raw is not None
                else reconstructed_nuns[local_i]
            )

            cfs = ensure_cf_batch(x_cf_raw, length, n_channels)
            if isinstance(result, dict) and "selected_cf_index" in result:
                selected_idx = int(result["selected_cf_index"])
            elif isinstance(result, dict) and "best cf index" in result:
                selected_idx = int(result["best cf index"])
            elif not metrics_df.empty and "best cf index" in metrics_df:
                selected_idx = int(metrics_df.iloc[local_i]["best cf index"])
            else:
                selected_idx = 0
            selected_idx = int(np.clip(selected_idx, 0, cfs.shape[0] - 1))
            x_cf = cfs[selected_idx]

            x_orig_list.append(x_orig)
            x_nun_list.append(x_nun)
            x_cf_list.append(x_cf)
            row = {
                "instance_id": test_index,
                "local_result_index": local_i,
                "original_label": int(context.y_pred_test[test_index]),
                "true_label": int(context.y_test[test_index]),
                "target_nun_label": int(desired_classes[local_i]) if len(desired_classes) > local_i else np.nan,
                "nun_distance": float(nun_distances[local_i]) if len(nun_distances) > local_i else np.nan,
                "selected_cf_index": selected_idx,
            }
            if not metrics_df.empty:
                for key, value in metrics_df.iloc[local_i].to_dict().items():
                    if isinstance(value, (np.integer, np.floating, np.bool_)):
                        value = value.item()
                    row[f"cf_metric_{key}"] = value
            rows.append(row)
        except Exception as exc:
            skipped.append(f"local_i={local_i}, test_index={test_index}: {exc}")

    if not x_orig_list:
        raise ValueError(f"No analyzable counterfactual rows found in {result_path}")
    return CounterfactualBundle(
        instance_ids=np.asarray([row["instance_id"] for row in rows], dtype=int),
        x_orig=np.asarray(x_orig_list),
        x_nun=np.asarray(x_nun_list),
        x_cf=np.asarray(x_cf_list),
        rows=rows,
        metrics_df=metrics_df,
        warnings=skipped,
    )


def compute_ae_scores(
    ae_calculator: AEOutlierCalculator,
    X_train: np.ndarray,
    x_orig: np.ndarray,
    x_nun: np.ndarray,
    x_cf: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "train_raw_reconstruction_error": ae_calculator._get_raw_outlier_scores(X_train).reshape(-1),
        "orig_raw_reconstruction_error": ae_calculator._get_raw_outlier_scores(x_orig).reshape(-1),
        "nun_raw_reconstruction_error": ae_calculator._get_raw_outlier_scores(x_nun).reshape(-1),
        "cf_raw_reconstruction_error": ae_calculator._get_raw_outlier_scores(x_cf).reshape(-1),
        "train": ae_calculator.get_outlier_scores(X_train).reshape(-1),
        "orig": ae_calculator.get_outlier_scores(x_orig).reshape(-1),
        "nun": ae_calculator.get_outlier_scores(x_nun).reshape(-1),
        "cf": ae_calculator.get_outlier_scores(x_cf).reshape(-1),
    }


def compute_threshold_summary(scores_df: pd.DataFrame, train_scores: np.ndarray, percentiles: list[float]) -> pd.DataFrame:
    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(train_scores, percentile))
        orig_atypical = scores_df["ae_outlier_score_orig"] > threshold
        nun_atypical = scores_df["ae_outlier_score_nun"] > threshold
        cf_atypical = scores_df["ae_outlier_score_cf"] > threshold

        nun_atyp = nun_atypical.sum()
        nun_typ = (~nun_atypical).sum()
        p_cf_given_atyp = float(cf_atypical[nun_atypical].mean()) if nun_atyp > 0 else np.nan
        p_cf_given_typ = float(cf_atypical[~nun_atypical].mean()) if nun_typ > 0 else np.nan
        risk_difference = p_cf_given_atyp - p_cf_given_typ if np.isfinite(p_cf_given_atyp) and np.isfinite(p_cf_given_typ) else np.nan

        a = int((nun_atypical & cf_atypical).sum())
        b = int((nun_atypical & ~cf_atypical).sum())
        c = int((~nun_atypical & cf_atypical).sum())
        d = int((~nun_atypical & ~cf_atypical).sum())
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

        rows.append(
            {
                "percentile": percentile,
                "threshold_value": threshold,
                "score_name": ANALYSIS_SCORE_NAME,
                "pct_orig_atypical": float(orig_atypical.mean() * 100),
                "pct_nun_atypical": float(nun_atypical.mean() * 100),
                "pct_cf_atypical": float(cf_atypical.mean() * 100),
                "p_cf_atypical_given_nun_atypical": p_cf_given_atyp,
                "p_cf_atypical_given_nun_typical": p_cf_given_typ,
                "risk_difference": risk_difference,
                "odds_ratio": float(odds_ratio),
                "n_nun_typical_cf_typical": d,
                "n_nun_typical_cf_atypical": c,
                "n_nun_atypical_cf_typical": b,
                "n_nun_atypical_cf_atypical": a,
            }
        )
    return pd.DataFrame(rows)


def compute_propagation_tables(scores_df: pd.DataFrame, train_scores: np.ndarray, percentiles: list[float]) -> dict[str, pd.DataFrame]:
    tables = {}
    for percentile in percentiles:
        threshold = float(np.percentile(train_scores, percentile))
        nun_atypical = scores_df["ae_outlier_score_nun"] > threshold
        cf_atypical = scores_df["ae_outlier_score_cf"] > threshold
        table = pd.DataFrame(
            [
                [
                    int((~nun_atypical & ~cf_atypical).sum()),
                    int((~nun_atypical & cf_atypical).sum()),
                ],
                [
                    int((nun_atypical & ~cf_atypical).sum()),
                    int((nun_atypical & cf_atypical).sum()),
                ],
            ],
            index=["NUN typical", "NUN atypical"],
            columns=["CF typical", "CF atypical"],
        )
        tables[format_percentile_label(percentile)] = table
    return tables


def format_percentile_label(percentile: float) -> str:
    return f"p{str(percentile).replace('.', '_')}"


def compute_correlations(scores_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("nun_ae_outlier_score_vs_cf_ae_outlier_score", "ae_outlier_score_nun", "ae_outlier_score_cf"),
        ("nun_ae_outlier_score_vs_cf_minus_orig", "ae_outlier_score_nun", "ae_outlier_score_cf_minus_orig"),
    ]
    rows = []
    for name, left, right in pairs:
        for method in ("spearman", "pearson"):
            valid = scores_df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = valid[left].corr(valid[right], method=method) if len(valid) >= 2 else np.nan
            rows.append({"comparison": name, "method": method, "correlation": corr, "n": len(valid)})
    return pd.DataFrame(rows)


def summarize_numeric(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out[f"{col}_mean"] = float(values.mean())
        out[f"{col}_median"] = float(values.median())
        out[f"{col}_std"] = float(values.std())
    return out


def compute_stratified_quality(scores_df: pd.DataFrame, train_scores: np.ndarray, percentiles: list[float]) -> pd.DataFrame:
    base_metric_cols = [
        "ae_outlier_score_cf",
        "ae_outlier_score_cf_minus_orig",
        "cf_metric_L2",
        "cf_metric_sparsity",
        "cf_metric_subsequences",
        "cf_metric_subsequences %",
        "cf_metric_valid",
        "cf_metric_proba",
        "cf_metric_AE_OS",
        "cf_metric_AE_IOS",
    ]
    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(train_scores, percentile))
        nun_atypical = scores_df["ae_outlier_score_nun"] > threshold
        cf_atypical = scores_df["ae_outlier_score_cf"] > threshold
        for group_name, mask in (("typical_nun", ~nun_atypical), ("atypical_nun", nun_atypical)):
            group = scores_df[mask].copy()
            row = {
                "percentile": percentile,
                "threshold_value": threshold,
                "score_name": ANALYSIS_SCORE_NAME,
                "nun_group": group_name,
                "n": int(len(group)),
                "cf_atypical_pct": float(cf_atypical[mask].mean() * 100) if len(group) else np.nan,
            }
            row.update(summarize_numeric(group, base_metric_cols))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_threshold_summary_from_flags(flag_df: pd.DataFrame, percentiles: list[float]) -> pd.DataFrame:
    rows = []
    for percentile in percentiles:
        df = flag_df[flag_df["percentile"] == percentile]
        orig_atypical = df["orig_atypical"]
        nun_atypical = df["nun_atypical"]
        cf_atypical = df["cf_atypical"]

        nun_atyp = nun_atypical.sum()
        nun_typ = (~nun_atypical).sum()
        p_cf_given_atyp = float(cf_atypical[nun_atypical].mean()) if nun_atyp > 0 else np.nan
        p_cf_given_typ = float(cf_atypical[~nun_atypical].mean()) if nun_typ > 0 else np.nan
        risk_difference = p_cf_given_atyp - p_cf_given_typ if np.isfinite(p_cf_given_atyp) and np.isfinite(p_cf_given_typ) else np.nan

        a = int((nun_atypical & cf_atypical).sum())
        b = int((nun_atypical & ~cf_atypical).sum())
        c = int((~nun_atypical & cf_atypical).sum())
        d = int((~nun_atypical & ~cf_atypical).sum())
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

        rows.append(
            {
                "dataset": "global",
                "percentile": percentile,
                "threshold_scope": "dataset_specific",
                "threshold_value": np.nan,
                "score_name": ANALYSIS_SCORE_NAME,
                "pct_orig_atypical": float(orig_atypical.mean() * 100),
                "pct_nun_atypical": float(nun_atypical.mean() * 100),
                "pct_cf_atypical": float(cf_atypical.mean() * 100),
                "p_cf_atypical_given_nun_atypical": p_cf_given_atyp,
                "p_cf_atypical_given_nun_typical": p_cf_given_typ,
                "risk_difference": risk_difference,
                "odds_ratio": float(odds_ratio),
                "n_nun_typical_cf_typical": d,
                "n_nun_typical_cf_atypical": c,
                "n_nun_atypical_cf_typical": b,
                "n_nun_atypical_cf_atypical": a,
            }
        )
    return pd.DataFrame(rows)


def compute_propagation_tables_from_flags(flag_df: pd.DataFrame, percentiles: list[float]) -> dict[str, pd.DataFrame]:
    tables = {}
    for percentile in percentiles:
        df = flag_df[flag_df["percentile"] == percentile]
        nun_atypical = df["nun_atypical"]
        cf_atypical = df["cf_atypical"]
        tables[format_percentile_label(percentile)] = pd.DataFrame(
            [
                [
                    int((~nun_atypical & ~cf_atypical).sum()),
                    int((~nun_atypical & cf_atypical).sum()),
                ],
                [
                    int((nun_atypical & ~cf_atypical).sum()),
                    int((nun_atypical & cf_atypical).sum()),
                ],
            ],
            index=["NUN typical", "NUN atypical"],
            columns=["CF typical", "CF atypical"],
        )
    return tables


def compute_stratified_quality_from_flags(flag_df: pd.DataFrame, percentiles: list[float]) -> pd.DataFrame:
    base_metric_cols = [
        "ae_outlier_score_cf",
        "ae_outlier_score_cf_minus_orig",
        "cf_metric_L2",
        "cf_metric_sparsity",
        "cf_metric_subsequences",
        "cf_metric_subsequences %",
        "cf_metric_valid",
        "cf_metric_proba",
        "cf_metric_AE_OS",
        "cf_metric_AE_IOS",
    ]
    rows = []
    for percentile in percentiles:
        df = flag_df[flag_df["percentile"] == percentile]
        for group_name, mask in (("typical_nun", ~df["nun_atypical"]), ("atypical_nun", df["nun_atypical"])):
            group = df[mask].copy()
            row = {
                "dataset": "global",
                "percentile": percentile,
                "threshold_scope": "dataset_specific",
                "threshold_value": np.nan,
                "score_name": ANALYSIS_SCORE_NAME,
                "nun_group": group_name,
                "n": int(len(group)),
                "cf_atypical_pct": float(group["cf_atypical"].mean() * 100) if len(group) else np.nan,
            }
            row.update(summarize_numeric(group, base_metric_cols))
            rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_specific_global_flags(
    analysis_results: list[AnalysisResult],
    percentiles: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag_frames = []
    threshold_rows = []
    for result in analysis_results:
        for percentile in percentiles:
            threshold = float(np.percentile(result.train_scores, percentile))
            df = result.scores_df.copy()
            df["percentile"] = percentile
            df["dataset_threshold_value"] = threshold
            df["orig_atypical"] = df["ae_outlier_score_orig"] > threshold
            df["nun_atypical"] = df["ae_outlier_score_nun"] > threshold
            df["cf_atypical"] = df["ae_outlier_score_cf"] > threshold
            flag_frames.append(df)
            threshold_rows.append(
                {
                    "dataset": result.dataset,
                    "percentile": percentile,
                    "threshold_value": threshold,
                    "n_train_reference_instances": int(len(result.train_scores)),
                    "n_explained_instances": int(len(result.scores_df)),
                    "score_name": ANALYSIS_SCORE_NAME,
                }
            )
    return pd.concat(flag_frames, ignore_index=True), pd.DataFrame(threshold_rows)


def build_ae_scores_df(bundle: CounterfactualBundle, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    df = pd.DataFrame(bundle.rows)
    df["ae_outlier_score_orig"] = scores["orig"]
    df["ae_outlier_score_nun"] = scores["nun"]
    df["ae_outlier_score_cf"] = scores["cf"]
    df["ae_raw_reconstruction_error_orig"] = scores["orig_raw_reconstruction_error"]
    df["ae_raw_reconstruction_error_nun"] = scores["nun_raw_reconstruction_error"]
    df["ae_raw_reconstruction_error_cf"] = scores["cf_raw_reconstruction_error"]
    df["ae_outlier_score_cf_minus_orig"] = df["ae_outlier_score_cf"] - df["ae_outlier_score_orig"]
    df["ae_outlier_score_nun_minus_orig"] = df["ae_outlier_score_nun"] - df["ae_outlier_score_orig"]
    df["ae_raw_reconstruction_error_cf_minus_orig"] = (
        df["ae_raw_reconstruction_error_cf"] - df["ae_raw_reconstruction_error_orig"]
    )
    df["ae_raw_reconstruction_error_nun_minus_orig"] = (
        df["ae_raw_reconstruction_error_nun"] - df["ae_raw_reconstruction_error_orig"]
    )
    return df


def build_plot_df(train_scores: np.ndarray, scores_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Train data": pd.Series(train_scores),
            "Original instances": pd.Series(scores_df["ae_outlier_score_orig"]),
            "Selected NUNs": pd.Series(scores_df["ae_outlier_score_nun"]),
            "Generated CFs": pd.Series(scores_df["ae_outlier_score_cf"]),
        }
    ).melt(var_name="group", value_name="ae_outlier_score").dropna()


def save_distribution_plots(output_dir: Path, train_scores: np.ndarray, scores_df: pd.DataFrame, save_pdf: bool, dpi: int) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plot_df = build_plot_df(train_scores, scores_df)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.histplot(
        data=plot_df,
        x="ae_outlier_score",
        hue="group",
        hue_order=PLOT_GROUP_ORDER,
        palette=PLOT_PALETTE,
        bins=HISTOGRAM_BINS,
        stat="probability",
        common_norm=False,
        element="bars",
        fill=True,
        alpha=0.28,
        kde=True,
        line_kws={"linewidth": 2.0},
        ax=ax,
    )
    ax.set_xlabel("AE outlier score")
    ax.set_ylabel("Probability")
    ax.set_title("AE Outlier Score Probability Distributions with KDE Fit")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "ae_outlier_score_histogram.png", dpi=dpi)
    if save_pdf:
        fig.savefig(output_dir / "ae_outlier_score_histogram.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.kdeplot(
        data=plot_df,
        x="ae_outlier_score",
        hue="group",
        hue_order=PLOT_GROUP_ORDER,
        palette=PLOT_PALETTE,
        common_norm=False,
        linewidth=2.2,
        ax=ax,
    )
    ax.set_xlabel("AE outlier score")
    ax.set_ylabel("Density")
    ax.set_title("AE Outlier Score KDE Distributions")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "ae_outlier_score_kde.png", dpi=dpi)
    if save_pdf:
        fig.savefig(output_dir / "ae_outlier_score_kde.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.boxplot(
        data=plot_df,
        x="group",
        y="ae_outlier_score",
        hue="group",
        order=PLOT_GROUP_ORDER,
        hue_order=PLOT_GROUP_ORDER,
        palette=PLOT_PALETTE,
        showfliers=False,
        width=0.58,
        linewidth=1.2,
        dodge=False,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df[plot_df["group"] != "Train data"],
        x="group",
        y="ae_outlier_score",
        order=PLOT_GROUP_ORDER,
        color="#222222",
        alpha=0.35,
        size=2.4,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("AE outlier score")
    ax.set_title("AE Outlier Score by Sample Group")
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "ae_outlier_score_boxplot.png", dpi=dpi)
    if save_pdf:
        fig.savefig(output_dir / "ae_outlier_score_boxplot.pdf")
    plt.close()


def save_propagation_heatmap(
    output_dir: Path,
    propagation_tables: dict[str, pd.DataFrame],
    threshold_summary: pd.DataFrame,
    save_pdf: bool,
    dpi: int,
) -> None:
    if not propagation_tables:
        return

    n_tables = len(propagation_tables)
    fig, axes = plt.subplots(n_tables, 1, figsize=(5.6, 3.8 * n_tables), squeeze=False)
    vmax = max(int(table.to_numpy().max()) for table in propagation_tables.values())
    threshold_lookup = {
        format_percentile_label(float(row["percentile"])): row
        for _, row in threshold_summary.iterrows()
    }

    for ax, (label, table) in zip(axes.flat, propagation_tables.items()):
        summary_row = threshold_lookup.get(label)
        title = label.replace("_", ".").upper()
        if summary_row is not None:
            title = f"{summary_row['percentile']:g}th percentile"
        sns.heatmap(
            table,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar=True,
            vmin=0,
            vmax=vmax,
            linewidths=0.8,
            linecolor="white",
            square=True,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Counterfactual")
        ax.set_ylabel("NUN")

    fig.suptitle("NUN-CF Atypicality Propagation Counts", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "propagation_heatmaps.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / "propagation_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def save_propagation_tables_csv(output_dir: Path, propagation_tables: dict[str, pd.DataFrame]) -> None:
    legacy_excel_path = output_dir / "propagation_tables.xlsx"
    if legacy_excel_path.exists():
        legacy_excel_path.unlink()
    for label, table in propagation_tables.items():
        table.to_csv(output_dir / f"propagation_table_{label}.csv")


def save_global_dataset_kde_subplots(
    output_dir: Path,
    analysis_results: list[AnalysisResult],
    dataset_thresholds: pd.DataFrame,
    save_pdf: bool,
    dpi: int,
) -> None:
    if not analysis_results:
        return

    n_datasets = len(analysis_results)
    n_cols = 3
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.3 * n_cols, 4.0 * n_rows), squeeze=False)
    threshold_percentiles = [90.0, 95.0, 99.0]
    train_color = PLOT_PALETTE["Train data"]

    for ax, result in zip(axes.flat, analysis_results):
        plot_df = build_plot_df(result.train_scores, result.scores_df)
        sns.kdeplot(
            data=plot_df,
            x="ae_outlier_score",
            hue="group",
            hue_order=PLOT_GROUP_ORDER,
            palette=PLOT_PALETTE,
            common_norm=False,
            linewidth=1.8,
            ax=ax,
            legend=False,
        )
        dataset_threshold_rows = dataset_thresholds[
            (dataset_thresholds["dataset"] == result.dataset)
            & (dataset_thresholds["percentile"].isin(threshold_percentiles))
        ]
        for _, row in dataset_threshold_rows.iterrows():
            threshold = float(row["threshold_value"])
            percentile = float(row["percentile"])
            ax.axvline(
                threshold,
                color=train_color,
                linestyle=(0, (4, 4)),
                linewidth=1.4,
                alpha=0.9,
            )
            y_min, y_max = ax.get_ylim()
            ax.text(
                threshold,
                y_max * 0.96,
                f"{percentile:g}",
                color=train_color,
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
            )
        ax.set_title(result.dataset)
        ax.set_xlabel("AE outlier score")
        ax.set_ylabel("Density")
        sns.despine(ax=ax)

    for ax in axes.flat[n_datasets:]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=PLOT_PALETTE[group], linewidth=2, label=group)
        for group in PLOT_GROUP_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color=train_color,
            linewidth=1.4,
            linestyle=(0, (4, 4)),
            label="Train percentile thresholds: 90, 95, 99",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=min(5, len(handles)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "ae_outlier_score_kde_by_dataset.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / "ae_outlier_score_kde_by_dataset.pdf", bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    dataset: str,
    result_path: Path,
    context: ReferenceContext,
    scores_df: pd.DataFrame,
    train_scores: np.ndarray,
    threshold_summary: pd.DataFrame,
    propagation_tables: dict[str, pd.DataFrame],
    correlation_summary: pd.DataFrame,
    stratified_quality: pd.DataFrame,
    percentiles: list[float],
    utility_name: str,
    mo_weights: np.ndarray | None,
    skipped_warnings: list[str],
    save_plots_flag: bool,
    save_pdf: bool,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_dir / "ae_outlier_scores.csv", index=False)
    threshold_summary.to_csv(output_dir / "threshold_summary.csv", index=False)
    correlation_summary.to_csv(output_dir / "correlation_summary.csv", index=False)
    stratified_quality.to_csv(output_dir / "stratified_cf_quality.csv", index=False)
    save_propagation_tables_csv(output_dir, propagation_tables)

    if save_plots_flag:
        save_distribution_plots(output_dir, train_scores, scores_df, save_pdf=save_pdf, dpi=dpi)
        save_propagation_heatmap(
            output_dir,
            propagation_tables,
            threshold_summary,
            save_pdf=save_pdf,
            dpi=dpi,
        )

    summary = {
        "dataset": dataset,
        "n_explained_instances": int(len(scores_df)),
        "percentiles": percentiles,
        "analysis_score_name": ANALYSIS_SCORE_NAME,
        "utility_name": utility_name,
        "mo_weights": None if mo_weights is None else [float(value) for value in mo_weights],
        "histogram_bins": HISTOGRAM_BINS,
        "paths": {
            "result_path": str(result_path),
            "counterfactuals_path": str(result_path / "counterfactuals.pickle"),
            "params_path": str(result_path / "params.json"),
            "ae_model_path": str(context.ae_model_path),
            "output_dir": str(output_dir),
        },
        "analysis_score_definition": "AEOutlierCalculator.get_outlier_scores: AE raw reconstruction error scaled by the training calibration range.",
        "raw_reconstruction_error_definition": "AEOutlierCalculator._get_raw_outlier_scores: mean absolute reconstruction error over time and channels.",
        "threshold_summary": threshold_summary.replace({np.nan: None}).to_dict(orient="records"),
        "correlation_summary": correlation_summary.replace({np.nan: None}).to_dict(orient="records"),
        "warnings": skipped_warnings,
    }
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)


def save_global_outputs(
    output_dir: Path,
    analysis_results: list[AnalysisResult],
    percentiles: list[float],
    utility_name: str,
    mo_weights: np.ndarray | None,
    save_plots_flag: bool,
    save_pdf: bool,
    dpi: int,
) -> None:
    if not analysis_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_df = pd.concat([result.scores_df for result in analysis_results], ignore_index=True)
    train_scores = np.concatenate([result.train_scores for result in analysis_results])
    flag_df, dataset_thresholds = build_dataset_specific_global_flags(analysis_results, percentiles)
    threshold_summary = compute_threshold_summary_from_flags(flag_df, percentiles)
    propagation_tables = compute_propagation_tables_from_flags(flag_df, percentiles)
    correlation_summary = compute_correlations(scores_df)
    correlation_summary.insert(0, "dataset", "global")
    stratified_quality = compute_stratified_quality_from_flags(flag_df, percentiles)

    scores_df.to_csv(output_dir / "ae_outlier_scores.csv", index=False)
    flag_df.to_csv(output_dir / "ae_outlier_scores_with_dataset_specific_atypicality.csv", index=False)
    dataset_thresholds.to_csv(output_dir / "dataset_specific_thresholds.csv", index=False)
    threshold_summary.to_csv(output_dir / "threshold_summary.csv", index=False)
    correlation_summary.to_csv(output_dir / "correlation_summary.csv", index=False)
    stratified_quality.to_csv(output_dir / "stratified_cf_quality.csv", index=False)
    save_propagation_tables_csv(output_dir, propagation_tables)

    dataset_summaries = pd.DataFrame(
        {
            "dataset": [result.dataset for result in analysis_results],
            "output_dir": [str(result.output_dir) for result in analysis_results],
            "n_explained_instances": [len(result.scores_df) for result in analysis_results],
            "n_train_reference_instances": [len(result.train_scores) for result in analysis_results],
        }
    )
    dataset_summaries.to_csv(output_dir / "analyzed_datasets.csv", index=False)
    pd.concat([result.threshold_summary for result in analysis_results], ignore_index=True).to_csv(
        output_dir / "per_dataset_threshold_summary.csv",
        index=False,
    )
    pd.concat([result.correlation_summary for result in analysis_results], ignore_index=True).to_csv(
        output_dir / "per_dataset_correlation_summary.csv",
        index=False,
    )
    pd.concat([result.stratified_quality for result in analysis_results], ignore_index=True).to_csv(
        output_dir / "per_dataset_stratified_cf_quality.csv",
        index=False,
    )

    if save_plots_flag:
        save_distribution_plots(output_dir, train_scores, scores_df, save_pdf=save_pdf, dpi=dpi)
        save_global_dataset_kde_subplots(
            output_dir,
            analysis_results,
            dataset_thresholds,
            save_pdf=save_pdf,
            dpi=dpi,
        )
        save_propagation_heatmap(
            output_dir,
            propagation_tables,
            threshold_summary,
            save_pdf=save_pdf,
            dpi=dpi,
        )

    summary = {
        "dataset": "global",
        "datasets": [result.dataset for result in analysis_results],
        "n_explained_instances": int(len(scores_df)),
        "n_train_reference_instances": int(len(train_scores)),
        "percentiles": percentiles,
        "analysis_score_name": ANALYSIS_SCORE_NAME,
        "global_threshold_scope": "dataset_specific",
        "utility_name": utility_name,
        "mo_weights": None if mo_weights is None else [float(value) for value in mo_weights],
        "histogram_bins": HISTOGRAM_BINS,
        "paths": {
            "output_dir": str(output_dir),
        },
        "analysis_score_definition": "AEOutlierCalculator.get_outlier_scores: AE raw reconstruction error scaled by each dataset's training calibration range.",
        "global_atypicality_definition": "Each dataset is thresholded against its own training-score percentile; atypicality labels are then pooled for global tables and heatmaps.",
        "threshold_summary": threshold_summary.replace({np.nan: None}).to_dict(orient="records"),
        "correlation_summary": correlation_summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)


def run_analysis(
    dataset: str,
    result_path: Path,
    output_root: Path,
    percentiles: list[float],
    mo_weights: np.ndarray | None,
    model_to_explain: str | None,
    ae_experiment: str,
    save_plots_flag: bool,
    save_pdf: bool,
    dpi: int,
) -> AnalysisResult:
    context = load_reference_data_and_autoencoder(
        dataset=dataset,
        result_path=result_path,
        model_to_explain=model_to_explain,
        ae_experiment=ae_experiment,
    )
    bundle = load_counterfactual_results(result_path, context, mo_weights=mo_weights)
    scores = compute_ae_scores(
        context.ae_calculator,
        context.X_train,
        bundle.x_orig,
        bundle.x_nun,
        bundle.x_cf,
    )
    scores_df = build_ae_scores_df(bundle, scores)
    threshold_summary = compute_threshold_summary(scores_df, scores["train"], percentiles)
    propagation_tables = compute_propagation_tables(scores_df, scores["train"], percentiles)
    correlation_summary = compute_correlations(scores_df)
    stratified_quality = compute_stratified_quality(scores_df, scores["train"], percentiles)
    scores_df.insert(0, "dataset", dataset)
    threshold_summary.insert(0, "dataset", dataset)
    correlation_summary.insert(0, "dataset", dataset)
    stratified_quality.insert(0, "dataset", dataset)

    utility_name = make_utility_name(mo_weights, UTILITY_NAME)
    output_dir = output_root / utility_name / dataset
    save_outputs(
        output_dir=output_dir,
        dataset=dataset,
        result_path=result_path,
        context=context,
        scores_df=scores_df,
        train_scores=scores["train"],
        threshold_summary=threshold_summary,
        propagation_tables=propagation_tables,
        correlation_summary=correlation_summary,
        stratified_quality=stratified_quality,
        percentiles=percentiles,
        utility_name=utility_name,
        mo_weights=mo_weights,
        skipped_warnings=bundle.warnings,
        save_plots_flag=save_plots_flag,
        save_pdf=save_pdf,
        dpi=dpi,
    )
    return AnalysisResult(
        dataset=dataset,
        output_dir=output_dir,
        scores_df=scores_df,
        train_scores=scores["train"],
        threshold_summary=threshold_summary,
        correlation_summary=correlation_summary,
        stratified_quality=stratified_quality,
    )


def main() -> None:
    mo_weights = get_configured_mo_weights(MO_WEIGHTS)
    utility_name = make_utility_name(mo_weights, UTILITY_NAME)
    analysis_results: list[AnalysisResult] = []

    for dataset in DATASETS:
        result_paths = configured_result_paths_for_dataset(dataset)
        if len(result_paths) > 1:
            warnings.warn(
                f"Dataset {dataset} has {len(result_paths)} matching result folders. "
                "Only the last one will be reflected in the dataset-level output folder."
            )

        for result_path in result_paths:
            result_path = result_path.resolve()
            if not (result_path / "counterfactuals.pickle").is_file():
                warnings.warn(f"Skipping {dataset}: missing counterfactuals.pickle in {result_path}")
                continue
            result = run_analysis(
                dataset=dataset,
                result_path=result_path,
                output_root=OUTPUT_ROOT,
                percentiles=list(PERCENTILES),
                mo_weights=mo_weights,
                model_to_explain=MODEL_TO_EXPLAIN or infer_model_from_result_path(result_path),
                ae_experiment=AE_EXPERIMENT,
                save_plots_flag=SAVE_PLOTS,
                save_pdf=SAVE_PDF,
                dpi=PLOT_DPI,
            )
            analysis_results.append(result)
            print(f"Wrote NUN atypicality analysis to {result.output_dir}")

    if not analysis_results:
        raise RuntimeError("No dataset analyses were completed. Check DATASETS, RESULT_PATHS, and experiment paths.")

    global_dir = OUTPUT_ROOT / utility_name / "global"
    save_global_outputs(
        global_dir,
        analysis_results,
        percentiles=list(PERCENTILES),
        utility_name=utility_name,
        mo_weights=mo_weights,
        save_plots_flag=SAVE_PLOTS,
        save_pdf=SAVE_PDF,
        dpi=PLOT_DPI,
    )
    print(f"Wrote global NUN atypicality analysis to {global_dir}")


if __name__ == "__main__":
    main()
