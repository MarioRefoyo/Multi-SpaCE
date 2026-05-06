import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METADATA_COLUMNS = {
    "dataset",
    "model_to_explain",
    "experiment_family",
    "experiment_hash",
    "method",
    "metrics_path",
    "params_path",
    "family_dir",
    "experiment_dir",
    "counterfactuals_path",
    "metrics_excel_path",
    "MO_EVAL_WEIGHTS",
    "X_test_indexes",
}


def find_results_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for candidate in [start, *start.parents]:
        results_root = candidate / "experiments" / "results"
        if results_root.is_dir():
            return results_root
    raise FileNotFoundError(f"Could not find experiments/results from {start}")


def stable_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), sort_keys=True)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return np.nan
    return value


def read_params_json(params_path):
    with open(params_path, "r") as fp:
        params = json.load(fp)
    return {key: stable_value(value) for key, value in params.items()}


def read_params_sheet(metrics_path):
    params_df = pd.read_excel(metrics_path, sheet_name="params")
    if not {"parameter", "value"}.issubset(params_df.columns):
        return {}
    return {
        row["parameter"]: stable_value(row["value"])
        for _, row in params_df.iterrows()
        if pd.notna(row["parameter"])
    }


def read_experiment_params(experiment_dir, metrics_path):
    params_path = experiment_dir / "params.json"
    if params_path.is_file():
        return read_params_json(params_path)
    return read_params_sheet(metrics_path)


def normalize_metric_columns(df):
    df = df.copy()
    rename_map = {}
    if "improvement_over_nun" in df.columns and "IoN" not in df.columns:
        rename_map["improvement_over_nun"] = "IoN"
    return df.rename(columns=rename_map)


def discover_metric_files(
    results_root,
    model_to_explain,
    experiment_families,
    datasets=None,
    weights_file="model_weights_10_28_42_20.xlsx",
):
    results_root = Path(results_root)
    dataset_dirs = [results_root / dataset for dataset in datasets] if datasets else sorted(results_root.iterdir())
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            continue
        model_dir = dataset_dir / model_to_explain
        if not model_dir.is_dir():
            continue
        family_dirs = [model_dir / family for family in experiment_families] if experiment_families else sorted(model_dir.iterdir())
        for family_dir in family_dirs:
            if not family_dir.is_dir():
                continue
            for experiment_dir in sorted(family_dir.iterdir()):
                metrics_path = experiment_dir / weights_file
                if experiment_dir.is_dir() and metrics_path.is_file():
                    yield dataset_dir.name, family_dir.name, experiment_dir.name, metrics_path


def load_instance_metrics(
    results_root,
    model_to_explain,
    experiment_families,
    datasets=None,
    weights_file="model_weights_10_28_42_20.xlsx",
):
    frames = []
    for dataset, experiment_family, experiment_hash, metrics_path in discover_metric_files(
        results_root,
        model_to_explain,
        experiment_families,
        datasets=datasets,
        weights_file=weights_file,
    ):
        experiment_dir = metrics_path.parent
        metrics_df = pd.read_excel(metrics_path, sheet_name="metrics")
        metrics_df = normalize_metric_columns(metrics_df)
        params = read_experiment_params(experiment_dir, metrics_path)

        metrics_df["dataset"] = metrics_df.get("dataset", dataset)
        metrics_df["model_to_explain"] = metrics_df.get("model_to_explain", model_to_explain)
        metrics_df["experiment_family"] = metrics_df.get("experiment_family", experiment_family)
        metrics_df["experiment_hash"] = metrics_df.get("experiment_hash", experiment_hash)
        metrics_df["metrics_path"] = str(metrics_path)
        metrics_df["experiment_dir"] = str(experiment_dir)

        for key, value in params.items():
            if key not in metrics_df.columns:
                metrics_df[key] = value

        frames.append(metrics_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def infer_parameter_columns(instance_df, metric_columns, extra_exclude=None):
    extra_exclude = set() if extra_exclude is None else set(extra_exclude)
    excluded = DEFAULT_METADATA_COLUMNS | set(metric_columns) | extra_exclude
    suffixes = ("_mean", "_std", "_median", "_rank")
    candidates = []
    for column in instance_df.columns:
        if column in excluded or column.endswith(suffixes):
            continue
        if instance_df[column].nunique(dropna=True) > 1:
            candidates.append(column)
    return candidates


def summarize_hash_metrics(instance_df, metric_columns, parameter_columns):
    available_metrics = [column for column in metric_columns if column in instance_df.columns]
    group_columns = [
        "dataset",
        "model_to_explain",
        "experiment_family",
        "experiment_hash",
        *[column for column in parameter_columns if column in instance_df.columns],
    ]
    group_columns = list(dict.fromkeys(group_columns))

    grouped = instance_df.groupby(group_columns, dropna=False)
    summary = grouped[available_metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_instances"] = grouped.size().to_numpy()

    if "validity" in instance_df.columns:
        summary["n_valid_cfs"] = grouped["validity"].sum().to_numpy()
    elif "valid" in instance_df.columns:
        summary["n_valid_cfs"] = grouped["valid"].sum().to_numpy()

    return summary


def add_parameter_combination(df, parameter_columns, combination_col="parameter_combination"):
    df = df.copy()
    parameter_columns = [column for column in parameter_columns if column in df.columns]
    if not parameter_columns:
        df[combination_col] = "all"
        return df

    def build_label(row):
        parts = []
        for column in parameter_columns:
            value = row[column]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            parts.append(f"{column}={value}")
        return " | ".join(parts)

    df[combination_col] = df.apply(build_label, axis=1)
    return df


def dataset_combination_means(hash_df, metric_mean_column, combination_col="parameter_combination"):
    return (
        hash_df.groupby(["dataset", combination_col], dropna=False)[metric_mean_column]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{metric_mean_column}_dataset_mean",
                "std": f"{metric_mean_column}_dataset_std",
                "median": f"{metric_mean_column}_dataset_median",
                "count": "n_hashes",
            }
        )
    )


def rank_combinations_across_datasets(dataset_combo_df, metric_dataset_mean_column, higher_is_better=True):
    ascending = not higher_is_better
    ranked = dataset_combo_df.copy()
    ranked["dataset_rank"] = ranked.groupby("dataset")[metric_dataset_mean_column].rank(
        method="average",
        ascending=ascending,
    )
    return ranked


def aggregate_combination_rule(
    dataset_combo_df,
    metric_dataset_mean_column,
    combination_col="parameter_combination",
    higher_is_better=True,
):
    ranking = (
        dataset_combo_df.groupby(combination_col, dropna=False)[metric_dataset_mean_column]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_across_dataset_means",
                "std": "stability_std_across_datasets",
                "median": "median_across_dataset_means",
                "count": "n_datasets",
            }
        )
    )
    ranking = ranking.sort_values(
        ["mean_across_dataset_means", "stability_std_across_datasets"],
        ascending=[not higher_is_better, True],
    ).reset_index(drop=True)
    ranking["overall_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def combination_metric_table(dataset_combo_df, value_column, combination_col="parameter_combination"):
    return dataset_combo_df.pivot(index="dataset", columns=combination_col, values=value_column)


def parameter_heatmap_data(hash_df, param_x, param_y, metric_mean_column):
    return (
        hash_df.groupby([param_y, param_x], dropna=False)[metric_mean_column]
        .mean()
        .reset_index()
        .pivot(index=param_y, columns=param_x, values=metric_mean_column)
    )
