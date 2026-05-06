import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.evaluation.evaluation_utils import calculate_method_metrics
from methods.outlier_calculators import AEOutlierCalculator, IFOutlierCalculator, LOFOutlierCalculator


DEFAULT_METRIC_DIRECTIONS = {
    "valid": "max",
    "validity": "max",
    "selected utility score": "max",
    "IoN": "max",
    "proximity": "min",
    "L1": "min",
    "L2": "min",
    "sparsity": "min",
    "NoS": "min",
    "subsequences": "min",
    "contiguity": "min",
    "subsequences %": "min",
    "(sparsity + subsequences %) / 2": "min",
    "AE_OS": "min",
    "AE_IOS": "min",
    "IF_OS": "min",
    "IF_IOS": "min",
    "LOF_OS": "min",
    "LOF_IOS": "min",
}


def make_model_weights_name(weights):
    weight_values = [str(int(round(float(value) * 100))) for value in weights.values()]
    return "model_weights_" + "_".join(weight_values)


def get_metrics_excel_filename(weights):
    return f"{make_model_weights_name(weights)}.xlsx"


def load_outlier_calculators(dataset, X_train, outlier_calculator_experiments):
    calculators = {}
    for calculator_name, experiment_name in outlier_calculator_experiments.items():
        model_folder = Path("experiments/models") / dataset / experiment_name
        try:
            if calculator_name == "AE":
                model_path = model_folder / "model.hdf5"
                if not model_path.is_file():
                    continue
                model = tf.keras.models.load_model(model_path, compile=False)
                calculators[calculator_name] = AEOutlierCalculator(model, X_train)
            elif calculator_name == "IF":
                model_path = model_folder / "model.pickle"
                if not model_path.is_file():
                    continue
                with open(model_path, "rb") as fp:
                    model = pickle.load(fp)
                calculators[calculator_name] = IFOutlierCalculator(model, X_train)
            elif calculator_name == "LOF":
                model_path = model_folder / "model.pickle"
                if not model_path.is_file():
                    continue
                with open(model_path, "rb") as fp:
                    model = pickle.load(fp)
                calculators[calculator_name] = LOFOutlierCalculator(model, X_train)
        except Exception as exc:
            print(f"Skipping {calculator_name} outlier calculator for {dataset}: {exc}")
    return calculators


def create_metrics_summary(metrics_df):
    summary = {
        "n_evaluated_instances": len(metrics_df),
        "n_valid_cfs": int(metrics_df["valid"].fillna(False).sum()) if "valid" in metrics_df else np.nan,
    }
    excluded_cols = {
        "ii", "dataset", "model_to_explain", "experiment_family", "experiment_hash",
        "original_class", "true_class", "target_class", "pred_class", "method",
    }
    metric_cols = [
        col for col in metrics_df.select_dtypes(include=[np.number, bool]).columns
        if col not in excluded_cols and not col.endswith("_rank")
    ]
    for col in metric_cols:
        values = pd.to_numeric(metrics_df[col], errors="coerce")
        summary[f"{col}_mean"] = values.mean()
        summary[f"{col}_std"] = values.std()
        summary[f"{col}_median"] = values.median()
    return pd.DataFrame([summary])


def add_summary_ranks(summary_df, family_path, excel_filename, current_exp_name, metric_directions=None):
    metric_directions = metric_directions or DEFAULT_METRIC_DIRECTIONS
    summaries = []
    current_summary = summary_df.copy()
    current_summary["experiment_hash"] = current_exp_name
    summaries.append(current_summary)

    for sibling_dir in family_path.iterdir() if family_path.is_dir() else []:
        if not sibling_dir.is_dir() or sibling_dir.name == current_exp_name:
            continue
        sibling_metrics_path = sibling_dir / excel_filename
        if not sibling_metrics_path.is_file():
            continue
        try:
            sibling_summary = pd.read_excel(sibling_metrics_path, sheet_name="summary")
        except Exception as exc:
            print(f"Skipping ranks from {sibling_metrics_path}: {exc}")
            continue
        if "experiment_hash" not in sibling_summary:
            sibling_summary["experiment_hash"] = sibling_dir.name
        summaries.append(sibling_summary)

    rank_df = pd.concat(summaries, ignore_index=True)
    for metric, direction in metric_directions.items():
        metric_col = f"{metric}_mean"
        if metric_col not in rank_df:
            continue
        rank_col = f"{metric}_rank"
        rank_df[rank_col] = pd.to_numeric(rank_df[metric_col], errors="coerce").rank(
            ascending=(direction == "min"),
            method="average",
        )

    current_ranked = rank_df[rank_df["experiment_hash"] == current_exp_name].tail(1)
    rank_cols = [col for col in current_ranked.columns if col.endswith("_rank")]
    summary_df = summary_df.copy()
    for col in rank_cols:
        summary_df[col] = current_ranked.iloc[0][col]
    return summary_df


def create_params_sheet(params, dataset, exp_name, experiment_family, result_path, excel_path,
                        model_to_explain, mo_eval_weights):
    rows = {
        **params,
        "dataset": dataset,
        "model_to_explain": model_to_explain,
        "experiment_family": experiment_family,
        "experiment_hash": exp_name,
        "counterfactuals_path": str(result_path / "counterfactuals.pickle"),
        "metrics_excel_path": str(excel_path),
        "MO_EVAL_WEIGHTS": json.dumps(mo_eval_weights, sort_keys=True),
    }
    return pd.DataFrame([{"parameter": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value}
                         for key, value in rows.items()])


def create_family_summary_records(family_path, excel_filename):
    records = []
    for experiment_dir in sorted(family_path.iterdir()) if family_path.is_dir() else []:
        if not experiment_dir.is_dir():
            continue
        metrics_path = experiment_dir / excel_filename
        params_path = experiment_dir / "params.json"
        if not metrics_path.is_file() or not params_path.is_file():
            continue

        try:
            summary_df = pd.read_excel(metrics_path, sheet_name="summary")
        except Exception as exc:
            print(f"Skipping family summary row from {metrics_path}: {exc}")
            continue
        if summary_df.empty:
            continue

        with open(params_path, "r") as fp:
            params = json.load(fp)

        row = summary_df.iloc[0].to_dict()
        row["experiment_hash"] = experiment_dir.name
        row.update(params)
        records.append(row)
    return records


def save_family_summary_excel(family_path, excel_filename):
    records = create_family_summary_records(family_path, excel_filename)
    if not records:
        return None

    family_df = pd.DataFrame.from_records(records)
    sort_col = "IoN_mean" if "IoN_mean" in family_df.columns else None
    if sort_col is not None:
        family_df = family_df.sort_values(sort_col, ascending=False, na_position="last")

    output_path = family_path / f"family_summary_{excel_filename}"
    with pd.ExcelWriter(output_path) as writer:
        family_df.to_excel(writer, sheet_name="experiments", index=False)

    print(f"Family summary Excel path: {output_path}")
    return output_path


def save_metrics_excel(dataset, exp_name, params, experiment_family, result_path,
                       X_train, X_test, y_test, y_pred_test, subset_idx, nuns, model_wrapper,
                       model_to_explain, mo_eval_weights, outlier_calculator_experiments,
                       metric_directions=None):
    weights_array = np.array(list(mo_eval_weights.values()))
    excel_filename = get_metrics_excel_filename(mo_eval_weights)
    excel_path = result_path / excel_filename

    with open(result_path / "counterfactuals.pickle", "rb") as fp:
        results = pickle.load(fp)

    outlier_calculators = load_outlier_calculators(dataset, X_train, outlier_calculator_experiments)
    if not outlier_calculators:
        outlier_calculators = None

    metrics_df = calculate_method_metrics(
        model_wrapper,
        outlier_calculators,
        X_test,
        nuns,
        results,
        y_pred_test,
        exp_name,
        mo_weights=weights_array,
    )
    metrics_df.insert(0, "ii", subset_idx.tolist())
    metrics_df.insert(0, "experiment_hash", exp_name)
    metrics_df.insert(0, "experiment_family", experiment_family)
    metrics_df.insert(0, "model_to_explain", model_to_explain)
    metrics_df.insert(0, "dataset", dataset)
    metrics_df["original_class"] = y_pred_test
    metrics_df["true_class"] = y_test
    metrics_df["validity"] = metrics_df["valid"].astype(float)
    metrics_df["proximity"] = metrics_df["L2"]
    metrics_df["NoS"] = metrics_df["subsequences"]
    metrics_df["contiguity"] = metrics_df["subsequences %"]

    summary_df = create_metrics_summary(metrics_df)
    summary_df.insert(0, "experiment_hash", exp_name)
    summary_df.insert(0, "experiment_family", experiment_family)
    summary_df.insert(0, "model_to_explain", model_to_explain)
    summary_df.insert(0, "dataset", dataset)
    summary_df = add_summary_ranks(
        summary_df,
        result_path.parent,
        excel_filename,
        exp_name,
        metric_directions=metric_directions,
    )
    params_df = create_params_sheet(
        params,
        dataset,
        exp_name,
        experiment_family,
        result_path,
        excel_path,
        model_to_explain,
        mo_eval_weights,
    )

    with pd.ExcelWriter(excel_path) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        params_df.to_excel(writer, sheet_name="params", index=False)

    print(f"Metrics Excel path: {excel_path}")
    return excel_path
