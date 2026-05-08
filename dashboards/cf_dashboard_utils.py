from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".npz",
    ".npy",
    ".json",
    ".csv",
    ".xlsx",
    ".parquet",
}

PRIORITY_PATH_PARTS = {
    "results",
    "outputs",
    "counterfactuals",
    "counterfactual",
    "cf",
    "multispace",
    "multi-space",
    "multisubspace",
    "experiments",
}

ARRAY_KEYS = {
    "x_orig": ["x_orig", "orig", "original", "x_original", "instance", "query"],
    "x_cf": ["x_cf", "cf", "counterfactual", "counterfactuals"],
    "x_cfs": ["cfs", "x_cfs", "counterfactuals"],
    "x_nun": ["nun", "x_nun", "nun_example", "nearest_unlike_neighbor"],
    "mask": ["mask", "change_mask", "cf_mask", "counterfactual_mask"],
}

LABEL_KEYS = {
    "y_true": ["y_true", "true_label", "x_orig_true_label", "label"],
    "predicted_orig": ["predicted_orig", "x_orig_pred_label", "orig_pred", "pred_orig"],
    "predicted_cf": ["predicted_cf", "cf_pred_label", "pred_class", "predicted_cls"],
    "predicted_cfs": ["cf_pred_labels", "predicted_cfs", "pred_classes"],
    "target_class": ["target_class", "desired_target", "desired_class", "nun_class"],
    "nun_idx": ["nun_idx", "nun_index", "nun_id"],
    "validity": ["validity", "valid", "is_valid"],
}

OBJECTIVE_NAMES = {
    "sparsity",
    "subsequences",
    "subsequences %",
    "L0",
    "L1",
    "L2",
    "IoS",
    "IOS",
    "OS",
    "AE_OS",
    "AE_IOS",
    "IF_OS",
    "IF_IOS",
    "LOF_OS",
    "LOF_IOS",
    "fitness",
    "utility",
    "proba",
    "probability",
    "desired_class_prob",
    "target_probability",
    "validity",
    "valid",
}

UTILITY_DEPENDENT_METRIC_COLUMNS = {
    "selected utility score",
    "IoN",
}

INSTANCE_LINK_COLUMNS = {
    "dataset",
    "model_to_explain",
    "experiment_family",
    "experiment_hash",
    "ii",
    "method",
    "best cf index",
}


def discover_candidate_files(root: str | Path = ".", limit: int = 500) -> list[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []

    if root_path.is_file():
        return [root_path] if root_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    candidates: list[Path] = []
    for path in root_path.rglob("*"):
        if len(candidates) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.startswith("~$"):
            continue
        candidates.append(path)

    def score(path: Path) -> tuple[int, str]:
        parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        priority = len(parts & PRIORITY_PATH_PARTS)
        if name in {"counterfactuals.pickle", "counterfactuals.pkl"}:
            priority += 5
        if "metric" in name or "weight" in name:
            priority += 1
        return -priority, str(path)

    return sorted(candidates, key=score)


def list_result_datasets(results_root: str | Path = "experiments/results") -> list[str]:
    root_path = Path(results_root).expanduser()
    if not root_path.is_dir():
        return []
    return sorted(path.name for path in root_path.iterdir() if path.is_dir())


def list_result_subfolders(dataset: str, results_root: str | Path = "experiments/results") -> list[str]:
    dataset_path = Path(results_root).expanduser() / dataset
    if not dataset_path.is_dir():
        return []
    return sorted(path.name for path in dataset_path.iterdir() if path.is_dir())


def list_experiment_options(
    dataset: str,
    subfolder: str,
    results_root: str | Path = "experiments/results",
) -> dict[str, list[str]]:
    subfolder_path = Path(results_root).expanduser() / dataset / subfolder
    if not subfolder_path.is_dir():
        return {"direct_experiments": [], "families": []}

    direct_experiments: list[str] = []
    families: list[str] = []
    for child in sorted(path for path in subfolder_path.iterdir() if path.is_dir()):
        if is_experiment_dir(child):
            direct_experiments.append(child.name)
            continue
        if any(is_experiment_dir(grandchild) for grandchild in child.iterdir() if grandchild.is_dir()):
            families.append(child.name)

    return {"direct_experiments": direct_experiments, "families": families}


def list_family_experiments(
    dataset: str,
    subfolder: str,
    family: str,
    results_root: str | Path = "experiments/results",
) -> list[str]:
    family_path = Path(results_root).expanduser() / dataset / subfolder / family
    if not family_path.is_dir():
        return []
    return sorted(path.name for path in family_path.iterdir() if path.is_dir() and is_experiment_dir(path))


def build_result_file_path(
    dataset: str,
    subfolder: str,
    experiment: str,
    family: str | None = None,
    filename: str = "counterfactuals.pickle",
    results_root: str | Path = "experiments/results",
) -> Path:
    root_path = Path(results_root).expanduser()
    if family:
        return root_path / dataset / subfolder / family / experiment / filename
    return root_path / dataset / subfolder / experiment / filename


def is_experiment_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "counterfactuals.pickle").is_file() or (path / "counterfactuals.pkl").is_file()


def load_result_file(path: str | Path) -> Any:
    source_path = Path(path).expanduser()
    suffix = source_path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with source_path.open("rb") as fp:
            return pickle.load(fp)
    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load this file.") from exc
        return joblib.load(source_path)
    if suffix == ".npz":
        loaded = np.load(source_path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    if suffix == ".npy":
        return np.load(source_path, allow_pickle=True)
    if suffix == ".json":
        with source_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix == ".xlsx":
        return pd.read_excel(source_path)
    if suffix == ".parquet":
        return pd.read_parquet(source_path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def find_nearby_params(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).expanduser()
    starts = [source_path.parent] if source_path.is_file() else [source_path]
    starts.extend(starts[0].parents)
    for parent in starts[:8]:
        params_path = parent / "params.json"
        if params_path.is_file():
            try:
                with params_path.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                return {}
    return {}


def find_nearby_metrics_workbook(path: str | Path) -> Path | None:
    source_path = Path(path).expanduser()
    starts = [source_path.parent] if source_path.is_file() else [source_path]
    starts.extend(starts[0].parents)
    for parent in starts[:4]:
        local = sorted(parent.glob("model_weights*.xlsx"))
        if local:
            return local[0]
    return None


def load_saved_instance_metrics(path: str | Path) -> dict[int, dict[str, Any]]:
    workbook = find_nearby_metrics_workbook(path)
    if workbook is None or not workbook.is_file():
        return {}

    try:
        metrics_df = pd.read_excel(workbook, sheet_name="metrics")
    except Exception:
        return {}

    if "ii" not in metrics_df.columns:
        return {}

    metrics_by_instance: dict[int, dict[str, Any]] = {}
    for _, row in metrics_df.iterrows():
        row_dict = row.dropna().to_dict()
        ii = row_dict.get("ii")
        try:
            ii_key = int(ii)
        except Exception:
            continue
        metrics_by_instance[ii_key] = row_dict
    return metrics_by_instance


def extract_saved_candidate_metrics(metrics_row: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    candidate_metrics: dict[str, float] = {}
    instance_metadata: dict[str, Any] = {}
    for key, value in metrics_row.items():
        if key in INSTANCE_LINK_COLUMNS or key in UTILITY_DEPENDENT_METRIC_COLUMNS:
            continue
        scalar = scalarize(value)
        if isinstance(scalar, (int, float, np.number, bool)) and not pd.isna(scalar):
            candidate_metrics[str(key)] = float(scalar)
        elif scalar is not None and not (isinstance(scalar, float) and np.isnan(scalar)):
            instance_metadata[str(key)] = scalar
    return candidate_metrics, instance_metadata


def infer_metadata_from_path(path: str | Path, params: dict[str, Any] | None = None) -> dict[str, Any]:
    source_path = Path(path).expanduser()
    params = params or {}
    parts = list(source_path.parts)
    metadata: dict[str, Any] = {"source_path": str(source_path)}

    if "results" in parts:
        result_i = parts.index("results")
        if len(parts) > result_i + 1:
            metadata["dataset_name"] = parts[result_i + 1]
        if len(parts) > result_i + 2:
            metadata["model_name"] = parts[result_i + 2]
        relative_parts = parts[result_i + 3 :]
        if len(relative_parts) >= 3:
            metadata["experiment_family"] = relative_parts[0]
            metadata["method_name"] = relative_parts[1]
            metadata["experiment_name"] = relative_parts[2]
        elif len(relative_parts) >= 2:
            metadata["method_name"] = relative_parts[0]
            metadata["experiment_name"] = relative_parts[1]

    metadata.setdefault("dataset_name", params.get("dataset"))
    metadata.setdefault("method_name", params.get("method") or params.get("method_name"))
    if params.get("metadata"):
        metadata["experiment_metadata"] = params["metadata"]
    return metadata


def load_companion_data(path: str | Path, metadata: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    dataset = metadata.get("dataset_name") or params.get("dataset")
    if not dataset:
        return {}

    data_dir = find_dataset_data_dir(str(dataset))
    if not data_dir.is_dir():
        return {}

    companion: dict[str, Any] = {}
    try:
        scaling = params.get("scaling", "none")
        X_test = np.load(data_dir / "X_test.npy", allow_pickle=True)
        X_train = np.load(data_dir / "X_train.npy", allow_pickle=True)
        if scaling in {"none", None}:
            companion["X_train"] = X_train
            companion["X_test"] = X_test
        else:
            companion["X_train"] = apply_scaling(X_train, X_train, scaling)
            companion["X_test"] = apply_scaling(X_train, X_test, scaling)
    except Exception as exc:
        companion["error"] = f"Could not load companion X arrays from {data_dir}: {exc}"
        return companion

    try:
        y_test = np.load(data_dir / "y_test.npy", allow_pickle=True)
        y_train_path = data_dir / "y_train.npy"
        if y_train_path.is_file():
            try:
                y_train = np.load(y_train_path, allow_pickle=True)
                y_train, y_test = encode_labels(y_train, y_test)
                companion["y_train"] = y_train
            except Exception:
                pass
        companion["y_test"] = y_test
    except Exception as exc:
        companion["label_error"] = f"Could not load companion labels from {data_dir}: {exc}"

    return companion


def find_dataset_data_dir(dataset: str) -> Path:
    candidates = [
        Path("experiments/data/UCR") / dataset,
        Path("experiments/data") / dataset,
        Path("data/UCR") / dataset,
        Path("data") / dataset,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def apply_scaling(X_train: np.ndarray, X_test: np.ndarray, scaling: str) -> np.ndarray:
    if scaling == "none" or scaling is None:
        return X_test
    if scaling == "standard":
        std = X_train.std()
        return (X_test - X_train.mean()) / std if std != 0 else X_test - X_train.mean()
    if scaling == "min_max":
        data_min = X_train.min()
        data_max = X_train.max()
        denom = data_max - data_min
        return (X_test - data_min) / denom if denom != 0 else X_test - data_min
    return X_test


def encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from experiments.data_utils import label_encoder
    except ImportError:
        from data_utils import label_encoder

    return label_encoder(y_train, y_test)


def normalize_loaded_object(
    raw: Any,
    source_path: str | Path,
    params: dict[str, Any] | None = None,
    load_companion: bool = False,
) -> dict[str, Any]:
    params = params or find_nearby_params(source_path)
    metadata = infer_metadata_from_path(source_path, params)
    companion = load_companion_data(source_path, metadata, params) if load_companion else {}
    instances = extract_instances(raw, params=params, companion=companion)

    return {
        "source_path": str(source_path),
        "dataset_name": metadata.get("dataset_name"),
        "method_name": metadata.get("method_name"),
        "metadata": {
            **metadata,
            "params": params,
            "companion_data": summarize_companion(companion),
        },
        "instances": instances,
        "raw": raw,
    }


def extract_instances(
    raw: Any,
    params: dict[str, Any],
    companion: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(raw, pd.DataFrame):
        return instances_from_dataframe(raw)

    if isinstance(raw, dict):
        direct = instance_from_mapping(raw, 0, params, companion, None)
        if direct["pareto"] or direct.get("x_orig") is not None:
            return [direct]

        instances = []
        for key, value in raw.items():
            if isinstance(value, (dict, list, tuple, np.ndarray, pd.DataFrame)):
                instances.append(instance_from_any(value, len(instances), params, companion, key))
        return instances

    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            return extract_instances(raw.tolist(), params, companion)
        return [instance_from_mapping({"cfs": raw}, 0, params, companion, None)]

    if isinstance(raw, (list, tuple)):
        return [instance_from_any(item, i, params, companion, None) for i, item in enumerate(raw)]

    return [instance_from_mapping({"raw_value": raw}, 0, params, companion, None)]


def instance_from_any(
    value: Any,
    position: int,
    params: dict[str, Any],
    companion: dict[str, Any],
    key: Any,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return instance_from_mapping(value, position, params, companion, key)
    if isinstance(value, pd.Series):
        return instance_from_mapping(value.to_dict(), position, params, companion, key)
    if isinstance(value, np.ndarray):
        return instance_from_mapping({"cfs": value}, position, params, companion, key)
    if isinstance(value, (list, tuple)):
        return instance_from_mapping({"pareto": value}, position, params, companion, key)
    return instance_from_mapping({"raw_value": value}, position, params, companion, key)


def instance_from_mapping(
    item: dict[str, Any],
    position: int,
    params: dict[str, Any],
    companion: dict[str, Any],
    key: Any,
) -> dict[str, Any]:
    original_indexes = params.get("X_test_indexes")
    instance_id = value_from_keys(item, ["instance_id", "original_index", "idx", "index"])
    if instance_id is None and key is not None and not isinstance(key, str):
        instance_id = key
    if instance_id is None and isinstance(original_indexes, list) and position < len(original_indexes):
        instance_id = original_indexes[position]
    if instance_id is None:
        instance_id = position

    x_orig = first_array(item, ARRAY_KEYS["x_orig"])
    x_nun = first_array(item, ARRAY_KEYS["x_nun"])
    if x_orig is None:
        x_orig = companion_x(companion, "X_test", instance_id)
    if x_orig is None:
        x_orig = companion_position_x(companion, "X_test", position)
    if x_nun is None:
        x_nun = companion_position_x(companion, "X_nun", position)

    y_true = value_from_keys(item, LABEL_KEYS["y_true"])
    if y_true is None:
        y_true = companion_y(companion, instance_id)
    if y_true is None:
        y_true = companion_position_y(companion, position)

    predicted_orig = value_from_keys(item, LABEL_KEYS["predicted_orig"])
    target_class = value_from_keys(item, LABEL_KEYS["target_class"])
    nun_idx = value_from_keys(item, LABEL_KEYS["nun_idx"])

    pareto = extract_pareto(item, x_orig=x_orig)
    predicted_cfs = value_from_keys(item, LABEL_KEYS["predicted_cfs"])
    if predicted_cfs is not None:
        for candidate, pred in zip(pareto, np.asarray(predicted_cfs).reshape(-1)):
            candidate["predicted_cf"] = scalarize(pred)

    return {
        "instance_id": scalarize(instance_id),
        "position": position,
        "x_orig": normalize_series_array(x_orig),
        "x_nun": normalize_series_array(x_nun),
        "y_true": scalarize(y_true),
        "predicted_orig": scalarize(predicted_orig),
        "target_class": scalarize(target_class),
        "nun_idx": scalarize(nun_idx),
        "metadata": metadata_from_mapping(item),
        "pareto": pareto,
        "raw": item,
    }


def extract_pareto(item: dict[str, Any], x_orig: np.ndarray | None = None) -> list[dict[str, Any]]:
    candidates_raw = value_from_keys(item, ["pareto", "candidates", "front", "solutions"])
    if candidates_raw is not None and isinstance(candidates_raw, (list, tuple)):
        return [candidate_from_any(candidate, i, x_orig) for i, candidate in enumerate(candidates_raw)]

    cfs = first_array(item, ARRAY_KEYS["x_cfs"])
    if cfs is None:
        cf = first_array(item, ARRAY_KEYS["x_cf"])
        cfs = np.expand_dims(cf, axis=0) if cf is not None and np.asarray(cf).ndim <= 2 else cf

    if cfs is None:
        return []

    cfs_arr = np.asarray(cfs)
    if cfs_arr.ndim <= 2:
        cfs_arr = np.expand_dims(cfs_arr, axis=0)

    candidates = []
    for i, x_cf in enumerate(cfs_arr):
        candidate = candidate_from_mapping(item, i, x_orig)
        candidate["x_cf"] = normalize_series_array(x_cf)
        enrich_candidate_metrics(candidate, x_orig)
        candidates.append(candidate)
    return candidates


def candidate_from_any(candidate: Any, candidate_id: int, x_orig: np.ndarray | None) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate_from_mapping(candidate, candidate_id, x_orig)
    return candidate_from_mapping({"x_cf": candidate}, candidate_id, x_orig)


def candidate_from_mapping(item: dict[str, Any], candidate_id: int, x_orig: np.ndarray | None) -> dict[str, Any]:
    x_cf = first_array(item, ARRAY_KEYS["x_cf"])
    mask = first_array(item, ARRAY_KEYS["mask"])
    predicted_cf = value_from_keys(item, LABEL_KEYS["predicted_cf"])
    validity = value_from_keys(item, LABEL_KEYS["validity"])

    objectives = numeric_fields(item, OBJECTIVE_NAMES)
    metrics = numeric_fields(item, set(item.keys()) - set(objectives.keys()))
    candidate = {
        "candidate_id": candidate_id,
        "x_cf": normalize_series_array(x_cf),
        "mask": normalize_series_array(mask),
        "predicted_cf": scalarize(predicted_cf),
        "validity": scalarize(validity),
        "objectives": objectives,
        "metrics": metrics,
        "metadata": metadata_from_mapping(item),
        "raw": item,
    }
    enrich_candidate_metrics(candidate, x_orig)
    return candidate


def enrich_candidate_metrics(candidate: dict[str, Any], x_orig: np.ndarray | None) -> None:
    x_cf = candidate.get("x_cf")
    if x_orig is None or x_cf is None:
        return

    x_orig_arr = normalize_series_array(x_orig)
    x_cf_arr = normalize_series_array(x_cf)
    if x_orig_arr is None or x_cf_arr is None or x_orig_arr.shape != x_cf_arr.shape:
        return

    diff = x_cf_arr - x_orig_arr
    mask = candidate.get("mask")
    if mask is None:
        mask = np.abs(diff) > 1e-12
        candidate["mask"] = mask.astype(int)
    mask_arr = np.asarray(mask).astype(bool)

    inferred = {
        "L1": float(np.linalg.norm(diff.reshape(-1), ord=1)),
        "L2": float(np.linalg.norm(diff.reshape(-1), ord=2)),
        "L0": int(mask_arr.sum()),
        "sparsity": float(mask_arr.sum() / mask_arr.size) if mask_arr.size else np.nan,
        "subsequences": int(np.count_nonzero(np.diff(mask_arr.astype(int), prepend=0, axis=0) == 1)),
        "subsequences %": float(np.count_nonzero(np.diff(mask_arr.astype(int), prepend=0, axis=0) == 1) / mask_arr.size) if mask_arr.size else np.nan,
        "proximity": float(np.linalg.norm(diff.reshape(-1), ord=2)),
        "NoS": int(np.count_nonzero(np.diff(mask_arr.astype(int), prepend=0, axis=0) == 1)),
        "contiguity": float(np.count_nonzero(np.diff(mask_arr.astype(int), prepend=0, axis=0) == 1) / mask_arr.size) if mask_arr.size else np.nan,
    }
    for key, value in inferred.items():
        candidate["metrics"].setdefault(key, value)
        if key in {"L1", "L2", "sparsity", "subsequences", "subsequences %", "proximity", "NoS", "contiguity"}:
            candidate["objectives"].setdefault(key, value)


def instances_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    instance_col = first_existing_column(df, ["instance_id", "ii", "index", "original_index"])
    if instance_col is None:
        grouped = [(0, df)]
    else:
        grouped = list(df.groupby(instance_col, sort=False))

    instances = []
    for instance_id, group in grouped:
        pareto = []
        for i, (_, row) in enumerate(group.iterrows()):
            row_dict = row.dropna().to_dict()
            pareto.append(candidate_from_mapping(row_dict, i, None))
        first_row = group.iloc[0].dropna().to_dict()
        instances.append(
            {
                "instance_id": scalarize(instance_id),
                "position": len(instances),
                "x_orig": None,
                "x_nun": None,
                "y_true": scalarize(value_from_keys(first_row, LABEL_KEYS["y_true"])),
                "predicted_orig": scalarize(value_from_keys(first_row, LABEL_KEYS["predicted_orig"])),
                "target_class": scalarize(value_from_keys(first_row, LABEL_KEYS["target_class"])),
                "nun_idx": scalarize(value_from_keys(first_row, LABEL_KEYS["nun_idx"])),
                "metadata": metadata_from_mapping(first_row),
                "pareto": pareto,
                "raw": group,
            }
        )
    return instances


def candidate_table(instance: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for candidate in instance.get("pareto", []):
        row = {
            "candidate_id": candidate.get("candidate_id"),
            "predicted_cf": candidate.get("predicted_cf"),
            "validity": candidate.get("validity"),
        }
        row.update(candidate.get("objectives", {}))
        for key, value in candidate.get("metrics", {}).items():
            row.setdefault(key, value)
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_instance_with_evaluation_metrics(instance: dict[str, Any], source_path: str | Path) -> dict[str, Any]:
    context = get_evaluation_context(str(source_path))
    companion = context.get("companion", {})

    x_orig = normalize_series_array(instance.get("x_orig"))
    if x_orig is None:
        x_orig = companion_x(companion, "X_test", instance.get("instance_id"))
    if x_orig is None:
        x_orig = companion_position_x(companion, "X_test", int(instance.get("position", 0)))
    if x_orig is None:
        return instance
    instance["x_orig"] = x_orig

    x_nun = normalize_series_array(instance.get("x_nun"))
    model_wrapper = context["model_wrapper"]
    predicted_orig = instance.get("predicted_orig")
    if model_wrapper is not None and predicted_orig is None:
        predicted_orig = int(np.argmax(model_wrapper.predict(np.expand_dims(x_orig, axis=0)), axis=1)[0])
        instance["predicted_orig"] = predicted_orig

    if x_nun is None and model_wrapper is not None and predicted_orig is not None:
        x_nun, nun_idx = retrieve_global_nun_for_dashboard(context, x_orig, int(predicted_orig))
        if x_nun is not None:
            instance["x_nun"] = x_nun
            instance["nun_idx"] = nun_idx

    target_class = instance.get("target_class")
    if model_wrapper is not None and target_class is None and x_nun is not None:
        target_class = int(np.argmax(model_wrapper.predict(np.expand_dims(x_nun, axis=0)), axis=1)[0])
        instance["target_class"] = target_class

    outlier_calculators = context.get("outlier_calculators", {})
    original_outlier_scores = {
        name: calculator.get_outlier_scores(np.expand_dims(x_orig, axis=0))[0]
        for name, calculator in outlier_calculators.items()
    }

    for candidate in instance.get("pareto", []):
        x_cf = normalize_series_array(candidate.get("x_cf"))
        if x_cf is None or x_cf.shape != x_orig.shape:
            continue

        predicted_probs = None
        if model_wrapper is not None:
            predicted_probs = model_wrapper.predict(np.expand_dims(x_cf, axis=0))
            predicted_cf = int(np.argmax(predicted_probs, axis=1)[0])
            candidate["predicted_cf"] = predicted_cf
            if predicted_orig is not None:
                candidate["validity"] = float(predicted_cf != predicted_orig)

        change_mask = calculate_change_mask_local(x_orig, x_cf, x_nun=x_nun)
        diff = x_cf - x_orig
        mask_arr = np.asarray(change_mask).astype(bool)
        l1 = float(np.linalg.norm(diff.reshape(-1), ord=1))
        l2 = float(np.linalg.norm(diff.reshape(-1), ord=2))
        sparsity = float(mask_arr.sum() / mask_arr.size) if mask_arr.size else np.nan
        subsequences = int(np.count_nonzero(np.diff(mask_arr.astype(int), prepend=0, axis=0) == 1))
        subsequences_pct = float(subsequences / (mask_arr.size / 2)) if mask_arr.size else np.nan

        candidate["metrics"].update(
            {
                "L1": l1,
                "L2": l2,
                "L0": int(mask_arr.sum()),
                "sparsity": sparsity,
                "subsequences": subsequences,
                "subsequences %": subsequences_pct,
            }
        )
        if candidate.get("validity") is not None:
            candidate["metrics"]["validity"] = candidate["validity"]
        if candidate.get("predicted_cf") is not None:
            candidate["metrics"]["predicted_cf"] = candidate["predicted_cf"]
        candidate["objectives"].update(
            {
                "L1": l1,
                "L2": l2,
                "sparsity": sparsity,
                "subsequences": subsequences,
                "subsequences %": subsequences_pct,
            }
        )
        if candidate.get("validity") is not None:
            candidate["objectives"]["validity"] = candidate["validity"]

        if predicted_probs is not None and target_class is not None:
            desired_prob = float(predicted_probs[0, int(target_class)])
            candidate["metrics"]["desired_class_prob"] = desired_prob
            candidate["objectives"]["desired_class_prob"] = desired_prob

        for name, calculator in outlier_calculators.items():
            score = float(calculator.get_outlier_scores(np.expand_dims(x_cf, axis=0))[0])
            increase = max(0.0, score - float(original_outlier_scores[name]))
            candidate["metrics"][f"{name}_OS"] = score
            candidate["metrics"][f"{name}_IOS"] = increase
            candidate["objectives"][f"{name}_OS"] = score
            candidate["objectives"][f"{name}_IOS"] = increase

    return instance


def retrieve_global_nun_for_dashboard(
    context: dict[str, Any],
    x_orig: np.ndarray,
    predicted_orig: int,
) -> tuple[np.ndarray | None, int | None]:
    X_train = context.get("companion", {}).get("X_train")
    model_wrapper = context.get("model_wrapper")
    if X_train is None or model_wrapper is None:
        return None, None

    y_pred_train = context.get("y_pred_train")
    if y_pred_train is None:
        try:
            y_pred_train = np.argmax(model_wrapper.predict(X_train, batch_size=16), axis=1)
            context["y_pred_train"] = y_pred_train
        except Exception:
            return None, None

    unlike_indexes = np.flatnonzero(np.asarray(y_pred_train).reshape(-1) != predicted_orig)
    if unlike_indexes.size == 0:
        return None, None

    candidates = np.asarray(X_train)[unlike_indexes]
    distances = np.linalg.norm((candidates - x_orig).reshape(candidates.shape[0], -1), axis=1)
    best_position = int(np.argmin(distances))
    train_idx = int(unlike_indexes[best_position])
    return np.asarray(X_train[train_idx]), train_idx


@lru_cache(maxsize=8)
def get_evaluation_context(source_path: str) -> dict[str, Any]:
    source = Path(source_path)
    params = find_nearby_params(source)
    metadata = infer_metadata_from_path(source, params)
    companion = load_companion_data(source, metadata, params)
    model_wrapper = load_model_wrapper(metadata, companion)
    outlier_calculators = load_dashboard_outlier_calculators(metadata, companion)
    return {
        "params": params,
        "metadata": metadata,
        "companion": companion,
        "model_wrapper": model_wrapper,
        "outlier_calculators": outlier_calculators,
    }


def load_model_wrapper(metadata: dict[str, Any], companion: dict[str, Any]):
    dataset = metadata.get("dataset_name")
    model_name = metadata.get("model_name")
    X_train = companion.get("X_train")
    y_train = companion.get("y_train")
    if not dataset or not model_name or X_train is None or y_train is None:
        return None
    try:
        from experiments.experiment_utils import load_model
    except ImportError:
        return None

    ts_length = X_train.shape[0] if X_train.ndim == 2 else X_train.shape[1]
    n_channels = X_train.shape[1] if X_train.ndim == 2 else X_train.shape[2]
    n_classes = len(np.unique(y_train))
    try:
        return load_model(f"experiments/models/{dataset}/{model_name}", dataset, n_channels, ts_length, n_classes)
    except Exception:
        return None


def load_dashboard_outlier_calculators(metadata: dict[str, Any], companion: dict[str, Any]) -> dict[str, Any]:
    dataset = metadata.get("dataset_name")
    X_train = companion.get("X_train")
    if not dataset or X_train is None:
        return {}
    try:
        from experiments.metrics_excel import load_outlier_calculators
    except ImportError:
        return {}
    default_experiments = {"AE": "ae_basic_train", "IF": "if_basic_train", "LOF": "lof_basic_train"}
    try:
        return load_outlier_calculators(dataset, X_train, default_experiments)
    except Exception:
        return {}


def calculate_change_mask_local(x_orig: np.ndarray, x_cf: np.ndarray, x_nun: np.ndarray | None = None) -> np.ndarray:
    orig_change_mask = (x_orig != x_cf).astype(int)
    orig_change_mask = orig_change_mask.T.reshape(-1, 1)
    if x_nun is None:
        return orig_change_mask.reshape(x_orig.shape, order="F")

    cv_xorig_nun = (x_orig == x_nun)
    cv_nun_cf = (x_nun == x_cf)
    cv_all = (cv_xorig_nun & cv_nun_cf).astype(int)
    cv_all = cv_all.T.reshape(-1, 1)
    start_end_mask = cv_all & get_start_end_subsequence_positions_local(orig_change_mask).astype(int)
    noise = np.random.normal(0, 1e-6, x_orig.shape)
    new_x_orig = x_orig + noise * start_end_mask.reshape(x_orig.shape, order="F")
    return (new_x_orig != x_cf).astype(int)


def get_start_end_subsequence_positions_local(orig_change_mask: np.ndarray) -> np.ndarray:
    ones_mask = np.in1d(orig_change_mask, 1).reshape(orig_change_mask.shape)
    before_ones_mask = np.roll(ones_mask, -1, axis=0)
    before_ones_mask[ones_mask.shape[0] - 1, :] = False
    after_ones_mask = np.roll(ones_mask, 1, axis=0)
    after_ones_mask[0, :] = False
    before_after_ones_mask = before_ones_mask + after_ones_mask
    before_after_ones_mask[ones_mask] = False
    return before_after_ones_mask


def available_numeric_columns(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return []
    numeric = []
    for col in table.columns:
        if col == "candidate_id":
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        if values.notna().any():
            numeric.append(col)
    return numeric


def weighted_candidate(table: pd.DataFrame, weights: dict[str, float], maximize: bool = False) -> int | None:
    if table.empty or not weights:
        return None
    score = pd.Series(0.0, index=table.index)
    used = False
    for column, weight in weights.items():
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        span = values.max() - values.min()
        scaled = (values - values.min()) / span if span != 0 else values * 0
        score = score + (float(weight) * scaled.fillna(scaled.max() if not maximize else scaled.min()))
        used = True
    if not used:
        return None
    row_index = score.idxmax() if maximize else score.idxmin()
    return int(table.loc[row_index, "candidate_id"])


def normalize_series_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def value_from_keys(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    lowered = {str(key).lower(): key for key in mapping.keys()}
    for key in keys:
        real_key = lowered.get(key.lower())
        if real_key is not None:
            return mapping[real_key]
    return None


def first_array(mapping: dict[str, Any], keys: list[str]) -> np.ndarray | None:
    value = value_from_keys(mapping, keys)
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def numeric_fields(mapping: dict[str, Any], allowed: set[str]) -> dict[str, float]:
    output = {}
    for key, value in mapping.items():
        if str(key) not in allowed:
            continue
        if hasattr(value, "shape") and np.asarray(value).size > 1:
            continue
        if isinstance(value, (list, tuple, dict)):
            continue
        scalar = scalarize(value)
        if isinstance(scalar, (int, float, np.number, bool)) and not isinstance(scalar, str):
            output[str(key)] = float(scalar)
    return output


def metadata_from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key, value in mapping.items():
        if str(key) in {"raw", "cfs", "cf", "x_cf", "x_orig", "nun", "x_nun", "mask"}:
            continue
        if is_small_metadata_value(value):
            metadata[str(key)] = scalarize(value)
    return metadata


def is_small_metadata_value(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool, type(None), np.number)):
        return True
    arr = np.asarray(value) if hasattr(value, "__array__") else None
    return bool(arr is not None and arr.size == 1)


def scalarize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return scalarize(value.reshape(-1)[0])
        return value.tolist()
    return value


def is_int_like(value: Any) -> bool:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        int(value)
        return True
    except Exception:
        return False


def companion_x(companion: dict[str, Any], key: str, instance_id: Any) -> np.ndarray | None:
    data = companion.get(key)
    if data is None:
        return None
    try:
        idx = int(instance_id)
        if 0 <= idx < len(data):
            return np.asarray(data[idx])
    except Exception:
        return None
    return None


def companion_position_x(companion: dict[str, Any], key: str, position: int) -> np.ndarray | None:
    data = companion.get(key)
    if data is None:
        return None
    try:
        if 0 <= int(position) < len(data):
            return np.asarray(data[int(position)])
    except Exception:
        return None
    return None


def companion_y(companion: dict[str, Any], instance_id: Any) -> Any:
    labels = companion.get("y_test")
    if labels is None:
        return None
    try:
        idx = int(instance_id)
        if 0 <= idx < len(labels):
            return scalarize(labels[idx])
    except Exception:
        return None
    return None


def companion_position_y(companion: dict[str, Any], position: int) -> Any:
    labels = companion.get("y_test")
    if labels is None:
        return None
    try:
        idx = int(position)
        if 0 <= idx < len(labels):
            return scalarize(labels[idx])
    except Exception:
        return None
    return None


def summarize_companion(companion: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in companion.items():
        if hasattr(value, "shape"):
            summary[key] = {"shape": tuple(value.shape), "dtype": str(value.dtype)}
        else:
            summary[key] = value
    return summary


def first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None
