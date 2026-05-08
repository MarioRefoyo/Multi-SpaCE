import json
import multiprocessing as mp
import os
import pickle
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "experiments/evaluation/.matplotlib_cache")

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm

from experiments.evaluation.evaluation_utils import calculate_change_mask, calculate_method_metrics
from experiments.experiment_utils import (
    generate_settings_combinations,
    load_model,
    load_parameters_from_json,
    prepare_experiment_robustness,
)
from experiments.metrics_excel import load_outlier_calculators
from methods.MultiSubSpaCE.FeatureImportanceInitializers import GraCAMPlusFI, NoneFI
from methods.MultiSubSpaCECF import MultiSubSpaCECFv2
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder, SecondBestGlobalNUNFinder
from methods.outlier_calculators import AEOutlierCalculator


DATASETS = [
    "ArticularyWordRecognition",
    # "SelfRegulationSCP1",
]

EXPERIMENT_FAMILY = "multisubspace_v2_robustness"
PARAMS_PATH = f"experiments/params_cf/{EXPERIMENT_FAMILY}.json"
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
OC_EXPERIMENT_NAME = "ae_basic_train"
OUTLIER_CALCULATOR_EXPERIMENTS = {"AE": OC_EXPERIMENT_NAME, "IF": "if_basic_train", "LOF": "lof_basic_train"}

# Order matches visualize_counterfactuals_mo_multivariate: [adv, sparsity, contiguity, plausibility].
MO_EVAL_WEIGHTS = {"adv": 0.1, "sparsity": 0.4 * 0.7, "contiguity": 0.6 * 0.7, "plausibility": 0.2}

N_ANCHORS = 50
K_NEIGHBORS = 3
ROBUSTNESS_RANDOM_SEED = 24
DISTANCE = "euclidean_flat"
NEIGHBOR_SELECTION_REGIMES = ["free", "same_nun_target"]
MULTIPROCESSING = True
POOL_SIZE = 20
MP_START_METHOD = "spawn"

_ROBUSTNESS_WORKER_CONTEXT = {}


def configure_tensorflow_runtime(log=False):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if log:
            print(f"TensorFlow {tf.__version__}: no GPU detected.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            if log:
                print(f"Could not enable memory growth for {gpu}: {exc}")

    if log:
        growth_enabled = tf.config.experimental.get_memory_growth(gpus[0])
        print(f"TensorFlow {tf.__version__}: GPU memory growth enabled={growth_enabled}")


def set_all_seeds(seed):
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_nun_strategy(params):
    if "nun_strategy" in params:
        return params["nun_strategy"]
    if params.get("independent_channels_nun", False):
        return "independent"
    return "global"


def get_neighbor_selection_regimes(params):
    regimes = params.get("neighbor_selection_regimes", NEIGHBOR_SELECTION_REGIMES)
    if isinstance(regimes, str):
        regimes = [regimes]
    supported = {"free", "same_nun_target"}
    unsupported = sorted(set(regimes) - supported)
    if unsupported:
        raise ValueError(f"Unsupported neighbor selection regimes: {unsupported}")
    return list(regimes)


def build_nun_finder(dataset, params, X_train, y_train, y_pred_train, n_classes):
    nun_strategy = get_nun_strategy(params)
    nun_model_wrapper = None
    if nun_strategy in {"independent", "second_best"}:
        model_folder = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}"
        nun_model_wrapper = load_model(model_folder, dataset, X_train.shape[2], X_train.shape[1], n_classes)

    if nun_strategy == "independent":
        return IndependentNUNFinder(
            X_train, y_train, y_pred_train, distance="euclidean",
            from_true_labels=False, backend="tf", n_neighbors=params["n_neighbors"], model=nun_model_wrapper
        )
    if nun_strategy == "second_best":
        return SecondBestGlobalNUNFinder(
            X_train, y_train, y_pred_train, distance="euclidean",
            from_true_labels=False, backend="tf", model=nun_model_wrapper
        )
    if nun_strategy == "global":
        return GlobalNUNFinder(
            X_train, y_train, y_pred_train, distance="euclidean",
            from_true_labels=False, backend="tf"
        )
    raise ValueError(f"Unsupported nun_strategy: {nun_strategy}")


def build_explainer(dataset, params, X_train, n_classes):
    ts_length = X_train.shape[1]
    n_channels = X_train.shape[2]
    model_folder = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}"
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)

    if params["plausibility_objective"] == "none":
        outlier_calculator = None
    else:
        ae_model = tf.keras.models.load_model(
            f"./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5",
            compile=False,
        )
        outlier_calculator = AEOutlierCalculator(ae_model, X_train)

    if params["init_fi"] == "none":
        fi_method = NoneFI("tf")
    elif params["init_fi"] == "gradcam++":
        fi_method = GraCAMPlusFI("tf", model_wrapper)
    else:
        raise ValueError("The provided init_fi is not valid.")

    grouped_channels_iter, individual_channels_iter, pruning_iter = params["max_iter"]
    return MultiSubSpaCECFv2(
        model_wrapper, outlier_calculator, fi_method,
        grouped_channels_iter, individual_channels_iter, pruning_iter,
        plausibility_objective=params["plausibility_objective"],
        population_size=params["population_size"],
        change_subseq_mutation_prob=params["change_subseq_mutation_prob"],
        add_subseq_mutation_prob=params["add_subseq_mutation_prob"],
        integrated_pruning_mutation_prob=params["integrated_pruning_mutation_prob"],
        final_pruning_mutation_prob=params["final_pruning_mutation_prob"],
        channel_mutation_prob=params["channel_mutation_prob"],
        init_pct=params["init_pct"],
        reinit=params["reinit"],
        init_random_mix_ratio=params["init_random_mix_ratio"],
        invalid_penalization=params["invalid_penalization"],
    )


def init_counterfactual_worker(worker_context):
    configure_tensorflow_runtime(log=False)
    _ROBUSTNESS_WORKER_CONTEXT.clear()
    _ROBUSTNESS_WORKER_CONTEXT.update(worker_context)
    set_all_seeds(worker_context["seed"])
    _ROBUSTNESS_WORKER_CONTEXT["explainer"] = build_explainer(
        worker_context["dataset"],
        worker_context["params"],
        worker_context["X_train"],
        worker_context["n_classes"],
    )


def generate_counterfactual_worker(test_index):
    context = _ROBUSTNESS_WORKER_CONTEXT
    test_index = int(test_index)
    set_all_seeds(context["seed"] + test_index if context["seed"] is not None else None)

    result = context["explainer"].generate_counterfactual(
        context["X_test"][test_index],
        context["desired_by_index"][test_index],
        nun_example=context["nuns_by_index"][test_index],
        y_true_orig=context["y_test"][test_index],
    )
    with open(result_file_for(context["raw_results_dir"], test_index), "wb") as fp:
        pickle.dump(result, fp, pickle.HIGHEST_PROTOCOL)
    return test_index


def select_anchor_indices(y_test, y_pred_test, n_anchors, seed):
    rng = np.random.default_rng(seed)
    correct_idx = np.where(y_test == y_pred_test)[0]
    fallback_idx = np.arange(len(y_test))
    source_idx = correct_idx if len(correct_idx) >= min(n_anchors, len(y_test)) else fallback_idx
    selected = rng.choice(source_idx, size=min(n_anchors, len(source_idx)), replace=False)
    return np.sort(selected)


def build_nun_metadata(dataset, X_train, X_test, y_test, y_pred_test, nuns_by_index,
                       desired_by_index, nun_distances_by_index):
    rows = []
    for test_index in range(len(X_test)):
        x_nun = nuns_by_index[test_index]
        rows.append(
            {
                "dataset": dataset,
                "test_index": int(test_index),
                "true_label": int(y_test[test_index]),
                "pred_label": int(y_pred_test[test_index]),
                "correctly_classified": bool(y_test[test_index] == y_pred_test[test_index]),
                "nun_index": infer_train_index(X_train, x_nun),
                "nun_label": int(desired_by_index[test_index]),
                "nun_l2_distance": float(np.asarray(nun_distances_by_index[test_index]).reshape(-1)[0]),
            }
        )
    return pd.DataFrame(rows)


def get_regime_rules(regime, anchor_idx, y_test, y_pred_test, correct, desired_by_index):
    if regime == "free":
        regime_mask = np.ones(len(y_test), dtype=bool)
        return [
            ("same_pred_correct", regime_mask & (y_pred_test == y_pred_test[anchor_idx]) & correct),
            ("fallback_same_true_correct", regime_mask & (y_test == y_test[anchor_idx]) & correct),
            ("fallback_correct_any_class", regime_mask & correct),
            ("fallback_any", regime_mask),
        ], regime_mask

    if regime == "same_nun_target":
        regime_mask = desired_by_index == desired_by_index[anchor_idx]
        return [
            ("same_nun_target_same_pred_correct", regime_mask & (y_pred_test == y_pred_test[anchor_idx]) & correct),
            ("fallback_same_nun_target_same_true_correct", regime_mask & (y_test == y_test[anchor_idx]) & correct),
            ("fallback_same_nun_target_correct_any_class", regime_mask & correct),
            ("fallback_same_nun_target_any", regime_mask),
        ], regime_mask

    raise ValueError(f"Unsupported neighbor selection regime: {regime}")


def nearest_neighbor_rows(dataset, anchor_indices, X_test, y_test, y_pred_test, k_neighbors,
                          regimes, desired_by_index, nun_distances_by_index,
                          nun_indices_by_index):
    flat = X_test.reshape(len(X_test), -1)
    correct = y_test == y_pred_test
    rows = []
    selected_neighbors = set()

    for regime in regimes:
        for anchor_idx in anchor_indices:
            distances = np.linalg.norm(flat - flat[anchor_idx], axis=1)
            distance_order = np.argsort(distances)
            rules, regime_mask = get_regime_rules(regime, anchor_idx, y_test, y_pred_test, correct, desired_by_index)
            regime_mask = regime_mask.copy()
            regime_mask[anchor_idx] = False
            n_candidates_available = int(regime_mask.sum())

            chosen = []
            chosen_set = set()
            for rule_name, rule_mask in rules:
                rule_mask = rule_mask.copy()
                rule_mask[anchor_idx] = False
                for neighbor_idx in distance_order:
                    if neighbor_idx == anchor_idx or neighbor_idx in chosen_set:
                        continue
                    if not rule_mask[neighbor_idx]:
                        continue
                    chosen.append((int(neighbor_idx), rule_name))
                    chosen_set.add(int(neighbor_idx))
                    if len(chosen) == k_neighbors:
                        break
                if len(chosen) == k_neighbors:
                    break

            for rank, (neighbor_idx, rule_name) in enumerate(chosen, start=1):
                selected_neighbors.add(neighbor_idx)
                same_nun_index = bool(nun_indices_by_index[anchor_idx] == nun_indices_by_index[neighbor_idx])
                same_nun_label = bool(desired_by_index[anchor_idx] == desired_by_index[neighbor_idx])
                rows.append(
                    {
                        "dataset": dataset,
                        "neighbor_selection_regime": regime,
                        "anchor_test_index": int(anchor_idx),
                        "neighbor_test_index": int(neighbor_idx),
                        "neighbor_rank_within_regime": rank,
                        "neighbor_rank": rank,
                        "original_l2_distance": float(distances[neighbor_idx]),
                        "original_euclidean_distance": float(distances[neighbor_idx]),
                        "anchor_true_label": int(y_test[anchor_idx]),
                        "neighbor_true_label": int(y_test[neighbor_idx]),
                        "anchor_pred_label": int(y_pred_test[anchor_idx]),
                        "neighbor_pred_label": int(y_pred_test[neighbor_idx]),
                        "anchor_predicted_label": int(y_pred_test[anchor_idx]),
                        "neighbor_predicted_label": int(y_pred_test[neighbor_idx]),
                        "same_pred_label": bool(y_pred_test[anchor_idx] == y_pred_test[neighbor_idx]),
                        "anchor_correctly_classified": bool(correct[anchor_idx]),
                        "neighbor_correctly_classified": bool(correct[neighbor_idx]),
                        "anchor_correct": bool(correct[anchor_idx]),
                        "neighbor_correct": bool(correct[neighbor_idx]),
                        "anchor_nun_index": int(nun_indices_by_index[anchor_idx]),
                        "neighbor_nun_index": int(nun_indices_by_index[neighbor_idx]),
                        "same_nun_index": same_nun_index,
                        "anchor_nun_label": int(desired_by_index[anchor_idx]),
                        "neighbor_nun_label": int(desired_by_index[neighbor_idx]),
                        "same_nun_label": same_nun_label,
                        "anchor_nun_l2_distance": float(np.asarray(nun_distances_by_index[anchor_idx]).reshape(-1)[0]),
                        "neighbor_nun_l2_distance": float(np.asarray(nun_distances_by_index[neighbor_idx]).reshape(-1)[0]),
                        "n_candidates_available_for_regime": n_candidates_available,
                        "selection_rule_used": rule_name,
                        "selection_rule": rule_name,
                    }
                )

    return pd.DataFrame(rows), selected_neighbors


def infer_train_index(X_train, x_nun):
    distances = np.linalg.norm((X_train - x_nun).reshape(len(X_train), -1), axis=1)
    return int(np.argmin(distances))


def result_file_for(raw_results_dir, test_index):
    return raw_results_dir / f"test_index_{int(test_index):06d}.pickle"


def array_file_for(arrays_dir, test_index):
    return arrays_dir / f"test_index_{int(test_index):06d}.npz"


def generate_missing_explanations(result_path, dataset, params, X_train, X_test, y_test, unique_test_indices,
                                  nuns_by_index, desired_by_index, n_classes):
    raw_results_dir = result_path / "raw_results"
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    existing = [idx for idx in unique_test_indices if result_file_for(raw_results_dir, idx).is_file()]
    missing = [idx for idx in unique_test_indices if idx not in set(existing)]

    print(f"Existing explanations skipped: {len(existing)}")
    print(f"Missing explanations to generate: {len(missing)}")
    if not missing:
        return existing, []

    seed = params.get("run_seed", params.get("seed"))
    use_multiprocessing = bool(params.get("multiprocessing", MULTIPROCESSING))
    pool_size = int(params.get("pool_size", POOL_SIZE))
    if use_multiprocessing and len(missing) > 1:
        worker_context = {
            "dataset": dataset,
            "params": params,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "nuns_by_index": nuns_by_index,
            "desired_by_index": desired_by_index,
            "n_classes": n_classes,
            "raw_results_dir": raw_results_dir,
            "seed": seed,
        }
        pool_size = min(pool_size, len(missing))
        print(
            f"Starting robustness counterfactual generation with multiprocessing "
            f"(start_method={MP_START_METHOD}, pool_size={pool_size}, instances={len(missing)})..."
        )
        ctx = mp.get_context(MP_START_METHOD)
        with ctx.Pool(
            pool_size,
            initializer=init_counterfactual_worker,
            initargs=(worker_context,),
        ) as pool:
            with tqdm(
                total=len(missing),
                desc=f"{dataset} Multi-SpaCE robustness",
                unit="cf",
                dynamic_ncols=True,
                mininterval=0.1,
                miniters=1,
                disable=False,
                file=sys.stderr,
            ) as progress:
                progress.refresh()
                generated = []
                async_results = []

                def update_progress(test_index):
                    generated.append(int(test_index))
                    progress.update(1)
                    progress.set_postfix_str(f"last={int(test_index)}")
                    progress.refresh()

                for test_index in missing:
                    async_results.append(
                        pool.apply_async(
                            generate_counterfactual_worker,
                            args=(int(test_index),),
                            callback=update_progress,
                        )
                    )

                for async_result in async_results:
                    async_result.get()
        generated = sorted(generated)
    else:
        worker_context = {
            "dataset": dataset,
            "params": params,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "nuns_by_index": nuns_by_index,
            "desired_by_index": desired_by_index,
            "n_classes": n_classes,
            "raw_results_dir": raw_results_dir,
            "seed": seed,
        }
        init_counterfactual_worker(worker_context)
        generated = []
        for test_index in tqdm(
            missing,
            desc=f"{dataset} Multi-SpaCE robustness",
            unit="cf",
            dynamic_ncols=True,
            mininterval=0.1,
            miniters=1,
            disable=False,
            file=sys.stderr,
        ):
            generated.append(generate_counterfactual_worker(test_index))

    return existing, generated


def load_results_in_order(raw_results_dir, unique_test_indices):
    results = []
    for test_index in unique_test_indices:
        with open(result_file_for(raw_results_dir, test_index), "rb") as fp:
            results.append(pickle.load(fp))
    return results


def build_instance_metadata(result_path, dataset, exp_name, params, X_train, X_test, y_test, y_pred_test,
                            y_pred_test_logits, unique_test_indices, anchor_indices, relationships_df,
                            nuns_by_index, desired_by_index, nun_distances_by_index, results, model_wrapper):
    arrays_dir = result_path / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    outlier_calculators = load_outlier_calculators(dataset, X_train, OUTLIER_CALCULATOR_EXPERIMENTS)
    metrics_df = calculate_method_metrics(
        model_wrapper,
        outlier_calculators,
        X_test[unique_test_indices],
        nuns_by_index[unique_test_indices],
        results,
        y_pred_test[unique_test_indices],
        "Multi-SpaCE",
        mo_weights=np.array(list(MO_EVAL_WEIGHTS.values())),
    )

    rows = []
    selected_results = []
    anchor_set = set(int(idx) for idx in anchor_indices)
    neighbor_set = set(relationships_df["neighbor_test_index"].astype(int).tolist()) if not relationships_df.empty else set()
    regime_by_index = {int(idx): set() for idx in unique_test_indices}
    for _, rel in relationships_df.iterrows():
        regime = str(rel["neighbor_selection_regime"])
        regime_by_index.setdefault(int(rel["anchor_test_index"]), set()).add(regime)
        regime_by_index.setdefault(int(rel["neighbor_test_index"]), set()).add(regime)

    for local_i, test_index in enumerate(unique_test_indices):
        result = results[local_i]
        metric_row = metrics_df.iloc[local_i].to_dict()
        selected_cf_index = int(metric_row["best cf index"])
        x_cf = np.asarray(result["cfs"])[selected_cf_index]
        x_nun = nuns_by_index[test_index]
        mask = calculate_change_mask(X_test[test_index], x_cf, x_nun=x_nun, verbose=0).astype(bool)
        array_path = array_file_for(arrays_dir, test_index)
        np.savez_compressed(
            array_path,
            x=X_test[test_index],
            x_cf=x_cf,
            mask=mask,
            x_nun=x_nun,
            all_cfs=np.asarray(result["cfs"]),
        )

        is_anchor = test_index in anchor_set
        is_neighbor = test_index in neighbor_set
        if is_anchor and is_neighbor:
            role = "anchor_and_neighbor"
        elif is_anchor:
            role = "anchor"
        else:
            role = "neighbor"
        regimes_appeared_in = sorted(regime_by_index.get(int(test_index), set()))

        selected_result = dict(result)
        selected_result["selected_cf_index"] = selected_cf_index
        selected_result["selected_cf"] = x_cf
        selected_result["selected_mask"] = mask
        selected_results.append(selected_result)

        pred_probs = np.asarray(y_pred_test_logits[test_index])
        row = {
            "dataset": dataset,
            "experiment_hash": exp_name,
            "test_index": int(test_index),
            "role": role,
            "is_anchor": bool(is_anchor),
            "is_neighbor": bool(is_neighbor),
            "appears_in_free_regime": "free" in regimes_appeared_in,
            "appears_in_same_nun_target_regime": "same_nun_target" in regimes_appeared_in,
            "regimes_appeared_in": ";".join(regimes_appeared_in),
            "true_label": int(y_test[test_index]),
            "pred_label": int(y_pred_test[test_index]),
            "predicted_label": int(y_pred_test[test_index]),
            "predicted_probability": float(pred_probs[int(y_pred_test[test_index])]),
            "correctly_classified": bool(y_test[test_index] == y_pred_test[test_index]),
            "nun_label": int(desired_by_index[test_index]),
            "target_nun_label": int(desired_by_index[test_index]),
            "nun_index": infer_train_index(X_train, x_nun),
            "nun_distance": float(np.asarray(nun_distances_by_index[test_index]).reshape(-1)[0]),
            "selected_counterfactual_index": selected_cf_index,
            "array_path": str(array_path),
            "raw_result_path": str(result_file_for(result_path / "raw_results", test_index)),
            "validity": bool(metric_row["valid"]),
            "generation_time": float(result["time"]),
        }
        for key, value in metric_row.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                value = value.item()
            row[key] = value
        rows.append(row)

    return pd.DataFrame(rows), selected_results, metrics_df


def save_relationship_links(relationships_df, instance_df):
    link_cols = ["test_index", "array_path", "raw_result_path"]
    links = instance_df[link_cols].rename(
        columns={
            "test_index": "anchor_test_index",
            "array_path": "anchor_array_path",
            "raw_result_path": "anchor_raw_result_path",
        }
    )
    relationships_df = relationships_df.merge(links, on="anchor_test_index", how="left")
    links = instance_df[link_cols].rename(
        columns={
            "test_index": "neighbor_test_index",
            "array_path": "neighbor_array_path",
            "raw_result_path": "neighbor_raw_result_path",
        }
    )
    return relationships_df.merge(links, on="neighbor_test_index", how="left")


def build_generation_summary(dataset, regimes, relationships_df, instance_df, n_anchors, k_neighbors, generated):
    rows = []
    for regime in regimes:
        regime_relationships = relationships_df[
            relationships_df["neighbor_selection_regime"] == regime
        ].copy()
        neighbors_per_anchor = regime_relationships.groupby("anchor_test_index").size()
        full_k = int((neighbors_per_anchor == k_neighbors).sum())
        fewer_than_k = int(n_anchors - full_k)
        involved = set(regime_relationships["anchor_test_index"].astype(int).tolist())
        involved.update(regime_relationships["neighbor_test_index"].astype(int).tolist())
        regime_instances = instance_df[instance_df["test_index"].astype(int).isin(involved)]
        valid_count = int(regime_instances["validity"].fillna(False).sum()) if "validity" in regime_instances else 0
        validity_rate = (
            float(regime_instances["validity"].mean())
            if len(regime_instances) and "validity" in regime_instances
            else np.nan
        )
        rows.append(
            {
                "dataset": dataset,
                "neighbor_selection_regime": regime,
                "n_anchors": int(n_anchors),
                "requested_k_neighbors": int(k_neighbors),
                "n_relationship_rows": int(len(regime_relationships)),
                "mean_neighbors_per_anchor": float(len(regime_relationships) / n_anchors) if n_anchors else np.nan,
                "n_anchors_with_full_k_neighbors": full_k,
                "n_anchors_with_fewer_than_k_neighbors": fewer_than_k,
                "n_unique_instances_involved": int(len(involved)),
                "n_unique_instances_generated": int(len(set(generated).intersection(involved))),
                "n_valid_cfes": valid_count,
                "validity_rate": validity_rate,
            }
        )
    return pd.DataFrame(rows)


def experiment_dataset(dataset, exp_name, params, experiment_family):
    result_path = Path("experiments/results") / dataset / MODEL_TO_EXPLAIN_EXPERIMENT_NAME / experiment_family / exp_name
    result_path.mkdir(parents=True, exist_ok=True)
    print(f"Result path: {result_path}")

    random_seed = params.get("robustness_random_seed", params.get("run_seed", params.get("seed", ROBUSTNESS_RANDOM_SEED)))
    n_anchors = int(params.get("n_anchors", N_ANCHORS))
    k_neighbors = int(params.get("k_neighbors", K_NEIGHBORS))
    regimes = get_neighbor_selection_regimes(params)
    set_all_seeds(random_seed)

    (
        X_train, y_train, X_test, y_test, test_idx, n_classes, model_wrapper,
        y_pred_train, y_pred_test, _y_pred_train_logits, y_pred_test_logits,
    ) = prepare_experiment_robustness(dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    anchor_indices = select_anchor_indices(y_test, y_pred_test, n_anchors, random_seed)
    print(f"Candidate anchors correctly classified: {int(np.sum(y_test == y_pred_test))}")
    print(f"Anchors selected: {len(anchor_indices)}")
    print(f"Neighbor selection regimes: {regimes}")

    nun_finder = build_nun_finder(dataset, params, X_train, y_train, y_pred_train, n_classes)
    print("Retrieving NUN metadata for the full test split...")
    nuns, desired_classes, nun_distances = nun_finder.retrieve_nuns(X_test, y_pred_test)
    nuns = nuns[:, 0, :, :]
    desired_classes = np.asarray(desired_classes).reshape(-1)
    nuns_by_index = nuns.astype(X_test.dtype, copy=False)
    desired_by_index = desired_classes.astype(int, copy=False)
    nun_distances_by_index = np.asarray(nun_distances).reshape(len(X_test), -1)[:, :1]

    nun_metadata_df = build_nun_metadata(
        dataset, X_train, X_test, y_test, y_pred_test,
        nuns_by_index, desired_by_index, nun_distances_by_index
    )
    nun_indices_by_index = nun_metadata_df.sort_values("test_index")["nun_index"].to_numpy()

    relationships_df, neighbor_indices = nearest_neighbor_rows(
        dataset, anchor_indices, X_test, y_test, y_pred_test, k_neighbors,
        regimes, desired_by_index, nun_distances_by_index, nun_indices_by_index
    )
    unique_test_indices = np.array(sorted(set(anchor_indices).union(neighbor_indices)), dtype=int)

    for regime in regimes:
        regime_relationships = relationships_df[relationships_df["neighbor_selection_regime"] == regime]
        neighbors_per_anchor = regime_relationships.groupby("anchor_test_index").size()
        full_k = int((neighbors_per_anchor == k_neighbors).sum())
        fewer_than_k = int(len(anchor_indices) - full_k)
        print(
            f"Regime {regime}: relationship_rows={len(regime_relationships)}, "
            f"anchors_with_{k_neighbors}_neighbors={full_k}, anchors_with_fewer={fewer_than_k}"
        )
        if regime == "same_nun_target" and fewer_than_k > 0:
            print(f"Regime {regime}: saved fewer than {k_neighbors} neighbors where same-NUN-label candidates were unavailable.")

    print(f"Unique neighbors selected across regimes: {len(neighbor_indices)}")
    print(f"Unique instances to explain across regimes: {len(unique_test_indices)}")

    existing, generated = generate_missing_explanations(
        result_path, dataset, params, X_train, X_test, y_test, unique_test_indices,
        nuns_by_index, desired_by_index, n_classes
    )
    results = load_results_in_order(result_path / "raw_results", unique_test_indices)
    instance_df, selected_results, _ = build_instance_metadata(
        result_path, dataset, exp_name, params, X_train, X_test, y_test, y_pred_test,
        y_pred_test_logits, unique_test_indices, anchor_indices, relationships_df,
        nuns_by_index, desired_by_index, nun_distances_by_index, results, model_wrapper
    )
    relationships_df = save_relationship_links(relationships_df, instance_df)
    summary_df = build_generation_summary(
        dataset, regimes, relationships_df, instance_df, len(anchor_indices), k_neighbors, generated
    )

    instance_df.to_csv(result_path / "explained_instances.csv", index=False)
    relationships_df.to_csv(result_path / "anchor_neighbor_relationships.csv", index=False)
    nun_metadata_df.to_csv(result_path / "nun_metadata.csv", index=False)
    summary_df.to_csv(result_path / "robustness_generation_summary.csv", index=False)
    with open(result_path / "counterfactuals.pickle", "wb") as fp:
        pickle.dump(results, fp, pickle.HIGHEST_PROTOCOL)
    with open(result_path / "selected_counterfactuals.pickle", "wb") as fp:
        pickle.dump(selected_results, fp, pickle.HIGHEST_PROTOCOL)

    params_out = dict(params)
    params_out.update(
        {
            "X_test_indexes": test_idx[unique_test_indices].tolist(),
            "robustness_unique_test_indices": unique_test_indices.tolist(),
            "robustness_anchor_test_indices": anchor_indices.tolist(),
            "n_anchors": n_anchors,
            "k_neighbors": k_neighbors,
            "robustness_random_seed": random_seed,
            "neighbor_selection_regimes": regimes,
            "distance": DISTANCE,
            "method": "Multi-SpaCE",
            "experiment_family": experiment_family,
            "MO_EVAL_WEIGHTS": MO_EVAL_WEIGHTS,
            "explained_instances_path": str(result_path / "explained_instances.csv"),
            "anchor_neighbor_relationships_path": str(result_path / "anchor_neighbor_relationships.csv"),
            "nun_metadata_path": str(result_path / "nun_metadata.csv"),
            "robustness_generation_summary_path": str(result_path / "robustness_generation_summary.csv"),
        }
    )
    with open(result_path / "params.json", "w") as fp:
        json.dump(params_out, fp, sort_keys=True, indent=2)
    with open(result_path / "robustness_generation_config.json", "w") as fp:
        json.dump(params_out, fp, sort_keys=True, indent=2)

    validity_rate = float(instance_df["validity"].mean()) if len(instance_df) else np.nan
    total_time = float(instance_df["generation_time"].sum()) if len(instance_df) else 0.0
    mean_time = float(instance_df["generation_time"].mean()) if len(instance_df) else np.nan
    print(f"Generated explanations: {len(generated)}")
    print(f"Already existing/skipped: {len(existing)}")
    print(f"Validity rate: {validity_rate:.3f}")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Mean generation time: {mean_time:.2f}s")
    return summary_df


if __name__ == "__main__":
    mp.freeze_support()
    configure_tensorflow_runtime(log=True)

    all_params = load_parameters_from_json(PARAMS_PATH)
    params_combinations = generate_settings_combinations(all_params)
    all_summaries = []
    for dataset in DATASETS:
        for experiment_name, experiment_params in params_combinations.items():
            experiment_name = f"v2_{experiment_name}"
            print(f"Starting robustness experiment {experiment_name} for dataset {dataset}...")
            summary_df = experiment_dataset(dataset, experiment_name, experiment_params, EXPERIMENT_FAMILY)
            summary_df.insert(1, "experiment_hash", experiment_name)
            all_summaries.append(summary_df)
    if all_summaries:
        family_summary = pd.concat(all_summaries, ignore_index=True)
        family_path = Path("experiments/results") / "_robustness_generation_summaries"
        family_path.mkdir(parents=True, exist_ok=True)
        family_summary.to_csv(family_path / f"{EXPERIMENT_FAMILY}_summary.csv", index=False)
        print(f"Global robustness generation summary: {family_path / f'{EXPERIMENT_FAMILY}_summary.csv'}")
    print("Finished")
