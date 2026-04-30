import json
import os
import pickle
import random
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from experiments.experiment_utils import (
    load_model,
    load_parameters_from_json,
    prepare_experiment,
    store_partial_cfs,
)
from experiments.results.results_concatenator import concatenate_result_files
from methods.MASCOTSCF import MASCOTSCF


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


DATASETS = [
    'ECG200', 'Gunpoint', 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', 'CinCECGTorso',
    'NonInvasiveFatalECGThorax2', 'CBF',
]
DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    # 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    # 'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]

PARAMS_PATH = "experiments/params_cf/baseline_mascots.json"
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"

MULTIPROCESSING = False
I_START = 0
THREAD_SAMPLES = 100
POOL_SIZE = 1


def build_explainer(model_wrapper, X_train, y_train, params, build_cache_dir):
    return MASCOTSCF(
        model_wrapper,
        X_train,
        y_train,
        borf_config=params.get("borf_config", "auto"),
        borf_args=params.get("borf_args", {}),
        build_args=params.get("build_args", {}),
        counterfactual_args=params.get("counterfactual_args", {}),
        train_subset=params.get("train_subset"),
        seed=params.get("seed"),
        build_cache_dir=build_cache_dir,
        restore_build=params.get("restore_build", False),
        save_build=params.get("save_build", True),
        force_rebuild=params.get("force_rebuild", False),
    )


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    X_train, y_train = sample_dict["train_data_tuple"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]
    n_classes = sample_dict["n_classes"]
    ts_length = x_orig_samples_worker.shape[1]
    n_channels = x_orig_samples_worker.shape[2]

    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        random.seed(params["seed"])

    model_folder = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}"
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    build_cache_dir = f"{model_folder}/mascot_build"
    cf_explainer = build_explainer(model_wrapper, X_train, y_train, params, build_cache_dir)

    results = []
    for i in tqdm(range(len(x_orig_samples_worker))):
        x_orig = x_orig_samples_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig)
        results.append(result)

    store_partial_cfs(
        results,
        first_sample_i,
        first_sample_i + len(x_orig_samples_worker) - 1,
        dataset,
        MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
        file_suffix_name=exp_name,
    )
    return 1


def experiment_dataset(dataset, exp_name, params):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        subset_idx,
        n_classes,
        model_wrapper,
        y_pred_train,
        y_pred_test,
    ) = prepare_experiment(dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    if MULTIPROCESSING:
        samples = []
        for i in range(I_START, len(X_test), THREAD_SAMPLES):
            x_orig_samples = X_test[i : i + THREAD_SAMPLES]
            samples.append(
                {
                    "dataset": dataset,
                    "train_data_tuple": (X_train, y_train),
                    "exp_name": exp_name,
                    "params": params,
                    "first_sample_i": i,
                    "x_orig_samples": x_orig_samples,
                    "n_classes": n_classes,
                }
            )

        print("Starting counterfactual generation using multiprocessing...")
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

        concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    else:
        build_cache_dir = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/mascot_build"
        cf_explainer = build_explainer(model_wrapper, X_train, y_train, params, build_cache_dir)

        results = []
        for i in tqdm(range(len(X_test))):
            result = cf_explainer.generate_counterfactual(X_test[i])
            results.append(result)

        with open(
            f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/counterfactuals.pickle",
            "wb",
        ) as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    params["X_test_indexes"] = subset_idx.tolist()
    with open(
        f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json",
        "w",
    ) as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    exp_name = "mascots_scalar_gpu"
    all_params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        os.makedirs(
            f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}",
            exist_ok=True,
        )
        print(f"Starting experiment for dataset {dataset}...")
        experiment_dataset(dataset, exp_name, all_params)
    print("Finished")
