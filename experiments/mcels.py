import json
import os
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
from methods.MCELSCF import MCELSCF

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

DATASETS = [
    'BasicMotions',
    # 'NATOPS',
    'UWaveGestureLibrary', 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]
"""
DATASETS = [
    # 'ECG200', 
    'Gunpoint', 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', 'CinCECGTorso',
    'NonInvasiveFatalECGThorax2', 'CBF',
]"""

PARAMS_PATH = "experiments/params_cf/baseline_mcels.json"
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 100
POOL_SIZE = 1


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    X_train, y_train = sample_dict["train_data_tuple"]
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

    cf_explainer = MCELSCF(
        model_wrapper,
        X_train,
        y_train,
        max_iter=params["max_iter"],
        lr=params["lr"],
        enable_lr_decay=params["enable_lr_decay"],
        lr_decay=params["lr_decay"],
        enable_budget=params["enable_budget"],
        enable_tvnorm=params["enable_tvnorm"],
        l_budget_coeff=params["l_budget_coeff"],
        l_tv_norm_coeff=params["l_tv_norm_coeff"],
        l_max_coeff=params["l_max_coeff"],
        tv_beta=params["tv_beta"],
        seed=params["seed"],
        dataset_name=dataset,
    )

    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        result = cf_explainer.generate_counterfactual(x_orig_samples_worker[i])
        results.append(result)

    store_partial_cfs(
        results,
        first_sample_i,
        first_sample_i + THREAD_SAMPLES - 1,
        dataset,
        MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
        file_suffix_name=exp_name,
    )
    return 1


def experiment_dataset(dataset, exp_name, params):
    X_train, y_train, X_test, y_test, subset_idx, n_classes, _, _, _ = prepare_experiment(
        dataset,
        params,
        MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
    )

    samples = []
    for i in range(I_START, len(X_test), THREAD_SAMPLES):
        sample_dict = {
            "dataset": dataset,
            "exp_name": exp_name,
            "params": params,
            "first_sample_i": i,
            "train_data_tuple": (X_train, y_train),
            "x_orig_samples": X_test[i:i + THREAD_SAMPLES],
            "n_classes": n_classes,
        }
        samples.append(sample_dict)

    if MULTIPROCESSING:
        print("Starting counterfactual generation using multiprocessing...")
        with Pool(POOL_SIZE) as pool:
            _ = list(tqdm(pool.imap(get_counterfactual_worker, samples), total=len(samples)))
    else:
        for sample in tqdm(samples):
            get_counterfactual_worker(sample)

    concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    params["X_test_indexes"] = subset_idx.tolist()
    with open(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json", "w") as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    exp_name = "mcels"
    params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}", exist_ok=True)
        print(f"Starting experiment for dataset {dataset}...")
        experiment_dataset(dataset, exp_name, params.copy())
    print("Finished")
