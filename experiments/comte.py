import os
import copy
import random
import pickle
import sys
import json
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from experiments.experiment_utils import local_data_loader, label_encoder, store_partial_cfs, \
    load_parameters_from_json, load_model, prepare_experiment
from experiments.results.results_concatenator import concatenate_result_files

from methods.COMTECF import COMTECF

DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    'ArticularyWordRecognition',
    'Epilepsy', 'PenDigits',
    'RacketSports', 'SelfRegulationSCP1'
    'PEMS-SF', 'Cricket',
]

ADDITIONAL_SUBSAMPLE_SUBSET = None
PARAMS_PATH = 'experiments/params_cf/baseline_comte.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'inceptiontime_noscaling'
MULTIPROCESSING = False
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 1
MP_START_METHOD = "spawn"
# INDEXES_TO_CALCULATE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# INDEXES_TO_CALCULATE = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# INDEXES_TO_CALCULATE = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
# INDEXES_TO_CALCULATE = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
# INDEXES_TO_CALCULATE = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
INDEXES_TO_CALCULATE = None


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


def get_counterfactual_worker(sample_dict):
    configure_tensorflow_runtime(log=False)

    dataset = sample_dict["dataset"]
    X_train, y_train = sample_dict["train_data_tuple"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]
    n_classes = sample_dict["n_classes"]
    ts_length = x_orig_samples_worker.shape[1]
    n_channels = x_orig_samples_worker.shape[2]

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        random.seed(params["seed"])

    # Load model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Instantiate the Counterfactual Explanation method
    cf_explainer = COMTECF(
        model_wrapper, X_train, y_train, params["number_distractors"],
        max_attempts=params["max_attempts"], max_iter=params["max_iter"],
        restarts=params["restarts"], reg=params["reg"])

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig = np.expand_dims(x_orig_samples_worker[i], axis=0)
        result = cf_explainer.generate_counterfactual(x_orig)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i + len(x_orig_samples_worker) - 1,
                      dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, file_suffix_name=exp_name)
    return 1


def experiment_dataset(dataset, exp_name, params):
    X_train, y_train, X_test, y_test, subset_idx, n_classes, model_wrapper, y_pred_train, y_pred_test = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    if INDEXES_TO_CALCULATE is not None:
        if THREAD_SAMPLES != 1:
            raise ValueError("Using specific indexes to calculate counterfactuals "
                             "does not support multiple instances per thread.")
        first_sample_list = copy.deepcopy(INDEXES_TO_CALCULATE)
    else:
        first_sample_list = list(range(I_START, len(X_test), THREAD_SAMPLES))

    # Get counterfactuals
    if MULTIPROCESSING:
        # Prepare dict to iterate optimization problem
        samples = []
        for i in range(len(first_sample_list)):
            # Init optimizer
            first_sample = first_sample_list[i]
            end_sample = first_sample_list[i] + THREAD_SAMPLES
            x_orig_samples = X_test[first_sample:end_sample]

            sample_dict = {
                "dataset": dataset,
                "train_data_tuple": (X_train, y_train),
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": first_sample_list[i],
                "x_orig_samples": x_orig_samples,
                "n_classes": n_classes
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        pool_size = min(POOL_SIZE, len(samples))
        if pool_size == 0:
            print("No test batches to process after applying I_START. Skipping multiprocessing run.")
            return

        print(
            f"Starting counterfactual generation using multiprocessing "
            f"(start_method={MP_START_METHOD}, pool_size={pool_size}, batches={len(samples)})..."
        )
        ctx = mp.get_context(MP_START_METHOD)
        with ctx.Pool(pool_size, maxtasksperchild=1) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

        # Concatenate the results
        concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    else:
        cf_explainer = COMTECF(
            model_wrapper, X_train, y_train, params["number_distractors"],
            max_attempts=params["max_attempts"], max_iter=params["max_iter"],
            restarts=params["restarts"], reg=params["reg"])

        # Generate counterfactuals
        results = []
        for i in tqdm(range(len(X_test))):
            x_orig = np.expand_dims(X_test[i], axis=0)
            result = cf_explainer.generate_counterfactual(x_orig)
            results.append(result)

        # Store experiment results
        with open(f'./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/results.pickle', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    with open(f'./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    mp.freeze_support()
    configure_tensorflow_runtime(log=True)

    # Load parameters
    exp_name = "comte_gpu"
    all_params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        if not os.path.isdir(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}"):
            os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}")
        print(f'Starting experiment for dataset {dataset}...')

        if dataset in ["PEMS-SF", "Cricket"]:
            all_params["additional_subsample_subset"] = ADDITIONAL_SUBSAMPLE_SUBSET
        experiment_dataset(
            dataset,
            exp_name,
            all_params
        )
    print('Finished')
