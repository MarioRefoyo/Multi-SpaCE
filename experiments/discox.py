import os
import copy
import random
import pickle
import sys
import json
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import classification_report

from experiments.experiment_utils import local_data_loader, label_encoder, store_partial_cfs, \
    load_parameters_from_json, generate_settings_combinations, get_subsample
from experiments.results.results_concatenator import concatenate_result_files

from methods.DiscoXCF import DiscoXCF


DATASETS = [
    "BasicMotions", "NATOPS", "UWaveGestureLibrary",
    'ArticularyWordRecognition', 'Cricket',
    'Epilepsy', 'PenDigits', 'PEMS-SF', 'RacketSports', 'SelfRegulationSCP1'
]
DATASETS = [
    # 'ECG200', 'Gunpoint', # 'Coffee',
    'ItalyPowerDemand',
    # 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    # 'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', # 'CinCECGTorso',
    # 'NonInvasiveFatalECGThorax2', 'CBF',
]
DATASETS = ['Coffee', 'CinCECGTorso']

PARAMS_PATH = 'experiments/params_cf/baseline_discox.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'inceptiontime_noscaling'
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 1
POOL_SIZE = 10
# INDEXES_TO_CALCULATE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# INDEXES_TO_CALCULATE = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# INDEXES_TO_CALCULATE = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
# INDEXES_TO_CALCULATE = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
# INDEXES_TO_CALCULATE = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
INDEXES_TO_CALCULATE = None


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    X_train, y_train = sample_dict["train_data_tuple"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        random.seed(params["seed"])

    # Load model
    model_worker = tf.keras.models.load_model(f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/model.hdf5')

    # Instantiate the Counterfactual Explanation method
    cf_explainer = DiscoXCF(model_worker, 'tf', X_train, y_train, window_pct=params["window_pct"])

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig = x_orig_samples_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i+THREAD_SAMPLES-1,
                      dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, file_suffix_name=exp_name)
    return 1


def experiment_dataset(dataset, exp_name, params):
    # Set seed
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        random.seed(params["seed"])

    # Load data
    scaling = params["scaling"]
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), scaling, backend="tf", data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)

    # Get a subset of testing data if specified
    if (params["subset"]) & (len(y_test) > params["subset_number"]):
        X_test, y_test, subset_idx = get_subsample(X_test, y_test, params["subset_number"], params["seed"])
    else:
        subset_idx = np.arange(len(X_test))

    # Load model
    model = tf.keras.models.load_model(f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/model.hdf5')

    # Predict
    y_pred_test_logits = model.predict(X_test, verbose=0)
    y_pred_train_logits = model.predict(X_train, verbose=0)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)
    # Classification report
    print(classification_report(y_test, y_pred_test))

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
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        print('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

        # Concatenate the results
        concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    else:
        cf_explainer = DiscoXCF(model, 'tf', X_train, y_train, window_pct=params["window_pct"])

        # Generate counterfactuals
        results = []
        for i in tqdm(range(len(X_test))):
            x_orig = X_test[i]
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
    # Load parameters
    exp_name = "discox"
    all_params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        if not os.path.isdir(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}"):
            os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}")
        print(f'Starting experiment for dataset {dataset}...')
        experiment_dataset(
            dataset,
            exp_name,
            all_params
        )
    print('Finished')
