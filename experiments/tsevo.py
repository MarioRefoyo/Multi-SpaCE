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
from methods.TSEvoCF import TSEvoCF


def configure_tensorflow_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


configure_tensorflow_memory_growth()


"""DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary', 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]"""
DATASETS = [
    'ECG200', 'Gunpoint', 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', 'CinCECGTorso',
    'NonInvasiveFatalECGThorax2', 'CBF',
]

DATASETS = ['CBF']

PARAMS_PATH = "experiments/params_cf/baseline_tsevo.json"
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 100
POOL_SIZE = 1


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    x_train, y_reference = sample_dict["train_data_tuple"]
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

    configure_tensorflow_memory_growth()

    model_folder = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}"
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)

    cf_explainer = TSEvoCF(
        model_wrapper,
        x_train,
        y_reference,
        transformer=params["transformer"],
        epochs=params["epochs"],
        verbose=params["verbose"],
    )

    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig = x_orig_samples_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig)
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
    x_train, y_train, x_test, y_test, subset_idx, n_classes, _, y_pred_train, _ = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME
    )

    if MULTIPROCESSING:
        samples = []
        for i in range(I_START, len(x_test), THREAD_SAMPLES):
            x_orig_samples = x_test[i:i + THREAD_SAMPLES]

            sample_dict = {
                "dataset": dataset,
                "train_data_tuple": (x_train, y_pred_train),
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "n_classes": n_classes,
            }
            samples.append(sample_dict)

        print("Starting counterfactual generation using multiprocessing...")
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))
    else:
        ts_length = x_test.shape[1]
        n_channels = x_test.shape[2]
        model_folder = f"experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}"
        model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
        cf_explainer = TSEvoCF(
            model_wrapper,
            x_train,
            y_pred_train,
            transformer=params["transformer"],
            epochs=params["epochs"],
            verbose=params["verbose"],
        )

        results = []
        for i in tqdm(range(len(x_test))):
            x_orig = x_test[i]
            result = cf_explainer.generate_counterfactual(x_orig)
            results.append(result)

        store_partial_cfs(
            results,
            0,
            len(x_test) - 1,
            dataset,
            MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
            file_suffix_name=exp_name,
        )

    concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    params["X_test_indexes"] = subset_idx.tolist()
    with open(
        f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json",
        "w",
    ) as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    exp_name = "tsevo"
    all_params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        os.makedirs(
            f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}",
            exist_ok=True,
        )
        print(f"Starting experiment for dataset {dataset}...")
        experiment_dataset(dataset, exp_name, all_params)
    print("Finished")
