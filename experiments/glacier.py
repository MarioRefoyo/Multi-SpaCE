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
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from experiments.experiment_utils import local_data_loader, label_encoder, store_partial_cfs, \
    generate_settings_combinations, load_parameters_from_json, load_model, prepare_experiment
from experiments.results.results_concatenator import concatenate_result_files

from methods.GlacierCF import GlacierCF

# Needed mode of tf for Glacier
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


DATASETS = [
    # 'ECG200', 'Gunpoint', 'Coffee',
    # 'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA',
    # 'HandOutlines',
    # 'CinCECGTorso',
    # 'CBF', 'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000',
    'NonInvasiveFatalECGThorax2'
]

ADDITIONAL_SUBSAMPLE_SUBSET = 20
PARAMS_PATH = 'experiments/params_cf/baseline_glacier.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'inceptiontime_noscaling'
OC_EXPERIMENT_NAME = 'ae_basic_train'
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 1


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

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        random.seed(params["seed"])

    # Get model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Load Autoencoder
    if params["autoencoder"]:
        ae_model_worker = tf.keras.models.load_model(
            f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5',
            compile=False)
    else:
        ae_model_worker = None

    # Instantiate the Counterfactual Explanation method
    cf_explainer = GlacierCF(
        model_wrapper, X_train, y_train, ae_model_worker,
        w_value=params["w_value"], tau_value=params["tau_value"], lr_list=params["lr_list"],
        w_type=params["w_type"], seed=params["seed"]
    )

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig = np.expand_dims(x_orig_samples_worker[i], axis=0)
        result = cf_explainer.generate_counterfactual(x_orig=x_orig)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i+THREAD_SAMPLES-1,
                      dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, file_suffix_name=exp_name)
    return 1


def experiment_dataset(dataset, exp_name, params):
    X_train, y_train, X_test, y_test, subset_idx, n_classes, model_wrapper, y_pred_train, y_pred_test = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    # Get counterfactuals
    if MULTIPROCESSING:
        # Prepare dict to iterate optimization problem
        samples = []
        for i in range(I_START, len(X_test), THREAD_SAMPLES):
            # Init optimizer
            x_orig_samples = X_test[i:i + THREAD_SAMPLES]

            sample_dict = {
                "dataset": dataset,
                "train_data_tuple": (X_train, y_train),
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "n_classes": n_classes
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        print('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

        # Concatenate the results
        concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    else:
        # Load Autoencoder
        if params["autoencoder"]:
            ae_model_worker = tf.keras.models.load_model(
                f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5',
                compile=False)
        else:
            ae_model_worker = None

        # Instantiate the Counterfactual Explanation method
        cf_explainer = GlacierCF(
            model_wrapper, X_train, y_train, ae_model_worker,
            w_value=params["w_value"], tau_value=params["tau_value"], lr_list=params["lr_list"],
            w_type=params["w_type"], seed=params["seed"]
        )

        # Generate counterfactuals
        results = []
        for i in tqdm(range(len(X_test))):
            x_orig = np.expand_dims(X_test[i], axis=0)
            result = cf_explainer.generate_counterfactual(x_orig=x_orig)
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
    all_params = load_parameters_from_json(PARAMS_PATH)
    params_combinations = generate_settings_combinations(all_params)
    for _, experiment_params in params_combinations.items():
        # Select name
        if experiment_params["autoencoder"]:
            exp_name = "glacier_gpu"
        else:
            exp_name = "glacier_NoAE_gpu"

        # Experiment for datasets
        for dataset in DATASETS:
            if not os.path.isdir(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}"):
                os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}")
            print(f'Starting experiment for dataset {dataset}...')

            if dataset in ["HandOutlines"]:
                experiment_params["additional_subsample_subset"] = ADDITIONAL_SUBSAMPLE_SUBSET

            try:
                experiment_dataset(
                    dataset,
                    exp_name,
                    experiment_params
                )
            except ValueError as msg:
                print(msg)

    print('Finished')
