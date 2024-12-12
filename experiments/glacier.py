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

from methods.GlacierCF import GlacierCF

# Needed mode of tf for Glacier
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)


DATASETS = [
    'ECG200', 'Gunpoint', 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'CBF', 'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', 'CinCECGTorso', 'NonInvasiveFatalECGThorax2'
]
PARAMS_PATH = 'experiments/params_cf/baseline_glacier.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'cls_basic_train'
OC_EXPERIMENT_NAME = 'ae_basic_train'
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 10


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

    # Load Autoencoder
    if params["autoencoder"]:
        ae_model_worker = tf.keras.models.load_model(f'experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5')
    else:
        ae_model_worker = None

    # Instantiate the Counterfactual Explanation method
    cf_explainer = GlacierCF(
        model_worker, 'tf', X_train, y_train, ae_model_worker,
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
    # Set seed
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        random.seed(params["seed"])

    # Load data
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), min_max_scaling=False, data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)

    # Glacier only works for binary classification problems
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"Glacier only works for binary classification problems. {dataset} has {len(np.unique(y_train))} distinct labels")

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
            ae_model = tf.keras.models.load_model(f'experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5')
        else:
            ae_model = None

        cf_explainer = GlacierCF(
            model, 'tf', X_train, y_train, ae_model,
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
            exp_name = "glacier"
        else:
            exp_name = "glacier_NoAE"

        # Experiment for datasets
        for dataset in DATASETS:
            if not os.path.isdir(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}"):
                os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}")
            print(f'Starting experiment for dataset {dataset}...')
            try:
                experiment_dataset(
                    dataset,
                    exp_name,
                    experiment_params
                )
            except ValueError as msg:
                print(msg)

    print('Finished')
