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

from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, store_partial_cfs, \
    ucr_data_loader, load_parameters_from_json, generate_settings_combinations, get_subsample
from experiments.results.results_concatenator import concatenate_result_files

from methods.SubSpaCECF import SubSpaCECFv2
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder

DATASETS = ['BasicMotions', 'NATOPS', 'UWaveGestureLibrary']
# DATASETS = ['UWaveGestureLibrary', 'NATOPS']
PARAMS_PATH = 'experiments/params/subspacev2_independent.json'
MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 10


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]
    nun_examples_worker = sample_dict["nun_examples"]
    desired_targets_worker = sample_dict["desired_targets"]

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        random.seed(params["seed"])

    # Get model
    model_worker = tf.keras.models.load_model(f'experiments/models/{dataset}/{dataset}_best_model.hdf5')

    # Get outlier calculator
    with open(f'experiments/models/{dataset}/{dataset}_outlier_calculator.pickle', 'rb') as f:
        outlier_calculator_worker = pickle.load(f)

    # Instantiate the Counterfactual Explanation method
    grouped_channels_iter, individual_channels_iter = params["max_iter"]
    cf_explainer = SubSpaCECFv2(
        model_worker, 'tf', outlier_calculator_worker, grouped_channels_iter, individual_channels_iter,
        population_size=params["population_size"], elite_number=params["elite_number"],
        offsprings_number=params["offsprings_number"],
        change_subseq_mutation_prob=params["change_subseq_mutation_prob"],
        add_subseq_mutation_prob=params["add_subseq_mutation_prob"],
        init_pct=params["init_pct"], reinit=params["reinit"],
        invalid_penalization=params["invalid_penalization"], alpha=params["alpha"], beta=params["beta"],
        eta=params["eta"], gamma=params["gamma"], sparsity_balancer=params["sparsity_balancer"],
    )

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig_worker = x_orig_samples_worker[i]
        nun_example_worker = nun_examples_worker[i]
        desired_target_worker = desired_targets_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig_worker, desired_target_worker, nun_example=nun_example_worker)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i+THREAD_SAMPLES-1,
                      dataset, file_suffix_name=exp_name)
    return 1


def experiment_dataset(dataset, exp_name, params):
    # Set seed
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        random.seed(params["seed"])

    # Load data
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)

    # Get a subset of testing data if specified
    if (params["subset"]) & (len(y_test) > params["subset_number"]):
        X_test, y_test, subset_idx = get_subsample(X_test, y_test, params["subset_number"], params["seed"])
    else:
        subset_idx = np.arange(len(X_test))

    # Load model
    model = tf.keras.models.load_model(f'experiments/models/{dataset}/{dataset}_best_model.hdf5')

    # Predict
    y_pred_test_logits = model.predict(X_test, verbose=0)
    y_pred_train_logits = model.predict(X_train, verbose=0)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)
    # Classification report
    print(classification_report(y_test, y_pred_test))

    # Get the NUNs
    if params["independent_channels_nun"]:
        nun_finder = IndependentNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf', n_neighbors=params["n_neighbors"], model=model
        )
    else:
        nun_finder = GlobalNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf'
        )
    nuns, desired_classes, distances = nun_finder.retrieve_nuns(X_test, y_pred_test)
    # ToDo: New SubSpaCe and evolutionary optimizers to support multiple nuns
    nuns = nuns[:, 0, :, :]

    # START COUNTERFACTUAL GENERATION
    if MULTIPROCESSING:
        # Prepare dict to iterate optimization problem
        samples = []
        for i in range(I_START, len(X_test), THREAD_SAMPLES):
            # Init optimizer
            x_orig_samples = X_test[i:i + THREAD_SAMPLES]
            nun_examples = nuns[i:i + THREAD_SAMPLES]
            desired_targets = desired_classes[i:i + THREAD_SAMPLES]

            sample_dict = {
                "dataset": dataset,
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "nun_examples": nun_examples,
                "desired_targets": desired_targets,
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        print('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

    # Concatenate the results
    concatenate_result_files(dataset, exp_name)

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    with open(f'./experiments/results/{dataset}/{exp_name}/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    # Load parameters
    all_params = load_parameters_from_json(PARAMS_PATH)
    params_combinations = generate_settings_combinations(all_params)
    for experiment_name, experiment_params in params_combinations.items():
        for dataset in DATASETS:
            print(f'Starting experiment {experiment_name} for dataset {dataset}...')
            experiment_dataset(
                dataset,
                experiment_name,
                experiment_params
            )
    print('Finished')