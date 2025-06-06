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
import torch
from sklearn.metrics import classification_report

from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, store_partial_cfs, \
    ucr_data_loader, load_parameters_from_json, generate_settings_combinations, get_subsample
from experiments.results.results_concatenator import concatenate_result_files
from methods.outlier_calculators import AEOutlierCalculator

from methods.MultiSubSpaCECF import MultiSubSpaCECF
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder
from methods.MultiSubSpaCE.FeatureImportanceInitializers import GraCAMPlusFI, NoneFI
from experiments.models.utils import load_model


"""DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary', 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]"""
DATASETS = [
    'ECG200', 'Gunpoint', # 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', # 'CinCECGTorso',
    'NonInvasiveFatalECGThorax2', 'CBF',
]
DATASETS = ['ECG200']

# PARAMS_PATH = 'experiments/params_cf/multisubspace_final.json'
# MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
PARAMS_PATH = 'experiments/params_cf/multisubspace_final_pytorch.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'fcn_pytorch'
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
    nun_examples_worker = sample_dict["nun_examples"]
    desired_targets_worker = sample_dict["desired_targets"]
    n_classes = sample_dict["n_classes"]
    ts_length = x_orig_samples_worker.shape[1]
    n_channels = x_orig_samples_worker.shape[2]

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        torch.manual_seed(params["seed"])
        torch.cuda.manual_seed(params["seed"])
        random.seed(params["seed"])

    # Get model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Get outlier calculator
    ae_model = tf.keras.models.load_model(f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5')
    outlier_calculator_worker = AEOutlierCalculator(ae_model, X_train)

    # Get FI method for initialization
    if params["init_fi"] == "none":
        fi_method = NoneFI('tf')
    elif params["init_fi"] == "gradcam++":
        fi_method = GraCAMPlusFI('tf', model_wrapper)
    else:
        raise ValueError("The provided init_fi is not valid.")

    # Instantiate the Counterfactual Explanation method
    grouped_channels_iter, individual_channels_iter, pruning_iter = params["max_iter"]
    cf_explainer = MultiSubSpaCECF(
        model_wrapper, outlier_calculator_worker, fi_method,
        grouped_channels_iter, individual_channels_iter, pruning_iter,
        plausibility_objective=params["plausibility_objective"],
        population_size=params["population_size"],
        change_subseq_mutation_prob=params["change_subseq_mutation_prob"],
        add_subseq_mutation_prob=params["add_subseq_mutation_prob"],
        integrated_pruning_mutation_prob=params["integrated_pruning_mutation_prob"],
        final_pruning_mutation_prob=params["final_pruning_mutation_prob"],
        init_pct=params["init_pct"], reinit=params["reinit"], init_random_mix_ratio=params["init_random_mix_ratio"],
        invalid_penalization=params["invalid_penalization"],
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
    store_partial_cfs(results, first_sample_i, first_sample_i + THREAD_SAMPLES - 1,
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
    ts_length = X_train.shape[1]
    n_channels = X_train.shape[2]
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Get a subset of testing data if specified
    if (params["subset"]) & (len(y_test) > params["subset_number"]):
        X_test, y_test, subset_idx = get_subsample(X_test, y_test, params["subset_number"], params["seed"])
    else:
        subset_idx = np.arange(len(X_test))

    # Get model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)

    # Predict
    y_pred_test_logits = model_wrapper.predict(X_test)
    y_pred_train_logits = model_wrapper.predict(X_train)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)
    # Classification report
    print(classification_report(y_test, y_pred_test))

    # Get the NUNs
    if params["independent_channels_nun"]:
        nun_finder = IndependentNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf', n_neighbors=params["n_neighbors"], model=model_wrapper
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
                "train_data_tuple": (X_train, y_train),
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "nun_examples": nun_examples,
                "desired_targets": desired_targets,
                "n_classes": n_classes
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        print('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

    # Concatenate the results
    concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    with open(f'./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json', 'w') as fp:
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
