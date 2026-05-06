import os
import copy
import random
import pickle
import sys
import json
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

"""if gpus:
    try:
        # Set visible devices to an empty list
        tf.config.set_visible_devices([], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)"""

import torch
from sklearn.metrics import classification_report

from experiments.experiment_utils import store_partial_cfs, load_parameters_from_json,generate_settings_combinations
from experiments.results.results_concatenator import concatenate_result_files

from methods.outlier_calculators import AEOutlierCalculator
from methods.MultiSubSpaCECF import MultiSubSpaCECF
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder, SecondBestGlobalNUNFinder
from methods.MultiSubSpaCE.FeatureImportanceInitializers import GraCAMPlusFI, NoneFI

from experiments.experiment_utils import prepare_experiment, load_model


DATASETS = [
    # 'BasicMotions',
    # 'NATOPS',
   # 'UWaveGestureLibrary', 'Cricket',
   # 'ArticularyWordRecognition', 'Epilepsy',
   # 'PenDigits',
    
   # 'RacketSports',
    'SelfRegulationSCP1',
    'PEMS-SF',
]

"""DATASETS = [
    'ECG200', 'Gunpoint', 'Coffee',
    'ItalyPowerDemand', 'ProximalPhalanxOutlineCorrect', 'Strawberry', 'FordA', 'HandOutlines',
    'Plane', 'TwoPatterns', 'FacesUCR', 'ECG5000', 'CinCECGTorso',
    'NonInvasiveFatalECGThorax2', 'CBF',
]"""

EXPERIMENT_FAMILY = 'multisubspace_second_nun'
# EXPERIMENT_FAMILY = 'multisubspace_final_gpu'
# EXPERIMENT_FAMILY = 'multisubspace_final_noplau_gpu'
PARAMS_PATH = f'experiments/params_cf/{EXPERIMENT_FAMILY}.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"
OC_EXPERIMENT_NAME = 'ae_basic_train'

MULTIPROCESSING = True
I_START = 0
THREAD_SAMPLES = 5
POOL_SIZE = 20
MP_START_METHOD = "spawn"


def get_nun_strategy(params):
    if "nun_strategy" in params:
        return params["nun_strategy"]
    if params.get("independent_channels_nun", False):
        return "independent"
    return "global"


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
    experiment_family = sample_dict["experiment_family"]
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params["seed"])
        random.seed(params["seed"])

    if "run_seed" in params and params["run_seed"] is not None:
        np.random.seed(params["run_seed"])
        tf.random.set_seed(params["run_seed"])
        torch.manual_seed(params["run_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params["run_seed"])
        random.seed(params["run_seed"])

    # Get model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Get outlier calculator only when plausibility is part of the optimization objective
    if params["plausibility_objective"] == "none":
        outlier_calculator_worker = None
    else:
        ae_model = tf.keras.models.load_model(f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5', compile=False)
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
                      dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, file_suffix_name=exp_name,
                      experiment_family=experiment_family)
    return 1


def experiment_dataset(dataset, exp_name, params, experiment_family):
    result_path = Path("experiments/results") / dataset / MODEL_TO_EXPLAIN_EXPERIMENT_NAME / experiment_family / exp_name
    print(f"Result path: {result_path}")

    X_train, y_train, X_test, y_test, subset_idx, n_classes, _, y_pred_train, y_pred_test = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    # Get the NUNs
    nun_strategy = get_nun_strategy(params)
    model_wrapper = None
    if nun_strategy in {"independent", "second_best"}:
        model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
        model_wrapper = load_model(model_folder, dataset, X_train.shape[2], X_train.shape[1], n_classes)

    if nun_strategy == "independent":
        nun_finder = IndependentNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf', n_neighbors=params["n_neighbors"], model=model_wrapper
        )
    elif nun_strategy == "second_best":
        nun_finder = SecondBestGlobalNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf', model=model_wrapper
        )
    elif nun_strategy == "global":
        nun_finder = GlobalNUNFinder(
            X_train, y_train, y_pred_train, distance='euclidean',
            from_true_labels=False, backend='tf'
        )
    else:
        raise ValueError(f"Unsupported nun_strategy: {nun_strategy}")
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
                "experiment_family": experiment_family,
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
    concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name, experiment_family=experiment_family)

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    params["experiment_family"] = experiment_family
    with open(result_path / 'params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    mp.freeze_support()
    configure_tensorflow_runtime(log=True)

    # Load parameters
    all_params = load_parameters_from_json(PARAMS_PATH)
    params_combinations = generate_settings_combinations(all_params)
    for experiment_name, experiment_params in params_combinations.items():
        for dataset in DATASETS:
            print(f'Starting experiment {experiment_name} for dataset {dataset}...')
            experiment_dataset(
                dataset,
                experiment_name,
                experiment_params,
                EXPERIMENT_FAMILY
            )
    print('Finished')
