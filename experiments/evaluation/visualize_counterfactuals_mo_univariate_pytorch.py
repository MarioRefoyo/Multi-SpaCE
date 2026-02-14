import os
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import seaborn as sns
import plotly.express as px

from experiments.evaluation.evaluation_utils import load_dataset_for_eval, calculate_metrics_for_dataset, calculate_metrics_for_dataset_mp

DATASETS = [
    'ECG200',
]
MO_UTILITY = np.array([0.1, 0.4*0.8, 0.6*0.8, 0.10])
model_to_explain = "fcn_pytorch"
scaling = "standard"
osc_names = {"AE": "ae_basic_train", "IF": "if_basic_train", "LOF": "lof_basic_train"}
methods = {
    "ng": "NG",
    # "abcf": "AB-CF",
    "d8a9658768838580632011a7329d8bbb0ce8a55a": "Multi-SpaCE",
}


if __name__ == "__main__":
    # Data and aux data dict
    data_dict = {}
    models_dict = {}
    outlier_calculators_dict = {}
    possible_nuns_dict = {}
    desired_classes_dict = {}
    original_classes_dict = {}

    # Results dicts
    mean_results_dict = {}
    methods_cfs_dict = {}
    results_all_datasets_df = pd.DataFrame()
    common_test_indexes_dict = {}

    # Iterate through datasets
    for dataset in DATASETS:
        print(f'Calculating metrics for {dataset}')
        # Load all info needed to get the counterfactual
        data_tuple, original_classes, model_wrapper, outlier_calculators, possible_nuns, desired_classes = load_dataset_for_eval(
            dataset, model_to_explain, osc_names, scaling=scaling)
        data_dict[dataset] = data_tuple
        models_dict[dataset] = model_wrapper
        outlier_calculators_dict[dataset] = outlier_calculators
        possible_nuns_dict[dataset] = possible_nuns
        desired_classes_dict[dataset] = desired_classes
        original_classes_dict[dataset] = original_classes

        # Get the metrics for all methods
        dataset_mean_std_df, dataset_results_df, method_cfs_dataset, common_test_indexes = calculate_metrics_for_dataset(
            dataset, methods, model_to_explain,
            data_tuple, original_classes, model_wrapper, outlier_calculators, possible_nuns,
            mo_weights=MO_UTILITY
        )
        mean_results_dict[dataset] = dataset_mean_std_df
        methods_cfs_dict[dataset] = method_cfs_dataset
        results_all_datasets_df = pd.concat([results_all_datasets_df, dataset_results_df])
        common_test_indexes_dict[dataset] = common_test_indexes

        # Store results
        # dataset_mean_std_df.to_csv(f'./experiments/evaluation/results_mean_metrics_{dataset}.csv', sep=";", index=False)

    # Store all results
    # results_all_datasets_df.to_csv(f'./experiments/evaluation/results_all.csv', sep=";", index=False)
