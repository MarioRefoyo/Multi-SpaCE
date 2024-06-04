import os
import pickle
import pandas as pd

from experiments.evaluation.evaluation_utils import calculate_metrics_for_dataset
from experiments.evaluation.evaluation_utils import load_dataset_for_eval

DATASETS = ["BasicMotions", 'UWaveGestureLibrary']
methods = ['subspace_individual', 'subspace_grouped', 'subspace_v2_individual', 'subspace_v2_grouped']
counterfactual_methods = [f"{method}.pickle" for method in methods]


if __name__ == "__main__":
    # Data and aux data dict
    data_dict = {}
    models_dict = {}
    outlier_calculators_dict = {}
    nuns_idx_dict = {}
    desired_classes_dict = {}
    original_classes_dict = {}

    # Results dicts
    mean_results_dict = {}
    methods_cfs_dict = {}
    results_all_datasets_df = pd.DataFrame()

    # Iterate through datasets
    for dataset in DATASETS:
        print(f'Calculating metrics for {dataset}')
        # Load all info needed to get the counterfactual
        data_tuple, original_classes, model, outlier_calculator, nuns_idx, desired_classes = load_dataset_for_eval(
            dataset)
        data_dict[dataset] = data_tuple
        models_dict[dataset] = model
        outlier_calculators_dict[dataset] = outlier_calculator
        nuns_idx_dict[dataset] = nuns_idx
        desired_classes_dict[dataset] = desired_classes
        original_classes_dict[dataset] = original_classes

        # Get the metrics for all methods
        dataset_mean_std_df, dataset_results_df, method_cfs_dataset = calculate_metrics_for_dataset(
            dataset, counterfactual_methods,
            data_tuple, original_classes, model, outlier_calculator, nuns_idx
        )
        mean_results_dict[dataset] = dataset_mean_std_df
        methods_cfs_dict[dataset] = method_cfs_dataset
        results_all_datasets_df = pd.concat([results_all_datasets_df, dataset_results_df])

        # Store results
        dataset_mean_std_df.to_csv(f'./experiments/evaluation/results_mean_metrics_{dataset}.csv', sep=";", index=False)

    # Store all results
    results_all_datasets_df.to_csv(f'./experiments/evaluation/results_all.csv', sep=";", index=False)
