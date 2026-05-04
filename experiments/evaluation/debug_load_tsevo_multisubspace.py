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
import plotly.io as pio

from experiments.evaluation.evaluation_utils import (
    load_dataset_for_eval,
    calculate_metrics_for_dataset,
    calculate_metrics_for_dataset_mp,
    calculate_pareto_front_metrics_for_dataset,
    generate_critical_difference_diagrams,
    generate_wilcoxon_holm_tables,
)

print(tf.__version__)


# DATASETS = ['CBF', 'chinatown', 'coffee', 'gunpoint', 'ECG200']
DATASETS = [
    "BasicMotions",
    "NATOPS",
    "UWaveGestureLibrary",
    'ArticularyWordRecognition', 'Cricket',
    'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]
DATASETS = ["CBF",]

MO_UTILITY = np.array([0.1, 0.4*0.7, 0.6*0.7, 0.2])
model_to_explain = "inceptiontime_noscaling"
scaling = "none"
osc_names = {"AE": "ae_basic_train", "IF": "if_basic_train", "LOF": "lof_basic_train"}
methods = {
    # Comparison between single objective and multi objective ordered by same weights
    # "abcf_gpu": "AB-CF",
    "tsevo": "TSEvo",
    # "574aca0df70f6f6d3ceac2fbf11faa9dc1b3b8ce": "Multi-SpaCE Global",
}


PENALIZATION_QUANTILES = ['none', 0.75, 0.95, 1.0]
PENALIZE_INVALID = True
PARETO_PLAUSIBILITY_OBJECTIVE = "AE"

# Data and aux data dict
data_dict = {}
models_dict = {}
outlier_calculators_dict = {}
possible_nuns_dict = {}
desired_classes_dict = {}
original_classes_dict = {}

# Results dicts keyed by penalization choice
mean_results_dict_all = {}
methods_cfs_dict_all = {}
results_all_datasets_df_all = {}
common_test_indexes_dict_all = {}

# Iterate through datasets
for dataset in DATASETS:
    print(f'Calculating metrics for {dataset}')
    data_tuple, original_classes, model, outlier_calculators, possible_nuns, desired_classes = load_dataset_for_eval(dataset, model_to_explain, osc_names, scaling=scaling)
    data_dict[dataset] = data_tuple
    models_dict[dataset] = model
    outlier_calculators_dict[dataset] = outlier_calculators
    possible_nuns_dict[dataset] = possible_nuns
    desired_classes_dict[dataset] = desired_classes
    original_classes_dict[dataset] = original_classes

    dataset_results = calculate_metrics_for_dataset(
        dataset, methods, model_to_explain,
        data_tuple, original_classes, model, outlier_calculators, possible_nuns,
        mo_weights=MO_UTILITY,
        penalize_invalid=PENALIZE_INVALID,
        penalization_quantile=PENALIZATION_QUANTILES,
    )

    mean_results_dict_all[dataset] = {key: result['mean_std_df'] for key, result in dataset_results.items()}
    methods_cfs_dict_all[dataset] = {key: result['method_cfs_dataset'] for key, result in dataset_results.items()}
    results_all_datasets_df_all[dataset] = {key: result['results_df'] for key, result in dataset_results.items()}
    common_test_indexes_dict_all[dataset] = {key: result['common_test_indexes'] for key, result in dataset_results.items()}


from IPython.display import display

selected_penalization_key = 0.95

mean_results_dict = {
    dataset: mean_results_dict_all[dataset][selected_penalization_key]
    for dataset in DATASETS
}
methods_cfs_dict = {
    dataset: methods_cfs_dict_all[dataset][selected_penalization_key]
    for dataset in DATASETS
}
common_test_indexes_dict = {
    dataset: common_test_indexes_dict_all[dataset][selected_penalization_key]
    for dataset in DATASETS
}
results_all_datasets_df = pd.concat(
    [results_all_datasets_df_all[dataset][selected_penalization_key] for dataset in DATASETS],
    ignore_index=True,
)

print(f'Selected penalization key: {selected_penalization_key}')


pareto_comparison_all = {}
for dataset in DATASETS:
    pareto_comparison_all[dataset] = calculate_pareto_front_metrics_for_dataset(
        dataset,
        methods,
        model_to_explain,
        data_dict[dataset],
        original_classes_dict[dataset],
        models_dict[dataset],
        outlier_calculators_dict[dataset],
        possible_nuns_dict[dataset],
        plausibility_objective=PARETO_PLAUSIBILITY_OBJECTIVE,
    )

pareto_summary_df = pd.concat(
    [pareto_comparison_all[dataset]["summary_df"] for dataset in DATASETS],
    ignore_index=True,
)
pareto_pairwise_df = pd.concat(
    [pareto_comparison_all[dataset]["pairwise_df"] for dataset in DATASETS],
    ignore_index=True,
)
