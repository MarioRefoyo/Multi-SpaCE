import os
import copy
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras

from methods.outlier_calculators import AEOutlierCalculator
from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, get_subsample
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder


def get_start_end_subsequence_positions(orig_change_mask):
    # ----- Get potential extension locations
    ones_mask = np.in1d(orig_change_mask, 1).reshape(orig_change_mask.shape)
    # Get before and after ones masks
    before_ones_mask = np.roll(ones_mask, -1, axis=0)
    before_ones_mask[ones_mask.shape[0] - 1, :] = False
    after_ones_mask = np.roll(ones_mask, 1, axis=0)
    after_ones_mask[0, :] = False
    # Generate complete mask of after and before ones (and set to False the places where the original ones exist)
    before_after_ones_mask = before_ones_mask + after_ones_mask
    before_after_ones_mask[ones_mask] = False
    return before_after_ones_mask


def calculate_change_mask(x_orig, x_cf, x_nun=None, verbose=0):
    # Get original change mask (could contain points with common values between NUN, x_orig and x_cf)
    orig_change_mask = (x_orig != x_cf).astype(int)
    orig_change_mask = orig_change_mask.T.reshape(-1, 1)

    # Find common values
    if x_nun is not None:
        cv_xorig_nun = (x_orig == x_nun)
        cv_nun_cf = (x_nun == x_cf)
        cv_all = (cv_xorig_nun & cv_nun_cf).astype(int)
        cv_all = cv_all.T.reshape(-1, 1)

        # Check if those common values are at the start or end of a current subsequence
        start_end_mask = cv_all & get_start_end_subsequence_positions(orig_change_mask).astype(int)
        if verbose==1:
            print(orig_change_mask.flatten())
            print(get_start_end_subsequence_positions(orig_change_mask).flatten())
            print(cv_all.flatten())
            print(start_end_mask.flatten())

        # Add noise to those original points that are common to original, NUN and cf
        # are at the beginning or end of a subsequence on the change mask
        noise = np.random.normal(0, 1e-6, x_orig.shape)
        new_x_orig = x_orig + noise * start_end_mask.reshape(x_orig.shape, order='F')

        # Calculate adjusted change mask
        change_mask = (new_x_orig != x_cf).astype(int)
    else:
        change_mask = orig_change_mask

    return change_mask


def load_dataset_for_eval(dataset, model_to_explain, ae_name):
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), min_max_scaling=False, data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)
    data_tuple = (X_train, y_train, X_test, y_test)

    # Load model
    model = keras.models.load_model(f'./experiments/models/{dataset}/{model_to_explain}/model.hdf5')
    # Predict
    y_pred_test_logits = model.predict(X_test, verbose=0)
    y_pred_train_logits = model.predict(X_train, verbose=0)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)

    # Load outlier calculator
    ae = keras.models.load_model(f'./experiments/models/{dataset}/{ae_name}/model.hdf5')
    outlier_calculator = AEOutlierCalculator(ae, X_train)

    # Get the NUNs
    possible_nuns = {}
    # Get nuns with global knn
    nun_finder = GlobalNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean',
        from_true_labels=False, backend='tf'
    )
    gknn_nuns, desired_classes, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    gknn_nuns = gknn_nuns[:, 0, :, :]
    possible_nuns['gknn'] = gknn_nuns
    # Get nuns with individual knn for channels
    nun_finder = IndependentNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean', n_neighbors=1,
        from_true_labels=False, backend='tf', model=model
    )
    iknn_nuns, desired_classes, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    iknn_nuns = iknn_nuns[:, 0, :, :]
    possible_nuns['iknn'] = iknn_nuns
    # NOTE: DESIRED CLASSES ARE ALWAYS THE SAME

    return data_tuple, y_pred_test, model, outlier_calculator, possible_nuns, desired_classes


def calculate_metrics_for_dataset(dataset, methods, model_to_explain,
                                  data_tuple, original_classes, model, outlier_calculator, possible_nuns):
    X_train, y_train, X_test, y_test = data_tuple

    results_df = pd.DataFrame()
    cf_solution_dirs = [fname for fname in os.listdir(f'./experiments/results/{dataset}/{model_to_explain}') if os.path.isdir(f'./experiments/results/{dataset}/{model_to_explain}/{fname}')]
    desired_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in cf_solution_dirs if cf_sol_dir in methods.keys()]
    valid_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in desired_cf_solution_dirs if os.path.isfile(f'./experiments/results/{dataset}/{model_to_explain}/{cf_sol_dir}/counterfactuals.pickle')]
    method_cfs_dataset = {}
    common_test_indexes = list(range(len(X_test)))
    for i, method_dir_name in enumerate(valid_cf_solution_dirs):
        # Load solution cfs
        with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/counterfactuals.pickle', 'rb') as f:
            print(method_dir_name)
            method_cfs = pickle.load(f)
        # Load params
        with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/params.json', 'r') as json_file:
            method_params = json.load(json_file)
            method_test_indexes = method_params["X_test_indexes"]

        # Get nuns used by the method depending on the name
        if "independent_channels_nun" in method_params:
            if method_params["independent_channels_nun"]:
                nuns = possible_nuns["iknn"]
            else:
                nuns = possible_nuns["gknn"]
        else:
            nuns = np.array([None]*len(X_test))

        # Calculate metrics
        method_name = methods[method_dir_name]
        method_metrics = calculate_method_metrics(model, outlier_calculator,
                                                  X_test[method_test_indexes], nuns[method_test_indexes], method_cfs,
                                                  original_classes[method_test_indexes], method_name, order=i + 1)
        method_metrics.insert(0, "ii", method_test_indexes)
        results_df = pd.concat([results_df, method_metrics])
        method_cfs_dataset[method_name] = method_cfs
        common_test_indexes = list(set(method_test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    # Calculate results table for the dataset
    means_df = results_df.groupby('method').mean()
    means_df = means_df.sort_values('order').drop('order', axis=1)
    stds_df = results_df.groupby('method').std()
    stds_df = stds_df.drop('order', axis=1)
    stds_df = stds_df.reindex(means_df.index)
    mean_std_df = means_df.round(2).astype(str) + " ± " + stds_df.round(2).astype(str)
    mean_std_df = mean_std_df.reset_index()
    results_df['dataset'] = dataset

    return mean_std_df, results_df, method_cfs_dataset, common_test_indexes


def calculate_method_valids(model, X_test, counterfactuals, original_classes):
    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    # Loop over counterfactuals
    valids = []
    for i in tqdm(range(len(X_test))):
        counterfactuals[i] = counterfactuals[i].reshape(length, n_channels)

        # Predict counterfactual class probability
        preds = model.predict(counterfactuals[i].reshape(-1, length, n_channels), verbose=0)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if pred_class != original_classes[i]:
            valids.append(True)
        else:
            valids.append(False)

    return valids


def calculate_method_metrics(model, outlier_calculator, X_test, nuns, solutions_in, original_classes,
                             method_name, order=None):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    counterfactuals = [solution['cf'] for solution in solutions]
    execution_times = [solution['time'] for solution in solutions]

    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    # Loop over counterfactuals
    nchanges = []
    l1s = []
    l2s = []
    pred_probas = []
    valids = []
    n_subsequences = []
    for i in tqdm(range(len(X_test))):

        counterfactuals[i] = counterfactuals[i].reshape(length, n_channels)

        # Predict counterfactual class probability
        preds = model.predict(counterfactuals[i].reshape(-1, length, n_channels), verbose=0)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if pred_class != original_classes[i]:
            valids.append(True)

            # Add class probability
            pred_proba = preds[0, pred_class]
            pred_probas.append(pred_proba)

            # Calculate l0
            # change_mask = (X_test[i] != counterfactuals[i]).astype(int)
            # print(X_test[i].shape, X_train[nuns_idx[i]].shape, counterfactuals[i].shape)
            change_mask = calculate_change_mask(X_test[i], counterfactuals[i], x_nun=nuns[i], verbose=0)
            nchanges.append(change_mask.sum())

            # Calculate l1
            l1 = np.linalg.norm((X_test[i].flatten() - counterfactuals[i].flatten()), ord=1)
            l1s.append(l1)

            # Calculate l2
            l2 = np.linalg.norm((X_test[i].flatten() - counterfactuals[i].flatten()), ord=2)
            l2s.append(l2)

            # Number of sub-sequences
            # print(change_mask.shape)
            subsequences = np.count_nonzero(np.diff(change_mask, prepend=0, axis=0) == 1, axis=(0,1))
            n_subsequences.append(subsequences)
        else:
            valids.append(False)
            # Append all NaNs to not being take into consideration
            pred_probas.append(np.nan)
            nchanges.append(np.nan)
            l1s.append(np.nan)
            l2s.append(np.nan)
            n_subsequences.append(np.nan)

    # Valid NUN classes
    if nuns[0] is None:
        valid_nuns = [np.nan]*len(nuns)
    else:
        nun_preds = model.predict(nuns, verbose=0)
        nun_pred_class = np.argmax(nun_preds, axis=1)
        valid_nuns = nun_pred_class != original_classes

    # Outlier scores
    # Increase in outlier score
    outlier_scores = outlier_calculator.get_outlier_scores(np.array(counterfactuals).reshape(-1, length, n_channels))
    outlier_scores_orig = outlier_calculator.get_outlier_scores(X_test)
    # outlier_scores_nuns = outlier_calculator.get_outlier_scores(nuns)
    increase_os = outlier_scores - outlier_scores_orig
    increase_os[increase_os < 0] = 0
    # Put to nan all the non valid cfs
    valids_array = np.array(valids).flatten()
    increase_os = increase_os.flatten()
    increase_os[valids_array == False] = np.nan
    outlier_scores = outlier_scores.flatten()
    outlier_scores[valids_array == False] = np.nan

    # Create dataframe
    results = pd.DataFrame()
    results["nchanges"] = nchanges
    results["sparsity"] = results["nchanges"] / (length * n_channels)
    results["L1"] = l1s
    results["L2"] = l2s
    results["proba"] = pred_probas
    results["valid"] = valids
    results["nuns_valid"] = valid_nuns
    results["outlier_score"] = outlier_scores.tolist()
    results["increase_outlier_score"] = increase_os.tolist()
    results['subsequences'] = n_subsequences
    results['subsequences %'] = np.array(n_subsequences) / ((length * n_channels) / 2)
    results['times'] = execution_times
    results['method'] = method_name
    if order is not None:
        results['order'] = order

    return results
