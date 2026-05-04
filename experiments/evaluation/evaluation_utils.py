import os
import copy
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras
from multiprocessing import Pool
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

from methods.outlier_calculators import AEOutlierCalculator, IFOutlierCalculator, LOFOutlierCalculator
from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, get_subsample
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder, SecondBestGlobalNUNFinder
from methods.MultiSubSpaCE.FitnessFunctions import fitness_function_mo, fitness_function_mo_no_plausibility
from experiments.experiment_utils import load_model


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


def thresholded_change_mask(x_orig, x_cf):
    proximity_values = np.abs(x_orig - x_cf)
    threshold_values = np.sqrt(np.abs(x_orig) * 0.0001)
    return (proximity_values > threshold_values).astype(int)

def normal_change_mask(x_orig, x_cf):
    orig_change_mask = (x_orig != x_cf).astype(int)
    return orig_change_mask


def calculate_change_mask(x_orig, x_cf, x_nun=None, verbose=0):
    # Get original change mask (could contain points with common values between NUN, x_orig and x_cf)
    # orig_change_mask = thresholded_change_mask(x_orig, x_cf)
    orig_change_mask = normal_change_mask(x_orig, x_cf)
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
        # change_mask = thresholded_change_mask(new_x_orig, x_cf)
        change_mask = normal_change_mask(new_x_orig, x_cf)
    else:
        change_mask = orig_change_mask.reshape(x_orig.shape, order='F')

    return change_mask


def infer_desired_class_for_ranking(predicted_probs, original_class, preferred_class=None):
    if preferred_class is not None:
        return int(preferred_class)

    predicted_classes = np.argmax(predicted_probs, axis=1)
    valid_mask = predicted_classes != original_class

    if np.any(valid_mask):
        valid_classes = predicted_classes[valid_mask]
        class_values, class_counts = np.unique(valid_classes, return_counts=True)
        max_count = class_counts.max()
        top_classes = class_values[class_counts == max_count]
        if len(top_classes) == 1:
            return int(top_classes[0])

        mean_probs = np.array(
            [predicted_probs[valid_mask, class_id].mean() for class_id in top_classes]
        )
        return int(top_classes[np.argmax(mean_probs)])

    mean_probs = predicted_probs.mean(axis=0).copy()
    mean_probs[int(original_class)] = -np.inf
    return int(np.argmax(mean_probs))


def ensure_cf_batch(counterfactuals_i, length, n_channels):
    arr = np.asarray(counterfactuals_i)
    if arr.ndim == 2:
        return arr.reshape(1, length, n_channels)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], length, n_channels)
    raise ValueError(f"Unexpected counterfactual shape: {arr.shape}")


def load_dataset_for_eval(dataset, model_to_explain, osc_names, scaling="none"):
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), scaling=scaling, backend="tf", data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)
    data_tuple = (X_train, y_train, X_test, y_test)
    ts_length = X_train.shape[1]
    n_channels = X_train.shape[2]
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Load model
    model_folder = f'./experiments/models/{dataset}/{model_to_explain}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Predict
    y_pred_test_logits = model_wrapper.predict(X_test)
    y_pred_train_logits = model_wrapper.predict(X_train)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)

    # Load outlier calculators
    outlier_calculators = {}
    for osc_name, osc_exp_names in osc_names.items():
        if osc_name == "AE":
            ae_model = keras.models.load_model(
                f'./experiments/models/{dataset}/{osc_exp_names}/model.hdf5',
                compile=False,
            )
            outlier_calculators[osc_name] = AEOutlierCalculator(ae_model, X_train)
        elif osc_name == "IF":
            with open(f'./experiments/models/{dataset}/{osc_exp_names}/model.pickle', 'rb') as f:
                if_model = pickle.load(f)
            outlier_calculators[osc_name] = IFOutlierCalculator(if_model, X_train)
        elif osc_name == "LOF":
            with open(f'./experiments/models/{dataset}/{osc_exp_names}/model.pickle', 'rb') as f:
                lof_model = pickle.load(f)
            outlier_calculators[osc_name] = LOFOutlierCalculator(lof_model, X_train)
        else:
            raise ValueError("Not valid name in outlier calculator names.")

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
    nun_finder = SecondBestGlobalNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean',
        from_true_labels=False, backend='tf', model=model_wrapper
    )
    sgknn_nuns, _, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    sgknn_nuns = sgknn_nuns[:, 0, :, :]
    possible_nuns['sgknn'] = sgknn_nuns
    # Get nuns with individual knn for channels
    nun_finder = IndependentNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean', n_neighbors=1,
        from_true_labels=False, backend='tf', model=model_wrapper
    )
    iknn_nuns, desired_classes, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    iknn_nuns = iknn_nuns[:, 0, :, :]
    possible_nuns['iknn'] = iknn_nuns
    # NOTE: desired_classes here follow the last finder executed and should not be assumed to be
    # shared across every NUN strategy.

    return data_tuple, y_pred_test, model_wrapper, outlier_calculators, possible_nuns, desired_classes


def get_method_nun_key(method_params):
    nun_strategy = method_params.get("nun_strategy")
    if nun_strategy == "global":
        return "gknn"
    if nun_strategy == "independent":
        return "iknn"
    if nun_strategy == "second_best":
        return "sgknn"
    if "independent_channels_nun" in method_params:
        if method_params["independent_channels_nun"]:
            return "iknn"
        return "gknn"
    return None


def apply_quantile_penalization(results_df, penalization_quantile):
    if not 0 <= penalization_quantile <= 1:
        raise ValueError("penalization_quantile must be between 0 and 1.")

    metric_columns = [
        column for column in results_df.columns
        if column not in {
            "ii", "valid", "nuns_valid", "times", "method", "best cf index", "order", "dataset"
        }
    ]

    penalized_results_df = results_df.copy()
    for column in metric_columns:
        if penalized_results_df[column].isna().any():
            quantile_value = penalized_results_df[column].quantile(penalization_quantile)
            if not pd.isna(quantile_value):
                penalized_results_df[column] = penalized_results_df[column].fillna(quantile_value)

    return penalized_results_df


def parse_penalization_choice(choice):
    if isinstance(choice, str):
        normalized_choice = choice.strip().lower()
        if normalized_choice == "none":
            return "none", None
        try:
            choice = float(choice)
        except ValueError as exc:
            raise ValueError(
                "penalization_quantile entries must be numeric values in [0, 1] or 'none'."
            ) from exc

    quantile = float(choice)
    if not 0 <= quantile <= 1:
        raise ValueError("penalization_quantile must be between 0 and 1.")

    return quantile, quantile


def get_penalization_choices(penalization_quantile):
    if isinstance(penalization_quantile, str) or np.isscalar(penalization_quantile):
        raw_choices = [penalization_quantile]
    else:
        raw_choices = penalization_quantile

    parsed_choices = []
    seen_keys = set()
    for raw_choice in raw_choices:
        key, quantile = parse_penalization_choice(raw_choice)
        if key not in seen_keys:
            parsed_choices.append((key, quantile))
            seen_keys.add(key)

    return parsed_choices


def should_return_quantile_dict(penalization_quantile):
    return not isinstance(penalization_quantile, str) and not np.isscalar(penalization_quantile)


def build_penalization_results_dict(results_df, dataset, method_cfs_dataset, common_test_indexes,
                                    penalize_invalid, penalization_quantile):
    if not penalize_invalid and not should_return_quantile_dict(penalization_quantile):
        keys_and_frames = [("none", results_df)]
    else:
        keys_and_frames = []
        for key, quantile in get_penalization_choices(penalization_quantile):
            if quantile is None:
                keys_and_frames.append((key, results_df))
            else:
                keys_and_frames.append((key, apply_quantile_penalization(results_df, quantile)))

    return {
        key: {
            "mean_std_df": mean_std_df,
            "results_df": output_results_df,
            "method_cfs_dataset": method_cfs_dataset,
            "common_test_indexes": common_test_indexes,
        }
        for key, frame in keys_and_frames
        for mean_std_df, output_results_df, _, _ in [
            build_dataset_metrics_output(
                frame,
                dataset,
                method_cfs_dataset,
                common_test_indexes,
            )
        ]
    }


def build_dataset_metrics_output(results_df, dataset, method_cfs_dataset, common_test_indexes):
    means_df = results_df.groupby('method').mean()
    means_df = means_df.sort_values('order').drop('order', axis=1)
    stds_df = results_df.groupby('method').std()
    stds_df = stds_df.drop('order', axis=1)
    stds_df = stds_df.reindex(means_df.index)
    mean_std_df = means_df.round(2).astype(str) + " ± " + stds_df.round(2).astype(str)
    mean_std_df = mean_std_df.reset_index()

    results_df = results_df.copy()
    results_df['dataset'] = dataset

    return mean_std_df, results_df, method_cfs_dataset, common_test_indexes


def process_method_dir(args):
    dataset, model_to_explain, method_dir_name, methods, model_wrapper, outlier_calculators, X_test, original_classes, possible_nuns, mo_weights, order = args
    results_df = pd.DataFrame()
    method_cfs_dataset = {}

    # Load solution cfs
    with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/counterfactuals.pickle',
              'rb') as f:
        print(method_dir_name)
        method_cfs = pickle.load(f)

    # Load params
    with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/params.json', 'r') as json_file:
        method_params = json.load(json_file)
        method_test_indexes = method_params["X_test_indexes"]

    # Get nuns used by the method depending on the name
    nun_key = get_method_nun_key(method_params)
    if nun_key is not None:
        nuns = possible_nuns[nun_key]
    else:
        nuns = np.array([None] * len(X_test))

    # Calculate metrics
    method_name = methods[method_dir_name]
    method_metrics = calculate_method_metrics(
        model_wrapper, outlier_calculators, X_test[method_test_indexes],
        nuns[method_test_indexes], method_cfs, original_classes[method_test_indexes],
        method_name, mo_weights=mo_weights, order=order
    )
    method_metrics.insert(0, "ii", method_test_indexes)

    results_df = pd.concat([results_df, method_metrics])
    method_cfs_dataset[method_name] = method_cfs
    common_test_indexes = list(method_test_indexes)

    return results_df, method_cfs_dataset, common_test_indexes


def calculate_metrics_for_dataset_mp(dataset, methods, model_to_explain,
                                     data_tuple, original_classes, model_wrapper, outlier_calculators, possible_nuns,
                                     mo_weights=None, penalize_invalid=False, penalization_quantile=0.95):
    X_train, y_train, X_test, y_test = data_tuple

    cf_solution_dirs = [fname for fname in os.listdir(f'./experiments/results/{dataset}/{model_to_explain}') if
                        os.path.isdir(f'./experiments/results/{dataset}/{model_to_explain}/{fname}')]
    desired_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in cf_solution_dirs if cf_sol_dir in methods.keys()]
    valid_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in desired_cf_solution_dirs if os.path.isfile(
        f'./experiments/results/{dataset}/{model_to_explain}/{cf_sol_dir}/counterfactuals.pickle')]

    # Prepare arguments for parallel processing
    args = [
        (dataset, model_to_explain, method_dir_name, methods, model_wrapper, outlier_calculators, X_test,
         original_classes, possible_nuns, mo_weights, i + 1)
        for i, method_dir_name in enumerate(valid_cf_solution_dirs)
    ]

    # Use multiprocessing pool to parallelize the processing of each method directory
    with Pool(10) as pool:
        results = pool.map(process_method_dir, args)

    # Collect results
    results_df = pd.DataFrame()
    method_cfs_dataset = {}
    common_test_indexes = list(range(len(X_test)))

    for res_df, method_cfs, test_indexes in results:
        results_df = pd.concat([results_df, res_df])
        method_cfs_dataset.update(method_cfs)
        common_test_indexes = list(set(test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    if penalize_invalid or should_return_quantile_dict(penalization_quantile):
        return build_penalization_results_dict(
            results_df, dataset, method_cfs_dataset, common_test_indexes,
            penalize_invalid, penalization_quantile
        )

    return build_dataset_metrics_output(results_df, dataset, method_cfs_dataset, common_test_indexes)


def calculate_metrics_for_dataset(dataset, methods, model_to_explain,
                                  data_tuple, original_classes, model, outlier_calculators, possible_nuns,
                                  mo_weights=None, penalize_invalid=False, penalization_quantile=0.95):
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
        nun_key = get_method_nun_key(method_params)
        if nun_key is not None:
            nuns = possible_nuns[nun_key]
        else:
            nuns = np.array([None]*len(X_test))

        # Calculate metrics
        method_name = methods[method_dir_name]
        method_metrics = calculate_method_metrics(model, outlier_calculators,
                                                  X_test[method_test_indexes], nuns[method_test_indexes], method_cfs,
                                                  original_classes[method_test_indexes], method_name,
                                                  mo_weights=mo_weights, order=i + 1)
        method_metrics.insert(0, "ii", method_test_indexes)
        results_df = pd.concat([results_df, method_metrics])
        method_cfs_dataset[method_name] = method_cfs
        common_test_indexes = list(set(method_test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    if penalize_invalid or should_return_quantile_dict(penalization_quantile):
        return build_penalization_results_dict(
            results_df, dataset, method_cfs_dataset, common_test_indexes,
            penalize_invalid, penalization_quantile
        )

    print(f"Common test indexes are {len(common_test_indexes)}: {common_test_indexes}")

    return build_dataset_metrics_output(results_df, dataset, method_cfs_dataset, common_test_indexes)


def get_method_objectives(model, outlier_calculator, X_test, nuns, solutions_in, original_classes):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    # Check if the solutions are single or multiple solutions
    if 'cfs' in solutions[0]:
        counterfactuals = [solution['cfs'] for solution in solutions]
    else:
        counterfactuals = [solution['cf'] for solution in solutions]
    execution_times = [solution['time'] for solution in solutions]

    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    all_objectives_list = []
    for i in tqdm(range(len(X_test))):
        x_orig_i = X_test[i]
        counterfactuals_i = ensure_cf_batch(counterfactuals[i], length, n_channels)

        # Calculate valids
        predicted_logits = model.predict(counterfactuals_i)
        predicted_classes = np.argmax(predicted_logits, axis=1)
        valids = (predicted_classes != original_classes[i]).astype(int)

        # Filter counterfactuals based on valids
        valid_idx = np.where(valids == 1)[0]
        if len(valid_idx) == 0:
            # Calculate objectives dict
            sample_objectives_dict = {
                "valids": [0],
                "desired_class_prob": [np.nan],
                "sparsity": [np.nan],
                "subsequences": [np.nan],
                "IoS": [np.nan],
                "L2": [np.nan],
                "execution_time": execution_times[i]
            }

        else:
            valid_counterfactuals_i = counterfactuals_i[valid_idx]
            n_counterfactuals_i = valid_counterfactuals_i.shape[0]

            # Calculate desired class based on NUN
            if nuns[i] is not None:
                desired_class = np.argmax(
                    model.predict(np.expand_dims(nuns[i], axis=0)), axis=1
                )[0]
            else:
                desired_class = infer_desired_class_for_ranking(
                    predicted_logits[valid_idx], original_classes[i]
                )

            # Calculate predicted probabilities
            desired_predicted_probs = predicted_logits[valid_idx, desired_class]

            # Use the same change-mask logic as the main metric evaluation.
            change_masks = np.stack(
                [
                    calculate_change_mask(x_orig_i, cf, x_nun=nuns[i], verbose=0)
                    for cf in valid_counterfactuals_i
                ],
                axis=0,
            )
            sparsity = change_masks.sum(axis=(1, 2)) / (length * n_channels)

            # Subsequences
            subsequences = np.count_nonzero(np.diff(change_masks, prepend=0, axis=1) == 1, axis=(1, 2))
            subsequences_pct = subsequences / ((length * n_channels) / 2)

            # Calculate outlier scores
            if outlier_calculator is not None:
                aux_outlier_scores = outlier_calculator.get_outlier_scores(valid_counterfactuals_i)
                aux_increase_outlier_score = aux_outlier_scores - outlier_calculator.get_outlier_scores(x_orig_i)[0]
            else:
                aux_increase_outlier_score = np.zeros((n_counterfactuals_i, 1))
            aux_increase_outlier_score[aux_increase_outlier_score < 0] = 0
            l2_proximity = np.linalg.norm(
                (valid_counterfactuals_i - x_orig_i).reshape(n_counterfactuals_i, -1),
                ord=2,
                axis=1,
            )

            # Calculate objectives dict
            sample_objectives_dict = {
                "valids": valids,
                "desired_class_prob": desired_predicted_probs,
                "sparsity": sparsity,
                "subsequences": subsequences_pct,
                "IoS": aux_increase_outlier_score,
                "L2": l2_proximity,
                "execution_time": execution_times[i]
            }
        # Append to list
        all_objectives_list.append(sample_objectives_dict)
    return all_objectives_list


def objective_dict_to_minimization_matrix(sample_objectives_dict):
    sparsity = np.asarray(sample_objectives_dict["sparsity"]).reshape(-1)
    subsequences = np.asarray(sample_objectives_dict["subsequences"]).reshape(-1)
    ios = np.asarray(sample_objectives_dict["IoS"]).reshape(-1)
    l2 = np.asarray(sample_objectives_dict["L2"]).reshape(-1)

    if sparsity.size == 0:
        return np.empty((0, 4), dtype=float)

    valid_mask = (
        np.isfinite(sparsity)
        & np.isfinite(subsequences)
        & np.isfinite(ios)
        & np.isfinite(l2)
    )
    if not np.any(valid_mask):
        return np.empty((0, 4), dtype=float)

    return np.column_stack(
        [
            sparsity[valid_mask],
            subsequences[valid_mask],
            ios[valid_mask],
            l2[valid_mask],
        ]
    )


def dominates_minimization(point_a, point_b, atol=1e-12):
    return np.all(point_a <= point_b + atol) and np.any(point_a < point_b - atol)


def filter_nondominated_minimization(points, atol=1e-12):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 0), dtype=float) if points.ndim == 1 else points.reshape(0, points.shape[-1])

    keep_mask = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not keep_mask[i]:
            continue
        for j in range(points.shape[0]):
            if i == j or not keep_mask[j]:
                continue
            if dominates_minimization(points[j], points[i], atol=atol):
                keep_mask[i] = False
                break
    return points[keep_mask]


def normalize_objective_fronts(fronts, atol=1e-12):
    non_empty_fronts = [front for front in fronts if front.size > 0]
    if len(non_empty_fronts) == 0:
        return fronts, None, None

    combined = np.vstack(non_empty_fronts)
    min_values = combined.min(axis=0)
    max_values = combined.max(axis=0)
    ranges = max_values - min_values
    ranges[ranges <= atol] = 1.0

    normalized_fronts = []
    for front in fronts:
        if front.size == 0:
            normalized_fronts.append(front.copy())
        else:
            normalized_fronts.append((front - min_values) / ranges)
    return normalized_fronts, min_values, ranges


def hypervolume_minimization(points, reference_point):
    points = np.asarray(points, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)

    if points.size == 0:
        return 0.0

    valid_mask = np.all(points <= reference_point, axis=1)
    points = points[valid_mask]
    if points.size == 0:
        return 0.0

    points = filter_nondominated_minimization(points)

    def _recursive(front, ref):
        if front.shape[0] == 0:
            return 0.0
        if front.shape[1] == 1:
            return max(ref[0] - np.min(front[:, 0]), 0.0)

        order = np.argsort(front[:, 0])
        front = front[order]
        volume = 0.0
        previous = ref[0]

        while front.shape[0] > 0:
            current = front[-1, 0]
            width = previous - current
            if width > 0:
                volume += width * _recursive(front[:, 1:], ref[1:])
                previous = current
            front = front[front[:, 0] < current]

        return volume

    return float(_recursive(points, reference_point))


def empirical_coverage(front_a, front_b, atol=1e-12):
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)

    if front_b.size == 0:
        return np.nan
    if front_a.size == 0:
        return 0.0

    dominated_count = 0
    for point_b in front_b:
        if any(dominates_minimization(point_a, point_b, atol=atol) for point_a in front_a):
            dominated_count += 1
    return dominated_count / front_b.shape[0]


def igd_plus(approximation_front, reference_front):
    approximation_front = np.asarray(approximation_front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)

    if reference_front.size == 0:
        return np.nan
    if approximation_front.size == 0:
        return np.inf

    distances = []
    for reference_point in reference_front:
        delta = np.maximum(approximation_front - reference_point, 0.0)
        dists = np.linalg.norm(delta, axis=1)
        distances.append(dists.min())
    return float(np.mean(distances))


def compare_pareto_fronts(front_a, front_b, hv_reference_point=None, normalize=True, hv_ref_margin=0.1):
    front_a = filter_nondominated_minimization(np.asarray(front_a, dtype=float))
    front_b = filter_nondominated_minimization(np.asarray(front_b, dtype=float))

    if normalize:
        [front_a, front_b], _, _ = normalize_objective_fronts([front_a, front_b])

    non_empty_fronts = [front for front in [front_a, front_b] if front.size > 0]
    if len(non_empty_fronts) == 0:
        return {
            "hypervolume_a": np.nan,
            "hypervolume_b": np.nan,
            "empirical_coverage_a_over_b": np.nan,
            "empirical_coverage_b_over_a": np.nan,
            "igd_plus_a": np.nan,
            "igd_plus_b": np.nan,
            "n_points_a": 0,
            "n_points_b": 0,
            "n_points_reference": 0,
        }

    union_front = filter_nondominated_minimization(np.vstack(non_empty_fronts))
    if hv_reference_point is None:
        reference_point = np.ones(union_front.shape[1], dtype=float) + hv_ref_margin
    else:
        reference_point = np.asarray(hv_reference_point, dtype=float)

    return {
        "hypervolume_a": hypervolume_minimization(front_a, reference_point),
        "hypervolume_b": hypervolume_minimization(front_b, reference_point),
        "empirical_coverage_a_over_b": empirical_coverage(front_a, front_b),
        "empirical_coverage_b_over_a": empirical_coverage(front_b, front_a),
        "igd_plus_a": igd_plus(front_a, union_front),
        "igd_plus_b": igd_plus(front_b, union_front),
        "n_points_a": int(front_a.shape[0]),
        "n_points_b": int(front_b.shape[0]),
        "n_points_reference": int(union_front.shape[0]),
    }


def obtain_cfs_objectives(dataset, methods, model_to_explain,
                          data_tuple, original_classes, model, outlier_calculator, possible_nuns):

    X_train, y_train, X_test, y_test = data_tuple

    cf_solution_dirs = [fname for fname in os.listdir(f'./experiments/results/{dataset}/{model_to_explain}') if os.path.isdir(f'./experiments/results/{dataset}/{model_to_explain}/{fname}')]
    desired_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in cf_solution_dirs if cf_sol_dir in methods.keys()]
    valid_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in desired_cf_solution_dirs if os.path.isfile(f'./experiments/results/{dataset}/{model_to_explain}/{cf_sol_dir}/counterfactuals.pickle')]
    method_cfs_dataset_dict = {}
    method_objectives_dataset_dict = {}
    method_test_indexes_dict = {}
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
        nun_key = get_method_nun_key(method_params)
        if nun_key is not None:
            nuns = possible_nuns[nun_key]
        else:
            nuns = np.array([None]*len(X_test))

        # Calculate metrics
        method_name = methods[method_dir_name]
        method_objectives = get_method_objectives(
            model, outlier_calculator,
            X_test[method_test_indexes], nuns[method_test_indexes], method_cfs,
            original_classes[method_test_indexes]
        )
        method_objectives_dataset_dict[method_name] = method_objectives
        method_cfs_dataset_dict[method_name] = method_cfs
        method_test_indexes_dict[method_name] = np.array(method_test_indexes)
        common_test_indexes = list(set(method_test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    return method_objectives_dataset_dict, method_cfs_dataset_dict, method_test_indexes_dict, common_test_indexes


def calculate_pareto_front_metrics_for_dataset(
    dataset,
    methods,
    model_to_explain,
    data_tuple,
    original_classes,
    model,
    outlier_calculator,
    possible_nuns,
    plausibility_objective="AE",
    normalize=True,
    hv_reference_point=None,
    hv_ref_margin=0.1,
):
    if isinstance(outlier_calculator, dict):
        if plausibility_objective not in outlier_calculator:
            raise ValueError(
                f'Plausibility calculator "{plausibility_objective}" not found. '
                f'Available keys: {sorted(outlier_calculator.keys())}'
            )
        selected_outlier_calculator = outlier_calculator[plausibility_objective]
    else:
        selected_outlier_calculator = outlier_calculator

    (
        method_objectives_dataset_dict,
        method_cfs_dataset_dict,
        method_test_indexes_dict,
        common_test_indexes,
    ) = obtain_cfs_objectives(
        dataset,
        methods,
        model_to_explain,
        data_tuple,
        original_classes,
        model,
        selected_outlier_calculator,
        possible_nuns,
    )

    pairwise_rows = []
    method_names = list(method_objectives_dataset_dict.keys())
    for method_a, method_b in combinations(method_names, 2):
        index_to_objectives_a = {
            int(sample_idx): method_objectives_dataset_dict[method_a][i]
            for i, sample_idx in enumerate(method_test_indexes_dict[method_a])
        }
        index_to_objectives_b = {
            int(sample_idx): method_objectives_dataset_dict[method_b][i]
            for i, sample_idx in enumerate(method_test_indexes_dict[method_b])
        }
        shared_indexes = sorted(set(index_to_objectives_a.keys()).intersection(index_to_objectives_b.keys()))

        for sample_idx in shared_indexes:
            front_a = objective_dict_to_minimization_matrix(index_to_objectives_a[sample_idx])
            front_b = objective_dict_to_minimization_matrix(index_to_objectives_b[sample_idx])
            metrics = compare_pareto_fronts(
                front_a,
                front_b,
                hv_reference_point=hv_reference_point,
                normalize=normalize,
                hv_ref_margin=hv_ref_margin,
            )
            pairwise_rows.append(
                {
                    "dataset": dataset,
                    "sample_idx": int(sample_idx),
                    "method_a": method_a,
                    "method_b": method_b,
                    **metrics,
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        summary_df = pd.DataFrame()
        pairwise_valid_df = pd.DataFrame()
        summary_valid_df = pd.DataFrame()
    else:
        summary_df = (
            pairwise_df.groupby(["dataset", "method_a", "method_b"])
            .agg(
                hypervolume_a_mean=("hypervolume_a", "mean"),
                hypervolume_b_mean=("hypervolume_b", "mean"),
                empirical_coverage_a_over_b_mean=("empirical_coverage_a_over_b", "mean"),
                empirical_coverage_b_over_a_mean=("empirical_coverage_b_over_a", "mean"),
                igd_plus_a_mean=("igd_plus_a", "mean"),
                igd_plus_b_mean=("igd_plus_b", "mean"),
                n_points_a_mean=("n_points_a", "mean"),
                n_points_b_mean=("n_points_b", "mean"),
                n_points_reference_mean=("n_points_reference", "mean"),
                valid_front_success_a=("n_points_a", lambda x: np.mean(np.asarray(x) > 0)),
                valid_front_success_b=("n_points_b", lambda x: np.mean(np.asarray(x) > 0)),
                n_shared_samples=("sample_idx", "count"),
            )
            .reset_index()
        )
        pairwise_valid_df = pairwise_df[
            (pairwise_df["n_points_a"] > 0) & (pairwise_df["n_points_b"] > 0)
        ].copy()
        if pairwise_valid_df.empty:
            summary_valid_df = pd.DataFrame()
        else:
            summary_valid_df = (
                pairwise_valid_df.groupby(["dataset", "method_a", "method_b"])
                .agg(
                    hypervolume_a_mean=("hypervolume_a", "mean"),
                    hypervolume_b_mean=("hypervolume_b", "mean"),
                    empirical_coverage_a_over_b_mean=("empirical_coverage_a_over_b", "mean"),
                    empirical_coverage_b_over_a_mean=("empirical_coverage_b_over_a", "mean"),
                    igd_plus_a_mean=("igd_plus_a", "mean"),
                    igd_plus_b_mean=("igd_plus_b", "mean"),
                    n_points_a_mean=("n_points_a", "mean"),
                    n_points_b_mean=("n_points_b", "mean"),
                    n_points_reference_mean=("n_points_reference", "mean"),
                    n_shared_samples=("sample_idx", "count"),
                )
                .reset_index()
            )

    return {
        "summary_df": summary_df,
        "summary_valid_df": summary_valid_df,
        "pairwise_df": pairwise_df,
        "pairwise_valid_df": pairwise_valid_df,
        "method_objectives_dataset_dict": method_objectives_dataset_dict,
        "method_cfs_dataset_dict": method_cfs_dataset_dict,
        "method_test_indexes_dict": method_test_indexes_dict,
        "common_test_indexes": common_test_indexes,
    }


def calculate_method_valids(model, X_test, counterfactuals, original_classes):
    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    # Loop over counterfactuals
    valids = []
    for i in tqdm(range(len(X_test))):
        # Predict counterfactual class probability
        counterfactuals_i = ensure_cf_batch(counterfactuals[i], length, n_channels)
        preds = model.predict(counterfactuals_i)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if pred_class != original_classes[i]:
            valids.append(True)
        else:
            valids.append(False)

    return valids


def calculate_method_metrics(model_wrapper, outlier_calculators, X_test, nuns, solutions_in, original_classes,
                             method_name, mo_weights=None, order=None):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    # Check if the solutions are single or multiple solutions
    if 'cfs' in solutions[0]:
        # If there are no mo_weights then there is no way to compare the counterfactuals
        if mo_weights is None:
            raise ValueError("There are multiple counterfactuals for a single input instance. "
                             "Weights for objectives must be passed to order the counterfactuals using a "
                             "specific utility function")
        counterfactuals = [solution['cfs'] for solution in solutions]
    else:
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
    outlier_scores_dict = {}
    increase_outlier_scores_dict = {}
    n_subsequences = []
    best_cf_is = []
    if nuns[0] is not None:
        desired_classes = np.argmax(model_wrapper.predict(nuns), axis=1)
    else:
        desired_classes = None
    for i in tqdm(range(len(X_test))):
        counterfactuals_i = ensure_cf_batch(counterfactuals[i], length, n_channels)
        x_orig_i = X_test[i]
        # If there are multiple counterfactuals apply mo_weights
        if counterfactuals_i.shape[0] > 1:
            # Sort by objective weights and take the best
            predicted_probs = model_wrapper.predict(counterfactuals_i)
            preferred_class = desired_classes[i] if desired_classes is not None else None
            desired_class = infer_desired_class_for_ranking(
                predicted_probs, original_classes[i], preferred_class=preferred_class
            )
            # Get outlier scores from AE to get the best CF
            if outlier_calculators is not None:
                aux_outlier_scores = outlier_calculators["AE"].get_outlier_scores(counterfactuals_i)
            else:
                aux_outlier_scores = np.zeros((predicted_probs.shape[0], 1))
            # Get fitness scores
            change_masks = (counterfactuals_i != x_orig_i).astype(int)
            if outlier_calculators is not None:
                original_outlier_score = outlier_calculators["AE"].get_outlier_scores(x_orig_i)[0]
            else:
                original_outlier_score = 0
            if len(mo_weights) == 3:
                objective_fitness = fitness_function_mo_no_plausibility(
                    change_masks, predicted_probs, desired_class, aux_outlier_scores,
                    original_outlier_score, 100
                )
            elif len(mo_weights) == 4:
                objective_fitness = fitness_function_mo(
                    change_masks, predicted_probs, desired_class, aux_outlier_scores,
                    original_outlier_score, 100
                )
            else:
                raise ValueError(f"Unsupported number of multi-objective weights: {len(mo_weights)}")
            fitness = (objective_fitness * mo_weights).sum(axis=1)
            best_cf_i = np.argsort(fitness)[-1]
            counterfactual_i = counterfactuals_i[best_cf_i].reshape(length, n_channels)
            best_cf_is.append(best_cf_i)
        else:
            best_cf_is.append(0)
            counterfactual_i = counterfactuals_i[0].reshape(length, n_channels)

        # Predict counterfactual class probability
        preds = model_wrapper.predict(counterfactual_i)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if (pred_class != original_classes[i]) and (~np.isnan(counterfactual_i).any()):
            valids.append(True)

            # Add class probability
            pred_proba = preds[0, pred_class]
            pred_probas.append(pred_proba)

            # Calculate l0
            # change_mask = (X_test[i] != counterfactuals[i]).astype(int)
            # print(X_test[i].shape, X_train[nuns_idx[i]].shape, counterfactuals[i].shape)
            change_mask = calculate_change_mask(x_orig_i, counterfactual_i, x_nun=nuns[i], verbose=0)
            nchanges.append(change_mask.sum())

            # Calculate l1
            l1 = np.linalg.norm((x_orig_i.flatten() - counterfactual_i.flatten()), ord=1)
            l1s.append(l1)

            # Calculate l2
            l2 = np.linalg.norm((x_orig_i.flatten() - counterfactual_i.flatten()), ord=2)
            l2s.append(l2)

            # Calculate outlier scores
            for oc_name, outlier_calculator in outlier_calculators.items():
                outlier_score_orig = outlier_calculator.get_outlier_scores(x_orig_i)[0]
                outlier_score = outlier_calculator.get_outlier_scores(counterfactual_i)[0]
                increase_outlier_score = outlier_score - outlier_score_orig
                if oc_name in outlier_scores_dict:
                    outlier_scores_dict[oc_name].append(outlier_score)
                else:
                    outlier_scores_dict[oc_name] = [outlier_score]
                if oc_name in increase_outlier_scores_dict:
                    increase_outlier_scores_dict[oc_name].append(increase_outlier_score)
                else:
                    increase_outlier_scores_dict[oc_name] = [increase_outlier_score]

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
            for oc_name, outlier_calculator in outlier_calculators.items():
                if oc_name in outlier_scores_dict:
                    outlier_scores_dict[oc_name].append(np.nan)
                else:
                    outlier_scores_dict[oc_name] = [np.nan]
                if oc_name in increase_outlier_scores_dict:
                    increase_outlier_scores_dict[oc_name].append(np.nan)
                else:
                    increase_outlier_scores_dict[oc_name] = [np.nan]

    # Valid NUN classes
    if nuns[0] is None:
        valid_nuns = [np.nan]*len(nuns)
    else:
        nun_preds = model_wrapper.predict(nuns)
        nun_pred_class = np.argmax(nun_preds, axis=1)
        valid_nuns = nun_pred_class != original_classes

    # Create dataframe
    results = pd.DataFrame()
    results["nchanges"] = nchanges
    results["sparsity"] = results["nchanges"] / (length * n_channels)
    results["L1"] = l1s
    results["L2"] = l2s
    results["proba"] = pred_probas
    results["valid"] = valids
    results["nuns_valid"] = valid_nuns
    # Create column for Outlier Scores for every calculator
    for oc_name, outlier_scores in outlier_scores_dict.items():
        outlier_scores = np.array(outlier_scores)
        results[f"{oc_name}_OS"] = outlier_scores
    for oc_name, increase_outlier_scores in increase_outlier_scores_dict.items():
        increase_os = np.array(increase_outlier_scores)
        increase_os[increase_os < 0] = 0
        results[f"{oc_name}_IOS"] = increase_os
    results['subsequences'] = n_subsequences
    results['subsequences %'] = np.array(n_subsequences) / ((length * n_channels) / 2)
    results['(sparsity + subsequences %) / 2'] = (
        results['sparsity'] + results['subsequences %']**0.25
    ) / 2
    results['times'] = execution_times
    results['method'] = method_name
    results['best cf index'] = best_cf_is
    if order is not None:
        results['order'] = order

    return results


def build_metric_rank_matrix(results_df, metric, higher_is_better=False, rank_tie_method="average",
                             drop_incomplete_datasets=True):
    metric_col = f"{metric}_mean"
    if metric_col not in results_df.columns:
        raise KeyError(f"Missing required column '{metric_col}' in results_df.")

    valid_tie_methods = {"average", "min", "max", "dense", "first"}
    if rank_tie_method not in valid_tie_methods:
        raise ValueError(
            f"Unsupported rank_tie_method='{rank_tie_method}'. "
            f"Choose one of {sorted(valid_tie_methods)}."
        )

    metric_matrix = results_df.pivot_table(
        index="dataset",
        columns="method",
        values=metric_col,
        aggfunc="mean",
    ).dropna(axis=1, how="all")
    if drop_incomplete_datasets:
        # Keep only datasets with complete values across the compared methods for this metric.
        metric_matrix = metric_matrix.dropna(axis=0, how="any")

    rank_matrix = metric_matrix.rank(axis=1, method=rank_tie_method, ascending=not higher_is_better)
    return metric_matrix, rank_matrix


def compute_pairwise_wilcoxon_holm(metric_matrix, higher_is_better=False, alpha=0.05,
                                   alternative="two-sided", zero_method="wilcox"):
    from scipy.stats import wilcoxon

    try:
        from statsmodels.stats.multitest import multipletests
    except Exception as exc:
        raise ImportError(
            "statsmodels is required for Holm correction. "
            "Install it with: pip install statsmodels"
        ) from exc

    methods = list(metric_matrix.columns)
    pairs = list(combinations(methods, 2))
    if len(pairs) == 0:
        empty_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
        return pd.DataFrame(), empty_matrix

    rows = []
    raw_p_values = []
    for method_a, method_b in pairs:
        vals_a = metric_matrix[method_a].to_numpy(dtype=float)
        vals_b = metric_matrix[method_b].to_numpy(dtype=float)

        if np.allclose(vals_a, vals_b, equal_nan=False):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                vals_a, vals_b,
                alternative=alternative,
                zero_method=zero_method,
                method="auto",
            )

        # Positive deltas mean method_a is better.
        if higher_is_better:
            deltas = vals_a - vals_b
        else:
            deltas = vals_b - vals_a
        deltas = deltas[np.isfinite(deltas)]

        wins_a = int(np.sum(deltas > 0))
        wins_b = int(np.sum(deltas < 0))
        ties = int(np.sum(np.isclose(deltas, 0)))
        median_delta = float(np.median(deltas)) if deltas.size > 0 else np.nan
        mean_delta = float(np.mean(deltas)) if deltas.size > 0 else np.nan

        if np.isclose(median_delta, 0):
            better_method = "tie"
        else:
            better_method = method_a if median_delta > 0 else method_b

        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "wilcoxon_statistic": float(statistic),
            "p_value_raw": float(p_value),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "mean_delta_a_minus_b": mean_delta,
            "median_delta_a_minus_b": median_delta,
            "better_method": better_method,
        })
        raw_p_values.append(float(p_value))

    reject, p_holm, _, _ = multipletests(raw_p_values, alpha=alpha, method="holm")
    for row, p_corr, rejected in zip(rows, p_holm, reject):
        row["p_value_holm"] = float(p_corr)
        row["reject_holm"] = bool(rejected)

    pairwise_df = pd.DataFrame(rows).sort_values(
        ["p_value_holm", "p_value_raw", "method_a", "method_b"]
    ).reset_index(drop=True)

    pvalue_matrix = pd.DataFrame(
        np.ones((len(methods), len(methods)), dtype=float),
        index=methods, columns=methods
    )
    for row in pairwise_df.to_dict("records"):
        pvalue_matrix.loc[row["method_a"], row["method_b"]] = row["p_value_holm"]
        pvalue_matrix.loc[row["method_b"], row["method_a"]] = row["p_value_holm"]

    return pairwise_df, pvalue_matrix


def generate_wilcoxon_holm_tables(results_df, metrics, higher_is_better_metrics=None, alpha=0.05,
                                  method_order=None, metric_name_map=None):
    higher_is_better_metrics = set(higher_is_better_metrics or [])
    metric_name_map = metric_name_map or {}
    summary_rows = []
    pairwise_tables = {}
    holm_pvalue_matrices = {}

    for metric in metrics:
        higher_is_better = metric in higher_is_better_metrics
        metric_matrix, _ = build_metric_rank_matrix(
            results_df, metric, higher_is_better=higher_is_better
        )

        if method_order is not None:
            ordered_cols = [m for m in method_order if m in metric_matrix.columns]
            if ordered_cols:
                metric_matrix = metric_matrix[ordered_cols]

        if metric_matrix.shape[0] < 2 or metric_matrix.shape[1] < 2:
            print(
                f"Skipping {metric}: need at least 2 datasets and 2 methods after filtering."
            )
            continue

        pairwise_df, holm_matrix = compute_pairwise_wilcoxon_holm(
            metric_matrix=metric_matrix,
            higher_is_better=higher_is_better,
            alpha=alpha,
        )

        display_metric = metric_name_map.get(metric, metric)
        pairwise_tables[metric] = pairwise_df
        holm_pvalue_matrices[metric] = holm_matrix
        summary_rows.append({
            "metric": metric,
            "display_metric": display_metric,
            "n_datasets": int(metric_matrix.shape[0]),
            "n_methods": int(metric_matrix.shape[1]),
            "n_pairs": int(pairwise_df.shape[0]),
            "n_significant_holm": int(pairwise_df["reject_holm"].sum()),
            "min_p_holm": float(pairwise_df["p_value_holm"].min()),
        })

    if len(summary_rows) == 0:
        summary_df = pd.DataFrame(
            columns=[
                "metric", "display_metric", "n_datasets", "n_methods",
                "n_pairs", "n_significant_holm", "min_p_holm"
            ]
        )
        return summary_df, pairwise_tables, holm_pvalue_matrices

    summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)
    return summary_df, pairwise_tables, holm_pvalue_matrices


def build_pairwise_shared_instance_metrics(
    results_df,
    method_a,
    method_b,
    metrics,
    higher_is_better_metrics=None,
    require_valid_nuns=False,
    require_both_valid_cfs=True,
    dataset_col="dataset",
    method_col="method",
    instance_col="ii",
    valid_col="valid",
    nun_valid_col="nuns_valid",
):
    higher_is_better_metrics = set(higher_is_better_metrics or [])
    rows = []

    for dataset in sorted(results_df[dataset_col].dropna().unique().tolist()):
        dataset_df = results_df.loc[results_df[dataset_col] == dataset].copy()
        method_a_df = dataset_df.loc[dataset_df[method_col] == method_a].copy()
        method_b_df = dataset_df.loc[dataset_df[method_col] == method_b].copy()

        if method_a_df.empty or method_b_df.empty:
            continue

        method_a_df = (
            method_a_df
            .sort_values(instance_col)
            .drop_duplicates(subset=[instance_col], keep="first")
        )
        method_b_df = (
            method_b_df
            .sort_values(instance_col)
            .drop_duplicates(subset=[instance_col], keep="first")
        )

        common_instances = sorted(
            set(method_a_df[instance_col].dropna().astype(int).tolist()).intersection(
                set(method_b_df[instance_col].dropna().astype(int).tolist())
            )
        )
        if len(common_instances) == 0:
            continue

        method_a_df = method_a_df.set_index(instance_col).loc[common_instances].reset_index()
        method_b_df = method_b_df.set_index(instance_col).loc[common_instances].reset_index()
        merged_df = method_a_df.merge(
            method_b_df,
            on=instance_col,
            suffixes=("_a", "_b"),
        )

        valid_nuns_mask = pd.Series(True, index=merged_df.index, dtype=bool)
        if require_valid_nuns:
            valid_nuns_mask = (
                merged_df[f"{nun_valid_col}_a"].fillna(False).astype(bool)
                & merged_df[f"{nun_valid_col}_b"].fillna(False).astype(bool)
            )

        both_valid_cfs_mask = (
            merged_df[f"{valid_col}_a"].fillna(False).astype(bool)
            & merged_df[f"{valid_col}_b"].fillna(False).astype(bool)
        )

        n_common_instances = int(len(common_instances))
        n_common_valid_nuns = int(valid_nuns_mask.sum())

        for metric in metrics:
            higher_is_better = metric in higher_is_better_metrics

            if metric == valid_col:
                eligible_mask = valid_nuns_mask.copy()
            else:
                eligible_mask = valid_nuns_mask.copy()
                if require_both_valid_cfs:
                    eligible_mask = eligible_mask & both_valid_cfs_mask
                eligible_mask = (
                    eligible_mask
                    & merged_df[f"{metric}_a"].notna()
                    & merged_df[f"{metric}_b"].notna()
                )

            eligible_df = merged_df.loc[eligible_mask].copy()
            vals_a = eligible_df[f"{metric}_a"].astype(float).to_numpy() if not eligible_df.empty else np.array([])
            vals_b = eligible_df[f"{metric}_b"].astype(float).to_numpy() if not eligible_df.empty else np.array([])
            n_metric_instances = int(len(eligible_df))

            if n_metric_instances == 0:
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "higher_is_better": higher_is_better,
                    "n_common_instances": n_common_instances,
                    "n_common_valid_nuns": n_common_valid_nuns,
                    "n_metric_instances": 0,
                    "method_a_mean": np.nan,
                    "method_b_mean": np.nan,
                    "mean_delta_a_minus_b": np.nan,
                    "median_delta_a_minus_b": np.nan,
                    "instance_wins_a": 0,
                    "instance_ties": 0,
                    "instance_wins_b": 0,
                    "dataset_winner": "n/a",
                })
                continue

            if higher_is_better:
                deltas = vals_a - vals_b
            else:
                deltas = vals_b - vals_a

            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            mean_delta = float(np.mean(deltas))
            median_delta = float(np.median(deltas))
            instance_wins_a = int(np.sum(deltas > 0))
            instance_wins_b = int(np.sum(deltas < 0))
            instance_ties = int(np.sum(np.isclose(deltas, 0)))

            if np.isclose(mean_delta, 0):
                dataset_winner = "tie"
            else:
                dataset_winner = method_a if mean_delta > 0 else method_b

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "higher_is_better": higher_is_better,
                "n_common_instances": n_common_instances,
                "n_common_valid_nuns": n_common_valid_nuns,
                "n_metric_instances": n_metric_instances,
                "method_a_mean": mean_a,
                "method_b_mean": mean_b,
                "mean_delta_a_minus_b": mean_delta,
                "median_delta_a_minus_b": median_delta,
                "instance_wins_a": instance_wins_a,
                "instance_ties": instance_ties,
                "instance_wins_b": instance_wins_b,
                "dataset_winner": dataset_winner,
            })

    return pd.DataFrame(rows)


def summarize_target_pairwise_comparisons(
    results_df,
    target_method,
    metrics,
    higher_is_better_metrics=None,
    competitor_methods=None,
    alpha=0.05,
    require_valid_nuns=False,
    require_both_valid_cfs=True,
):
    higher_is_better_metrics = set(higher_is_better_metrics or [])
    available_methods = sorted(results_df["method"].dropna().unique().tolist())
    if competitor_methods is None:
        competitor_methods = [method for method in available_methods if method != target_method]

    summary_rows = []
    detailed_tables = {}

    for competitor in competitor_methods:
        pairwise_df = build_pairwise_shared_instance_metrics(
            results_df=results_df,
            method_a=target_method,
            method_b=competitor,
            metrics=metrics,
            higher_is_better_metrics=higher_is_better_metrics,
            require_valid_nuns=require_valid_nuns,
            require_both_valid_cfs=require_both_valid_cfs,
        )
        if pairwise_df.empty:
            continue

        detailed_tables[competitor] = pairwise_df

        for metric in metrics:
            metric_df = pairwise_df.loc[pairwise_df["metric"] == metric].copy()
            eligible_df = metric_df.loc[metric_df["n_metric_instances"] > 0].copy()
            wins = int((eligible_df["dataset_winner"] == target_method).sum())
            ties = int((eligible_df["dataset_winner"] == "tie").sum())
            losses = int((eligible_df["dataset_winner"] == competitor).sum())
            n_datasets = int(len(eligible_df))

            if n_datasets >= 2:
                metric_matrix = (
                    eligible_df
                    .set_index("dataset")[["method_a_mean", "method_b_mean"]]
                    .rename(columns={"method_a_mean": target_method, "method_b_mean": competitor})
                )
                wilcoxon_df, _ = compute_pairwise_wilcoxon_holm(
                    metric_matrix=metric_matrix,
                    higher_is_better=(metric in higher_is_better_metrics),
                    alpha=alpha,
                )
                wilcoxon_row = wilcoxon_df.iloc[0].to_dict()
            else:
                wilcoxon_row = {
                    "wilcoxon_statistic": np.nan,
                    "p_value_raw": np.nan,
                    "wins_a": np.nan,
                    "wins_b": np.nan,
                    "ties": np.nan,
                    "mean_delta_a_minus_b": np.nan,
                    "median_delta_a_minus_b": np.nan,
                    "better_method": np.nan,
                    "p_value_holm": np.nan,
                    "reject_holm": False,
                }

            summary_rows.append({
                "metric": metric,
                "target_method": target_method,
                "competitor_method": competitor,
                "n_datasets": n_datasets,
                "win": wins,
                "tie": ties,
                "loss": losses,
                "mean_shared_instances": float(eligible_df["n_metric_instances"].mean()) if n_datasets > 0 else np.nan,
                "min_shared_instances": int(eligible_df["n_metric_instances"].min()) if n_datasets > 0 else np.nan,
                "max_shared_instances": int(eligible_df["n_metric_instances"].max()) if n_datasets > 0 else np.nan,
                **wilcoxon_row,
            })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["metric", "competitor_method"]
        ).reset_index(drop=True)

    return summary_df, detailed_tables


def generate_critical_difference_diagrams(results_df, metrics, higher_is_better_metrics=None,
                                          alpha=0.05, method_order=None, metric_name_map=None,
                                          posthoc="nemenyi",
                                          rank_tie_method="average",
                                          use_precomputed_ranks=False,
                                          ranked_results_df=None,
                                          rank_col_suffix="_rank",
                                          drop_incomplete_rank_datasets=False,
                                          method_name_map=None,
                                          highlight_method=None,
                                          highlight_method_color="tab:blue",
                                          highlight_method_fontsize=None,
                                          highlight_method_fontweight=None,
                                          highlight_connector=True,
                                          highlight_connector_color=None,
                                          highlight_connector_linewidth=None,
                                          figsize=None,
                                          fig_height=3.2,
                                          method_spacing=1.25,
                                          min_fig_width=10.0,
                                          width_padding=2.0,
                                          method_label_fontsize=10,
                                          rank_number_fontsize=10,
                                          title_fontsize=12,
                                          text_h_margin=0.01,
                                          left_only=False,
                                          save_plots=False,
                                          save_dir=".",
                                          save_ext="png",
                                          save_suffix=None,
                                          save_dpi=300):
    try:
        import scikit_posthocs as sp
    except Exception as exc:
        raise ImportError(
            "scikit-posthocs is required for CD diagrams. "
            "Install it with: pip install scikit-posthocs"
        ) from exc

    from scipy.stats import friedmanchisquare

    higher_is_better_metrics = set(higher_is_better_metrics or [])
    metric_name_map = metric_name_map or {}
    method_name_map = method_name_map or {}
    rank_source_df = ranked_results_df if ranked_results_df is not None else results_df
    valid_tie_methods = {"average", "min", "max", "dense", "first"}
    if rank_tie_method not in valid_tie_methods:
        raise ValueError(
            f"Unsupported rank_tie_method='{rank_tie_method}'. "
            f"Choose one of {sorted(valid_tie_methods)}."
        )
    summary_rows = []
    save_dir_path = None
    if save_plots:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        higher_is_better = metric in higher_is_better_metrics
        metric_col = f"{metric}_mean"
        if metric_col not in results_df.columns:
            raise KeyError(f"Missing required column '{metric_col}' in results_df.")

        metric_matrix_all = results_df.pivot_table(
            index="dataset",
            columns="method",
            values=metric_col,
            aggfunc="mean",
        ).dropna(axis=1, how="all")
        if method_order is not None:
            ordered_cols = [m for m in method_order if m in metric_matrix_all.columns]
            if ordered_cols:
                metric_matrix_all = metric_matrix_all[ordered_cols]

        available_datasets = metric_matrix_all.index.tolist()
        metric_matrix = metric_matrix_all.dropna(axis=0, how="any")
        used_datasets = metric_matrix.index.tolist()
        dropped_datasets = sorted(set(available_datasets) - set(used_datasets))
        rank_matrix = metric_matrix.rank(
            axis=1,
            method=rank_tie_method,
            ascending=not higher_is_better
        )

        if rank_matrix.shape[0] < 2 or rank_matrix.shape[1] < 2:
            print(
                f"Skipping {metric}: need at least 2 datasets and 2 methods after filtering."
            )
            continue

        if use_precomputed_ranks:
            rank_col = f"{metric}{rank_col_suffix}"
            if rank_col not in rank_source_df.columns:
                raise KeyError(
                    f"Missing required precomputed rank column '{rank_col}' in ranked_results_df/results_df."
                )

            rank_matrix_for_plot = rank_source_df.pivot_table(
                index="dataset",
                columns="method",
                values=rank_col,
                aggfunc="mean",
            ).dropna(axis=1, how="all")
            if method_order is not None:
                ordered_rank_cols = [m for m in method_order if m in rank_matrix_for_plot.columns]
                if ordered_rank_cols:
                    rank_matrix_for_plot = rank_matrix_for_plot[ordered_rank_cols]
            if drop_incomplete_rank_datasets:
                rank_matrix_for_plot = rank_matrix_for_plot.dropna(axis=0, how="any")

            avg_ranks = rank_matrix_for_plot.mean(axis=0, skipna=True).sort_values()
            avg_ranks = avg_ranks[np.isfinite(avg_ranks.to_numpy(dtype=float))]
        else:
            avg_ranks = rank_matrix.mean(axis=0).sort_values()

        samples = [rank_matrix[col].to_numpy() for col in rank_matrix.columns]
        friedman_stat, friedman_p = friedmanchisquare(*samples)

        if posthoc == "nemenyi":
            sig_matrix = sp.posthoc_nemenyi_friedman(metric_matrix.to_numpy())
            sig_matrix.index = metric_matrix.columns
            sig_matrix.columns = metric_matrix.columns
        elif posthoc == "wilcoxon_holm":
            _, sig_matrix = compute_pairwise_wilcoxon_holm(
                metric_matrix=metric_matrix,
                higher_is_better=higher_is_better,
                alpha=alpha,
            )
        else:
            raise ValueError("posthoc must be one of: {'nemenyi', 'wilcoxon_holm'}")

        # Keep only methods present in both the average-rank vector and the
        # significance matrix used by the CD plotting routine.
        common_methods = [m for m in avg_ranks.index if m in sig_matrix.index]
        if len(common_methods) < 2:
            print(
                f"Skipping {metric}: need at least 2 common methods between ranks and significance matrix."
            )
            continue
        avg_ranks = avg_ranks.loc[common_methods]
        sig_matrix = sig_matrix.loc[common_methods, common_methods]

        # Apply paper/display names to methods for plotting (must remain unique).
        plot_method_names = [method_name_map.get(m, m) for m in avg_ranks.index]
        if len(set(plot_method_names)) != len(plot_method_names):
            raise ValueError(
                "method_name_map produces duplicate method labels. "
                "Please ensure unique display names."
            )
        avg_ranks_plot = avg_ranks.copy()
        avg_ranks_plot.index = plot_method_names
        sig_matrix_plot = sig_matrix.copy()
        sig_matrix_plot.index = [method_name_map.get(m, m) for m in sig_matrix_plot.index]
        sig_matrix_plot.columns = [method_name_map.get(m, m) for m in sig_matrix_plot.columns]

        summary_rows.append({
            "metric": metric,
            "rank_source": "precomputed" if use_precomputed_ranks else "computed",
            "rank_tie_method": rank_tie_method,
            "n_datasets_available": int(len(available_datasets)),
            "n_datasets": int(rank_matrix.shape[0]),
            "n_datasets_dropped": int(len(dropped_datasets)),
            "dropped_datasets": ", ".join(str(ds) for ds in dropped_datasets),
            "n_methods": int(len(avg_ranks)),
            "friedman_statistic": friedman_stat,
            "friedman_p": friedman_p,
            "posthoc_test": posthoc,
        })

        display_name = metric_name_map.get(metric, metric)
        if figsize is None:
            fig_w = max(min_fig_width, method_spacing * rank_matrix.shape[1] + width_padding)
            local_figsize = (fig_w, fig_height)
        else:
            local_figsize = figsize
        _, ax = plt.subplots(figsize=local_figsize)
        artists = sp.critical_difference_diagram(
            ranks=avg_ranks_plot,
            sig_matrix=sig_matrix_plot,
            alpha=alpha,
            ax=ax,
            label_fmt_left="{label} ({rank:.2f})",
            label_fmt_right="({rank:.2f}) {label}",
            label_props={"fontsize": method_label_fontsize},
            text_h_margin=text_h_margin,
            left_only=left_only,
        )

        # Force monochrome styling post-hoc to avoid any c/color alias conflicts
        # inside scikit-posthocs internals.
        for text_obj in ax.texts:
            text_obj.set_color("black")
        for line_obj in ax.lines:
            line_obj.set_color("black")
        for coll in ax.collections:
            if hasattr(coll, "set_facecolor"):
                coll.set_facecolor("black")
            if hasattr(coll, "set_edgecolor"):
                coll.set_edgecolor("black")

        ax.tick_params(axis='x', labelsize=rank_number_fontsize, colors="black")

        if highlight_method is not None:
            highlight_label = method_name_map.get(highlight_method, highlight_method)
            connector_color = (
                highlight_connector_color
                if highlight_connector_color is not None
                else highlight_method_color
            )

            target_text_ys = []
            for text_obj in ax.texts:
                txt = text_obj.get_text().strip()
                # Exact label patterns generated by label_fmt_left/right:
                #   "{label} ({rank:.2f})" or "({rank:.2f}) {label}"
                is_target = txt.startswith(f"{highlight_label} (") or txt.endswith(f") {highlight_label}")
                if is_target:
                    _, y_pos = text_obj.get_position()
                    if np.isfinite(y_pos):
                        target_text_ys.append(float(y_pos))
                    if highlight_method_color is not None:
                        text_obj.set_color(highlight_method_color)
                    if highlight_method_fontweight is not None:
                        text_obj.set_fontweight(highlight_method_fontweight)
                    if highlight_method_fontsize is not None:
                        text_obj.set_fontsize(highlight_method_fontsize)

            if highlight_connector and (highlight_label in avg_ranks_plot.index) and (connector_color is not None):
                highlight_rank = float(avg_ranks_plot.loc[highlight_label])
                x_left, x_right = ax.get_xlim()
                tol = max(1e-9, 0.02 * abs(x_right - x_left))

                selected_line = None
                method_labels = list(avg_ranks_plot.index)

                # Preferred path: if the plotting backend returns one elbow artist per method
                # (in the same order as ranks), pick the exact method connector by index.
                if isinstance(artists, dict):
                    elbow_artists = artists.get("elbows", [])
                    if isinstance(elbow_artists, list) and len(elbow_artists) == len(method_labels):
                        method_idx = method_labels.index(highlight_label)
                        maybe_line = elbow_artists[method_idx]
                        if hasattr(maybe_line, "get_xdata") and hasattr(maybe_line, "get_ydata"):
                            selected_line = maybe_line

                # Fallback path: score all plausible connectors and choose exactly one.
                if selected_line is None:
                    candidate_lines = []
                    if isinstance(artists, dict):
                        for key in ("elbows", "elbow", "elbow_lines", "label_lines"):
                            vals = artists.get(key, [])
                            if isinstance(vals, list):
                                for obj in vals:
                                    if hasattr(obj, "get_xdata") and hasattr(obj, "get_ydata"):
                                        candidate_lines.append(obj)
                    if len(candidate_lines) == 0:
                        candidate_lines = [
                            ln for ln in ax.lines
                            if hasattr(ln, "get_xdata") and hasattr(ln, "get_ydata")
                        ]

                    scored_lines = []
                    for ln in candidate_lines:
                        xdata = np.asarray(ln.get_xdata(), dtype=float)
                        ydata = np.asarray(ln.get_ydata(), dtype=float)
                        if xdata.size == 0 or ydata.size == 0:
                            continue
                        if np.all(~np.isfinite(xdata)) or np.all(~np.isfinite(ydata)):
                            continue

                        xdelta = float(np.nanmin(np.abs(xdata - highlight_rank)))
                        yspan = float(np.nanmax(ydata) - np.nanmin(ydata))
                        if yspan <= 0:
                            continue

                        if len(target_text_ys) > 0:
                            ydelta = min(
                                float(np.nanmin(np.abs(ydata - y_target)))
                                for y_target in target_text_ys
                            )
                        else:
                            ydelta = 0.0

                        # Prefer lines nearest in rank and nearest to the highlighted text y.
                        scored_lines.append((xdelta, ydelta, -yspan, ln))

                    if len(scored_lines) > 0:
                        scored_lines.sort(key=lambda t: (t[0], t[1], t[2]))
                        best_within_tol = [it for it in scored_lines if it[0] <= tol]
                        selected_line = (best_within_tol[0] if best_within_tol else scored_lines[0])[3]

                if selected_line is not None:
                    selected_line.set_color(connector_color)
                    if highlight_connector_linewidth is not None:
                        selected_line.set_linewidth(highlight_connector_linewidth)

        ax.set_title(
            f"{display_name} Critical Difference Diagram",
            fontsize=title_fontsize,
        )
        plt.tight_layout()
        saved_path = None
        if save_plots:
            metric_fname = str(metric).replace("/", "_").replace("\\", "_")
            suffix = str(save_suffix).strip() if save_suffix is not None else ""
            if suffix:
                safe_suffix = suffix.replace("/", "_").replace("\\", "_").replace(" ", "_")
                out_name = f"img_cd_plot_{metric_fname}_{safe_suffix}.{save_ext.lstrip('.')}"
            else:
                out_name = f"img_cd_plot_{metric_fname}.{save_ext.lstrip('.')}"
            saved_path = save_dir_path / out_name
            plt.savefig(saved_path, dpi=save_dpi, bbox_inches="tight")
        plt.show()
        plt.close()

        summary_rows[-1]["saved_path"] = str(saved_path) if saved_path is not None else np.nan

    if len(summary_rows) == 0:
        return pd.DataFrame(
            columns=[
                "metric", "rank_source", "rank_tie_method", "n_datasets_available", "n_datasets",
                "n_datasets_dropped", "dropped_datasets", "n_methods", "friedman_statistic",
                "friedman_p", "posthoc_test", "saved_path"
            ]
        )

    return pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)
