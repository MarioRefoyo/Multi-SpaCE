import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def calculate_change_mask(x_orig, x_nun, x_cf, verbose=0):
    # Get original change mask (could contain points with common values between NUN, x_orig and x_cf)
    orig_change_mask = (x_orig != x_cf).astype(int)

    # Find common values
    cv_xorig_nun = (x_orig == x_nun)
    cv_nun_cf = (x_nun == x_cf)
    cv_all = (cv_xorig_nun & cv_nun_cf).astype(int)

    # Check if thos common values are at the start or end of a current subsequence
    start_end_mask = cv_all & get_start_end_subsequence_positions(orig_change_mask).astype(int)
    if verbose==1:
        print(orig_change_mask.flatten())
        print(get_start_end_subsequence_positions(orig_change_mask).flatten())
        print(cv_all.flatten())
        print(start_end_mask.flatten())

    # Add noise to those original points that are common to original, NUN and cf
    # are at the beginning or end of a subsequence on the change mask
    noise = np.random.normal(0, 1e-6, x_orig.shape)
    new_x_orig = x_orig + noise * start_end_mask

    # Calculate adjusted change mask
    new_change_mask = (new_x_orig != x_cf).astype(int)
    return new_change_mask


# Calculate metrics for each method
def calculate_metrics(model, outlier_calculator, X_train, X_test, nuns_idx, solutions_in, original_classes,
                      method_name, order=None):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    counterfactuals = [solution['cf'] for solution in solutions]
    execution_times = [solution['time'] for solution in solutions]

    # Get size of the input
    length = X_train.shape[1]
    n_channels = X_train.shape[2]

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
            change_mask = calculate_change_mask(X_test[i], X_train[nuns_idx[i]], counterfactuals[i], verbose=0)
            nchanges.append(change_mask.sum())

            # Calculate l1
            l1 = np.linalg.norm((X_test[i].flatten() - counterfactuals[i].flatten()), ord=1)
            l1s.append(l1)

            # Calculate l2
            l2 = np.linalg.norm((X_test[i].flatten() - counterfactuals[i].flatten()), ord=2)
            l2s.append(l2)

            # Number of sub-sequences
            # print(change_mask.shape)
            subsequences = np.count_nonzero(np.diff(change_mask, prepend=0, axis=0) == 1, axis=0)[0]
            n_subsequences.append(subsequences)
        else:
            valids.append(False)
            # Append all NaNs to not being take into consideration
            pred_probas.append(np.nan)
            nchanges.append(np.nan)
            l1s.append(np.nan)
            l2s.append(np.nan)
            n_subsequences.append(np.nan)

    # Outlier scores
    # Increase in outlier score
    outlier_scores = outlier_calculator.get_outlier_scores(np.array(counterfactuals).reshape(-1, length, n_channels))
    outlier_scores_orig = outlier_calculator.get_outlier_scores(X_test)
    outlier_scores_nuns = outlier_calculator.get_outlier_scores(X_train[nuns_idx])
    increase_os = outlier_scores - (outlier_scores_orig + outlier_scores_nuns) / 2
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
    results["outlier_score"] = outlier_scores.tolist()
    results["increase_outlier_score"] = increase_os.tolist()
    results['subsequences'] = n_subsequences
    results['subsequences %'] = np.array(n_subsequences) / (length / 2)
    results['times'] = execution_times
    results['method'] = method_name
    if order is not None:
        results['order'] = order

    return results
