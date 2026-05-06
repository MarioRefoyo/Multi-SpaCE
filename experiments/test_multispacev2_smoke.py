import random

import numpy as np
import tensorflow as tf
import torch

from experiments.experiment_utils import prepare_experiment
from methods.MultiSubSpaCE.FeatureImportanceInitializers import NoneFI
from methods.MultiSubSpaCECF import MultiSubSpaCECFv2
from methods.nun_finders import GlobalNUNFinder


DATASET = "RacketSports"
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_noscaling"


def main():
    params = {
        "seed": 24,
        "subset": True,
        "subset_number": 5,
        "additional_subsample_subset": 5,
        "scaling": "none",
        "plausibility_objective": "none",
        "independent_channels_nun": False,
        "nun_strategy": "global",
        "n_neighbors": 1,
    }

    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])

    X_train, y_train, X_test, y_test, subset_idx, n_classes, model_wrapper, y_pred_train, y_pred_test = prepare_experiment(
        DATASET, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME
    )

    nun_finder = GlobalNUNFinder(
        X_train, y_train, y_pred_train, distance="euclidean",
        from_true_labels=False, backend="tf"
    )
    nuns, desired_classes, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    nuns = nuns[:, 0, :, :]

    cf_explainer = MultiSubSpaCECFv2(
        model_wrapper, None, NoneFI("tf"),
        grouped_channels_iter=2, individual_channels_iter=2, pruning_iter=1,
        plausibility_objective="none",
        population_size=20,
        change_subseq_mutation_prob=0.05,
        add_subseq_mutation_prob=0.05,
        integrated_pruning_mutation_prob=0.05,
        final_pruning_mutation_prob=0.05,
        channel_mutation_prob=0.05,
        init_pct=0.4, reinit=False, init_random_mix_ratio=0.5,
        invalid_penalization=100,
    )

    sample_idx = 0
    x_orig = X_test[sample_idx]
    nun_example = nuns[sample_idx]
    desired_target = desired_classes[sample_idx]
    original_class = y_pred_test[sample_idx]

    result = cf_explainer.generate_counterfactual(
        x_orig, desired_target=desired_target, nun_example=nun_example
    )
    returned_cfs = result["cfs"]
    returned_predictions = np.argmax(model_wrapper.predict(returned_cfs), axis=1)
    valid_solutions = returned_predictions == desired_target

    print("MultiSpaCECFv2 smoke test")
    print(f"dataset: {DATASET}")
    print(f"subset_index: {subset_idx[sample_idx]}")
    print(f"original_class: {original_class}")
    print(f"desired_nun_class: {desired_target}")
    print(f"num_returned_solutions: {len(returned_cfs)}")
    print(f"counterfactual_shape: {returned_cfs.shape}")
    print(f"valid_solution_count: {int(valid_solutions.sum())}")
    print(f"has_valid_solution: {bool(valid_solutions.any())}")

    if len(returned_cfs) == 0:
        raise RuntimeError("No counterfactuals were returned.")


if __name__ == "__main__":
    main()
