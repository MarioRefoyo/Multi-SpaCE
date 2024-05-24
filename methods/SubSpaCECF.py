import numpy as np
import copy

from .SubSpaCE.EvolutionaryOptimizers import NSubsequenceEvolutionaryOptimizer
from .SubSpaCE.FitnessFunctions import fitness_function_final
from .counterfactual_common import CounterfactualMethod
from .utils import calculate_heatmap


class SubSpaCECF(CounterfactualMethod):
    def __init__(self, model, backend, outlier_calculator,
                 population_size=100, elite_number=4, offsprings_number=96, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0,
                 init_pct=0.4, reinit=True,
                 invalid_penalization=100, alpha=0.2, beta=0.6, eta=0.2, gamma=0.25, sparsity_balancer=0.4):
        super().__init__(model, backend)

        # Init Genetic Optimizer
        self.optimizer = NSubsequenceEvolutionaryOptimizer(
            fitness_function_final, self.predict_function,
            population_size, elite_number, offsprings_number, max_iter,
            change_subseq_mutation_prob, add_subseq_mutation_prob,
            init_pct, reinit,
            invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer,
            self.feature_axis
        )
        self.outlier_calculator = outlier_calculator

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        # Calculate importance heatmap
        heatmap_x_orig = calculate_heatmap(self.model, x_orig)
        heatmap_nun = calculate_heatmap(self.model, nun_example)
        combined_heatmap = (heatmap_x_orig + heatmap_nun) / 2

        # We have to load again the model to avoid using the same object in parallel
        self.optimizer.init(
            x_orig, nun_example, desired_target,
            self.model,
            outlier_calculator=self.outlier_calculator,
            importance_heatmap=combined_heatmap
        )

        # Calculate counterfactual
        found_counterfactual_mask, desired_class_prob = self.optimizer.optimize()
        if found_counterfactual_mask is None:
            print(f'Failed to converge for sample')
            x_cf = copy.deepcopy(np.expand_dims(x_orig, axis=0))
        else:
            x_cf = self.optimizer.get_counterfactuals(
                x_orig, nun_example, np.expand_dims(found_counterfactual_mask, axis=0)
            )

        result = {'cf': x_cf, 'fitness_evolution': self.optimizer.fitness_evolution}

        return result
