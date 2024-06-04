from abc import ABC, abstractmethod
import random
import numpy as np


class EvolutionaryOptimizer(ABC):
    def __init__(self, fitness_func, prediction_func, population_size, elite_number, offsprings_number, max_iter,
                 init_pct, reinit,
                 invalid_penalization, alpha, beta, eta, gamma, sparsity_balancer,
                 feature_axis, multivariate_mode):
        # Asert elite numbers and replacement count do not surpass population size
        if elite_number + offsprings_number > population_size:
            raise ValueError('Elites and offsprings counts must not be greater than population size')
        # Assert valid offspring number
        if offsprings_number % 2 != 0:
            raise ValueError('Offspring number must be even')

        self.population_size = population_size
        self.elite_number = elite_number
        self.offsprings_number = offsprings_number
        self.rest_number = population_size - elite_number - offsprings_number

        self.fitness_func = fitness_func
        self.invalid_penalization = invalid_penalization
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.sparsity_balancer = sparsity_balancer

        self.prediction_func = prediction_func
        self.max_iter = max_iter
        self.init_pct = init_pct

        self.feature_axis = feature_axis
        self.multivariate_mode = multivariate_mode

        self.reinit = reinit

    def init_population(self, importance_heatmap=None):
        # Threat all channels as an individual instance (crossover and mutate them independently)
        if self.multivariate_mode == 'individual':
            # Init population
            random_data = np.random.uniform(
                0, 1,
                (self.population_size,) + self.x_orig.shape
            )
            if importance_heatmap is not None:
                mix_ratio = 0.6
                inducted_data = (mix_ratio * random_data + (1 - mix_ratio) * importance_heatmap) / 2
            else:
                inducted_data = random_data
        # Work on instance level, mutate and crossover all channels at the same time
        elif self.multivariate_mode == 'grouped':
            # Init population
            random_data = np.random.uniform(
                0, 1,
                (self.population_size,) + self.x_orig.mean(axis=self.feature_axis-1).shape + (1,)
            )
            if importance_heatmap is not None:
                importance_heatmap_mean = importance_heatmap.mean(axis=self.feature_axis-1).reshape(self.ts_length, 1)
                mix_ratio = 0.6
                inducted_data = (mix_ratio * random_data + (1 - mix_ratio) * importance_heatmap_mean) / 2
            else:
                inducted_data = random_data
        else:
            raise ValueError("Multivariate mode is not supported.")

        # Calculate quantile and population
        quantile = np.quantile(inducted_data.flatten(), 1 - self.init_pct)
        population = (inducted_data > quantile).astype(int)

        return population

    def init(self, x_orig, nun_example, desired_class, model, outlier_calculator=None, importance_heatmap=None):
        self.x_orig = x_orig
        self.nun_example = nun_example
        self.desired_class = desired_class
        self.model = model
        self.outlier_calculator = outlier_calculator
        self.importance_heatmap = importance_heatmap
        self.fitness_evolution = []

        # Get dimensionality attributes
        if self.feature_axis == 2:
            self.n_features = x_orig.shape[1]
            self.ts_length = x_orig.shape[0]
        else:
            raise ValueError('Feature Axis Value is not valid. Only 2 is supported (for tf models)')

        # Init population
        population = self.init_population(self.importance_heatmap)
        self.population = population

        # Compute initial outlier scores
        self.outlier_scores_orig = self.outlier_calculator.get_outlier_scores(self.x_orig)
        self.outlier_score_nun = self.outlier_calculator.get_outlier_scores(self.nun_example)

    def __call__(self):
        return self.optimize()

    @abstractmethod
    def mutate(self, sub_population):
        pass

    @staticmethod
    def get_single_crossover_mask(subpopulation):
        split_points = np.random.randint(0, subpopulation.shape[1], size=subpopulation.shape[0] // 2)
        mask = np.arange(subpopulation.shape[1]) < split_points[:, np.newaxis]
        return mask

    def produce_offsprings(self, subpopulation, number):
        # Put channels as individual examples
        # Swap axis if features are in axis 2
        if self.feature_axis == 2:
            # Get sample population
            adapted_subpopulation = np.swapaxes(subpopulation, 2, 1)
        else:
            adapted_subpopulation = subpopulation
        subpopulation_n_features = subpopulation.shape[self.feature_axis]
        adapted_number = number * subpopulation_n_features
        adapted_subpopulation = adapted_subpopulation.reshape(adapted_number, -1)

        # Generate random split points and create mask
        mask = self.get_single_crossover_mask(adapted_subpopulation)

        # Generate random matches
        matches = np.random.choice(np.arange(adapted_subpopulation.shape[0]),
                                   size=(adapted_subpopulation.shape[0] // 2, 2),
                                   replace=False)

        # Create the two partial offsprings
        offsprings1 = np.empty((adapted_number//2, adapted_subpopulation.shape[1]))
        offsprings1[mask] = adapted_subpopulation[matches[:, 0]][mask]
        offsprings1[~mask] = adapted_subpopulation[matches[:, 1]][~mask]
        offsprings2 = np.zeros((adapted_number//2, adapted_subpopulation.shape[1]))
        offsprings2[mask] = adapted_subpopulation[matches[:, 1]][mask]
        offsprings2[~mask] = adapted_subpopulation[matches[:, 0]][~mask]
        # Calculate adapted offspring
        adapted_offsprings = np.concatenate([offsprings1, offsprings2])

        # Mutate offsprings
        adapted_offsprings = self.mutate(adapted_offsprings)

        # Get final offsprings (matching original dimensionality)
        adapted_offsprings = adapted_offsprings.reshape(number, subpopulation_n_features, -1)
        if self.feature_axis == 2:
            offsprings = np.swapaxes(adapted_offsprings, 2, 1)
        else:
            offsprings = adapted_offsprings

        return offsprings

    def get_counterfactuals(self, x_orig, nun_example, population):
        population_size = population.shape[0]
        # Transform mask to original dara dimensions
        if self.multivariate_mode == 'individual':
            population_mask = population.astype(bool)
        elif self.multivariate_mode == 'grouped':
            population = np.repeat(population, self.n_features, axis=self.feature_axis)
            population_mask = population.astype(bool)
        else:
            raise ValueError("Multivariate mode is not supported.")

        # Replicate x_orig and nun_example in array
        x_orig_ext = np.tile(x_orig, (population_size, 1, 1))
        nun_ext = np.tile(nun_example, (population_size, 1, 1))

        # Generate counterfactuals
        counterfactuals = np.zeros(population_mask.shape)
        counterfactuals[~population_mask] = x_orig_ext[~population_mask]
        counterfactuals[population_mask] = nun_ext[population_mask]

        return counterfactuals

    def compute_fitness(self):
        # Get counterfactuals
        population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, self.population)

        # Get desired class probs
        predicted_probs = self.prediction_func(population_cfs)

        # Get outlier scores
        if self.outlier_calculator is not None:
            outlier_scores = self.outlier_calculator.get_outlier_scores(population_cfs)
            increase_outlier_score = outlier_scores - (self.outlier_scores_orig + self.outlier_score_nun) / 2
        else:
            outlier_scores = None
            increase_outlier_score = None

        # Get fitness function
        fitness, desired_class_probs = self.fitness_func(self.population, predicted_probs, self.desired_class, increase_outlier_score,
                                                         self.invalid_penalization, self.alpha, self.beta, self.eta,
                                                         self.gamma, self.sparsity_balancer)
        return fitness, desired_class_probs

    @staticmethod
    def roulette(fitness, number):
        scaled_fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-5)
        selection_probs = scaled_fitness / scaled_fitness.sum()
        if np.isnan(selection_probs).any():
            print('NaN found in candidate probabilities')
            print(f'Fitness {fitness}')
            print(f'Selection probs: {selection_probs}')
        selected_indexes = np.random.choice(scaled_fitness.shape[0], number, p=selection_probs)
        return selected_indexes

    def select_candidates(self, population, fitness, number):
        selected_indexes = self.roulette(fitness, number)
        return population[selected_indexes]

    def optimize(self):
        # Keep track of the best solution
        best_score = -100
        best_sample = None
        best_classification_prob = 0

        # Compute initial fitness
        fitness, _ = self.compute_fitness()
        i = np.argsort(fitness)[-1]
        self.fitness_evolution.append(fitness[i])

        # Run evolution
        iteration = 0
        while iteration < self.max_iter:

            # Init new population
            new_population = np.empty(self.population.shape)

            # Elites: Select elites and add to new population
            elites_idx = np.argsort(fitness)[-self.elite_number:]
            new_population[:self.elite_number, :] = self.population[elites_idx]

            # Cross-over and mutation
            # Select parents
            candidate_population = self.select_candidates(self.population, fitness, self.offsprings_number)
            # Produce offsprings
            offsprings = self.produce_offsprings(candidate_population, self.offsprings_number)
            # Add to the population
            new_population[self.elite_number:self.offsprings_number + self.elite_number] = offsprings

            # The rest of the population is random selected
            random_indexes = np.random.randint(self.population_size, size=self.rest_number)
            if self.rest_number > 0:
                new_population[-self.rest_number:] = self.population[random_indexes]

            # Change population
            self.population = new_population.astype(int)
            # Keep track of the best solution
            if len(self.population) < self.population_size:
                print('what???')
            fitness, class_probs = self.compute_fitness()
            i = np.argsort(fitness)[-1]
            self.fitness_evolution.append(fitness[i])
            if fitness[i] > best_score:
                best_score = fitness[i]
                best_sample = self.population[i]
                best_classification_prob = class_probs[i]

            # Handle while loop updates
            if self.reinit and (iteration == 50) and (self.init_pct < 1) and (fitness[i] < -self.invalid_penalization+1):
                print('Failed to find a valid counterfactual in 50 iterations. '
                      f'Restarting process with more activations in init. Current init_pct: {self.init_pct:.2f}')
                iteration = 0
                self.init_pct = self.init_pct + 0.2
                self.population = self.init_population(self.importance_heatmap)
                fitness, class_probs = self.compute_fitness()
            else:
                iteration += 1

            # Reinit if all fitness are equal
            if np.all(fitness == fitness[0]):
                print(f'Found convergence of solutions in {iteration} iteration.')
                if best_classification_prob > 0.5:
                    break
                else:
                    print(f'Final prob {best_classification_prob:.2f}. '
                          'Restarting process with more activations in init.')
                    iteration = 0
                    self.init_pct = self.init_pct + 0.2
                    self.population = self.init_population(self.importance_heatmap)
                    fitness, class_probs = self.compute_fitness()

        return best_sample, best_classification_prob


class BasicEvolutionaryOptimizer(EvolutionaryOptimizer):
    def __init__(self, fitness_func, prediction_func,
                 population_size=100, elite_number=4, offsprings_number=96, max_iter=100,
                 mutation_prob=0.1,
                 init_pct=0.4, reinit=True,
                 invalid_penalization=100, alpha=0.2, beta=0.6, eta=0.2, gamma=0.25, sparsity_balancer=0.4,
                 feature_axis=2, multivariate_mode='individual'):
        super().__init__(fitness_func, prediction_func, population_size, elite_number, offsprings_number, max_iter,
                         init_pct, reinit, invalid_penalization,
                         alpha, beta, eta, gamma, sparsity_balancer,
                         feature_axis, multivariate_mode)
        self.mutation_prob = mutation_prob

    def mutate(self, sub_population):
        # Compute mutation values
        mutation_mask = (np.random.uniform(0, 1, sub_population.shape) > 1-self.mutation_prob).astype(int)
        mutated_sub_population = (sub_population + mutation_mask) % 2
        return mutated_sub_population


class NSubsequenceEvolutionaryOptimizer(EvolutionaryOptimizer):

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, elite_number=4, offsprings_number=96, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0,
                 init_pct=0.4, reinit=True,
                 invalid_penalization=100, alpha=0.2, beta=0.6, eta=0.2, gamma=0.25, sparsity_balancer=0.4,
                 feature_axis=2, multivariate_mode='individual'):
        super().__init__(fitness_func, prediction_func, population_size, elite_number, offsprings_number, max_iter,
                         init_pct, reinit, invalid_penalization,
                         alpha, beta, eta, gamma, sparsity_balancer,
                         feature_axis, multivariate_mode)
        self.change_subseq_mutation_prob = change_subseq_mutation_prob
        self.add_subseq_mutation_prob = add_subseq_mutation_prob

    @ staticmethod
    def add_subsequence_mutation(population, mutation_prob):
        # ----- Get potential extension locations
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        # Get before and after ones masks
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        # Generate complete mask of after and before ones (and set to False the places where the original ones exist)
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False

        # Get potential positions mask
        possibilities_mask = ~(before_after_ones_mask + ones_mask)

        # Get new subsequences
        new_subsequences = np.zeros(population.shape).astype(int)
        for i, row in enumerate(possibilities_mask):
            # Flip a coin to mutate or not
            if np.random.random() < mutation_prob:
                valid_idx = np.where(row == True)[0]
                # Get random index and length to add subsequence
                if len(valid_idx) > 0:
                    chosen_idx = np.random.choice(valid_idx)
                    subseq_len = min(population.shape[1] - chosen_idx, np.random.randint(2, 6))
                    new_subsequences[i, chosen_idx:chosen_idx + subseq_len] = 1

        # Get mutated population
        mutated_population = np.clip(population + new_subsequences, 0, 1)
        return mutated_population

    @staticmethod
    def extend_mutation(population, mutation_prob):
        # ----- Get potential extension locations
        ones_mask = np.in1d(population, 1).reshape(population.shape)
        # Get before and after ones masks
        before_ones_mask = np.roll(ones_mask, -1, axis=1)
        before_ones_mask[:, ones_mask.shape[1] - 1] = False
        after_ones_mask = np.roll(ones_mask, 1, axis=1)
        after_ones_mask[:, 0] = False
        # Generate complete mask of after and before ones (and set to False the places where the original ones exist)
        before_after_ones_mask = before_ones_mask + after_ones_mask
        before_after_ones_mask[ones_mask] = False

        # ------ Generate mutation
        # Get random matrix
        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[before_after_ones_mask] = random_mutations[before_after_ones_mask]
        mutated_population = (population + valid_mutations) % 2

        return mutated_population

    @staticmethod
    def shrink_mutation(population, mutation_prob):
        # ----- Get potential shrinking locations
        # Get mask of the subsequence begginings and endings
        mask_beginnings = np.diff(population, 1, prepend=0)
        mask_beginnings = np.in1d(mask_beginnings, 1).reshape(mask_beginnings.shape)
        mask_endings = np.flip(np.diff(np.flip(population, axis=1), 1, prepend=0), axis=1)
        mask_endings = np.in1d(mask_endings, 1).reshape(mask_endings.shape)
        # Generate complete mask
        beginnings_endings_mask = mask_beginnings + mask_endings

        # ------ Generate mutation
        # Get random matrix
        random_mutations = (np.random.uniform(0, 1, population.shape) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros(population.shape).astype(int)
        valid_mutations[beginnings_endings_mask] = random_mutations[beginnings_endings_mask]
        mutated_population = (population + valid_mutations) % 2
        return mutated_population

    def mutate(self, sub_population):
        # Compute mutation values
        mutated_sub_population = self.shrink_mutation(sub_population, self.change_subseq_mutation_prob)
        mutated_sub_population = self.extend_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
        if self.add_subseq_mutation_prob > 0:
            mutated_sub_population = self.add_subsequence_mutation(mutated_sub_population, self.add_subseq_mutation_prob)
        return mutated_sub_population


if __name__ == "__main__":
    b = np.array([[1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 0]])
    shrank_mutation = NSubsequenceEvolutionaryOptimizer.shrink_mutation(b, 0.1)
    extended_mutation = NSubsequenceEvolutionaryOptimizer.extend_mutation(b, 0.1)
