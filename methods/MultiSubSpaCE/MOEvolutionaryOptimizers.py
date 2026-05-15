import copy
from abc import ABC, abstractmethod
import random
import numpy as np


class MOEvolutionaryOptimizer(ABC):
    def __init__(self, fitness_func, prediction_func, population_size, max_iter,
                 init_pct, reinit, init_random_mix_ratio,
                 invalid_penalization,
                 individual_channel_search,
                 mask_window_pct=None):

        self.population_size = population_size

        self.fitness_func = fitness_func
        self.invalid_penalization = invalid_penalization

        self.prediction_func = prediction_func
        self.max_iter = max_iter
        self.original_init_pct = init_pct

        self.feature_axis = 2   # Works for tf data ordering
        self.individual_channel_search = individual_channel_search

        self.reinit = reinit
        self.init_random_mix_ratio = init_random_mix_ratio
        self.mask_window_pct = mask_window_pct
        # self.mask_window_size = 1
        # self.ts_length_orig = None
        # self.uses_reduced_mask = False

    def increase_init_pct(self):
        if self.init_pct >= 1:
            self.init_pct = 1
            return
        next_init_pct = self.init_pct + 0.2
        if next_init_pct >= 1 and not self.tried_almost_full_init:
            self.init_pct = 0.95
            self.tried_almost_full_init = True
        else:
            self.init_pct = min(1, next_init_pct)

    def get_mask_shape(self):
        if self.individual_channel_search:
            return self.population_size, self.ts_length, self.n_features
        return self.population_size, self.ts_length, 1

    def init_population(self, importance_heatmap=None):
        reduced_heatmap = self.reduce_original_heatmap(importance_heatmap)
        # Threat all channels as an individual instance (crossover and mutate them independently)
        if self.individual_channel_search:
            # Init population
            random_data = np.random.uniform(
                0, 1,
                self.get_mask_shape()
            )
            if reduced_heatmap is not None:
                inducted_data = (self.init_random_mix_ratio * random_data + (1 - self.init_random_mix_ratio) * reduced_heatmap) / 2
            else:
                inducted_data = random_data
        # Work on instance level, mutate and crossover all channels at the same time
        else:
            # Init population
            random_data = np.random.uniform(
                0, 1,
                (self.population_size,) + (self.ts_length, 1)
            )
            if reduced_heatmap is not None:
                importance_heatmap_mean = reduced_heatmap.mean(axis=self.feature_axis-1).reshape(self.ts_length, 1)
                inducted_data = (self.init_random_mix_ratio * random_data + (1 - self.init_random_mix_ratio) * importance_heatmap_mean) / 2
            else:
                inducted_data = random_data

        # Calculate quantile and population
        quantile = np.quantile(inducted_data.flatten(), 1 - self.init_pct)
        population = (inducted_data > quantile).astype(int)

        return population

    def init(self, x_orig, nun_example, desired_class, model_wrapper,
             init_mask=None, outlier_calculator=None, importance_heatmap=None):
        self.x_orig = x_orig
        self.nun_example = nun_example
        self.desired_class = desired_class
        self.model_wrapper = model_wrapper
        self.outlier_calculator = outlier_calculator
        self.importance_heatmap = importance_heatmap
        self.init_pct = copy.deepcopy(self.original_init_pct)
        self.tried_almost_full_init = False
        self.init_mask = init_mask

        # Compute initial outlier scores when plausibility is enabled. Otherwise, use zeros so the
        # plausibility objective becomes neutral without requiring outlier model inference.
        if self.outlier_calculator is not None:
            self.outlier_scores_orig = self.outlier_calculator.get_outlier_scores(self.x_orig)
        else:
            self.outlier_scores_orig = np.zeros((1,))
        # self.outlier_score_nun = self.outlier_calculator.get_outlier_scores(self.nun_example)

        # Get dimensionality attributes
        if self.feature_axis == 2:
            self.n_features = x_orig.shape[1]
            self.ts_length_orig = x_orig.shape[0]
            if self.mask_window_pct is None:
                self.mask_window_size = 1
                self.ts_length = self.ts_length_orig
                self.uses_reduced_mask = False
            else:
                if (self.mask_window_pct <= 0) or (self.mask_window_pct > 1):
                    raise ValueError("mask_window_pct must be > 0 and <= 1.")
                self.mask_window_size = max(1, int(np.ceil(self.ts_length_orig * self.mask_window_pct)))
                self.ts_length = int(np.ceil(self.ts_length_orig / self.mask_window_size))
                self.uses_reduced_mask = True
        else:
            raise ValueError('Feature Axis Value is not valid. Only 2 is supported (for tf models)')

        # Init population
        if init_mask is not None:
            init_mask = self.normalize_init_mask(init_mask)

            # Get population (select randomly population_size elements from the pareto solutions in init_mask)
            random_idx = np.random.randint(len(init_mask), size=self.population_size)
            past_population = init_mask[random_idx]

            # Perform a simple step of crossover and mutation to ensure diversity in the first iteration.
            # Generate offsprings population, ranks, crowding_distances, number
            offsprings_population = self.produce_offsprings(past_population, self.population_size)
            # Add offsprings to the original population
            complete_population = np.vstack((past_population, offsprings_population))
            # Compute ordering of population
            objectives_fitness = self.compute_fitness(complete_population)
            fronts = self.fast_non_dominated_sorting(objectives_fitness)
            sorted_idx, sorted_population, sorted_objectives, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
                fronts, complete_population, objectives_fitness, self.population_size
            )
            best_front_individuals = np.where(np.array(sorted_ranks) == 0)[0]
            # sample from best_front_individuals until reaching the size of population_size
            random_choices = np.random.choice(len(best_front_individuals), self.population_size)
            selected_population_individuals = best_front_individuals[random_choices]
            population = sorted_population[selected_population_individuals]

        else:
            # Init masks randomly
            population = self.init_population(self.importance_heatmap)

        # Set population attribute
        self.population = population

    def __call__(self):
        return self.optimize()

    def expand_reduced_mask(self, population):
        if population is None:
            return None
        if not self.uses_reduced_mask:
            return population

        expanded_population = np.repeat(population, self.mask_window_size, axis=1)
        expanded_population = expanded_population[:, :self.ts_length_orig, :]
        return expanded_population

    def reduce_original_mask(self, population):
        if population is None:
            return None
        if not self.uses_reduced_mask:
            return population
        if population.shape[1] != self.ts_length_orig:
            raise ValueError(
                f"Original mask temporal length must be {self.ts_length_orig}. Got {population.shape[1]}."
            )

        reduced_population = np.zeros((population.shape[0], self.ts_length, population.shape[2]), dtype=int)
        for i in range(self.ts_length):
            start = i * self.mask_window_size
            end = min((i + 1) * self.mask_window_size, self.ts_length_orig)
            reduced_population[:, i, :] = population[:, start:end, :].any(axis=1).astype(int)
        return reduced_population

    def normalize_init_mask(self, init_mask):
        if init_mask.ndim != 3:
            raise ValueError("Init mask must have 3 dimentions")

        expected_channels = self.n_features if self.individual_channel_search else 1
        if init_mask.shape[2] != expected_channels:
            raise ValueError(
                f"Init mask must have {expected_channels} channel(s) for the current optimizer mode. "
                f"Got {init_mask.shape[2]}."
            )

        if init_mask.shape[1] == self.ts_length:
            return init_mask.astype(int)

        if init_mask.shape[1] == self.ts_length_orig:
            return self.reduce_original_mask(init_mask).astype(int)

        raise ValueError(
            f"Init mask temporal length must be either {self.ts_length} "
            f"(optimizer mask length) or {self.ts_length_orig} (original time length). "
            f"Got {init_mask.shape[1]}."
        )

    def reduce_original_heatmap(self, importance_heatmap):
        if importance_heatmap is None:
            return None
        if not self.uses_reduced_mask:
            return importance_heatmap
        if importance_heatmap.shape[0] != self.ts_length_orig:
            raise ValueError(
                f"Importance heatmap temporal length must be {self.ts_length_orig}. Got {importance_heatmap.shape[0]}."
            )

        reduced_heatmap = np.zeros((self.ts_length, importance_heatmap.shape[1]), dtype=importance_heatmap.dtype)
        for i in range(self.ts_length):
            start = i * self.mask_window_size
            end = min((i + 1) * self.mask_window_size, self.ts_length_orig)
            reduced_heatmap[i] = importance_heatmap[start:end].mean(axis=0)
        return reduced_heatmap

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
        population = self.expand_reduced_mask(population)
        population_size = population.shape[0]
        # Transform mask to original dara dimensions
        if self.individual_channel_search:
            population_mask = population.astype(bool)
        else:
            population = np.repeat(population, self.n_features, axis=self.feature_axis)
            population_mask = population.astype(bool)

        # Replicate x_orig and nun_example in array
        x_orig_ext = np.tile(x_orig, (population_size, 1, 1))
        nun_ext = np.tile(nun_example, (population_size, 1, 1))

        # Generate counterfactuals
        counterfactuals = np.zeros(population_mask.shape)
        counterfactuals[~population_mask] = x_orig_ext[~population_mask]
        counterfactuals[population_mask] = nun_ext[population_mask]

        return counterfactuals

    def compute_fitness(self, population):
        # Get counterfactuals
        population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, population)

        # Get desired class probs
        predicted_probs = self.prediction_func(population_cfs)

        # Get outlier scores
        if self.outlier_calculator is not None:
            outlier_scores = self.outlier_calculator.get_outlier_scores(population_cfs)
            # increase_outlier_score = outlier_scores - self.outlier_scores_orig
        else:
            outlier_scores = np.zeros((predicted_probs.shape[0], 1))

        # Get fitness function
        expanded_population = self.expand_reduced_mask(population)
        fitness = self.fitness_func(expanded_population, predicted_probs, self.desired_class, outlier_scores,
                                    self.outlier_scores_orig, self.invalid_penalization)
        return fitness

    def select_candidates(self, population, fitness, number):
        selected_indexes = self.roulette(fitness, number)
        return population[selected_indexes]

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

    def select_candidates_mo(self, population, ranks, crowding_distances, number):
        selected_indexes = self.tournament_mo(ranks, crowding_distances, number)
        return population[selected_indexes]

    @staticmethod
    def tournament_mo(ranks, crowding_distances, number):
        population_size = len(ranks)

        # Generate random individual matches
        random_pairs = np.random.randint(population_size, size=(number, 2))

        # Calculate winners based on rank (lower rank is better)
        random_pairs_ranks = np.ones(random_pairs.shape) * -1
        random_pairs_ranks[:, 0] = np.array(ranks)[random_pairs[:, 0]]
        random_pairs_ranks[:, 1] = np.array(ranks)[random_pairs[:, 1]]
        random_pairs_ranks_winners = np.argmin(random_pairs_ranks, axis=1)
        pairs_ranks_winners = random_pairs[range(population_size), random_pairs_ranks_winners]

        # Calculate winners based on crowding distance (higher is better)
        random_pairs_cdist = np.ones(random_pairs.shape) * -1
        random_pairs_cdist[:, 0] = np.array(crowding_distances)[random_pairs[:, 0]]
        random_pairs_cdist[:, 1] = np.array(crowding_distances)[random_pairs[:, 1]]
        random_pairs_cdist_winners = np.argmax(random_pairs_cdist, axis=1)
        pairs_cdist_winners = random_pairs[range(population_size), random_pairs_cdist_winners]

        # If ranks are equal, then decide winner by crowing distance
        aux = np.diff(random_pairs_ranks, axis=1)
        equal_ranks_indexes = np.where(aux == 0)[0]
        selected_indexes = pairs_ranks_winners.copy()
        if len(equal_ranks_indexes) > 0:
            selected_indexes[equal_ranks_indexes] = pairs_cdist_winners[equal_ranks_indexes]

        return selected_indexes

    @staticmethod
    def fast_non_dominated_sorting(objectives_fitness):
        population_size = objectives_fitness.shape[0]
        # Extend matrices to vectorize the condition calculations
        repeated_objectives = np.repeat(objectives_fitness, repeats=population_size, axis=0)
        stacked_objectives = np.tile(objectives_fitness, (population_size, 1))

        # Calculate "dominated by" matrix
        l_eq_matrix = repeated_objectives <= stacked_objectives
        l_matrix = repeated_objectives < stacked_objectives
        and_cond_vector = l_eq_matrix.all(axis=1)
        or_cond_vector = l_matrix.any(axis=1)
        dominated_vector = and_cond_vector * or_cond_vector
        dominated_by_matrix = dominated_vector.reshape(population_size, population_size, order='C')

        # Iteratively extract fronts
        fronts = []
        current_excluded_individuals = []
        dominated_by_matrix_copy = dominated_by_matrix.copy()
        i = 0
        while dominated_by_matrix_copy.sum() > 0:
            domination_count = dominated_by_matrix_copy.sum(axis=1)
            original_front_individuals = np.where(domination_count == 0)[0].tolist()
            front_individuals = list(set(original_front_individuals) - set(current_excluded_individuals))

            # Update auxiliary variables for next iteration
            fronts.append(front_individuals)
            current_excluded_individuals = current_excluded_individuals + front_individuals
            dominated_by_matrix_copy[:, front_individuals] = False
            i += 1

        # Last front
        final_front_individuals = list(set(range(len(objectives_fitness))) - set(current_excluded_individuals))
        fronts.append(final_front_individuals)

        # Sanity check for individual number
        list_idx = []
        for front in fronts:
            list_idx += front
        set_idx = set(list_idx)
        if len(list_idx) != len(set_idx):
            raise ValueError("Number of individuals differ in non-dominated sorting")

        return fronts

    @staticmethod
    def calculate_front_crowding_distance(objectives_fitness, front_indexes):
        front_len = len(front_indexes)
        n_objectives = objectives_fitness.shape[1]
        if front_len > 0:
            front_all_objectives_fitness = objectives_fitness[front_indexes, :]
            front_distance = np.ones((front_len, n_objectives)) * np.inf
            for o in range(n_objectives):
                front_objective_fitness = front_all_objectives_fitness[:, o]
                order_by_objective_fitness = np.argsort(front_objective_fitness)
                o_min = front_objective_fitness[order_by_objective_fitness[0]]
                o_max = front_objective_fitness[order_by_objective_fitness[-1]]
                norm_o = o_max - o_min

                # Ignore objectives with no variation inside the front. This keeps a constant
                # objective, such as plausibility in "none" mode, from introducing NaNs and
                # distorting NSGA-II crowding distance.
                if norm_o == 0:
                    front_distance[:, o] = 0
                    continue

                for n in range(1, front_len-1):
                    idx_minus = order_by_objective_fitness[n-1]
                    idx = order_by_objective_fitness[n]
                    idx_plus = order_by_objective_fitness[n+1]
                    front_distance[idx, o] = front_objective_fitness[idx_plus] - front_objective_fitness[idx_minus]

                # Normalize distance
                front_distance[:, o] = front_distance[:, o] / norm_o

            # Calculate total crowing distance
            crowding_distance = front_distance.sum(axis=1)

        else:
            raise ValueError("Length of front individuals is 0.")

        return crowding_distance

    @staticmethod
    def crowing_distance_sorting(fronts, population, population_objective_fitness, desired_size):
        ranks = []
        cdists = []
        sorted_idx = []

        current_population_count = 0
        for i_front, front in enumerate(fronts):
            front_len = len(front)
            front_crowing_distance = MOEvolutionaryOptimizer.calculate_front_crowding_distance(population_objective_fitness, front)
            sort_idx_crowing_distance = np.argsort(front_crowing_distance)[::-1]
            sorted_front = np.array(front)[sort_idx_crowing_distance]

            if (current_population_count + front_len) > desired_size:
                front_needed_len = desired_size - current_population_count
                reduced_sorted_front = sorted_front[:front_needed_len]
                ranks = ranks + [i_front]*front_needed_len
                cdists = cdists + front_crowing_distance[sort_idx_crowing_distance[:front_needed_len]].tolist()
                sorted_idx = sorted_idx + reduced_sorted_front.tolist()
                break
            else:
                ranks = ranks + [i_front] * front_len
                cdists = cdists + front_crowing_distance[sort_idx_crowing_distance].tolist()
                sorted_idx = sorted_idx + sorted_front.tolist()
            current_population_count += front_len

        sorted_population = population[sorted_idx]
        sorted_objectives = population_objective_fitness[sorted_idx]
        return sorted_idx, sorted_population, sorted_objectives, ranks, cdists

    @staticmethod
    def calculate_front_avg_fitness(front, objective_fitness):
        # Get front individual objective fitness
        front_all_objectives_fitness = objective_fitness[front, :]
        return front_all_objectives_fitness.mean()

    def optimize(self):
        # Keep track of the best solution
        best_score = -100
        best_individuals = None
        best_avg_fitness_evolution = []

        # Compute initial ordering
        objectives_fitness = self.compute_fitness(self.population)
        fronts = self.fast_non_dominated_sorting(objectives_fitness)
        sorted_idx, sorted_population, sorted_objectives, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
            fronts, self.population, objectives_fitness, self.population_size
        )

        # Get initial avg objective fitness
        best_avg_fitness = self.calculate_front_avg_fitness(fronts[0], objectives_fitness)
        best_avg_fitness_evolution.append(best_avg_fitness)

        # Run evolution
        iteration = 0
        while iteration < self.max_iter:

            # Generate offsprings population, ranks, crowding_distances, number
            selected_candidates = self.select_candidates_mo(sorted_population, sorted_ranks, sorted_cdists, self.population_size)
            offsprings_population = self.produce_offsprings(selected_candidates, self.population_size)
            # Add offsprings to the original population
            complete_population = np.vstack((sorted_population, offsprings_population))

            # Compute ordering of population
            objectives_fitness = self.compute_fitness(complete_population)
            fronts = self.fast_non_dominated_sorting(objectives_fitness)
            sorted_idx, sorted_population, sorted_objectives, sorted_ranks, sorted_cdists = self.crowing_distance_sorting(
                fronts, complete_population, objectives_fitness, self.population_size
            )

            # Change population
            self.population = sorted_population

            # Keep track of the best solution
            best_avg_fitness = self.calculate_front_avg_fitness(fronts[0], objectives_fitness)
            best_avg_fitness_evolution.append(best_avg_fitness)
            if best_avg_fitness > best_score:
                best_score = best_avg_fitness
                best_front_individuals = np.where(np.array(sorted_ranks) == 0)[0]
                best_individuals = sorted_population[best_front_individuals]
            """fitness = self.compute_fitness(auxiliar_population)
            i = np.argsort(fitness)[-1]
            cdist_evolution.append(fitness[i])
            if fitness[i] > best_score:
                best_score = fitness[i]
                best_sample = self.population[i]"""

            # Handle while loop updates
            if self.reinit and (iteration == 50) and (self.init_pct < 1) and (best_avg_fitness < -self.invalid_penalization+1):
                print('Failed to find a valid counterfactual in 50 iterations. '
                      f'Restarting process with more activations in init. Current init_pct: {self.init_pct:.2f}')
                iteration = 0
                self.increase_init_pct()
                try:
                    self.population = self.init_population(self.importance_heatmap)
                except Exception as e:
                    print(f"Error in initialization. Ending cf search. {e}")
                    return self.expand_reduced_mask(best_individuals), best_avg_fitness_evolution

                objectives_fitness = self.compute_fitness(self.population)
                fronts = self.fast_non_dominated_sorting(objectives_fitness)
                sorted_idx, sorted_population, sorted_objectives, ranks, cdists = self.crowing_distance_sorting(
                    fronts, self.population, objectives_fitness, self.population_size
                )
            else:
                iteration += 1

            # Reinit if all solutions are equal
            if np.all((self.population == self.population[0])):
                print(f'Found convergence of solutions in {iteration} iteration.')
                best_classification_prob = objectives_fitness[0, 0]
                # Get counterfactuals
                population_cfs = self.get_counterfactuals(self.x_orig, self.nun_example, self.population)
                predicted_class = np.argmax(self.prediction_func(population_cfs), axis=1)[0]
                if predicted_class == self.desired_class:
                    break
                else:
                    print(f'Final prob {best_classification_prob:.2f}, class {predicted_class}. '
                          'Restarting process with more activations in init.')
                    iteration = 0
                    self.increase_init_pct()
                    try:
                        self.population = self.init_population(self.importance_heatmap)
                    except Exception:
                        print("Error in initialization. Ending cf search.")
                        return self.expand_reduced_mask(best_individuals), best_avg_fitness_evolution
                    objectives_fitness = self.compute_fitness(self.population)
                    fronts = self.fast_non_dominated_sorting(objectives_fitness)
                    sorted_idx, sorted_population, sorted_objectives, ranks, cdists = self.crowing_distance_sorting(
                        fronts, self.population, objectives_fitness, self.population_size
                    )
        # Return best front individuals
        return self.expand_reduced_mask(best_individuals), best_avg_fitness_evolution


class NSubsequenceEvolutionaryOptimizer(MOEvolutionaryOptimizer):

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0,
                 init_pct=0.4, reinit=True, init_random_mix_ratio=0.5,
                 invalid_penalization=100,
                 mask_window_pct=None,
                 feature_axis=2, individual_channel_search=False):
        super().__init__(
            fitness_func, prediction_func, population_size, max_iter,
            init_pct, reinit, init_random_mix_ratio,
            invalid_penalization,
            individual_channel_search,
            mask_window_pct
        )
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


class SubsequencePruningEvolutionaryOptimizer(MOEvolutionaryOptimizer):

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, max_iter=100,
                 remove_subseq_mutation_prob=0.05,
                 init_pct=0.4, reinit=False, init_random_mix_ratio=0.5,
                 invalid_penalization=100,
                 mask_window_pct=None,
                 feature_axis=2):

        # By definition, pruning works individually for each channel
        individual_channel_search = True
        super().__init__(
            fitness_func, prediction_func, population_size, max_iter,
            init_pct, reinit, init_random_mix_ratio,
            invalid_penalization,
            individual_channel_search,
            mask_window_pct
        )
        self.remove_subseq_mutation_prob = remove_subseq_mutation_prob

    @staticmethod
    def remove_subsequence_mutation(population, mutation_prob):

        # Get mask of the subsequence begginings
        subseq_diff = np.diff(population, 1, prepend=0, append=0)
        mask_beginnings = np.in1d(subseq_diff, 1).reshape(subseq_diff.shape)
        end_point_mask = np.in1d(subseq_diff, -1).reshape(subseq_diff.shape)

        # ------ Generate mutation
        # Get random matrix
        random_mutations = (np.random.uniform(0, 1, (population.shape[0], population.shape[1]+1)) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros((population.shape[0], population.shape[1] + 1)).astype(int)
        valid_mutations[mask_beginnings] = random_mutations[mask_beginnings]
        valid_mutations[end_point_mask] = -random_mutations[mask_beginnings]

        subseq_diff_mutated = subseq_diff - valid_mutations
        mutated_population = np.cumsum(subseq_diff_mutated, axis=1)
        mutated_population = mutated_population[:, :-1]

        return mutated_population

    def mutate(self, sub_population):
        # Compute mutation values
        mutated_sub_population = self.remove_subsequence_mutation(sub_population, self.remove_subseq_mutation_prob)
        return mutated_sub_population


class IntegratedPruningNSubsequenceEvolutionaryOptimizer(MOEvolutionaryOptimizer):

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0, remove_subseq_mutation_prob=0.05,
                 init_pct=0.4, reinit=True, init_random_mix_ratio=0.5,
                 invalid_penalization=100,
                 individual_channel_search=False,
                 mask_window_pct=None):
        super().__init__(
            fitness_func, prediction_func, population_size, max_iter,
            init_pct, reinit, init_random_mix_ratio,
            invalid_penalization,
            individual_channel_search,
            mask_window_pct
        )
        self.change_subseq_mutation_prob = change_subseq_mutation_prob
        self.add_subseq_mutation_prob = add_subseq_mutation_prob
        self.remove_subseq_mutation_prob = remove_subseq_mutation_prob

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

    @staticmethod
    def remove_subsequence_mutation(population, mutation_prob):

        # Get mask of the subsequence begginings
        subseq_diff = np.diff(population, 1, prepend=0, append=0)
        mask_beginnings = np.in1d(subseq_diff, 1).reshape(subseq_diff.shape)
        end_point_mask = np.in1d(subseq_diff, -1).reshape(subseq_diff.shape)

        # ------ Generate mutation
        # Get random matrix
        random_mutations = (
                    np.random.uniform(0, 1, (population.shape[0], population.shape[1] + 1)) < mutation_prob).astype(int)
        # Get mutated population
        valid_mutations = np.zeros((population.shape[0], population.shape[1] + 1)).astype(int)
        valid_mutations[mask_beginnings] = random_mutations[mask_beginnings]
        valid_mutations[end_point_mask] = -random_mutations[mask_beginnings]

        subseq_diff_mutated = subseq_diff - valid_mutations
        mutated_population = np.cumsum(subseq_diff_mutated, axis=1)
        mutated_population = mutated_population[:, :-1]

        return mutated_population

    def mutate(self, sub_population):
        mutated_sub_population = sub_population.copy()
        # Subsequence Extension / Compression mutation
        if self.change_subseq_mutation_prob > 0:
            mutated_sub_population = self.shrink_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
            mutated_sub_population = self.extend_mutation(mutated_sub_population, self.change_subseq_mutation_prob)
        # Subsequence creation mutation
        if self.add_subseq_mutation_prob > 0:
            mutated_sub_population = self.add_subsequence_mutation(mutated_sub_population, self.add_subseq_mutation_prob)
        # Subsequence removal mutation
        if self.remove_subseq_mutation_prob > 0:
            mutated_sub_population = self.remove_subsequence_mutation(mutated_sub_population, self.remove_subseq_mutation_prob)
        return mutated_sub_population


class SubsequenceChannelEvolutionaryOptimizer(IntegratedPruningNSubsequenceEvolutionaryOptimizer):

    def __init__(self, fitness_func, prediction_func,
                 population_size=100, max_iter=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0, remove_subseq_mutation_prob=0.05,
                 channel_mutation_prob=0.05,
                 init_pct=0.4, reinit=True, init_random_mix_ratio=0.5,
                 invalid_penalization=100,
                 mask_window_pct=None,
                 temporal_mutation_mode="grouped"):
        super().__init__(
            fitness_func, prediction_func,
            population_size, max_iter,
            change_subseq_mutation_prob, add_subseq_mutation_prob, remove_subseq_mutation_prob,
            init_pct, reinit, init_random_mix_ratio,
            invalid_penalization,
            True,
            mask_window_pct
        )
        self.temporal_mutation_mode = temporal_mutation_mode
        self.channel_mutation_prob = channel_mutation_prob


    def init_population(self, importance_heatmap=None):
        """Initialize only temporal positions, then activate all channels.

        The initial population starts from global temporal subsequences shared by
        all channels. Channel-specific structure is introduced later by mutations.
        """
        random_data = np.random.uniform(
            0, 1,
            (self.population_size, self.ts_length, 1)
        )

        reduced_heatmap = self.reduce_original_heatmap(importance_heatmap)

        if reduced_heatmap is not None:
            importance_heatmap_mean = reduced_heatmap.mean(
                axis=self.feature_axis - 1
            ).reshape(self.ts_length, 1)

            inducted_data = (
                    self.init_random_mix_ratio * random_data
                    + (1 - self.init_random_mix_ratio) * importance_heatmap_mean
            ) / 2
        else:
            inducted_data = random_data

        quantile = np.quantile(inducted_data.flatten(), 1 - self.init_pct)
        temporal_population = (inducted_data > quantile).astype(int)

        # Expand temporal mask to all channels: (P, L, 1) -> (P, L, C)
        population = np.repeat(
            temporal_population,
            self.n_features,
            axis=self.feature_axis
        )

        return population.astype(int)

    @staticmethod
    def mutate_channel_gate(population, mutation_prob):
        """Mutate channel gates at the global-subsequence level."""
        mutated_population = population.copy().astype(bool)
        population_size, ts_length, n_channels = mutated_population.shape

        if (mutation_prob <= 0) or (n_channels == 0):
            return mutated_population.astype(int)

        time_mask = mutated_population.any(axis=2)

        if not time_mask.any():
            return mutated_population.astype(int)

        prev_time_mask = np.roll(time_mask, 1, axis=1)
        prev_time_mask[:, 0] = False
        starts = time_mask & ~prev_time_mask

        labels = np.cumsum(starts, axis=1) * time_mask
        max_labels = labels.max()

        if max_labels == 0:
            return mutated_population.astype(int)

        label_ids = np.arange(1, max_labels + 1)
        label_mask = labels[:, :, None] == label_ids[None, None, :]

        gate = (
            label_mask[:, :, :, None]
            & mutated_population[:, :, None, :]
        ).any(axis=1)

        existing_labels = label_mask.any(axis=1)

        flips = np.random.uniform(0, 1, gate.shape) < mutation_prob
        flips &= existing_labels[:, :, None]

        mutated_gate = np.logical_xor(gate, flips)

        rebuilt = np.einsum(
            "plk,pkc->plc",
            label_mask.astype(int),
            mutated_gate.astype(int)
        ).astype(bool)

        return rebuilt.astype(int)


    @staticmethod
    def _to_channelwise_rows(population):
        """Convert (P, L, C) to (P*C, L)."""
        population_size, ts_length, n_channels = population.shape
        adapted = np.swapaxes(population, 2, 1)
        adapted = adapted.reshape(population_size * n_channels, ts_length)
        return adapted


    @staticmethod
    def _from_channelwise_rows(adapted_population, population_shape):
        """Convert (P*C, L) back to (P, L, C)."""
        population_size, ts_length, n_channels = population_shape
        adapted = adapted_population.reshape(population_size, n_channels, ts_length)
        return np.swapaxes(adapted, 2, 1)


    def _apply_channelwise_old_mutation(self, population, mutation_func, mutation_prob):
        """Apply old 2D mutation to each channel independently."""
        adapted = self._to_channelwise_rows(population)
        adapted = mutation_func(adapted, mutation_prob)
        return self._from_channelwise_rows(adapted, population.shape)


    @staticmethod
    def _expand_time_mask_with_neighbor_gates(original_population, old_time_mask, new_time_mask):
        """Convert mutated global time mask back to (P, L, C).

        Removed timestamps are zeroed.
        Newly activated timestamps copy the closest active neighbor gate.
        Existing active timestamps keep their original channel pattern.
        """
        mutated = original_population.copy().astype(int)
        population_size, ts_length, n_channels = mutated.shape

        removed = old_time_mask & ~new_time_mask
        mutated[removed, :] = 0

        added = new_time_mask & ~old_time_mask

        for sample_idx in range(population_size):
            added_idx = np.where(added[sample_idx])[0]
            active_old_idx = np.where(old_time_mask[sample_idx])[0]

            if len(added_idx) == 0:
                continue

            if len(active_old_idx) == 0:
                # Fallback: activate one random channel if a new global subsequence
                # appears from an originally empty mask.
                for t in added_idx:
                    channel_idx = np.random.randint(n_channels)
                    mutated[sample_idx, t, channel_idx] = 1
                continue

            for t in added_idx:
                # Prefer direct neighbor, otherwise closest old active timestamp.
                if (t > 0) and old_time_mask[sample_idx, t - 1]:
                    donor_t = t - 1
                elif (t < ts_length - 1) and old_time_mask[sample_idx, t + 1]:
                    donor_t = t + 1
                else:
                    donor_t = active_old_idx[np.argmin(np.abs(active_old_idx - t))]

                mutated[sample_idx, t, :] = original_population[sample_idx, donor_t, :]

        return mutated.astype(int)


    def _apply_grouped_old_mutation(self, population, mutation_func, mutation_prob):
        """Apply old 2D mutation to the global temporal skeleton.

        The old mutation changes only a (P, L) time mask. Then the result is
        expanded back to (P, L, C) using existing neighbor channel gates.
        """
        old_time_mask = population.any(axis=2).astype(int)
        new_time_mask = mutation_func(old_time_mask, mutation_prob).astype(bool)
        old_time_mask = old_time_mask.astype(bool)

        return self._expand_time_mask_with_neighbor_gates(
            population,
            old_time_mask,
            new_time_mask
        )


    def _apply_temporal_old_mutation(self, population, mutation_func, mutation_prob):
        """Dispatch old mutation according to temporal_mutation_mode."""
        if self.temporal_mutation_mode == "independent":
            return self._apply_channelwise_old_mutation(
                population,
                mutation_func,
                mutation_prob
            )

        if self.temporal_mutation_mode == "grouped":
            return self._apply_grouped_old_mutation(
                population,
                mutation_func,
                mutation_prob
            )

        raise ValueError(
            "temporal_mutation_mode must be 'independent' or 'grouped'. "
            f"Got {self.temporal_mutation_mode}"
        )


    def mutate(self, population):
        """Apply old temporal mutations, then optional channel-gate mutation."""
        mutated_population = population.copy()

        if self.change_subseq_mutation_prob > 0:
            mutated_population = self._apply_temporal_old_mutation(
                mutated_population,
                self.shrink_mutation,
                self.change_subseq_mutation_prob
            )
            mutated_population = self._apply_temporal_old_mutation(
                mutated_population,
                self.extend_mutation,
                self.change_subseq_mutation_prob
            )

        if self.add_subseq_mutation_prob > 0:
            mutated_population = self._apply_temporal_old_mutation(
                mutated_population,
                self.add_subsequence_mutation,
                self.add_subseq_mutation_prob
            )

        if self.remove_subseq_mutation_prob > 0:
            mutated_population = self._apply_temporal_old_mutation(
                mutated_population,
                self.remove_subsequence_mutation,
                self.remove_subseq_mutation_prob
            )

        if self.channel_mutation_prob > 0:
            mutated_population = self.mutate_channel_gate(
                mutated_population,
                self.channel_mutation_prob
            )

        return mutated_population.astype(int)


    def produce_offsprings(self, subpopulation, number):
        """Crossover complete multivariate masks along the time axis."""
        population_size, ts_length, n_channels = subpopulation.shape

        n_pairs = int(np.ceil(number / 2))

        parent_idx = np.random.choice(
            population_size,
            size=(n_pairs, 2),
            replace=True
        )
        parents_a = subpopulation[parent_idx[:, 0]]
        parents_b = subpopulation[parent_idx[:, 1]]

        split_points = np.random.randint(1, ts_length, size=n_pairs)

        time_idx = np.arange(ts_length)[None, :, None]
        crossover_mask = time_idx < split_points[:, None, None]

        offsprings1 = np.where(crossover_mask, parents_a, parents_b)
        offsprings2 = np.where(crossover_mask, parents_b, parents_a)

        offsprings = np.concatenate([offsprings1, offsprings2], axis=0)
        offsprings = offsprings[:number]

        offsprings = self.mutate(offsprings)
        return offsprings.astype(int)


if __name__ == "__main__":
    class _DummyOptimizer(MOEvolutionaryOptimizer):
        def mutate(self, sub_population):
            return sub_population

    dummy_optimizer = _DummyOptimizer(
        fitness_func=lambda *args, **kwargs: None,
        prediction_func=lambda x: x,
        population_size=2,
        max_iter=1,
        init_pct=0.4,
        reinit=False,
        init_random_mix_ratio=0.5,
        invalid_penalization=100,
        individual_channel_search=True
    )
    dummy_optimizer.uses_reduced_mask = False
    unchanged_population = np.array([[[0], [1], [0]]])
    assert np.array_equal(dummy_optimizer.expand_reduced_mask(unchanged_population), unchanged_population)

    reduced_optimizer = _DummyOptimizer(
        fitness_func=lambda *args, **kwargs: None,
        prediction_func=lambda x: x,
        population_size=2,
        max_iter=1,
        init_pct=0.4,
        reinit=False,
        init_random_mix_ratio=0.5,
        invalid_penalization=100,
        individual_channel_search=True,
        mask_window_pct=0.3
    )
    reduced_optimizer.n_features = 1
    reduced_optimizer.ts_length_orig = 10
    reduced_optimizer.mask_window_size = 3
    reduced_optimizer.ts_length = 4
    reduced_optimizer.uses_reduced_mask = True

    reduced_population = np.array([
        [[1], [0], [1], [0]],
        [[0], [1], [0], [1]]
    ])
    expanded_population = reduced_optimizer.expand_reduced_mask(reduced_population)
    assert expanded_population.shape == (2, 10, 1)
    assert np.array_equal(expanded_population[0, :, 0], np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0]))
    assert np.array_equal(expanded_population[1, :, 0], np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1]))

    original_population = np.array([
        [[0], [1], [0], [0], [0], [1], [0], [0], [0], [1]],
        [[0], [0], [0], [1], [0], [0], [0], [0], [1], [0]]
    ])
    reduced_from_original = reduced_optimizer.reduce_original_mask(original_population)
    assert reduced_from_original.shape == (2, 4, 1)
    assert np.array_equal(reduced_from_original[0, :, 0], np.array([1, 1, 0, 1]))
    assert np.array_equal(reduced_from_original[1, :, 0], np.array([0, 1, 1, 0]))
    normalized_init_mask = reduced_optimizer.normalize_init_mask(original_population)
    assert np.array_equal(normalized_init_mask, reduced_from_original)

    x_orig = np.zeros((10, 2))
    nun_example = np.ones((10, 2))
    grouped_optimizer = _DummyOptimizer(
        fitness_func=lambda *args, **kwargs: None,
        prediction_func=lambda x: x,
        population_size=2,
        max_iter=1,
        init_pct=0.4,
        reinit=False,
        init_random_mix_ratio=0.5,
        invalid_penalization=100,
        individual_channel_search=False,
        mask_window_pct=0.3
    )
    grouped_optimizer.n_features = 2
    grouped_optimizer.ts_length_orig = 10
    grouped_optimizer.mask_window_size = 3
    grouped_optimizer.ts_length = 4
    grouped_optimizer.uses_reduced_mask = True
    grouped_population = np.array([
        [[1], [0], [1], [0]],
        [[0], [1], [0], [1]]
    ])
    counterfactuals = grouped_optimizer.get_counterfactuals(x_orig, nun_example, grouped_population)
    assert counterfactuals.shape == (2, 10, 2)

    individual_optimizer = _DummyOptimizer(
        fitness_func=lambda population, *args, **kwargs: population.reshape(population.shape[0], -1).sum(axis=1, keepdims=True),
        prediction_func=lambda x: np.zeros((x.shape[0], 2)),
        population_size=2,
        max_iter=1,
        init_pct=0.4,
        reinit=False,
        init_random_mix_ratio=0.5,
        invalid_penalization=100,
        individual_channel_search=True,
        mask_window_pct=0.3
    )
    individual_optimizer.init(
        x_orig=np.zeros((10, 2)),
        nun_example=np.ones((10, 2)),
        desired_class=1,
        model_wrapper=None,
        init_mask=np.repeat(original_population, 2, axis=2),
        outlier_calculator=None,
        importance_heatmap=np.zeros((10, 2))
    )
    assert individual_optimizer.population.shape[1] == 4

    np.random.seed(0)
    corr = 0.0
    n = 20
    population = np.random.random((n, 300, 10)) > 0.5
    population_objective_fitness = np.random.multivariate_normal([0, 0], np.array([[1, corr], [corr, 1]]), size=n)
    fronts = MOEvolutionaryOptimizer.fast_non_dominated_sorting(population_objective_fitness)

    import matplotlib.pyplot as plt
    for i, front in enumerate(fronts):
        o1 = population_objective_fitness[front, 0]
        o2 = population_objective_fitness[front, 1]
        plt.scatter(o1, o2, label=i)
    plt.legend()
    plt.show()

    # Generate new population
    sorted_idx, sorted_population, sorted_objectives, ranks, cdists = MOEvolutionaryOptimizer.crowing_distance_sorting(
        fronts, population, population_objective_fitness, n//2
    )
    print("finished")
