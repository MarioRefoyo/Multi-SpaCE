import random
import time

import numpy as np
from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import convergence, diversity, hypervolume

import methods.TSEvo.EvoUtils as EvoUtils
from methods.TSEvo.EvoUtils import (
    create_logbook,
    create_mstats,
    evaluate_pop,
    pareto_eq,
    recombine,
)
from methods.TSEvo.Problem import MultiObjectiveCounterfactuals

log = False

MUT_TYPES = ["freq", "auth", "mean"]


class EvolutionaryOptimization:
    def __init__(
        self,
        model,
        observation_x,
        original_y,
        target_y,
        reference_set,
        neighborhood,
        window,
        channels,
        backend,
        transformer="authentic_opposing_information",
        epochs=500,
        verbose=0,
        mode="feat",
    ):
        self.MU = 100
        self.NGEN = epochs
        self.CXPB = 0.9
        self.MUTPB = 0.6

        self.verbose = verbose
        self.mop = MultiObjectiveCounterfactuals(
            model,
            observation_x,
            original_y,
            target_y,
            reference_set,
            neighborhood,
            window,
            backend,
            channels,
            mode=mode,
        )

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            if channels == 1:
                creator.create(
                    "Individual",
                    list,
                    fitness=creator.FitnessMin,
                    window=np.random.randint(
                        1, 0.5 * np.array(observation_x).reshape(-1).shape[0]
                    ),
                    channels=np.ones(channels),
                    mutation=None,
                )
            else:
                creator.create(
                    "Individual",
                    list,
                    fitness=creator.FitnessMin,
                    window=np.random.randint(
                        1, 0.5 * np.array(observation_x).reshape(channels, -1).shape[1]
                    ),
                    channels=np.ones(channels),
                    mutation=None,
                )
                if self.verbose == 1:
                    self.verbose = 2

        self.toolbox = base.Toolbox()

        def init_pop():
            if len(neighborhood) > 1:
                index = np.random.choice(neighborhood.shape[0], 1)
                neig = list(neighborhood[index])
                neig = np.asarray(neig, dtype=np.float64)
            else:
                neig = np.asarray(observation_x, dtype=np.float64).reshape(channels, -1)

            return neig

        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, init_pop
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", recombine)
        self.toolbox.register(
            "mutate", getattr(EvoUtils, transformer), reference_set=reference_set
        )

    def run(self):
        pop = self.toolbox.population(n=self.MU)
        window = []
        mutation = []
        for ind in pop:
            ind.window = np.random.randint(1, 0.5 * np.array(ind).shape[1])
            ind.mutation = random.choice(MUT_TYPES)

        hof = tools.ParetoFront(similar=pareto_eq)
        best = tools.HallOfFame(1, similar=pareto_eq)

        pop = evaluate_pop(pop, self.mop)
        pop = self.toolbox.select(pop, self.MU)

        rf = [1.0, 1.0, 1.0]
        gen = 0

        timestr = time.strftime("%Y%m%d-%H%M%S")
        if log:
            log_file = open("log" + timestr + ".txt", "w")

        mstats = create_mstats()
        logbook = create_logbook()

        record = mstats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        while gen < self.NGEN:
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = algorithms.varAnd(
                offspring, self.toolbox, cxpb=self.CXPB, mutpb=self.MUTPB
            )

            offspring = evaluate_pop(offspring, self.mop)
            pop = self.toolbox.select(offspring + pop, self.MU)

            if log:
                log_file.write(
                    str(gen)
                    + ","
                    + str(hypervolume(pop, rf))
                    + ","
                    + str(diversity(pop, [0, 0, 0], rf))
                    + ","
                    + str(convergence(pop, [[0, 0, 0]]))
                    + "\n"
                )

            record = mstats.compile(pop)
            logbook.record(gen=gen, evals=len(pop), **record)
            hof.update(pop)
            best.update(pop)
            if self.verbose != 0:
                print(logbook.stream)
            gen = gen + 1

            if self.verbose == 2 and pop[0].mutation is not None:
                mean = 0
                freq = 0
                auth = 0
                for ind in pop:
                    if ind.mutation == "freq":
                        freq += 1
                    elif ind.mutation == "auth":
                        auth += 1
                    elif ind.mutation == "mean":
                        mean += 1
                mutation.append([freq, auth, mean])
            if self.verbose == 2:
                add = []
                for ind in pop:
                    add.append(ind.window)
                window.append([np.mean(add), np.std(add)])

        for item in best:
            label, output = self.mop.predict(np.array([item]), full=True)
            item.output = output

        if self.verbose == 2:
            return best, logbook, window, mutation

        return best, best[0].output
