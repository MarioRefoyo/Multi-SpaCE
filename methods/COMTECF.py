import numpy as np
import copy

from .COMTE.Optimization import BruteForceSearch, OptimizedSearch
from .counterfactual_common import CounterfactualMethod


class COMTECF(CounterfactualMethod):
    def __init__(self, model, backend, x_train, y_train, number_distractors,
                 max_attempts=1000, max_iter=1000, restarts=5, reg=0.8, silent=False):
        if backend == 'tf':
            change = True
        else:
            raise "COMTE only supports tf models right now"

        super().__init__(model, backend, change=change)

        if change:
            self.x_train = np.swapaxes(x_train, 2, 1)
        else:
            self.x_train = x_train
        self.y_train = y_train
        self.number_distractors = number_distractors
        self.max_attempts = max_attempts
        self.max_iter = max_iter
        self.silent = silent
        self.restarts = restarts
        self.reg = reg

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        opt = OptimizedSearch(
            self.predict_function,
            self.x_train,
            self.y_train,
            silent=self.silent,
            threads=1,
            num_distractors=self.number_distractors,
            max_attempts=self.max_attempts,
            maxiter=self.max_iter,
            restarts=self.restarts,
            reg=self.reg,
        )
        if self.change:
            x_orig = np.swapaxes(x_orig, 2, 1)
            x_cf, predicted_label = opt.explain(x_orig, to_maximize=desired_target)
            x_cf = np.swapaxes(x_cf, 2, 1)
        else:
            x_cf, predicted_label = opt.explain(x_orig, to_maximize=desired_target)
        result = {'cf': x_cf}

        return result
