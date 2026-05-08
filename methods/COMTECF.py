import numpy as np
import copy

from .COMTE.Optimization import BruteForceSearch, OptimizedSearch
from .counterfactual_common import CounterfactualMethod


class COMTECF(CounterfactualMethod):
    def __init__(self, model_wrapper, x_train, y_train, number_distractors,
                 max_attempts=1000, max_iter=1000, restarts=5, reg=0.8, silent=False):
        super().__init__(model_wrapper)
        self.x_train = self._to_comte_layout(x_train)
        self.y_train = y_train
        self.number_distractors = number_distractors
        self.max_attempts = max_attempts
        self.max_iter = max_iter
        self.silent = silent
        self.restarts = restarts
        self.reg = reg

    @staticmethod
    def _to_comte_layout(x):
        """Convert project layout (batch, time, channels) to COMTE layout (batch, channels, time)."""
        return np.swapaxes(np.asarray(x), 1, 2)

    @staticmethod
    def _from_comte_layout(x):
        """Convert COMTE layout (batch, channels, time) back to project layout (batch, time, channels)."""
        return np.swapaxes(np.asarray(x), 1, 2)

    def predict_function_comte(self, inputs):
        return self.model_wrapper.predict(self._from_comte_layout(inputs))

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        opt = OptimizedSearch(
            self.predict_function_comte,
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

        x_orig_comte = self._to_comte_layout(x_orig)
        try:
            x_cf, predicted_label = opt.explain(x_orig_comte, to_maximize=desired_target)
        except AttributeError as msg:
            print(f'{msg}')
            print(f'COMTE failed to find a NUN so original instance is returned')
            x_cf = copy.deepcopy(x_orig_comte)

        if x_cf is None:
            print('COMTE failed to find a counterfactual so original instance is returned')
            x_cf = copy.deepcopy(x_orig_comte)

        result = {'cf': self._from_comte_layout(x_cf)}

        return result
