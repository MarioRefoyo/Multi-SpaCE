import time
from abc import ABC, abstractmethod
import numpy as np


class CounterfactualMethod(ABC):

    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    """def predict_function_tf(self, inputs):
        # Predict
        if self.change:
            inputs = np.swapaxes(inputs, 2, 1)
        predicted_probs = self.model.predict(inputs, verbose=0)
        return predicted_probs"""

    def predict_function(self, inputs):
        predicted_probs = self.model_wrapper.predict(inputs)
        return predicted_probs

    @abstractmethod
    def generate_counterfactual_specific(self, x_orig, desired_target=None, **kwargs):
        pass

    def generate_counterfactual(self, x_orig, desired_target=None, **kwargs):
        # Call to the specific counterfactual generation function and measure time of execution
        start = time.time()
        result = self.generate_counterfactual_specific(x_orig, desired_target, **kwargs)
        end = time.time()
        result = {'time': end-start, **result}

        # ToDo: Assert x_cf output of same size as input
        # print(result['cf'].shape)
        # if x_orig.shape != result['cf'].shape:
        #     raise ValueError('Generated counterfactual must have the same shape than the input.')
        return result
