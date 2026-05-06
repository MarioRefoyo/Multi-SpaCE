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

    def _predict_labels(self, inputs):
        array_inputs = np.asarray(inputs)
        if array_inputs.ndim == 2:
            array_inputs = np.expand_dims(array_inputs, axis=0)
        predicted_probs = self.predict_function(array_inputs)
        return np.argmax(predicted_probs, axis=1)

    @abstractmethod
    def generate_counterfactual_specific(self, x_orig, desired_target=None, **kwargs):
        pass

    def generate_counterfactual(self, x_orig, desired_target=None, **kwargs):
        # Call to the specific counterfactual generation function and measure time of execution
        start = time.time()
        result = self.generate_counterfactual_specific(x_orig, desired_target, **kwargs)
        end = time.time()
        result = {'time': end-start, **result}

        if "x_orig" not in result:
            result["x_orig"] = np.array(x_orig, copy=True)

        nun_example = kwargs.get("nun_example")
        if nun_example is not None and "nun" not in result:
            result["nun"] = np.array(nun_example, copy=True)

        y_true_orig = kwargs.get("y_true_orig")
        if y_true_orig is not None:
            result["x_orig_true_label"] = int(y_true_orig)
        result["x_orig_pred_label"] = int(self._predict_labels(x_orig)[0])

        if "cf" in result:
            result["cf_pred_label"] = int(self._predict_labels(result["cf"])[0])
        if "cfs" in result:
            result["cf_pred_labels"] = self._predict_labels(result["cfs"])

        # ToDo: Assert x_cf output of same size as input
        # print(result['cf'].shape)
        # if x_orig.shape != result['cf'].shape:
        #     raise ValueError('Generated counterfactual must have the same shape than the input.')
        return result
