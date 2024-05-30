import time
from abc import ABC, abstractmethod


class CounterfactualMethod(ABC):

    def __init__(self, model, backend='tf'):
        self.model = model
        self.backend = backend
        if backend == 'tf':
            self.predict_function = self.predict_function_tf
            self.feature_axis = 2
        else:
            raise ValueError('Backend not supported')

    def predict_function_tf(self, inputs):
        # Predict
        predicted_probs = self.model.predict(inputs, verbose=0)
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
