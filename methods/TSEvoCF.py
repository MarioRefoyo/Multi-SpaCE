import copy

import numpy as np

from .counterfactual_common import CounterfactualMethod
from .TSEvo.Evo import EvolutionaryOptimization


class TSEvoCF(CounterfactualMethod):
    def __init__(
        self,
        model_wrapper,
        x_reference,
        y_reference,
        transformer="authentic_opposing_information",
        orig=False,
        epochs=500,
        verbose=0,
    ):
        super().__init__(model_wrapper)
        self.x_reference = x_reference
        self.y_reference = y_reference
        self.transformer = transformer
        self.orig = orig
        self.epochs = epochs
        self.verbose = verbose
        self.backend = "PYT" if model_wrapper.backend == "torch" else "TF"

    def _get_reference_set(self, original_output, desired_target):
        if desired_target is not None:
            reference_mask = self.y_reference == desired_target
        else:
            original_label = int(np.argmax(original_output))
            reference_mask = self.y_reference != original_label
        return self.x_reference[reference_mask]

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        squeeze_output = len(x_orig.shape) == 2
        if squeeze_output:
            x_input = np.expand_dims(x_orig, axis=0)
        else:
            x_input = x_orig

        original_output = self.model_wrapper.predict(x_input)[0]
        reference_set = self._get_reference_set(original_output, desired_target)

        # Upstream TSEvo works internally in (channels, time). Repo data is (time, channels).
        original_x_internal = np.swapaxes(x_input, 2, 1)
        reference_internal = np.swapaxes(reference_set, 1, 2)

        window = original_x_internal.shape[-1]
        channels = original_x_internal.shape[-2]
        neighborhood = []

        optimizer = EvolutionaryOptimization(
            self.model_wrapper,
            original_x_internal,
            original_output,
            desired_target,
            reference_internal,
            neighborhood,
            window,
            channels,
            self.backend,
            self.transformer,
            verbose=self.verbose,
            epochs=self.epochs,
            mode="time",
        )

        try:
            explanations, outputs = optimizer.run(orig=self.orig)
            pareto_cfs = np.asarray(
                [np.swapaxes(np.asarray(explanation), 1, 0) for explanation in explanations]
            )
            if pareto_cfs.ndim == 2:
                pareto_cfs = np.expand_dims(pareto_cfs, axis=0)
            x_cf = pareto_cfs[0]
            if not squeeze_output:
                x_cf = np.expand_dims(x_cf, axis=0)
        except Exception as exc:
            print(f"TSEvo failed for sample: {exc}")
            x_cf = copy.deepcopy(x_input)
            pareto_cfs = copy.deepcopy(x_input)
            outputs = np.expand_dims(original_output, axis=0)

        return {"cf": x_cf, "cfs": pareto_cfs, "output": outputs[0], "outputs": outputs}
