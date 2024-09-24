import numpy as np
import copy

from .Glacier._guided import get_global_weights
from .Glacier.help_functions import find_best_lr
from .counterfactual_common import CounterfactualMethod


class GlacierCF(CounterfactualMethod):
    def __init__(self, model, backend, x_train, y_train, ae_model, w_value, tau_value, lr_list, w_type, seed):
        super().__init__(model, backend, change=False)
        self.x_train = x_train
        self.y_train = y_train
        self.ae_model = ae_model
        self.n_timesteps_padded = x_train.shape[1]
        self.n_features = x_train.shape[1]

        self.w_value = w_value
        self.tau_value = tau_value
        self.w_type = w_type
        self.lr_list = lr_list
        self.seed = seed

        # Set step weights based on constraints type
        if self.w_type == "global":
            step_weights = get_global_weights(
                x_train,
                y_train,
                self.model,
                random_state=self.seed,
            )

        elif self.w_type == "uniform":
            step_weights = np.ones((1, self.n_timesteps_padded, self.n_features))
        elif self.w_type.lower() == "local":
            step_weights = "local"
        elif self.w_type == "unconstrained":
            step_weights = np.zeros((1, self.n_timesteps_padded, self.n_features))
        else:
            raise NotImplementedError(
                "A.w_type not implemented, please choose 'local', 'global', 'uniform', or 'unconstrained'."
            )
        self.step_weights = step_weights

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        # Get pred labels
        pred_logits = self.predict_function_tf(x_orig)
        pred_labels = np.argmax(pred_logits, axis=1)

        # Here x_orig and desired target are a list of inputs, not just one. ToDo: adapt the method to work as expected
        best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
            self.model,
            X_samples=x_orig,
            pred_labels=pred_labels,
            autoencoder=self.ae_model,
            lr_list=self.lr_list,
            pred_margin_weight=self.w_value,
            step_weights=self.step_weights,
            random_state=self.seed,
            padding_size=0, # Forced by experiment execution
            target_prob=self.tau_value,
        )

        results = {"cf": best_cf_samples}

        return results
