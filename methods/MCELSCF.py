import copy
from types import SimpleNamespace

import numpy as np

from .MCELS import CFExplainer, TFMCELSExplainer
from .counterfactual_common import CounterfactualMethod


class MCELSCF(CounterfactualMethod):
    def __init__(
        self,
        model_wrapper,
        background_data,
        background_label,
        max_iter=1000,
        lr=0.1,
        enable_lr_decay=False,
        lr_decay=0.999,
        enable_budget=True,
        enable_tvnorm=True,
        l_budget_coeff=1.0,
        l_tv_norm_coeff=1.0,
        l_max_coeff=1.0,
        tv_beta=3.0,
        seed=None,
        dataset_name="unknown",
        nun_strategy="second_best",
        enable_wandb=False,
        use_cuda=False,
        save_dir="/tmp",
        use_torch_explainer_for_tf=False,
        break_tf_gradients_with_numpy=False,
    ):
        super().__init__(model_wrapper)
        self.background_data = background_data
        self.background_label = background_label
        self.max_iter = max_iter
        self.lr = lr
        self.enable_lr_decay = enable_lr_decay
        self.lr_decay = lr_decay
        self.enable_budget = enable_budget
        self.enable_tvnorm = enable_tvnorm
        self.l_budget_coeff = l_budget_coeff
        self.l_tv_norm_coeff = l_tv_norm_coeff
        self.l_max_coeff = l_max_coeff
        self.tv_beta = tv_beta
        self.seed = seed
        self.dataset_name = dataset_name
        self.nun_strategy = nun_strategy
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.save_dir = save_dir
        self.use_torch_explainer_for_tf = use_torch_explainer_for_tf
        self.break_tf_gradients_with_numpy = break_tf_gradients_with_numpy

    def _build_args(self):
        return SimpleNamespace(
            algo="cf",
            dataset=self.dataset_name,
            lr=self.lr,
            enable_lr_decay=self.enable_lr_decay,
            lr_decay=self.lr_decay,
            enable_budget=self.enable_budget,
            enable_tvnorm=self.enable_tvnorm,
            l_budget_coeff=self.l_budget_coeff,
            l_tv_norm_coeff=self.l_tv_norm_coeff,
            l_max_coeff=self.l_max_coeff,
            tv_beta=self.tv_beta,
            max_itr=self.max_iter,
            nun_strategy=self.nun_strategy,
            break_tf_gradients_with_numpy=self.break_tf_gradients_with_numpy,
        )

    @staticmethod
    def _transpose_sample_to_mcels(x):
        return np.transpose(x, (1, 0))

    @staticmethod
    def _transpose_background_to_mcels(x):
        return np.transpose(x, (0, 2, 1))

    @staticmethod
    def _mask_history_from_metrics(metrics):
        if not metrics or "L_Total" not in metrics:
            return []

        loss_history = []
        for i in range(len(metrics["L_Total"])):
            loss_history.append(
                {
                    "iter": i,
                    "loss": metrics["L_Total"][i],
                    "l_maximize": metrics["L_Maximize"][i],
                    "l_budget": metrics["L_Budget"][i],
                    "l_tv": metrics["L_TV_Norm"][i],
                    "target_prob": metrics["CF_Prob"][i],
                }
            )
        return loss_history

    def _torch_predict_fn(self):
        def predict_fn(x):
            x = x.to(self.model_wrapper.device)
            return self.model_wrapper.model(x)

        return predict_fn

    def _tf_predict_fn(self):
        import tensorflow as tf

        def predict_fn(x):
            x_tf = tf.transpose(x, perm=[0, 2, 1])
            return self.model_wrapper.model(x_tf, training=False)

        return predict_fn

    def _torch_predict_fn_for_tf_model(self):
        import torch

        def predict_fn(x):
            x_np = x.detach().cpu().numpy()
            x_np = np.transpose(x_np, (0, 2, 1))
            pred = self.model_wrapper.model(x_np, training=False)
            pred_np = pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)
            # The TF models in this repo already output probabilities.
            # Return log-probabilities so the Torch MCELS softmax recovers them
            # instead of applying a second softmax over probabilities.
            pred_np = np.clip(pred_np, 1e-12, 1.0)
            return torch.tensor(np.log(pred_np), dtype=torch.float32, device=x.device)

        return predict_fn

    def _get_target_tensor(self, x_orig):
        if self.model_wrapper.backend == "torch":
            import torch

            x_batch = np.expand_dims(self._transpose_sample_to_mcels(x_orig), axis=0)
            x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(self.model_wrapper.device)
            with torch.no_grad():
                return self.model_wrapper.model(x_tensor)

        if self.model_wrapper.backend == "tf":
            if self.use_torch_explainer_for_tf:
                import torch
                import tensorflow as tf

                x_batch = tf.convert_to_tensor(np.expand_dims(x_orig, axis=0), dtype=tf.float32)
                pred = self.model_wrapper.model(x_batch, training=False)
                pred_np = pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)
                return torch.tensor(pred_np, dtype=torch.float32)

            import tensorflow as tf

            x_batch = tf.convert_to_tensor(np.expand_dims(x_orig, axis=0), dtype=tf.float32)
            return self.model_wrapper.model(x_batch, training=False)

        raise NotImplementedError("Unsupported backend for MCELS.")

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None, y_true_orig=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        x_orig_copy = copy.deepcopy(x_orig)

        args = self._build_args()
        bg_data_mcels = self._transpose_background_to_mcels(self.background_data)
        x_orig_mcels = self._transpose_sample_to_mcels(x_orig)

        if self.model_wrapper.backend == "torch":
            explainer = CFExplainer(
                background_data=bg_data_mcels,
                background_label=self.background_label,
                predict_fn=self._torch_predict_fn(),
                enable_wandb=self.enable_wandb,
                use_cuda=self.use_cuda,
                args=args,
            )
        elif self.model_wrapper.backend == "tf" and self.use_torch_explainer_for_tf:
            explainer = CFExplainer(
                background_data=bg_data_mcels,
                background_label=self.background_label,
                predict_fn=self._torch_predict_fn_for_tf_model(),
                enable_wandb=self.enable_wandb,
                use_cuda=False,
                args=args,
            )
        elif self.model_wrapper.backend == "tf":
            explainer = TFMCELSExplainer(
                background_data=bg_data_mcels,
                background_label=self.background_label,
                predict_fn=self._tf_predict_fn(),
                enable_wandb=self.enable_wandb,
                use_cuda=False,
                args=args,
            )
        else:
            raise NotImplementedError("MCELS currently supports only 'tf' and 'torch' backends.")

        target_tensor = self._get_target_tensor(x_orig)
        mask_flat, x_cf_mcels, target_prob = explainer.generate_saliency(
            x_orig_mcels,
            y_true_orig,
            target=target_tensor,
            save_dir=self.save_dir,
        )

        if self.model_wrapper.backend == "tf" and self.use_torch_explainer_for_tf:
            # Match the upstream Torch-style MCELS behavior more closely:
            # keep the continuous perturbation instead of snapping it back
            # through a hard thresholded mask.
            mask = np.asarray(mask_flat).reshape(x_orig_mcels.shape).T
            x_cf = np.asarray(x_cf_mcels).T
        else:
            mask = (np.asarray(mask_flat).reshape(x_orig_mcels.shape).T > 0.5).astype(int)
            x_cf = np.asarray(x_cf_mcels).T
            # Set values where mask is 0 to the original sample.
            x_cf = np.where(mask == 0, x_orig_copy, x_cf)
        loss_history = self._mask_history_from_metrics(explainer.metrics)

        return {
            "cf": np.expand_dims(x_cf, axis=0),
            "mask": mask,
            "target_prob": target_prob,
            "loss_history": loss_history,
        }
