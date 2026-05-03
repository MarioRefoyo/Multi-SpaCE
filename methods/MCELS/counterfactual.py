from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries

from .losses import tv_norm, tv_norm_tf

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    class _WandbStub:
        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()


def save_timeseries_mul(mask, time_series, perturbated_output, save_dir, dataset, algo,
                        blurred=None, enable_wandb=False, raw_mask=None, category=None):
    return None


class Saliency:
    def __init__(self, background_data, background_label, predict_fn):
        self.background_data = background_data
        self.background_label = background_label
        self.predict_fn = predict_fn


class CFExplainer(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(CFExplainer, self).__init__(
            background_data=background_data,
            background_label=background_label,
            predict_fn=predict_fn,
        )
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.conf_threshold = 0.8
        self.eps = None
        self.eps_decay = 0.9991
        self.metrics = None
        self.cf_label = None

    def native_guide_retrieval(self, query, distance, n_neighbors, target_label=None, predicted_label=None):
        dim_nums, ts_length = query.shape[0], query.shape[1]
        df = pd.DataFrame(self.background_label, columns=["label"])
        if target_label is not None:
            candidate_indexes = df[df["label"] == target_label].index.values
        elif predicted_label is not None:
            candidate_indexes = df[df["label"] != predicted_label].index.values
        else:
            raise ValueError("Either target_label or predicted_label must be provided.")

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(self.background_data[list(candidate_indexes)])

        dist, ind = knn.kneighbors(query.reshape(1, dim_nums, ts_length), return_distance=True)
        selected_indexes = candidate_indexes[ind[0][:]]
        selected_label = int(df.loc[selected_indexes[0], "label"])
        return dist, selected_indexes, selected_label

    def cf_label_fun(self, instance):
        output = self.softmax_fn(self.predict_fn(instance.reshape(1, instance.shape[0], instance.shape[1]).float()))
        target = torch.argsort(output, descending=True)[0, 1].item()
        return target

    def generate_saliency(self, data, label, **kwargs):
        self.mode = "Explore"
        query = data.copy()

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if self.use_cuda:
            data = data.cuda()

        top_prediction_class = np.argmax(kwargs["target"].detach().cpu().numpy())

        predicted_label = int(top_prediction_class)
        nun_strategy = getattr(self.args, "nun_strategy", "second_best")
        if nun_strategy == "second_best":
            cf_label = self.cf_label_fun(data)
            _, idx, selected_label = self.native_guide_retrieval(
                query, "euclidean", 1, target_label=cf_label
            )
        elif nun_strategy == "global":
            _, idx, selected_label = self.native_guide_retrieval(
                query, "euclidean", 1, predicted_label=predicted_label
            )
            cf_label = selected_label
        else:
            raise ValueError(f"Unsupported nun_strategy: {nun_strategy}")
        self.cf_label = cf_label

        NUN = self.background_data[idx.item()]

        self.eps = 1.0

        mask_init = np.random.uniform(size=data.shape, low=0, high=1)
        mask_tensor = torch.from_numpy(mask_init).float()
        if self.use_cuda:
            mask_tensor = mask_tensor.cuda()
        mask = Variable(mask_tensor, requires_grad=True)

        optimizer = torch.optim.Adam([mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        print(f"{self.args.algo}: Optimizing... ")
        metrics = defaultdict(lambda: [])

        max_iterations_without_improvement = 100
        imp_threshold = 0.001
        best_loss = float("inf")
        counter = 0

        i = 0
        while i <= self.args.max_itr:
            Rt = torch.tensor(NUN, dtype=torch.float32, device=data.device)

            perturbated_input = data.mul(1 - mask) + Rt.mul(mask)

            pred_outputs = self.softmax_fn(
                self.predict_fn(perturbated_input.reshape(1, perturbated_input.shape[0], perturbated_input.shape[1]).float())
            )

            l_maximize = 1 - pred_outputs[0][cf_label]
            l_budget_loss = torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)

            loss = (self.args.l_budget_coeff * l_budget_loss) + \
                   (self.args.l_tv_norm_coeff * l_tv_norm_loss) + \
                   (self.args.l_max_coeff * l_maximize)

            if best_loss - loss < imp_threshold:
                counter += 1
            else:
                counter = 0
                best_loss = loss

            optimizer.zero_grad()
            loss.backward()

            metrics["L_Maximize"].append(float(l_maximize.item()))
            metrics["L_Budget"].append(float(l_budget_loss.item()))
            metrics["L_TV_Norm"].append(float(l_tv_norm_loss.item()))
            metrics["L_Total"].append(float(loss.item()))
            metrics["CF_Prob"].append(float(pred_outputs[0][cf_label].item()))

            optimizer.step()

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            mask.data.clamp_(0, 1)

            if self.enable_wandb:
                _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"}}
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]["eval_metrics"]}
                wandb.log(_mets)

            if counter >= max_iterations_without_improvement:
                print("Early stopping triggered: 'total loss' metric didn't improve much")
                break
            else:
                i += 1

        self.metrics = metrics

        mask = mask.detach().cpu().numpy()

        threshold = 0.5
        converted_mask = np.where(mask >= threshold, mask, 0)
        Rt = torch.tensor(NUN, dtype=torch.float32, device=data.device)
        converted_mask_torch = torch.tensor(converted_mask, dtype=torch.float32, device=data.device)
        perturbated_input = data.mul(1 - converted_mask_torch) + Rt.mul(converted_mask_torch)

        pred_outputs = self.softmax_fn(
            self.predict_fn(perturbated_input.reshape(1, perturbated_input.shape[0], perturbated_input.shape[1]).float())
        )
        target_prob = float(pred_outputs[0][cf_label].item())
        converted_mask = converted_mask_torch.detach().cpu().numpy().flatten()

        save_timeseries_mul(
            mask=converted_mask,
            raw_mask=None,
            time_series=data.detach().cpu().numpy(),
            perturbated_output=perturbated_input.detach().cpu().numpy(),
            save_dir=kwargs["save_dir"],
            enable_wandb=self.enable_wandb,
            algo=self.args.algo,
            dataset=self.args.dataset,
            category=top_prediction_class,
        )

        perturbated_input = perturbated_input.detach().cpu().numpy()

        return converted_mask, perturbated_input, target_prob


class TFMCELSExplainer(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(TFMCELSExplainer, self).__init__(
            background_data=background_data,
            background_label=background_label,
            predict_fn=predict_fn,
        )
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = lambda x: tf.nn.softmax(x, axis=-1)
        self.perturbation_manager = None
        self.conf_threshold = 0.8
        self.eps = None
        self.eps_decay = 0.9991
        self.metrics = None
        self.cf_label = None

    def _predict_with_optional_numpy_break(self, input_tensor):
        reshaped = tf.reshape(input_tensor, (1, input_tensor.shape[0], input_tensor.shape[1]))
        if getattr(self.args, "break_tf_gradients_with_numpy", False):
            return self.predict_fn(reshaped.numpy())
        return self.predict_fn(reshaped)

    def native_guide_retrieval(self, query, distance, n_neighbors, target_label=None, predicted_label=None):
        dim_nums, ts_length = query.shape[0], query.shape[1]
        df = pd.DataFrame(self.background_label, columns=["label"])
        if target_label is not None:
            candidate_indexes = df[df["label"] == target_label].index.values
        elif predicted_label is not None:
            candidate_indexes = df[df["label"] != predicted_label].index.values
        else:
            raise ValueError("Either target_label or predicted_label must be provided.")

        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(self.background_data[list(candidate_indexes)])

        dist, ind = knn.kneighbors(query.reshape(1, dim_nums, ts_length), return_distance=True)
        selected_indexes = candidate_indexes[ind[0][:]]
        selected_label = int(df.loc[selected_indexes[0], "label"])
        return dist, selected_indexes, selected_label

    def cf_label_fun(self, instance):
        output = self.softmax_fn(self._predict_with_optional_numpy_break(instance))
        target = int(tf.argsort(output, direction="DESCENDING")[0, 1].numpy())
        return target

    def generate_saliency(self, data, label, **kwargs):
        self.mode = "Explore"
        query = data.copy()

        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data, dtype=tf.float32)

        target_array = kwargs["target"].numpy() if hasattr(kwargs["target"], "numpy") else np.asarray(kwargs["target"])
        top_prediction_class = int(np.argmax(target_array))

        predicted_label = top_prediction_class
        nun_strategy = getattr(self.args, "nun_strategy", "second_best")
        if nun_strategy == "second_best":
            cf_label = self.cf_label_fun(data)
            _, idx, selected_label = self.native_guide_retrieval(
                query, "euclidean", 1, target_label=cf_label
            )
        elif nun_strategy == "global":
            _, idx, selected_label = self.native_guide_retrieval(
                query, "euclidean", 1, predicted_label=predicted_label
            )
            cf_label = selected_label
        else:
            raise ValueError(f"Unsupported nun_strategy: {nun_strategy}")
        self.cf_label = cf_label

        NUN = self.background_data[idx.item()]

        self.eps = 1.0

        mask_init = np.random.uniform(size=data.shape, low=0, high=1)
        mask = tf.Variable(mask_init.astype(np.float32), trainable=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

        print(f"{self.args.algo}: Optimizing... ")
        metrics = defaultdict(lambda: [])

        max_iterations_without_improvement = 100
        imp_threshold = 0.001
        best_loss = float("inf")
        counter = 0

        i = 0
        while i <= self.args.max_itr:
            Rt = tf.convert_to_tensor(NUN, dtype=tf.float32)

            with tf.GradientTape() as tape:
                perturbated_input = data * (1 - mask) + Rt * mask
                pred_outputs = self.softmax_fn(self._predict_with_optional_numpy_break(perturbated_input))

                l_maximize = 1 - pred_outputs[0][cf_label]
                l_budget_loss = tf.reduce_mean(tf.abs(mask)) * float(self.args.enable_budget)
                l_tv_norm_loss = tv_norm_tf(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)

                loss = (self.args.l_budget_coeff * l_budget_loss) + \
                       (self.args.l_tv_norm_coeff * l_tv_norm_loss) + \
                       (self.args.l_max_coeff * l_maximize)

            if best_loss - float(loss.numpy()) < imp_threshold:
                counter += 1
            else:
                counter = 0
                best_loss = float(loss.numpy())

            grads = tape.gradient(loss, [mask])
            optimizer.apply_gradients(zip(grads, [mask]))

            if self.args.enable_lr_decay:
                optimizer.learning_rate.assign(optimizer.learning_rate * self.args.lr_decay)

            mask.assign(tf.clip_by_value(mask, 0.0, 1.0))

            metrics["L_Maximize"].append(float(l_maximize.numpy()))
            metrics["L_Budget"].append(float(l_budget_loss.numpy()))
            metrics["L_TV_Norm"].append(float(l_tv_norm_loss.numpy()))
            metrics["L_Total"].append(float(loss.numpy()))
            metrics["CF_Prob"].append(float(pred_outputs[0][cf_label].numpy()))

            if self.enable_wandb:
                _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"}}
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]["eval_metrics"]}
                wandb.log(_mets)

            if counter >= max_iterations_without_improvement:
                print("Early stopping triggered: 'total loss' metric didn't improve much")
                break
            else:
                i += 1

        self.metrics = metrics

        mask_np = mask.numpy()

        threshold = 0.5
        converted_mask = np.where(mask_np >= threshold, mask_np, 0)
        Rt = tf.convert_to_tensor(NUN, dtype=tf.float32)
        converted_mask_tf = tf.convert_to_tensor(converted_mask, dtype=tf.float32)
        perturbated_input = data * (1 - converted_mask_tf) + Rt * converted_mask_tf

        pred_outputs = self.softmax_fn(self._predict_with_optional_numpy_break(perturbated_input))
        target_prob = float(pred_outputs[0][cf_label].numpy())
        converted_mask = converted_mask_tf.numpy().flatten()

        save_timeseries_mul(
            mask=converted_mask,
            raw_mask=None,
            time_series=data.numpy(),
            perturbated_output=perturbated_input.numpy(),
            save_dir=kwargs["save_dir"],
            enable_wandb=self.enable_wandb,
            algo=self.args.algo,
            dataset=self.args.dataset,
            category=top_prediction_class,
        )

        perturbated_input = perturbated_input.numpy()

        return converted_mask, perturbated_input, target_prob
