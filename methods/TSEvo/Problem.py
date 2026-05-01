import numpy as np
import torch
from pymop import Problem


class MultiObjectiveCounterfactuals(Problem):
    def __init__(
        self,
        model,
        observation,
        original_y,
        target,
        reference_set,
        neighborhood,
        window,
        backend="torch",
        channels=1,
        mode="feat",
    ):
        super().__init__(n_var=channels, n_obj=3, n_constr=0, evaluation_of="auto")

        self.model = model
        self.mode = mode
        self.window = window
        self.observation = observation
        self.target = target
        self.original_y = original_y
        if type(original_y) is int:
            self.original_label = original_y
        elif isinstance(original_y, np.ndarray) and original_y.ndim == 1:
            self.original_label = int(np.argmax(original_y))
        elif len(original_y) > 1:
            self.original_label = np.argmax(original_y, axis=1)[0]
        else:
            self.original_label = original_y

        self.reference_set = reference_set
        self.neighborhood = neighborhood
        self.backend = backend
        self.uses_model_wrapper = hasattr(model, "predict") and hasattr(model, "backend")

        if self.backend == "PYT" and not self.uses_model_wrapper:
            self.model.eval()

        if type(original_y) == np.int64:
            if self.backend == "PYT":
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_binary
        elif self.target is None:
            if self.backend == "PYT":
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_multi
        else:
            if self.backend == "PYT":
                self.predict = self.get_prediction_torch
            if self.backend == "TF":
                self.predict = self.get_prediction_tensorflow
            self.output_distance = self.output_distance_target
        self.label, self.output = self.predict(observation, full=True)

    def _evaluate(self, explanation, out, *args, **kwargs):
        label, output = self.predict(explanation)

        output_distance = self.output_distance(output, label)
        x_distance = self.mean_delta(self.observation, explanation)
        num_changed_features = np.count_nonzero(self.observation - explanation) / (
            self.observation.shape[-1] * self.observation.shape[-2]
        )

        out["F"] = np.column_stack([output_distance, x_distance, num_changed_features])

    def evaluate_population(self, individuals, invalid_penalty=1.0):
        if len(individuals) == 0:
            return []

        batch = np.asarray([np.asarray(ind, dtype=np.float64) for ind in individuals], dtype=np.float64)
        outputs = self.predict_batch(batch)
        labels = outputs.argmax(axis=1)
        pred_scores = outputs[np.arange(len(outputs)), labels]

        if type(self.original_y) == np.int64:
            output_distances = 1 - pred_scores
            output_distances[labels == self.original_label] = invalid_penalty
        elif self.target is None:
            output_distances = pred_scores - self.output[labels]
            output_distances[labels == self.original_label] = invalid_penalty
        else:
            output_distances = pred_scores - self.output[self.target]
            output_distances[labels != self.target] = invalid_penalty

        observation = np.asarray(self.observation, dtype=np.float64).reshape(1, batch.shape[1], batch.shape[2])
        x_distances = np.mean(np.abs(batch - observation), axis=(1, 2))
        changed_features = np.count_nonzero(batch != observation, axis=(1, 2)) / (
            self.observation.shape[-1] * self.observation.shape[-2]
        )

        invalid_mask = labels == self.original_label
        x_distances[invalid_mask] = invalid_penalty
        changed_features[invalid_mask] = invalid_penalty

        return list(zip(output_distances, x_distances, changed_features))

    def feasible(self, individual):
        prediction, _ = self.predict(np.asarray(individual))
        return prediction != self.original_label

    def mean_delta(self, first, second):
        return np.sum(np.abs(first - second)) / (first.shape[-1] * first.shape[-2])

    def _prepare_wrapper_input(self, individual):
        individual = np.array(individual.tolist(), dtype=np.float64)
        return np.swapaxes(individual, -1, -2).reshape(
            1, individual.shape[-1], individual.shape[-2]
        )

    def _prepare_wrapper_batch_input(self, individuals):
        return np.swapaxes(individuals, 1, 2)

    def predict_batch(self, individuals):
        if self.backend == "PYT":
            return self.get_prediction_torch_batch(individuals)
        return self.get_prediction_tensorflow_batch(individuals)

    def get_prediction_torch(self, individual, full=False):
        if self.uses_model_wrapper:
            output = self.model.predict(
                self._prepare_wrapper_input(individual), input_data_format="tf"
            )
            idx = output.argmax()
            if full:
                return idx, output[0]
            return idx, output[0][idx]

        if self.mode == "time":
            individual = np.swapaxes(individual, -1, -2).reshape(
                1, individual.shape[-1], individual.shape[-2]
            )
        else:
            individual = individual.reshape(
                1, individual.shape[-2], individual.shape[-1]
            )
        individual = np.array(individual.tolist(), dtype=np.float64)
        input_ = torch.from_numpy(individual).float()

        with torch.no_grad():
            output = torch.nn.functional.softmax(self.model(input_)).detach().numpy()

        idx = output.argmax()

        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def get_prediction_torch_batch(self, individuals):
        if self.uses_model_wrapper:
            return self.model.predict(
                self._prepare_wrapper_batch_input(individuals), input_data_format="tf"
            )

        if self.mode == "time":
            individuals = np.swapaxes(individuals, -1, -2)
        input_ = torch.from_numpy(np.asarray(individuals, dtype=np.float64)).float()

        with torch.no_grad():
            output = torch.nn.functional.softmax(self.model(input_), dim=1).detach().numpy()
        return output

    def get_prediction_tensorflow(self, individual, full=False):
        if self.uses_model_wrapper:
            output = self.model.predict(
                self._prepare_wrapper_input(individual), input_data_format="tf"
            )
        else:
            individual = np.array(individual.tolist(), dtype=np.float64)
            output = self.model.predict(
                np.swapaxes(individual, -1, -2).reshape(1, self.window, -1), verbose=0
            )
        idx = output.argmax()

        if full:
            return idx, output[0]
        return idx, output[0][idx]

    def get_prediction_tensorflow_batch(self, individuals):
        if self.uses_model_wrapper:
            return self.model.predict(
                self._prepare_wrapper_batch_input(individuals), input_data_format="tf"
            )

        return self.model.predict(
            np.swapaxes(individuals, -1, -2),
            verbose=0,
        )

    def output_distance_binary(self, output, label):
        output_distance = 1 - output

        if label == self.original_label:
            output_distance = 1.0

        return output_distance

    def output_distance_multi(self, output, label):
        output_distance = output - self.output[label]

        if label == self.original_label:
            output_distance = 1.0
        return output_distance

    def output_distance_target(self, output, label):
        output_distance = output - self.output[self.target]
        if label != self.target:
            output_distance = 1.0
        return output_distance
