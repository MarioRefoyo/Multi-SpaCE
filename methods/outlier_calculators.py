from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_samples


class OutlierCalculator(ABC):
    def __init__(self, model, calibration_data):
        self.length = calibration_data.shape[1]
        self.n_channels = calibration_data.shape[2]
        self.model = model

    @abstractmethod
    def _get_raw_outlier_scores(self, data):
        pass

    def _scale_score(self, data):
        scaled_scores = (data - self.min_score) / (self.max_score - self.min_score)
        return scaled_scores

    def _clip_score(self, data):
        scaled_scores = np.clip(data, a_min=self.min_score, a_max=self.max_score)
        return scaled_scores

    def get_outlier_scores(self, data):
        data_reconstruction_errors = self._get_raw_outlier_scores(data)
        scores = self._scale_score(data_reconstruction_errors).flatten()
        # scores = self._clip_score(scores)
        return scores


class AEOutlierCalculator(OutlierCalculator):
    def __init__(self, model, calibration_data):
        super().__init__(model, calibration_data)
        self._build_infer()

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = min(0, calibration_scores.min())
        self.max_score = calibration_scores.max()

    def _build_infer(self):
        model = self.model

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, self.length, self.n_channels], dtype=tf.float32)
            ],
            reduce_retracing=True,
        )
        def infer(x):
            return model(x, training=False)

        self._infer = infer

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_infer", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_infer()

    def _get_raw_outlier_scores(self, data):
        data = np.asarray(data, dtype=np.float32).reshape(-1, self.length, self.n_channels)
        data_reconstruction = self._infer(data)
        reconstruction_errors = np.mean(np.abs(data - data_reconstruction), axis=(1, 2))
        return reconstruction_errors


class IFOutlierCalculator(OutlierCalculator):

    def __init__(self, model, calibration_data):
        super().__init__(model, calibration_data)

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = calibration_scores.min()
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        data_flat = data.reshape(-1, self.length*self.n_channels)
        outlier_score = self.model.score_samples(data_flat)
        # Multiply by minus one as lower is more anomaly
        outlier_score = -1 * outlier_score
        return outlier_score


class LOFOutlierCalculator(OutlierCalculator):
    def __init__(self, model, calibration_data):
        super().__init__(model, calibration_data)

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_scores = self._get_raw_outlier_scores(calibration_data)
        self.min_score = calibration_scores.min()
        self.max_score = calibration_scores.max()

    def _get_raw_outlier_scores(self, data):
        data_flat = data.reshape(-1, self.length*self.n_channels)
        outlier_score = self.model.score_samples(data_flat)
        # Multiply by minus one as lower is more anomaly
        outlier_score = -1 * outlier_score
        return outlier_score
