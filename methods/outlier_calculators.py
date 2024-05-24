import numpy as np


class AEOutlierCalculator:
    def __init__(self, ae_model, calibration_data):
        self.length = calibration_data.shape[1]
        self.n_channels = calibration_data.shape[2]
        self.ae_model = ae_model

        # Calibrate to get outlier score as a number between 0 and 1
        calibration_reconstruction_errors = self._get_reconst_errors(calibration_data)
        self.min_error = 0
        self.max_error = calibration_reconstruction_errors.max()

    def _get_reconst_errors(self, data):
        data = data.reshape(-1, self.length, self.n_channels)
        data_reconstruction = self.ae_model.predict(data, verbose=0)
        reconstruction_errors = np.mean(np.abs(data - data_reconstruction), axis=(1, 2))
        return reconstruction_errors

    def _scale_score(self, data):
        scaled_scores = (data - self.min_error) / (self.max_error - self.min_error)
        return scaled_scores

    def get_outlier_scores(self, data):
        data_reconstruction_errors = self._get_reconst_errors(data)
        scores = self._scale_score(data_reconstruction_errors).flatten()
        return scores
