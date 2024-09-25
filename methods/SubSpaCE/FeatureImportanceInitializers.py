from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR


class FeatureImportanceMethod(ABC):

    def __init__(self, backend='tf'):
        self.backend = backend
        if backend == 'tf':
            self.feature_axis = 2
        else:
            raise ValueError('Backend not supported')

    @abstractmethod
    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        pass


class NoneFI(FeatureImportanceMethod):

    def __init__(self, backend):
        super().__init__(backend)

    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        heatmap = np.zeros(instance.shape)
        return heatmap


class GraCAMPlusFI(FeatureImportanceMethod):

    def __init__(self, backend, model):
        super().__init__(backend)

        # Detect convolutional layers
        conv_layer_names = [layer.name for layer in model.layers if 'conv1d' in layer.name]
        if not conv_layer_names:
            raise ValueError("Model has not convolutional layers. "
                             "GradCAM++ needs convolutional layers to generate an explanation")
        else:
            self.last_conv_layer_name = conv_layer_names[-1]
        conv_layer = model.get_layer(self.last_conv_layer_name)
        self.heatmap_model = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        # Calculate importance heatmap
        data_tensor = np.expand_dims(instance, axis=0)

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = self.heatmap_model(data_tensor)
                    if desired_target is None:
                        category_id = np.argmax(predictions[0])
                    output = predictions[:, category_id]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

        global_sum = np.sum(conv_output, axis=(0, 1))

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=0)
        alphas /= alpha_normalization_constant
        alphas = np.where(np.isnan(alphas), 0, alphas)

        weights = np.maximum(conv_first_grad[0], 0)

        deep_linearization_weights = np.sum(weights * alphas, axis=0)
        grad_map = deep_linearization_weights * conv_output[0]
        grad_cam_map = np.sum(grad_map, axis=1)
        heatmap = np.maximum(grad_cam_map, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        # Expand heatmap
        expanded_heatmap = np.interp(np.linspace(0, 1, instance.shape[0]), np.linspace(0, 1, len(heatmap)), heatmap)
        expanded_heatmap = np.tile(expanded_heatmap.reshape(-1, 1), (1, instance.shape[1]))

        if np.isnan(expanded_heatmap).any():
            expanded_heatmap = np.zeros(expanded_heatmap.shape)
        return expanded_heatmap


class TSRFI(FeatureImportanceMethod):

    def __init__(self, backend, model, ts_length, n_channels, method):
        super().__init__(backend)
        self.tsr = TSR(model, ts_length, n_channels, method=method, mode='time')

    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        data_tensor = np.expand_dims(instance, axis=0)
        if desired_target is None:
            desired_target = np.argmax(self.tsr.model(data_tensor), axis=1)[0]
        heatmap = self.tsr.explain(data_tensor, labels=desired_target, TSR=True)
        return heatmap