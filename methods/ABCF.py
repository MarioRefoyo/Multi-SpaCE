import numpy as np
import copy

from .AB.utils import sliding_window_3d, entropy, target_adapted, native_guide_retrieval
from .counterfactual_common import CounterfactualMethod


class ABCF(CounterfactualMethod):
    def __init__(self, model, backend, x_train, y_train, window_pct):
        if backend == 'tf':
            change = True
        else:
            raise "ABCF only supports tf models right now"

        super().__init__(model, backend, change=change)

        if change:
            self.x_train = np.swapaxes(x_train, 2, 1)
        else:
            self.x_train = x_train
        self.TS_nums = self.x_train.shape[0]
        self.dim_nums = self.x_train.shape[1]
        self.ts_length = self.x_train.shape[2]
        self.y_train = y_train

        self.window_pct = window_pct
        self.window_size = int(self.x_train[0].shape[1] * window_pct)
        self.stride = self.window_size

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None, y_true_orig=None):
        if self.change:
            x_orig = np.swapaxes(x_orig, 1, 0)
        y_pred_orig_proba = self.predict_function_tf(np.expand_dims(x_orig, axis=0))
        y_pred_orig = np.argmax(y_pred_orig_proba, axis=1)[0]

        subsequences = sliding_window_3d(x_orig, self.window_size, self.stride)
        padded_subsequences = np.pad(
            subsequences,
            ((0, 0), (0, 0), (0, self.ts_length - subsequences.shape[2])),
            mode='constant'
        )
        predict_proba = self.predict_function_tf(padded_subsequences)
        entropies = []
        for j in range(len(predict_proba)):
            entro = entropy(predict_proba[j])
            entropies.append(entro)
        indices = np.argsort(entropies)[:10]
        # print(indices)
        min_entropy_index = np.argmin(entropies)
        if y_true_orig is None:
            raise ValueError("you must pass y_true_orig label to generate counterfactual with AB-CF")
        if y_pred_orig != y_true_orig:
            target = y_true_orig
        else:
            target = target_adapted(self.model, x_orig)
        idx = native_guide_retrieval(x_orig, target, 'dtw', 1, self.x_train, self.y_train)

        nun = self.x_train[idx.item()]
        cf = x_orig.copy()
        num_dim_changed = []
        k = 1
        for index in indices:
            start = index * self.stride
            end = start + self.window_size
            columns_toreplace = list(range(start, end))
            cf[:, columns_toreplace] = nun[:, columns_toreplace]
            cf = cf.reshape(1, cf.shape[0], cf.shape[1])
            cf_pred = self.predict_function_tf(cf)
            # if model.predict(np.swapaxes(cf, 2, 1)) == target:
            if np.argmax(cf_pred, axis=1)[0] == target:
                print("success")
                # print(k)
                target_proba = cf_pred[0][target]
                num_dim_changed.append(k)
                final_cf = cf.copy()
                break
            else:
                # Modification of original code: add if-else to store the cf at maximum index
                if index == indices[-1]:
                    print('Method failed')
                    final_cf = cf.copy()
                else:
                    cf = cf.reshape(cf.shape[1], cf.shape[2])
                    k = k + 1

        result = {'cf': np.swapaxes(final_cf, 1, 2)}
        return result
