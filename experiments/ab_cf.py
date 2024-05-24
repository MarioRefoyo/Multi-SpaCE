import os
import copy
import pickle
import time
import sys
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow import keras

from experiments.experiment_utils import local_data_loader, label_encoder
from methods.ABCF.utils import sliding_window_3d, entropy, target_adapted, native_guide_retrieval


DATASETS = ['CBF', 'chinatown', 'coffee', 'gunpoint', 'ECG200']

experiments = {
    'ab_cf': {
        'params': {
        },
    },
}


def experiment_dataset(dataset, exp_name, params):
    # Load dataset data
    # X_train, y_train, X_test, y_test = ucr_data_loader(DATASET, store=True)
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="./data")
    y_train, y_test = label_encoder(y_train, y_test)

    # Load model
    model = keras.models.load_model(f'models/{dataset}/{dataset}_best_model.hdf5')

    # Predict on x test
    y_pred_logits = model.predict(X_test, verbose=0)
    y_preds = np.argmax(y_pred_logits, axis=1)

    # AB-CF implementation
    X_train = np.swapaxes(X_train, 2, 1)
    X_test = np.swapaxes(X_test, 2, 1)
    TS_nums, dim_nums, ts_length = X_train.shape[0], X_train.shape[1], X_train.shape[2]

    min_entropy_indices = []
    window_size = int(X_test[0].shape[1] * 0.1)
    stride = window_size

    cfs = []
    target_probas = []
    times = []
    for i in tqdm(range(len(X_test))):  # len(X_test)
        start_time = time.time()
        subsequences = sliding_window_3d(X_test[i], window_size, stride)
        padded_subsequences = np.pad(
            subsequences,
            ((0, 0), (0, 0), (0, ts_length - subsequences.shape[2])),
            mode='constant'
        )
        # predict_proba = model.predict_proba(padded_subsequences)
        # pred = model.predict(X_test[i].reshape(1, X_test[i].shape[0], X_test[i].shape[1]))
        predict_proba = model.predict(np.swapaxes(padded_subsequences, 2, 1))
        pred = y_preds[i]
        entropies = []
        for j in range(len(predict_proba)):
            entro = entropy(predict_proba[j])
            entropies.append(entro)
        indices = np.argsort(entropies)[:10]
        # print(indices)
        min_entropy_index = np.argmin(entropies)
        if pred != y_test[i]:
            target = y_test[i]
        else:
            target = target_adapted(model, X_test[i])
        idx = native_guide_retrieval(X_test[i], target, 'dtw', 1, X_train, y_train)

        nun = X_train[idx.item()]
        cf = X_test[i].copy()
        num_dim_changed = []
        k = 1
        for index in indices:
            start = index * stride
            end = start + window_size
            columns_toreplace = list(range(start, end))
            cf[:, columns_toreplace] = nun[:, columns_toreplace]
            cf = cf.reshape(1, cf.shape[0], cf.shape[1])
            cf_pred = model.predict(np.swapaxes(cf, 2, 1))
            # if model.predict(np.swapaxes(cf, 2, 1)) == target:
            if np.argmax(cf_pred, axis=1)[0] == target:
                # print("success")
                # print(k)
                target_proba = cf_pred[0][target]
                target_probas.append(target_proba)
                end = time.time() - start_time
                times.append(end)
                num_dim_changed.append(k)
                cfs.append(cf)
                break
            else:
                if index == indices[-1]:
                    cfs.append(cf)
                else:
                    cf = cf.reshape(cf.shape[1], cf.shape[2])
                    k = k + 1

    # Adapt counterfactual result to our format
    results = [{'cf': np.swapaxes(cf, 1, 2), 'time': -1} for cf in cfs]
    # Store concatenated file
    with open(f'./results/{dataset}/{exp_name}.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for experiment_name, experiment_params in experiments.items():
        for dataset in DATASETS:
            print(f'Starting experiment {experiment_name} for dataset {dataset}...')
            experiment_dataset(
                dataset,
                experiment_name,
                experiment_params["params"]
            )
    print('Finished')
