import os
import random
import json
import pickle
import itertools
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.datasets import UCR_UEA_datasets


def get_subsample(X_test, y_test, n_instances, seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    subset_idx = np.random.choice(len(X_test), n_instances, replace=False)
    subset_idx = np.sort(subset_idx)
    X_test = X_test[subset_idx]
    y_test = y_test[subset_idx]
    return X_test, y_test, subset_idx


def get_hash_from_params(params):
    params_str = ''.join(f'{key}={value},' for key, value in sorted(params.items()))
    params_hash = hashlib.sha1(params_str.encode()).hexdigest()
    return params_hash


def generate_settings_combinations(original_dict):
    # Create a list of keys with lists as values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    # Generate all possible combinations
    combinations = list(itertools.product(*[original_dict[key] for key in list_keys]))
    # Create a set of experiments dictionaries with unique combinations
    result = {}
    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        experiment_hash = get_hash_from_params(new_dict)
        result[experiment_hash] = new_dict
    return result


def load_parameters_from_json(json_filename):
    with open(json_filename, 'r') as json_file:
        params = json.load(json_file)
    return params


def store_partial_cfs(results, s_start, s_end, dataset, file_suffix_name):
    # Create folder for dataset if it does not exist
    os.makedirs(f'./experiments/results/{dataset}/', exist_ok=True)
    os.makedirs(f'./experiments/results/{dataset}/{file_suffix_name}', exist_ok=True)
    with open(f'./experiments/results/{dataset}/{file_suffix_name}/{file_suffix_name}_{s_start:04d}-{s_end:04d}.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def local_data_loader(dataset, data_path="../../data"):
    X_train = np.load(f'{data_path}/UCR/{dataset}/X_train.npy', allow_pickle=True)
    X_test = np.load(f'{data_path}/UCR/{dataset}/X_test.npy', allow_pickle=True)
    y_train = np.load(f'{data_path}/UCR/{dataset}/y_train.npy', allow_pickle=True)
    y_test = np.load(f'{data_path}/UCR/{dataset}/y_test.npy', allow_pickle=True)
    return X_train, y_train, X_test, y_test


def label_encoder(training_labels, testing_labels):
    # If label represent integers, try to cast it. If it is not possible, then resort to Label Encoding
    try:
        y_train = []
        for label in training_labels:
            y_train.append(int(float(label)))
        y_test = []
        for label in testing_labels:
            y_test.append(int(float(label)))
        y_train, y_test = np.array(y_train).reshape(-1, 1), np.array(y_test).reshape(-1, 1)
        # Add class 0 in case it does not exist
        classes = np.unique(y_train)
        if 0 not in classes:
            y_train = y_train - 1
            y_test = y_test - 1
    except Exception:
        le = LabelEncoder()
        le.fit(np.concatenate((training_labels, testing_labels), axis=0))
        y_train = le.transform(training_labels)
        y_test = le.transform(testing_labels)
    return y_train, y_test


def nun_retrieval(query, predicted_label, distance, n_neighbors, X_train, y_train, y_pred, from_true_labels=False):
    df_init = pd.DataFrame(y_train, columns=['true_label'])
    df_init["pred_label"] = y_pred
    df_init.index.name = 'index'

    if from_true_labels:
        label_name = 'true_label'
    else:
        label_name = 'pred_label'
    df = df_init[[label_name]]
    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_train[list(df[df[label_name] != predicted_label].index.values)])
    dist, ind = knn.kneighbors(np.expand_dims(query, axis=0), return_distance=True)
    distances = dist[0]
    index = df[df[label_name] != predicted_label].index[ind[0][:]]
    label = df[df.index.isin(index.tolist())].values[0]
    return distances, index, label


def ucr_data_loader(dataset, store_path="../../data/UCR"):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    np.save(f"{store_path}/{dataset}/X_train.npy", X_train)
    np.save(f"{store_path}/{dataset}/X_test.npy", X_test)
    np.save(f"{store_path}/{dataset}/y_train.npy", y_train)
    np.save(f"{store_path}/{dataset}/y_test.npy", y_test)
    return X_train, y_train, X_test, y_test
