import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.datasets import UCR_UEA_datasets


def store_partial_cfs(results, s_start, s_end, dataset, file_suffix_name):
    # Create folder for dataset if it does not exist
    os.makedirs(f'./results/{dataset}/', exist_ok=True)
    with open(f'./results/{dataset}/{file_suffix_name}_{s_start:04d}-{s_end:04d}.pickle', 'wb') as f:
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
