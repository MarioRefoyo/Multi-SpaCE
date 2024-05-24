import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tslearn.neighbors import KNeighborsTimeSeries


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
    le = LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)
    return y_train, y_test


def nun_retrieval(query, predicted_label, distance, n_neighbors, X_train, y_train):
    df = pd.DataFrame(y_train, columns=['label'])
    df.index.name = 'index'

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_train[list(df[df['label'] != predicted_label].index.values)])
    dist, ind = knn.kneighbors(np.expand_dims(query, axis=0), return_distance=True)
    distances = dist[0]
    index = df[df['label'] != predicted_label].index[ind[0][:]]
    label = df[df.index.isin(index.tolist())].values[0]
    return distances, index, label
