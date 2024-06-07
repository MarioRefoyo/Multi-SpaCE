import pandas as pd
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeries


class NUNFinder:
    def __init__(self, X_train, y_train, y_pred, distance, n_neighbors, from_true_labels,
                 independent_channels, backend):
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred = y_pred
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.independent_channels = independent_channels
        self.backend = backend
        if backend == 'tf':
            self.feature_axis = 2
            self.n_channels = X_train.shape[2]
            self.ts_length = X_train.shape[1]
        else:
            raise ValueError('Backend not supported')

        # Get df with index from true training and predicted labels
        df_init = pd.DataFrame(y_train, columns=['true_label'])
        df_init["pred_label"] = y_pred
        df_init.index.name = 'index'
        if from_true_labels:
            label_name = 'true_label'
        else:
            label_name = 'pred_label'
        self.label_name = label_name
        self.df_index = df_init[[label_name]]

        # Train knn for get the NUNs of every possible output class
        classes = np.unique(y_train)
        classes_knn_dict = {}
        for c in classes:
            knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
            diff_class_idxs = list(self.df_index[self.df_index[label_name] != c].index.values)
            knn.fit(X_train[diff_class_idxs])
            classes_knn_dict[c] = knn
        self.classes_knn_dict = classes_knn_dict

        # Train a knn per channel and class
        classes_channels_knn_dict = {}
        for c in classes:
            feature_knn_dict = {}
            for feature in range(self.n_channels):
                knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
                diff_class_idxs = list(self.df_index[self.df_index[label_name] != c].index.values)
                knn.fit(X_train[diff_class_idxs, feature])
                feature_knn_dict[feature] = knn
            classes_channels_knn_dict[c] = feature_knn_dict
        self.classes_channels_knn_dict = classes_channels_knn_dict

    def retrieve_nun(self, x_orig, predicted_label):
        # Get the closes neighbor on the knn training data
        dist, ind = self.classes_knn_dict[predicted_label].kneighbors(
            np.expand_dims(x_orig, axis=0), return_distance=True
        )
        distances = dist[0]
        # Transform the index to the original index in the complete X_train
        index = self.df_index[self.df_index[self.label_name] != predicted_label].index[ind[0][:]]
        # Get label of the closest neighbor
        label = self.df_index[self.df_index.index.isin(index.tolist())].values[0]
        return distances, index, label
