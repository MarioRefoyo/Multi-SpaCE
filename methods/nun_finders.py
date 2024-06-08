import copy

import pandas as pd
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeries


class NUNFinder:
    def __init__(self, X_train, y_train, y_pred, distance, n_neighbors, from_true_labels,
                 independent_channels, backend):
        if backend == 'tf':
            self.backend = backend
            self.feature_axis = 2
            self.n_channels = X_train.shape[2]
            self.ts_length = X_train.shape[1]
            self.data_shape = (self.ts_length, self.n_channels)
        else:
            raise ValueError('Backend not supported')
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred = y_pred
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.independent_channels = independent_channels
        self.backend = backend

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
        diff_class_index_dict = {}
        classes_knn_dict = {}
        for c in classes:
            knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
            diff_class_index = self.df_index[self.df_index[label_name] != c]
            knn.fit(self.X_train[diff_class_index.index.values])
            diff_class_index_dict[c] = diff_class_index
            classes_knn_dict[c] = knn
        self.diff_class_index_dict = diff_class_index_dict
        self.classes_knn_dict = classes_knn_dict

        # Train a knn per channel and class if independent channels flag is active
        if independent_channels:
            classes_feature_knn_dict = {}
            for c in classes:
                diff_class_index = self.diff_class_index_dict[c]
                feature_knn_dict = {}
                for feature in range(self.n_channels):
                    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
                    knn.fit(self.X_train[diff_class_index.index.values, :, feature])
                    feature_knn_dict[feature] = knn
                classes_feature_knn_dict[c] = feature_knn_dict
            self.classes_feature_knn_dict = classes_feature_knn_dict

    def get_nn_ind(self, knn, x_orig, original_label):
        # Get the closes neighbor on the knn training data with general KNN!
        # The idea is to obtain the label of the closest example.
        # Then use that label to find the closest channel of that class if needed
        dist, ind = knn.kneighbors(
            np.expand_dims(x_orig, axis=0), return_distance=True
        )
        # Transform the index to the original index in the complete X_train
        index = self.diff_class_index_dict[original_label].index[ind[0][:]]
        # Get label of the closest neighbor
        label = self.df_index[self.df_index.index.isin(index.tolist())].values[0][0]

        return index[0], label, dist[0]

    def retrieve_single_nun(self, x_orig, original_label):
        global_nun_index, label, global_dist = self.get_nn_ind(
            self.classes_knn_dict[original_label], x_orig, original_label
        )

        # Retrieve NUN
        if self.independent_channels:
            nun = np.zeros(self.data_shape)
            for feature in range(self.n_channels):
                feature_nun_idx, feature_label, feature_dist = self.get_nn_ind(
                    self.classes_feature_knn_dict[original_label][feature],
                    x_orig[:, feature], original_label
                )
                nun[:, feature] = self.X_train[feature_nun_idx, :, feature]
            dist = np.linalg.norm(x_orig - nun)

        else:
            nun = self.X_train[global_nun_index]
            dist = global_dist

        return nun, label, dist

    def retrieve_nuns(self, x_origs, original_labels):
        # Check for shape errors
        if not len(x_origs.shape) == 3:
            if not ((x_origs.shape[0] == self.X_train.shape[1]) & (x_origs.shape[1] == self.X_train.shape[2])):
                raise ValueError(f"The input must have shape of (n_instances, {self.X_train.shape[1]}, {self.X_train.shape[2]}), "
                                 f"but input has shape of {x_origs.shape}")
            else:
                # Expand dimension and treat input as a single dataset sample
                x_origs = copy.deepcopy(x_origs)
                x_origs = np.expand_dims(x_origs, axis=0)

        # Find all the NUNs
        nuns = []
        nun_labels = []
        distances = []
        for instance_idx in range(len(x_origs)):
            nun, nun_label, distance = self.retrieve_single_nun(x_origs[instance_idx], original_labels[instance_idx])
            nuns.append(nun)
            nun_labels.append(nun_label)
            distances.append(distance)

        return np.array(nuns), np.array(nun_labels), np.array(distances)




