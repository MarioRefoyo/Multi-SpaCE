import copy
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeries


class NUNFinder(ABC):
    def __init__(self, X_train, y_train, y_pred, distance, n_neighbors, from_true_labels, backend):
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

    def get_nns_indexes(self, knn, x_orig, knn_training_index_df):
        # Get the closests neighbors on the knn training data
        dist, ind = knn.kneighbors(
            np.expand_dims(x_orig, axis=0), return_distance=True
        )
        # Get only the first sample of the batch (batch_size=1 since we are only working with x_orig)
        dist, ind = dist[0], ind[0]
        # Transform the index to the original index in the complete X_train
        index = knn_training_index_df.index[ind]
        # Get label of the closest neighbor (even if nn is greater than 1)
        label = self.df_index[self.df_index.index.isin(index.tolist())].values[0][0]

        return index, label, dist

    @abstractmethod
    def retrieve_single_nun_specific(self, x_orig, original_label):
        pass

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
            nun, nun_label, distance = self.retrieve_single_nun_specific(x_origs[instance_idx], original_labels[instance_idx])
            nuns.append(nun)
            nun_labels.append(nun_label)
            distances.append(distance)

        return np.array(nuns), np.array(nun_labels), np.array(distances)


class GlobalNUNFinder(NUNFinder):
    def __init__(self, X_train, y_train, y_pred, distance, n_neighbors, from_true_labels,  backend):
        super().__init__(X_train, y_train, y_pred, distance, n_neighbors, from_true_labels, backend)

    def retrieve_single_nun_specific(self, x_orig, original_label):
        global_nun_indexes, nn_label, global_dists = self.get_nns_indexes(
            self.classes_knn_dict[original_label], x_orig, self.diff_class_index_dict[original_label]
        )
        # global_nun_index, global_dist = global_nun_indexes[0], global_dists[0]

        # Retrieve NUN
        nun = self.X_train[global_nun_indexes]
        dist = global_dists

        return nun, nn_label, dist


class IndependentNUNFinder(NUNFinder):
    def __init__(self, X_train, y_train, y_pred, distance, n_neighbors, from_true_labels, backend):
        super().__init__(X_train, y_train, y_pred, distance, n_neighbors, from_true_labels, backend)

        # Train a knn per channel and class if independent channels flag is active
        classes = np.unique(y_train)

        same_class_index_dict = {}
        classes_feature_knn_dict = {}
        for c in classes:
            feature_knn_dict = {}
            for feature in range(self.n_channels):
                knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)

                same_class_index = self.df_index[self.df_index[self.label_name] == c]
                knn.fit(self.X_train[same_class_index.index.values, :, feature])

                same_class_index_dict[c] = same_class_index
                feature_knn_dict[feature] = knn
            classes_feature_knn_dict[c] = feature_knn_dict
        self.same_class_index_dict = same_class_index_dict
        self.classes_feature_knn_dict = classes_feature_knn_dict

    def retrieve_single_nun_specific(self, x_orig, original_label):
        global_nun_indexes, nn_label, global_dists = self.get_nns_indexes(
            self.classes_knn_dict[original_label], x_orig, self.diff_class_index_dict[original_label]
        )
        global_nun_index, global_dist = global_nun_indexes[0], global_dists[0]

        # Retrieve NUN
        if self.n_neighbors > 1:
            # ToDo: Generate NUN from multiple neighbors
            channel_indexes = []
            for feature in range(self.n_channels):
                feature_nun_idxes, feature_label, _ = self.get_nns_indexes(
                    self.classes_feature_knn_dict[nn_label][feature],
                    x_orig[:, feature], self.same_class_index_dict[nn_label]
                )
                channel_indexes.append(feature_nun_idxes)

            # Generate all possible permutations

            # Get validity of the permutations

            # Order by proximity and get the nun

        else:
            nuns = np.zeros((self.n_neighbors,) + self.data_shape)
            feature_labels = []
            for feature in range(self.n_channels):
                feature_nun_idxes, feature_label, _ = self.get_nns_indexes(
                    self.classes_feature_knn_dict[nn_label][feature],
                    x_orig[:, feature], self.same_class_index_dict[nn_label]
                )
                feature_nun_idx = feature_nun_idxes[0]
                nuns[:, :, feature] = self.X_train[feature_nun_idx, :, feature]
                feature_labels.append(feature_label)
            dist = np.linalg.norm(x_orig - nuns)
            if len(set(feature_labels)) != 1:
                print(f"Channel labels correspond to instances of classes: {feature_labels}. Desired class is {nn_label}")

        return nuns, nn_label, dist

