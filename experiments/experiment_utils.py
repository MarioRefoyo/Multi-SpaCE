import os
import random
import json
import pickle
import itertools
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tslearn.neighbors import KNeighborsTimeSeries
import torch
import tensorflow as tf

from experiments.data_utils import local_data_loader, ucr_data_loader, label_encoder
from experiments.models.pytorch_utils import model_selector


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


def store_partial_cfs(results, s_start, s_end, dataset, model_to_explain_name, file_suffix_name):
    # Create folder for dataset if it does not exist
    os.makedirs(f'./experiments/results/{dataset}/', exist_ok=True)
    os.makedirs(f'./experiments/results/{dataset}/{model_to_explain_name}/', exist_ok=True)
    os.makedirs(f'./experiments/results/{dataset}/{model_to_explain_name}/{file_suffix_name}/', exist_ok=True)
    with open(f'./experiments/results/{dataset}/{model_to_explain_name}/{file_suffix_name}/{file_suffix_name}_{s_start:04d}-{s_end:04d}.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


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


def prepare_experiment(dataset, params, model_to_explain):
    # Set seed
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        random.seed(params["seed"])

    # Load dataset data
    scaling = params["scaling"]
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), scaling, backend="tf",
                                                         data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)
    ts_length = X_train.shape[1]
    n_channels = X_train.shape[2]
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Get a subset of testing data if specified
    if (params["subset"]) & (len(y_test) > params["subset_number"]):
        X_test, y_test, subset_idx = get_subsample(X_test, y_test, params["subset_number"], params["seed"])
    else:
        subset_idx = np.arange(len(X_test))

    # Get model
    model_folder = f'experiments/models/{dataset}/{model_to_explain}'
    model_wrapper = load_model(model_folder, dataset, n_channels, ts_length, n_classes)

    # Predict
    y_pred_test_logits = model_wrapper.predict(X_test)
    y_pred_train_logits = model_wrapper.predict(X_train)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)
    # Classification report
    print(classification_report(y_test, y_pred_test))

    return X_train, y_train, X_test, y_test, subset_idx, n_classes, model_wrapper, y_pred_train, y_pred_test


def load_model(model_folder, dataset, n_channels, ts_length, n_classes):
    if os.path.exists(f'{model_folder}/model.hdf5'):
        backend = "tf"
        model = tf.keras.models.load_model(f'{model_folder}/model.hdf5')

    elif os.path.exists(f'{model_folder}/model_weights.pth'):
        backend = "torch"
        # Load train params
        with open(f"{model_folder}/train_params.json") as f:
            train_params = json.load(f)
        model, _, _, _ = model_selector(dataset, n_channels, ts_length, n_classes, train_params)
        model_weights = torch.load(f'{model_folder}/model_weights.pth', weights_only=True)
        model.load_state_dict(model_weights)

    else:
        raise ValueError("Not valid model path or backend")
    model_wrapper = ModelWrapper(model, backend)
    return model_wrapper


class ModelWrapper:
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend.lower()

        # Prepare for backend
        if self.backend == "torch":
            self.framework = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        elif self.backend == "tf":
            self.framework = tf
        else:
            raise ValueError("Unsupported backend: choose 'torch' or 'tf'.")

    def predict(self, x: np.ndarray, input_data_format="tf") -> np.ndarray:
        assert input_data_format in ["tf", "torch"]

        # Append
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)
        if self.backend == "torch":
            if input_data_format == "tf":
                # Swap axes: from (B, T, F) to (B, F, T)
                x = np.transpose(x, (0, 2, 1))
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
                output = torch.nn.functional.softmax(output, dim=1)
            return output.detach().cpu().numpy()

        elif self.backend == "tf":
            if input_data_format == "torch":
                # Swap axes: from (B, F, T) to (B, T, F)
                x = np.transpose(x, (0, 2, 1))
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            output = self.model.predict(x_tensor, verbose=0)
            return output
