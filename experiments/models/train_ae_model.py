import copy
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from experiments.experiment_utils import local_data_loader, ucr_data_loader
from experiments.models.utils import build_ae_model
from methods.outlier_calculators import AEOutlierCalculator


"""DATASETS = ["NATOPS"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(64, 5), (32, 5), (16, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["UWaveGestureLibrary"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(32, 7), (16, 7), (8, 5), (4, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

DATASETS = ["BasicMotions"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(32, 5), (4, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}


def train_ae_experiment(dataset, params):
    # Set seed
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])
    random.seed(2)

    # Create model folder if it does not exist
    if not os.path.isdir(f"./experiments/models/{dataset}"):
        os.makedirs(f"./experiments/models/{dataset}")

    # Load data
    if os.path.isdir(f"./experiments/data/UCR/{dataset}"):
        X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="./experiments/data")
    else:
        os.makedirs(f"./experiments/data/UCR/{dataset}")
        X_train, y_train, X_test, y_test = ucr_data_loader(dataset, store_path="./experiments/data/UCR")

    # Define model architecture
    input_shape = X_train.shape[1:]
    ae = build_ae_model(
        input_shape,
        params["encoder_filters_kernels"],
        params['temporal_strides']
    )
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    print(ae.summary())

    # Model fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["es_patience"], verbose=1)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=params["lrs_patience"], verbose=1)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(f'./experiments/models/{dataset}/{dataset}_ae.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    training_history = ae.fit(
        X_train, X_train,
        batch_size=params['batch_size'],
        shuffle=True,
        validation_split=0.1,
        epochs=params["epochs"],
        callbacks=[early_stopping, lr_scheduler, mcp_save],
    )
    pd.DataFrame(training_history.history)[['loss', 'val_loss']].plot()
    with open(f'./experiments/models/{dataset}/train_ae_params.json', "w") as outfile:
        json.dump(params, outfile)

    # Evaluation on test set
    X_test_recons = ae.predict(X_test)
    n_features = X_test.shape[2]
    for i in np.random.choice(len(X_test), 10):
        fig = plt.figure(figsize=(8, 6))
        f, axs = plt.subplots(n_features, 1)
        for j in range(n_features):
            axs[j].plot(list(range(X_test.shape[1])), X_test[i, :, j].flatten())
            axs[j].plot(list(range(X_test.shape[1])), X_test_recons[i, :, j].flatten())
        plt.title(f"Label: {y_test[i]}")
        plt.show()

    # Create Outlier calculator and store it
    outlier_calculator = AEOutlierCalculator(ae, X_train)
    with open(f'./experiments/models/{dataset}/{dataset}_outlier_calculator.pickle', 'wb') as f:
        pickle.dump(outlier_calculator, f, pickle.HIGHEST_PROTOCOL)

    # Create test for AE reconstruction errors
    # For test set
    X_test_recons = ae.predict(X_test)
    recons_errors_base = np.mean(np.abs(X_test - X_test_recons), axis=(1, 2))

    # For same true class channel permutations
    classes = np.unique(y_train)
    class_indexes = {c: np.where(y_test == c)[0] for c in classes}
    X_perm_same = []
    for i, instance in enumerate(X_test):
        c = y_test[i]
        idx_of_interest = class_indexes[c]
        # Get random sample to use
        perm_idx = random.choice(idx_of_interest)
        # Create permutation mask
        perm_channels = np.random.randint(2, size=X_train.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_train.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_same.append(new_instance)
    X_perm_same = np.array(X_perm_same)
    X_perm_same_recons = ae.predict(X_perm_same)
    recons_errors_perm_same = np.mean(np.abs(X_perm_same - X_perm_same_recons), axis=(1, 2))

    # For different true class channel permutations
    classes = np.unique(y_train)
    class_indexes = {c: np.where(y_test != c)[0] for c in classes}
    X_perm_diff = []
    for i, instance in enumerate(X_test):
        c = y_test[i]
        idx_of_interest = class_indexes[c]
        # Get random sample to use
        perm_idx = random.choice(idx_of_interest)
        # Create permutation mask
        perm_channels = np.random.randint(2, size=X_train.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_train.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_diff.append(new_instance)
    X_perm_diff = np.array(X_perm_diff)
    X_perm_diff_recons = ae.predict(X_perm_diff)
    recons_errors_perm_diff = np.mean(np.abs(X_perm_diff - X_perm_diff_recons), axis=(1, 2))

    # Compare
    recons_errors_df = pd.DataFrame()
    recons_errors_df['base'] = recons_errors_base
    recons_errors_df['perm_same'] = recons_errors_perm_same
    recons_errors_df['perm_diff'] = recons_errors_perm_diff
    plt.figure()
    recons_errors_df.hist(layout=(3, 1), sharex=True, sharey=True)
    plt.show()
    print('Finished')



if __name__ == "__main__":
    for dataset in DATASETS:
        print(f'Starting training for dataset {dataset}...')
        train_ae_experiment(dataset, TRAIN_PARAMS)
    print('Finished')

