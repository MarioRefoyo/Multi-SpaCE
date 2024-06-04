import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from experiments.experiment_utils import local_data_loader, ucr_data_loader, label_encoder
from experiments.models.utils import build_classification_model


"""DATASETS = ["NATOPS", "UWaveGestureLibrary"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(32, 3), (64, 3), (128, 3)],
    'dense_units': [128],
    'dropout': 0.25,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["BasicMotions"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(32, 3), (64, 3)],
    'dense_units': [64],
    'dropout': 0.,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

DATASETS = ["Epilepsy"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(16, 7), (32, 7), (64, 3)],
    'dense_units': [128],
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}


def train_experiment(dataset, params):
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
    y_train, y_test = label_encoder(y_train, y_test)

    # Define model architecture
    input_shape = X_train.shape[1:]
    classes = np.unique(y_train)
    n_classes = len(classes)
    model = build_classification_model(
        input_shape, n_classes,
        params["conv_filters_kernels"], params["dense_units"], params["dropout"]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    print(model.summary())

    # Model fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["es_patience"], verbose=1)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=params["lrs_patience"], verbose=1)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(f'./experiments/models/{dataset}/{dataset}_best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    training_history = model.fit(
        X_train, y_train,
        batch_size=params['batch_size'],
        shuffle=True,
        validation_split=0.1,
        epochs=params["epochs"],
        callbacks=[early_stopping, lr_scheduler, mcp_save],
    )
    pd.DataFrame(training_history.history)[['acc', 'val_acc']].plot()
    with open(f'./experiments/models/{dataset}/train_params.json', "w") as outfile:
        json.dump(params, outfile)

    # Evaluation on test set
    y_probs = model.predict(X_test)
    predicted_labels = np.argmax(y_probs, axis=1)
    print(classification_report(y_test, predicted_labels))
    cm = confusion_matrix(y_test, predicted_labels)
    cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(12, 12))
    cmp.plot(ax=ax)


if __name__ == "__main__":
    for dataset in DATASETS:
        print(f'Starting training for dataset {dataset}...')
        train_experiment(dataset, TRAIN_PARAMS)
    print('Finished')

