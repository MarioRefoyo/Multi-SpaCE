import pickle
import os
import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt

from methods.outlier_calculators import AEOutlierCalculator
from experiments.experiment_utils import local_data_loader

if __name__ == "__main__":
    # Load data
    dataset = 'gunpoint'
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="../../data")

    if not os.path.isfile(f'./{dataset}_ae.hdf5'):
        # Define AE model
        encoder = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                keras.layers.Conv1D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"),
                keras.layers.Dropout(rate=0.2),
                keras.layers.Conv1D(filters=8, kernel_size=3, strides=2, padding="same", activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(80)
            ])
        decoder = keras.Sequential([
            keras.layers.Dense(encoder.layers[-2].output_shape[1]),
            keras.layers.Reshape(encoder.layers[-3].output_shape[1:]),
            keras.layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, padding="same", activation="relu"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"),
            keras.layers.Conv1DTranspose(filters=1, kernel_size=3, padding="same"),
            keras.layers.Cropping1D(cropping=(1, 1))
        ])

        ae = keras.Sequential([encoder, decoder])
        ae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        ae.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
        print(ae.summary())

        # Train AE model
        history = ae.fit(
            X_train,
            X_train,
            epochs=20,
            batch_size=1,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
            ],
        )
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()

        # Reconstruct X_test and plot reconstructions from x_test
        X_test_reconst = ae.predict(X_test)
        n_features = X_test.shape[2]
        for i in np.random.choice(len(X_test), 20):
            fig = plt.figure(figsize=(8, 6))
            f, axs = plt.subplots(n_features, 1)
            if n_features == 1:
                axs = [axs]
            for j in range(n_features):
                axs[j].plot(list(range(X_test.shape[1])), X_test[i, :, j].flatten())
                axs[j].plot(list(range(X_test.shape[1])), X_test_reconst[i, :, j].flatten())
            plt.title(f"Label: {y_test[i]}")
            plt.show()

        # Store keras model
        ae.save(f'./{dataset}_ae.hdf5')

    else:
        ae = keras.models.load_model(f'./{dataset}_ae.hdf5')

    # Create Outlier calculator and store it
    outlier_calculator = AEOutlierCalculator(ae, X_train)
    with open(f'./{dataset}_outlier_calculator.pickle', 'wb') as f:
        pickle.dump(outlier_calculator, f, pickle.HIGHEST_PROTOCOL)

