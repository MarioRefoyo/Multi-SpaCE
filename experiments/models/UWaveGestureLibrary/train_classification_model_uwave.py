import pickle
import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
sys.path.append('../../../')

from experiments.experiment_utils import local_data_loader

if __name__ == "__main__":
    # Load data
    dataset = 'UWaveGestureLibrary'
    X_train, y_train, X_test, y_test = local_data_loader(str(dataset), data_path="../../data")
    class_enc = OneHotEncoder(sparse=False).fit(y_train.reshape(-1, 1))
    y_train = class_enc.transform(y_train.reshape(-1, 1))
    y_test = class_enc.transform(y_test.reshape(-1, 1))
    n_classes = y_train.shape[1]
    classes_list = list(range(n_classes))

    # Define AE model
    model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),

            keras.layers.Conv1D(filters=16, kernel_size=21, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation='relu'),
            keras.layers.MaxPool1D(2, padding="same"),

            keras.layers.Conv1D(filters=32, kernel_size=7, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation='relu'),
            keras.layers.MaxPool1D(2, padding="same"),


            keras.layers.Flatten(),

            keras.layers.Dense(150),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation='relu'),

            keras.layers.Dense(n_classes, activation='softmax')
        ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    # Train AE model
    history = model.fit(
        X_train,
        y_train,
        shuffle=True,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
        ],
    )
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # Reconstruct X_test and plot reconstructions from x_test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes, labels=classes_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_list)
    disp.plot()
    plt.show()
    print(classification_report(y_test_classes, y_pred_classes))

    # Store keras model
    model.save(f'./{dataset}/{dataset}_best_model.hdf5')


