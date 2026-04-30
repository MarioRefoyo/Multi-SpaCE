from pathlib import Path
from typing import Callable, Literal

import numpy as np
from aeon.datasets import load_classification
from downtime import load_dataset
from numpy.typing import NDArray

Datanames = Literal["CBF", "FaultDetectionA"]


def get_data(
    name: Datanames,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    return data_fns[name]()


def _get_data_CBF() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    X_train_an, y_train, X_test_an, y_test = load_dataset("CBF")()
    return np.array(X_train_an), y_train, np.array(X_test_an), y_test


def _get_data_FaultDetectionA() -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    if not (Path("data") / "FaultDetectionA").exists():
        raise RuntimeError(
            "download FaultDetecionA from https://www.timeseriesclassification.com/dataset.php into `data` directory."
        )
    X_train, y_train = load_classification(
        "FaultDetectionA", extract_path="data", split="train"
    )
    X_test, y_test = load_classification(
        "FaultDetectionA", extract_path="data", split="test"
    )
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


data_fns: dict[
    str,
    Callable[
        [],
        tuple[
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
        ],
    ],
] = {"CBF": _get_data_CBF, "FaultDetectionA": _get_data_FaultDetectionA}
