from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)


def get_eval_funcs() -> (
    dict[str, Callable[[NDArray[np.int64], NDArray[np.int64]], float]]
):
    average = "micro"
    return {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        f"f1 ({average})": make_scorer(f1_score, average="micro"),
        f"recall ({average})": make_scorer(recall_score, average="micro"),
        f"precision ({average})": make_scorer(
            precision_score, average="micro"
        ),
    }


def full_score(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64]
) -> dict[str, float]:
    if np.unique(y_true).shape[0] > 2:
        average = "micro"
    else:
        average = "binary"

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"f1 ({average})": f1_score(y_true, y_pred, average=average),
        f"recall ({average})": recall_score(y_true, y_pred, average=average),
        f"precision ({average})": precision_score(
            y_true, y_pred, average=average
        ),
    }
