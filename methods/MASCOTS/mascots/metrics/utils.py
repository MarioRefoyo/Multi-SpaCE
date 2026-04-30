from datetime import datetime

import numpy as np
from numpy.typing import NDArray


def norm(
    x: NDArray[np.float64],
    loc: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:

    return (x - loc[np.newaxis, :, np.newaxis]) / scale[
        np.newaxis, :, np.newaxis
    ]


def get_current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
