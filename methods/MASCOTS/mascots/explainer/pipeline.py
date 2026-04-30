from typing import Any

from fast_borf.pipeline.reshaper import ReshapeTo2D
from fast_borf.pipeline.to_scipy import ToScipySparse
from fast_borf.pipeline.zero_columns_remover import ZeroColumnsRemover
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline


def get_generic_pipeline(borf: FeatureUnion, on_top_model: Any) -> Pipeline:
    return make_pipeline(
        borf,
        on_top_model,
    )


def get_default_borf_args() -> dict[str, Any]:
    return {
        "min_window_to_signal_std_ratio": 0,
        "pipeline_objects": [
            (ReshapeTo2D, dict(keep_unraveled_index=True)),
            (ZeroColumnsRemover, dict(axis=0)),
            (ToScipySparse, dict()),
        ],
        "n_jobs": -1,
        "n_jobs_numba": -1,
    }


def get_borf_config(data_shape: tuple[int, ...]) -> list[dict[str, Any]]:
    n_samples, n_feats, n_timestamps = data_shape
    word_sizes = []
    for i in range(3, n_timestamps):
        if 2**i > n_timestamps:
            break
        word_sizes.append(2**i)

    config = []
    for word_size in word_sizes:
        n_symbols = [4]
        for i in range(len(n_symbols)):
            if n_symbols[i] > word_size:
                break
            # stride = word_size // n_symbols[i]
            config.append(
                {
                    "window_size": word_size,
                    "stride": 1,
                    "dilation": 1,
                    "word_length": n_symbols[i],
                    "alphabet_size": 3,
                }
            )

    return config
