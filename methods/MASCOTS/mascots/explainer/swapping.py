import numpy as np
from numpy.typing import NDArray
from scipy import stats


def scalar_swap(
    X_sub: NDArray[np.float64],
    word: NDArray[np.int64],
    alphabet_size: int,
    symbol_size: int,
    quantille_bound: float = 2.0,
) -> NDArray[np.float64]:

    current_word = retreive_word(X_sub, alphabet_size, symbol_size)

    X_sub_copy = X_sub.copy()
    mean_org, std_org = X_sub_copy.mean(), X_sub_copy.std()

    X_sub_norm = (X_sub_copy - mean_org) / std_org

    qs = get_qunatiles(alphabet_size, quantille_bound)

    requested_positions = np.vstack([qs[word], qs[word + 1]]).mean(axis=0)

    diffs = requested_positions.reshape(-1) - X_sub_norm.reshape(
        -1, symbol_size
    ).mean(axis=1)
    # diffs = diffs * (current_word != word).astype(int)

    X_sub_copy += ((np.repeat(diffs, symbol_size) * std_org) + mean_org) * (
        np.repeat(current_word != word, symbol_size)
    ).astype(int)

    return X_sub_copy


def gaussian_swap(
    X_sub: NDArray[np.float64],
    word: NDArray[np.int64],
    alphabet_size: int,
    symbol_size: int,
    quantille_bound: float = 2.0,
    train: bool = True,
) -> NDArray[np.float64]:
    import torch
    from mascots.explainer.gp import ExactGPModel

    X_sub_copy = X_sub.copy()
    mean_org, std_org = X_sub_copy.mean(), X_sub_copy.std()

    X_sub_norm = (X_sub_copy - mean_org) / std_org

    qs = get_qunatiles(alphabet_size, quantille_bound)

    word_org = retreive_word(X_sub_norm, alphabet_size, symbol_size)

    means_org = np.array(
        [
            X_sub_norm[symbol_size * i : symbol_size * (i + 1)].mean()
            for i in range(len(word))
        ]
    )

    requested_positions = np.array(
        [
            (
                (qs[word[i]] + qs[word[i] + 1]) / 2
                if word[i] != word_org[i]
                else None
            )
            for i in range(len(word))
        ]
    )

    shifts = np.array(
        [
            (
                requested_positions[i] - means_org[i]
                if requested_positions[i]
                else 0
            )
            for i in range(len(word))
        ]
    )

    for i in range(len(word)):
        X_sub_norm[symbol_size * i : symbol_size * (i + 1)] += shifts[i]

    gp = ExactGPModel(
        torch.arange(len(word) * symbol_size), torch.Tensor(X_sub_norm)
    )
    if train == True:
        gp.fit(torch.arange(len(word) * symbol_size), torch.Tensor(X_sub_norm))
    gp.eval()
    gp.likelihood.eval()

    for i in range(len(word)):
        if word[i] != word_org[i]:
            X_sub_norm[symbol_size * i : symbol_size * (i + 1)] = (
                gp(torch.arange(symbol_size * i, symbol_size * (i + 1)))
                .mean.detach()
                .numpy()
            )

    return X_sub_norm * std_org + mean_org


def retreive_word(
    X_sub: NDArray[np.float64],
    alphabet_size: int,
    symbol_size: int,
) -> NDArray[np.int64]:
    n_symbols = X_sub.shape[0] // symbol_size

    current_word = []

    qs = get_qunatiles(alphabet_size)

    X_sub = X_sub.copy()
    X_sub = (X_sub - X_sub.mean()) / X_sub.std()

    for i in range(n_symbols):
        symbol = np.digitize(
            X_sub[i * symbol_size : symbol_size * (i + 1)].mean(),
            qs,
            right=False,
        )
        current_word.append(symbol)

    return np.array(current_word) - 1


def get_qunatiles(
    n_bins: int, bound_scalar: float | None = None
) -> NDArray[np.float64]:
    qs = stats.norm.ppf(np.linspace(0, 1, num=n_bins + 1))

    if bound_scalar is not None:
        qs = np.clip(
            qs, a_min=qs[1] * bound_scalar, a_max=qs[-2] * bound_scalar
        )

    return qs
