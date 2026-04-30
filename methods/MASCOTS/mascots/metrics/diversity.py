from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cosine

from mascots.explainer.borf import BorfExplainer
from mascots.metrics.utils import norm


class DiversityEvaluator:

    def __init__(self, X: NDArray[np.float64], borf_exp: BorfExplainer):
        """
        Args:
            X (NDArray[np.float64]): Full (training) dataset
                in shape (n_obs, n_features, n_timestamps). Used for normalization
            borf_exp (BorfExplainer): BoRF explainer
        """
        self.loc = X.mean(axis=(0, 2))
        self.scale = X.std(axis=(0, 2))
        self.borf_exp = borf_exp

    def _pariwise_distance(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        distance_fn: Callable[
            [NDArray[np.float64], NDArray[np.float64]], float
        ],
        normalize: bool,
        use_transformed_data: bool,
    ) -> float:

        diffs = []

        n_counterfactuals = counterfactuals.shape[0]

        if use_transformed_data:
            obs = self.borf_exp.borf.transform(obs).toarray()
            counterfactuals = self.borf_exp.borf.transform(
                counterfactuals
            ).toarray()

        if normalize and not use_transformed_data:
            obs = norm(obs, self.loc, self.scale)
            counterfactuals = norm(counterfactuals, self.loc, self.scale)
        counterfactuals_diff: NDArray[np.float64] = counterfactuals - obs

        n_counterfactuals = counterfactuals_diff.shape[0]
        idx_pairs = np.array(np.tril_indices(n_counterfactuals, -1)).T
        for idx1, idx2 in idx_pairs:
            diff = distance_fn(
                counterfactuals_diff[idx1], counterfactuals_diff[idx2]
            )
            diffs.append(diff)

        return float(np.mean(diffs))

    def pairwise_l2_distance(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        normalize: bool = True,
        use_transformed_data: bool = False,
    ) -> float:
        """
        Count average pairwise L2 distance of `obs - counterfactuals` vectors.

        Args:
            obs (NDArray[np.float64]): explained observation
            counterfactuals (NDArray[np.float64]): set of counterfactuals
            normalize (bool, optional): If true, normalized (on dataset level) vectors will be used instead.
                Defaults to True.
            used_transformed_data (bool, optional): If true, use BoRF features instead of original data.
                Defaults to False.

        Returns:
            float: average pairwise L2 distance
        """

        return self._pariwise_distance(
            obs, counterfactuals, self._l2, normalize, use_transformed_data
        )

    def pairwise_cosine_distance(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        normalize: bool = True,
        use_transformed_data: bool = False,
    ) -> float:
        """
        Count average pairwise cosine similarity
        distance of `obs - counterfactuals` vectors.

        Args:
            obs (NDArray[np.float64]): explained observation
            counterfactuals (NDArray[np.float64]): set of counterfactuals
            normalize (bool, optional): If true, normalized (on dataset level) vectors will be used instead.
                Defaults to True.
            used_transformed_data (bool, optional): If true, use BoRF features instead of original data.
                Defaults to False.

        Returns:
            float: 1 - average pairwise cosine similarity distance
        """

        return self._pariwise_distance(
            obs, counterfactuals, cosine, normalize, use_transformed_data
        )

    def _l2(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(x - y, ord=2))

    def evaluate(
        self, obs: NDArray[np.float64], counterfactuals: NDArray[np.float64]
    ) -> dict[str, float]:
        """Calculate all defined metrics for single observaiton
        and possibly multiple counterfactuals. Observation in shape
        (1, n_features, n_timestamps), counterfactuals in shape
        (n_counterfactuals, n_features, n_timestamps).

        Args:
            obs (NDArray[np.float64]): explained observation
            counterfactuals (NDArray[np.float64]): set of counterfactuals

        Returns:
            dict[str, NDArray[np.float64 | np.int64]]: dictionary with all gathered metrics.
        """
        res = {
            r"L_2": self.pairwise_l2_distance(obs, counterfactuals),
            r"$L_2^{borf}$": self.pairwise_l2_distance(
                obs, counterfactuals, use_transformed_data=True
            ),
        }

        return res
