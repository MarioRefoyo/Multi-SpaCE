import numpy as np
from numpy.typing import NDArray

from mascots.explainer.borf import BorfExplainer
from mascots.metrics.utils import norm


class SparsityEvaluator:

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

    def _distance(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        order: int | float,
    ) -> NDArray[np.float64]:

        obs = norm(obs, self.loc, self.scale)
        counterfactuals = norm(counterfactuals, self.loc, self.scale)
        return np.linalg.norm(obs - counterfactuals, ord=order, axis=(1, 2))

    def mse(
        self, obs: NDArray[np.float64], counterfactuals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._distance(obs, counterfactuals, order=2)

    def l_0(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        eps: float = 0.001,
    ) -> NDArray[np.float64]:
        obs = norm(obs, self.loc, self.scale)
        counterfactuals = norm(counterfactuals, self.loc, self.scale)
        return (
            (np.abs(obs - counterfactuals) > eps).astype(int).mean(axis=(1, 2))
        )

    def l_inf(
        self, obs: NDArray[np.float64], counterfactuals: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        obs = norm(obs, self.loc, self.scale)
        counterfactuals = norm(counterfactuals, self.loc, self.scale)

        return np.abs(obs - counterfactuals).max(axis=(1, 2))

    def n_borf_changes(
        self, obs: NDArray[np.float64], counterfactuals: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        obs_tr = self.borf_exp.borf.transform(obs).toarray()
        counterfactuals_tr = self.borf_exp.borf.transform(counterfactuals)
        n_counterfactuals = counterfactuals_tr.shape[0]

        return np.abs(
            np.repeat(obs_tr, repeats=n_counterfactuals, axis=0)
            - counterfactuals_tr
        ).mean(axis=(1))

    def evaluate(
        self,
        obs: NDArray[np.float64],
        counterfactuals: NDArray[np.float64],
        aggregate: bool = True,
    ) -> dict[str, NDArray[np.number]]:
        """Calculate all defined metrics for single observaiton
        and possibly multiple counterfactuals. Observation in shape
        (1, n_features, n_timestamps), counterfactuals in shape
        (n_counterfactuals, n_features, n_timestamps).

        Args:
            obs (NDArray[np.float64]): explained observation
            counterfactuals (NDArray[np.float64]): set of counterfactuals
            aggregate (bool, optional): If set as True, the method aggregates results
                along the counterfactuals. Otherwise, returned vectors have value for
                each counterfactual in order from `counterfactuals`. Defaults to True.

        Returns:
            dict[str, NDArray[np.float64 | np.int64]]: dictionary with all gathered metrics.
        """
        res = {
            r"$MSE$": self.mse(obs, counterfactuals),
            r"$L_0$": self.l_0(obs, counterfactuals),
            r"$L_{\infty}$": self.l_inf(obs, counterfactuals),
            r"$n~borf~changes$": self.n_borf_changes(obs, counterfactuals),
        }

        if aggregate:
            for key in res.keys():
                res[key] = res[key].mean()

        return res
