import os
import pickle as pkl
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from mascots.explainer.borf import BorfExplainer
from mascots.metrics.diversity import DiversityEvaluator
from mascots.metrics.plots import (
    plot_cls_success_rate,
    plot_counterfactuals,
    plot_metrics,
)
from mascots.metrics.sparsity import SparsityEvaluator
from mascots.metrics.utils import get_current_timestamp


class Evaluator:

    def __init__(
        self,
        borf_explainer_init_args: dict[str, Any],
        X: NDArray[np.float64] | None = None,
        y: NDArray[np.int64] | None = None,
        data_path: str | None = None,
        data_parse_fn: (
            Callable[[str], Tuple[NDArray[np.float64], NDArray[np.int64]]]
            | None
        ) = None,
    ) -> None:
        self.black_box_fn = borf_explainer_init_args["prediction_fn"]
        self.borf_explainer_init_args = borf_explainer_init_args
        self.X = X
        self.y = y
        self.data_path = data_path
        self.data_parse_fn = data_parse_fn

        self.out: dict[int, Any] = {}

    def build(
        self,
        borf_explainer_build_args: dict[str, Any],
    ) -> dict[str, Any]:
        if self.data_path and self.data_parse_fn:
            self.X, self.y = self.data_parse_fn(self.data_path)
        elif self.X is None or self.y is None:
            raise TypeError

        self.explainer = BorfExplainer(**self.borf_explainer_init_args)
        res = self.explainer.build(self.X, **borf_explainer_build_args)

        self.sparsity_eval = SparsityEvaluator(self.X, self.explainer)
        self.divesity_eval = DiversityEvaluator(self.X, self.explainer)

        self.build_res = {"borf_build_out": res, "shape": self.X.shape}

        return self.build_res

    def evaluate(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        y_target: NDArray[np.int64],
        n_counterfactuals: int,
        borf_explainer_counterfactual_args: dict[str, Any],
        seed: int | None = None,
        n_jobs: int | None = None,
    ) -> dict[int, Any]:

        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

        temp_dir = Path(
            f"temp_res/{formatted_datetime}-TASK-ID={os.getenv('TASK_ID', '0')}"
        )
        logger.info(f"{temp_dir=}")
        temp_dir.mkdir(exist_ok=True, parents=True)

        np.random.seed(seed)

        self.out = {}
        self._out_df = None

        if self.X is None or self.y is None:
            raise RuntimeError

        def __evaluate(idx: int, obs_y_tuple: tuple) -> None:
            obs, y_target = obs_y_tuple
            obs = obs[np.newaxis, :, :]

            counterfactuals, meta = self.explainer.counterfactual(
                obs,
                target_cls=y_target,
                returns_meta=True,
                n_restarts=n_counterfactuals,
                **borf_explainer_counterfactual_args,
            )

            if isinstance(counterfactuals, tuple):
                raise RuntimeError

            # sparsity_scores = self.sparsity_eval.evaluate(obs, counterfactuals)
            # diversity_scores = self.divesity_eval.evaluate(
            # obs, counterfactuals
            # )

            el = {
                # "sparsity": sparsity_scores,
                # "diversity": diversity_scores,
                "counterfactuals": counterfactuals,
                "observation": obs,
                "target_cls": y_target,
                "predicted_cls": self.black_box_fn(counterfactuals),
                "original_cls": y[idx],
                "meta": meta,
            }

            with open(f"{temp_dir}/out_{idx:03}.pkl", "wb") as f:
                pkl.dump(el, f)

        idxs = np.arange(X.shape[0])
        for idx in idxs:
            __evaluate(idx, (X[idx], y_target[idx]))

        for idx, path in enumerate(temp_dir.glob("*")):
            with open(path, "rb") as f:
                obj = pkl.load(f)
            self.out[idx] = obj

        return self.out

    def save_results(self, path: str | Path | None = None) -> None:
        self._init_df_if_not_exist()

        if path is None:
            path = f"results/{get_current_timestamp()}"

        self._save_plots(Path(path) / "plots")
        self._save_assets(Path(path) / "assets")

    def _save_plots(self, path: Path) -> None:
        df = self._out_df
        out = self.out

        path.mkdir(parents=True, exist_ok=True)
        (path / "counterfactuals").mkdir(parents=False, exist_ok=True)

        for idx, val in enumerate(out.values()):
            fig = plot_counterfactuals(
                val["observation"],
                val["counterfactuals"],
                val["target_cls"],
                val["predicted_cls"],
            )
            fig.savefig((path / "counterfactuals" / f"{idx:03}.png"))

        # fig = plot_metrics(df)
        # fig.savefig((path / "metrics.png"))

        # fig = plot_cls_success_rate(df)
        # fig.savefig((path / "cls_success_rate.png"))

    def _save_assets(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        self.out_df.to_csv(path / "out.csv", index=False)

        with open(path / "build_res.pkl", "wb") as f:
            pkl.dump(self.build_res, f)

        with open(path / "out.pkl", "wb") as f:
            pkl.dump(self.out, f)

    def _init_df_if_not_exist(self) -> None:
        if self.out is None:
            raise AttributeError
        if self._out_df is None:
            self._results_to_df()

    def _results_to_df(self) -> pd.DataFrame:

        res: list[dict[str, Any]] = []
        for key in self.out.keys():
            entry = {}
            # for metric, val in self.out[key]["sparsity"].items():
            #     entry[f"sparsity-{metric}"] = val
            # for metric, val in self.out[key]["diversity"].items():
            #     entry[f"diversity-{metric}"] = val
            entry["target_cls"] = int(self.out[key]["target_cls"])
            entry["original_cls"] = int(self.out[key]["original_cls"])
            entry["success_rate"] = (
                (self.out[key]["predicted_cls"] == self.out[key]["target_cls"])
                .astype(int)
                .mean()
            )
            res.append(entry)

        df = pd.DataFrame(res)
        self._out_df = df
        return df

    @property
    def out_df(self) -> pd.DataFrame:
        self._init_df_if_not_exist()
        return self._out_df


class GroupEvaluator:

    def __init__(
        self, paths: list[str] | None = None, root_path: str | None = None
    ) -> None:
        if paths is not None:
            self.paths = list(map(lambda p: Path(p), paths))
        elif root_path is not None:
            self.paths = list(Path(root_path).glob("*"))
        else:
            raise RuntimeError

    def compare(self, out_path: str) -> None:

        out_ppath = Path(out_path)
        out_ppath.mkdir(parents=True, exist_ok=True)

        dfs = []
        for path in self.paths:
            datafile = list(path.rglob("out.csv"))[0]
            df = pd.read_csv(datafile)
            df["config"] = datafile.parent.parent.stem
            dfs.append(df)

        self.df = pd.concat(dfs)

        fig = plot_cls_success_rate(self.df)
        fig.savefig(out_ppath / "cls_success_rate.png")

        fig = plot_metrics(self.df)
        fig.savefig(out_ppath / "metrics.png")
