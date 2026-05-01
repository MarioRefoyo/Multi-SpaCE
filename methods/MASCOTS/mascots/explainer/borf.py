import json
from typing import Any, Callable, Literal

import numpy as np
from fast_borf.classes.bag_of_receptive_fields_sax.borf_multi import (
    BorfPipelineBuilder,
)
from fast_borf.xai.mapping import BagOfReceptiveFields
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mascots.attributions.attributions import attribution_mapping
from mascots.explainer.pipeline import get_default_borf_args, get_generic_pipeline
from mascots.explainer.swapping import scalar_swap


class BorfExplainer:

    def __init__(
        self,
        prediction_fn: Callable[[Any], NDArray[np.int64]],
        prediction_fn_proba: Callable[[Any], NDArray[np.float64]],
        borf_config: dict[str, Any] | str = "config/126_borf_full.json",
        borf_args: dict[str, Any] = {},
    ) -> None:
        """
        Args:
            prediction_fn (Callable[[Any], NDArray[np.float64]]): Prediction function that accepts
                numeric vector/matrix and returns the predicted labels. TODO: implement handling proba
                responses.
            borf_config (dict[str, Any] | str, optional): Configuration of the BoRF. Might be a dictionary
                or a path file to .json config. Defaults to "config/126_borf_full.json".
            borf_args (dict[str, Any], optional): Additional parameters pf the BoRF. If the parameter is not set,
                defaults will be used. Defaults to {}.
        """
        self.prediction_fn = prediction_fn
        self.prediction_fn_proba = prediction_fn_proba
        self.borf_config = self._parse_borf_config(borf_config)
        borf_args = self._parse_borf_args(borf_args)

        logger.info(f"Borf #configs: {len(self.borf_config)}")
        logger.info(f"Borf #configs: {self.borf_config}")
        # raise Exception

        self.borf_builder = BorfPipelineBuilder(
            configs=self.borf_config, **borf_args
        )

    def build(
        self,
        X: NDArray[np.float64],
        on_top_model: Pipeline,
        attribution_name: str = "shap",
        attribution_args: dict[str, Any] = {},
        n_folds: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Train BoRF, builds attention objects, and create some
        meta-objects for further operaitons.

        Args:
            X (NDArray[np.float64]): Matrix of observations
            on_top_model (Pipeline): any model that is compatible with `scikit-learn`
                `Pipeline` class (has `train` and `fit` funcs).
            attribution_name (str, optional): Name of the attribution method.
                Possible methods are listed in `borf.attributions.attributions.attribution_mapping`.
                Defaults to "shap".
            attribution_args (dict[str, Any], optional): Arguments to the `__init__` of the
                attribution methods. Defaults to {}.
            n_folds (int, optional): Number of folds in CV in the evaluation phase. Defaults to 3.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            dict[str, Any]: Evaluation results (metrics, times, etc.).
        """

        print("[MASCOTS] Building BoRF pipeline...", flush=True)
        self.borf = self.borf_builder.build(X)

        print("[MASCOTS] Fitting and transforming BoRF features...", flush=True)
        X_transformed = self.borf.fit_transform(X)
        print(f"[MASCOTS] BoRF feature matrix shape: {X_transformed.shape}", flush=True)

        print("[MASCOTS] Querying black-box model for surrogate targets...", flush=True)
        y = self.prediction_fn_proba(X)

        if on_top_model == "torch_mlp":
            from bin.utils.models import MLP

            on_top_model = MLP(
                input_size=X_transformed.shape[1],
                output_size=y.shape[1],
                hidden_size=256,
                n_layers=2,
                lr=1e-4,
            )

        self.borf_pipeline = get_generic_pipeline(self.borf, on_top_model)

        print("[MASCOTS] Training surrogate model...", flush=True)
        results = self._train_borf_pipeline(X_transformed, y, n_folds, seed)
        print(f"[MASCOTS] Surrogate build results: {results}", flush=True)

        print("[MASCOTS] Building attribution method...", flush=True)
        self._create_attribution_method(
            X_transformed.toarray(), attribution_name, attribution_args
        )

        print("[MASCOTS] Mapping BoRF features back to time-series positions...", flush=True)
        mapper, mapper_info = self._map_borf_features(X[[0]])

        self.mapper = mapper
        self.mapper_info = mapper_info

        return results

    def counterfactual(
        self,
        X_obs: NDArray[np.float64],
        target_cls: int,
        swap_method: Literal["scalar", "gaussian"] = "scalar",
        max_borf_changes: int = 100,
        min_word_len_change: int = -1,
        max_word_len_change: int = -1,
        min_symbol_size: int = -1,
        max_symbol_size: int = -1,
        allow_only_shapes: tuple[tuple[int], ...] = tuple(),
        allow_only_idx: tuple[int, ...] = tuple(),
        select_top_k: int = 5,
        C: float = 0.1,
        n_restarts: int = 5,
        returns_meta: bool = True,
        seed: int | None = None,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], dict[int, Any]]:
        """
        Generte the counterfactual(s) for a given observation.

        Args:
            X_obs (NDArray[np.float64]): Observation in shape (1, n_signals, n_timestamps)
            target_cls (int): Target class for the couterfactuals
            swap_method (Literal[&quot;scalar&quot;, &quot;gaussian&quot;], optional): Method of swapping.
                Defaults to "scalar".
            max_borf_changes (int, optional): Maximum number of the method
                iterations. Defaults to 100.
            min_word_len_change (int, optional): Minimal length for the single change.
                Bigger changes tend to be be more plausible but less effective. Defaults to -1.
            max_word_len_change (int, optional): Maximal length for the single change.
                If negative, this param is ignored. Defaults to -1.
            min_symbol_size (int, optional): Minimal length of the single symbol in the change.
                Bigger symbol tend to preserve the original shape better. If negative, this param is
                ignored. Defaults to -1.
            max_symbol_size (int, optional): Maximal length of the single symbol in the change. If negative,
                this param is ignored. Defaults to -1.
            allow_only_shapes (tuple[tuple[int], ...], optional): Sequence of the shapes to be removed.
                Useful if the counterfactual should have impact only on the specific subsequences of the
                original observation. If empty, this param is ignored. Defaults to tuple().
            allow_only_idx (tuple[int, ...], optional): Sequence of the signals (featureS) to be altered.
                If empty, this param is ignored. Defaults to tuple().
            select_top_k (int, optional): The number of the most promising swaps from which the actual swap
                is chosen. This parameter controls the diversity of the counterfactuals by introducing
                the randomness. Defaults to 5.
            C (float, optional): Penalty for the size of the change. The penalty is calculated by counting
                differences on the original and the new word sequence to be swapped. Bigger `C` promotes
                changes that are smaller. Defaults to 0.1.
            n_restarts (int, optional): Number of counterfactuals. Defaults to 5.
            returns_meta (bool, optional): If set to `True` the meta-data about the
                generating the counterfactuals is returned as the second output's element.
                Defaults to True.
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            NDArray[np.float64] | tuple[NDArray[np.float64], dict[int, Any]]: First element is always
                a collection of ounterfactuals in shape (n_counterfactuals, n_sginals, n_timestamps).
                All counterfactuals are returned, even if they fail to flip the prediction. If `return_meta` is
                set to `True`, the second element contains meta-data frpm the counterfactuals generation process.
        """

        np.random.seed(seed)
        assert X_obs.shape[0] == 1
        y = self.prediction_fn(X_obs)

        counterfactuals = []
        meta_changes: dict[int, Any] = {}

        for idx_r in range(n_restarts):
            new_X = X_obs.copy()
            new_X_transformed = self.borf.transform(new_X)
            meta_changes[idx_r] = {}
            failed_iter = 0
            for idx in range(1, max_borf_changes + 1):
                new_X_proposal, change_record = self._heuristic_swap_step(
                    new_X,
                    new_X_transformed,
                    y,
                    target_cls,
                    C,
                    min_word_len_change,
                    max_word_len_change,
                    min_symbol_size,
                    max_symbol_size,
                    allow_only_shapes,
                    allow_only_idx,
                    select_top_k,
                    swap_method,
                )

                assert y != target_cls

                if (
                self.prediction_fn_proba(new_X_proposal)[0, target_cls]
                <= self.prediction_fn_proba(new_X)[0, target_cls]
                ):
                    # print(f"Iteration {idx} skipped due to negative outcome")
                    failed_iter += 1
                    if select_top_k == 1:
                        break
                else:
                    new_X = new_X_proposal

                    new_X_transformed = self.borf.transform(new_X)

                    if returns_meta:
                        meta_changes[idx_r][idx] = change_record

                if self.prediction_fn(new_X) != y:
                    logger.info(
                        f"Success after {idx} iterations (inluding {failed_iter} failed)"
                    )
                    break

                elif idx == max_borf_changes - 1:
                    logger.warning(
                        f"Cnt not created after {idx} iterations (inluding {failed_iter} failed)"
                    )

            counterfactuals.append(new_X)

            meta_changes[idx_r]["effective_iter"] = idx - failed_iter

        if returns_meta:
            return np.vstack(counterfactuals), meta_changes
        else:
            return np.vstack(counterfactuals)

    def _create_attribution_method(
        self,
        X_transformed: NDArray[np.int64],
        attribution_name: str,
        attribution_args: dict[str, Any],
    ) -> None:
        attribution_method = attribution_mapping[attribution_name](
            self.borf_pipeline[-1], X_transformed, **attribution_args
        )
        self.attribution_method = attribution_method
        self.attribution_method.build()

    def _map_borf_features(
        self, X: NDArray[np.float64]
    ) -> tuple[BagOfReceptiveFields, dict[int, dict[str, int]]]:

        mapper = BagOfReceptiveFields(self.borf)
        mapper.build(np.array(X))
        logger.info("create inner representation")
        mapper.task_ = "classification"

        mapper_info: dict[int, dict[str, int]] = {}

        for idx in range(len(mapper.receptive_fields_)):
            n_symbols = mapper.receptive_fields_[idx].alignments[0].shape[1]
            alphabet_size = mapper.receptive_fields_[idx].alphabet_size
            symbol_size = mapper.receptive_fields_[idx].alignments[0].shape[2]
            signal_idx = mapper.receptive_fields_[idx].signal_idx
            word_length = symbol_size * n_symbols
            dilation = mapper.receptive_fields_[idx].dilation
            stride = mapper.receptive_fields_[idx].stride
            mapper_info[idx] = {
                "word_length": word_length,
                "alphabet_size": alphabet_size,
                "n_symbols": n_symbols,
                "symbol_size": symbol_size,
                "signal_idx": signal_idx,
                "dilation": dilation,
                "stride": stride,
            }

        return mapper, mapper_info

    def _train_borf_pipeline(
        self,
        X_transformed: NDArray[np.int64],
        y: NDArray[np.float64],
        n_folds: int,
        seed: int,
    ) -> dict[str, Any]:
        # folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # scores_cv = cross_validate(
        #     self.borf_pipeline[-1],
        #     X_transformed,
        #     y,
        #     cv=folds,
        #     n_jobs=-1,
        #     scoring=get_eval_funcs(),
        # )

        X_train, X_val, y_train, y_val = train_test_split(X_transformed, y)

        try:
            self.borf_pipeline[-1].fit(X_train, y_train, X_val, y_val)
        except TypeError:
            self.borf_pipeline[-1].fit(X_train, y_train)
        y_pred = self.borf_pipeline[-1].predict(X_transformed)

        # scores_full = full_score(y, y_pred)

        try:
            res = {
                "accuracy": (
                    np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)
                ).mean(),
                "cross-entropy": (y * np.log(y_pred + 1e-6)).mean(),
                "mse": ((y - y_pred) ** 2).mean(),
                "r2": r2_score(y, y_pred),
                # "cv_scores": scores_cv,
                # "all_data_scores": scores_full
            }
        except Exception as e:
            logger.error(f"Error during surrogate evaluation: {e}")
            res = {}

        return res

    def _heuristic_swap_step(
        self,
        X_org: NDArray[np.float64],
        X_transformed: csr_matrix,
        y: NDArray[np.int64],
        target_cls: int,
        C: float,
        min_word_len_change: int,
        max_word_len_change: int,
        min_symbol_size: int,
        max_symbol_size: int,
        allow_only_shapes: tuple[tuple[int], ...],
        allow_only_idx: tuple[int, ...],
        select_top_k: int,
        swap_method: Literal["scalar", "gaussian"],
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:

        self.min_word_len_change = min_word_len_change
        self.max_word_len_change = max_word_len_change
        self.min_symbol_size = min_symbol_size
        self.max_symbol_size = max_symbol_size
        self.allow_only_shapes = allow_only_shapes
        self.allow_only_idx = allow_only_idx

        meta: dict[str, list | NDArray[np.float64]] = {
            "removed_shapes_candidates": [],
            "added_shapes_candidates": [],
            "added_channel_candidates": [],
            "selected_swap": [],
            "expected_gains": [],
            "windows_size": [],
            "indicies": [],
        }

        self.mapper.build(X_org, y, self.prediction_fn(X_org))
        self.mapper.task_ = "classification"

        F = self.attribution_method.explain(X_transformed)
        importance = np.ravel(F[target_cls, :, :] - F[y[0], :, :])

        n_feat = importance.shape[0]

        existing_shapes_idx = np.arange(X_transformed.shape[1])[
            np.ravel((X_transformed[0, :] > 0).toarray())
        ]
        existing_shapes_idx_set = set(existing_shapes_idx.tolist())

        max_imp = float(importance.max() - importance.min())

        compatibility_groups_new: dict[tuple[int, ...], list[int]] = {}
        compatibility_groups_old: dict[tuple[int, ...], list[int]] = {}

        for idx_feat in range(n_feat):
            if self._check_if_word_compatible(idx_feat):
                key = tuple(
                    self.mapper_info[idx_feat][constraint_name]
                    for constraint_name in self.constraint_names
                )
                compatibility_groups_new.setdefault(key, []).append(idx_feat)

            if idx_feat in existing_shapes_idx_set:
                key = tuple(
                    self.mapper_info[idx_feat][constraint_name]
                    for constraint_name in self.constraint_names
                )
                compatibility_groups_old.setdefault(key, []).append(idx_feat)

        pair_scores = []
        pair_flat_idx = []
        idx_pairs = []

        # Preserve the original score exactly, but only for feasible pairs.
        for key, new_indices in compatibility_groups_new.items():
            old_indices = compatibility_groups_old.get(key)
            if old_indices is None:
                continue

            for new_idx in new_indices:
                new_importance = importance[new_idx]
                for old_idx in old_indices:
                    flat_idx = new_idx * n_feat + old_idx
                    if flat_idx % n_feat == 0:
                        continue

                    pairwise_gain = new_importance - importance[old_idx]
                    if pairwise_gain < 0:
                        pairwise_gain = 0.0
                    if max_imp > 0:
                        pairwise_gain /= max_imp

                    score = -pairwise_gain + C * self._word_diff_measure(
                        new_idx, old_idx
                    )
                    pair_scores.append(score)
                    pair_flat_idx.append(flat_idx)
                    idx_pairs.append((new_idx, old_idx))

        if pair_scores:
            pair_scores_arr = np.asarray(pair_scores, dtype=np.float64)
            pair_flat_idx_arr = np.asarray(pair_flat_idx, dtype=np.int64)
            # Match the original flattened pair enumeration on equal scores.
            imp_sorted = np.lexsort((pair_flat_idx_arr, pair_scores_arr))
            idx_pairs = [idx_pairs[i] for i in imp_sorted]
        else:
            idx_pairs = []

        # if np.all(importance <= 0):
        #     return X_org, {"skipped": "no feasilble pairs"}

        # importance_sorted_idx = np.argsort(-importance)
        # importance_sorted_idx = importance_sorted_idx[
        #     importance[importance_sorted_idx] >= 0
        # ]

        top_k_changes: list[NDArray[np.float64]] = []

        for new_idx, old_idx in idx_pairs:
            if len(top_k_changes) == select_top_k:
                break
            if not self._check_if_word_compatible(new_idx):
                continue
            if not self._check_if_words_compatible(new_idx, old_idx):
                continue
            if old_idx not in existing_shapes_idx:
                continue

            (
                alphabet_size,
                symbol_size,
                target_word,
                signal_idx,
            ) = self._get_word_info_from_idx(new_idx)

            n_old_seq = (
                self.mapper.receptive_fields_[old_idx].alignments[0].shape[0]
            )
            indices = (
                self.mapper.receptive_fields_[old_idx]
                .alignments[0][np.random.choice(n_old_seq)]
                .reshape(-1)
            )

            if swap_method == "scalar":

                proposal = scalar_swap(
                    X_org[0, signal_idx, indices],
                    target_word,
                    alphabet_size,
                    symbol_size,
                    quantille_bound=1,
                )

            else:
                from mascots.explainer.swapping import gaussian_swap

                proposal = gaussian_swap(
                    X_org[0, signal_idx, indices],
                    target_word,
                    alphabet_size,
                    symbol_size,
                )

            new_obs = X_org.copy()
            new_obs[0, signal_idx, indices] = proposal
            top_k_changes.append(new_obs)

            meta["added_shapes_candidates"].append(target_word)
            meta["removed_shapes_candidates"].append(
                self.mapper.receptive_fields_[old_idx].word_array
            )
            meta["added_channel_candidates"].append(signal_idx)
            meta["expected_gains"].append(
                importance[new_idx] - importance[old_idx]
            )
            meta["windows_size"].append(
                self.mapper_info[new_idx]["word_length"]
            )
            meta["indicies"].append(indices)

        if len(top_k_changes) == 0:
            meta["skipped"] = "no feasible swap candidates"
            meta["new_obs"] = X_org
            return X_org, meta

        returned_obs_idx = np.random.choice(len(top_k_changes))
        meta["selected_swap"].append(returned_obs_idx)
        meta["new_obs"] = top_k_changes[returned_obs_idx]

        return (top_k_changes[returned_obs_idx], meta)

    def _get_word_info_from_idx(self, word_idx: int) -> tuple[Any, ...]:
        alphabet_size = self.mapper_info[word_idx]["alphabet_size"]
        symbol_size = self.mapper_info[word_idx]["symbol_size"]
        word_array = self.mapper.receptive_fields_[word_idx].word_array
        signal_idx = self.mapper_info[word_idx]["signal_idx"]

        return (alphabet_size, symbol_size, word_array, signal_idx)

    @property
    def constraint_names(self) -> list[str]:
        return [
            "word_length",
            "alphabet_size",
            "n_symbols",
            "symbol_size",
            "signal_idx",
            "dilation",
        ]

    def _check_if_words_compatible(
        self, word_idx_1: int, word_idx_2: int
    ) -> bool:
        constraint_names = self.constraint_names
        for contraint_name in constraint_names:
            if (
                self.mapper_info[word_idx_1][contraint_name]
                != self.mapper_info[word_idx_2][contraint_name]
            ):
                return False

        return True

    def _check_if_word_compatible(
        self,
        word_idx: int,
    ) -> bool:

        if self.mapper_info[word_idx]["dilation"] != 1:
            return False

        if self.min_word_len_change > 0:
            if (
                self.mapper_info[word_idx]["word_length"]
                < self.min_word_len_change
            ):
                return False

        if self.max_word_len_change > 0:
            if (
                self.mapper_info[word_idx]["word_length"]
                > self.max_word_len_change
            ):
                return False

        if self.min_symbol_size > 0:
            if (
                self.mapper_info[word_idx]["symbol_size"]
                < self.min_symbol_size
            ):
                return False

        if self.max_symbol_size > 0:
            if (
                self.mapper_info[word_idx]["symbol_size"]
                > self.max_symbol_size
            ):
                return False

        if len(self.allow_only_shapes) != 0:
            is_contained = (
                (
                    np.stack(self.allow_only_shapes)
                    - self.mapper.receptive_fields_[word_idx].word_array
                ).sum(axis=0)
                == 0
            ).any()
            if not is_contained:
                return False

        if len(self.allow_only_idx) != 0:
            if (
                not self.mapper_info[word_idx]["signal_idx"]
                in self.allow_only_idx
            ):
                return False

        return True

    def _word_diff_measure(self, idx1: int, idx2: int) -> float:

        word1 = self.mapper.receptive_fields_[idx1].word_array
        word2 = self.mapper.receptive_fields_[idx2].word_array

        if word1.shape != word2.shape:
            return float("inf")

        return float(np.abs(word1 - word2).sum())

    def _parse_borf_config(
        self, borf_config: dict[str, Any] | str
    ) -> dict[str, Any]:
        if isinstance(borf_config, str):
            with open(borf_config, "r") as f:
                return json.load(f)
        else:
            return borf_config

    def _parse_borf_args(self, borf_args: dict[str, Any]) -> dict[str, Any]:
        default_borf_args = get_default_borf_args()
        for key, val in borf_args.items():
            default_borf_args[key] = val
        return default_borf_args
