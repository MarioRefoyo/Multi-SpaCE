import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .counterfactual_common import CounterfactualMethod


MASCOTS_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "MASCOTS")
if MASCOTS_SOURCE_PATH not in sys.path:
    sys.path.insert(0, MASCOTS_SOURCE_PATH)

from mascots.explainer.borf import BorfExplainer
from mascots.explainer.pipeline import get_borf_config


def get_paper_borf_config(data_shape):
    _, _, n_timestamps = data_shape
    config = []
    window_size = 8
    while window_size <= n_timestamps:
        for word_length in (2, 4):
            if word_length <= window_size:
                config.append(
                    {
                        "window_size": window_size,
                        "stride": window_size // word_length,
                        "dilation": 1,
                        "word_length": word_length,
                        "alphabet_size": 3,
                    }
                )
        window_size *= 2
    return config


class MASCOTSCF(CounterfactualMethod):
    def __init__(
        self,
        model_wrapper,
        X_train,
        y_train=None,
        borf_config="auto",
        borf_args=None,
        build_args=None,
        counterfactual_args=None,
        train_subset=None,
        seed=None,
        build_cache_dir=None,
        restore_build=False,
        save_build=False,
        force_rebuild=False,
        warmup_numba=True,
    ):
        super().__init__(model_wrapper)
        self.seed = seed
        self.y_train = y_train
        self.counterfactual_args = counterfactual_args or {}
        self.build_cache_dir = build_cache_dir
        self.build_cache_path = (
            os.path.join(build_cache_dir, "mascots_build.pickle")
            if build_cache_dir is not None
            else None
        )

        X_borf = self._to_borf_batch(X_train)
        if train_subset is not None and len(X_borf) > train_subset:
            rng = np.random.default_rng(seed)
            subset_idx = np.sort(rng.choice(len(X_borf), train_subset, replace=False))
            X_borf = X_borf[subset_idx]

        if borf_config == "auto":
            borf_config = get_borf_config(X_borf.shape)
        elif borf_config == "paper":
            borf_config = get_paper_borf_config(X_borf.shape)
        elif borf_config == "original":
            borf_config = os.path.join(
                MASCOTS_SOURCE_PATH,
                "config",
                "126_borf_full.json",
            )

        self.explainer = BorfExplainer(
            prediction_fn=self._predict_cls_borf,
            prediction_fn_proba=self._predict_proba_borf,
            borf_config=borf_config,
            borf_args=borf_args or {},
        )

        build_args = dict(build_args or {})
        surrogate = build_args.pop("surrogate", "torch_mlp")
        if surrogate == "torch_mlp":
            on_top_model = "torch_mlp"
        elif surrogate == "random_forest":
            on_top_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
            )
        else:
            on_top_model = build_args.pop("on_top_model", None)
            if on_top_model is None:
                raise ValueError(
                    "Unsupported surrogate. Use 'torch_mlp', 'random_forest', "
                    "or pass an explicit 'on_top_model'."
                )

        attribution_args = build_args.pop(
            "attribution_args",
            {"mode": "deep", "scope": "local"} if surrogate == "torch_mlp" else {"mode": "tree", "scope": "local"},
        )
        attribution_name = build_args.pop("attribution_name", "shap")
        n_folds = build_args.pop("n_folds", 3)
        build_seed = build_args.pop("seed", seed if seed is not None else 42)

        if (
            restore_build
            and not force_rebuild
            and self.build_cache_path is not None
            and os.path.exists(self.build_cache_path)
        ):
            print(f"Loading cached MASCOTS build from {self.build_cache_path}")
            self.build_results = self._load_build(self.build_cache_path)
        else:
            self.build_results = self.explainer.build(
                X_borf,
                on_top_model=on_top_model,
                attribution_name=attribution_name,
                attribution_args=attribution_args,
                n_folds=n_folds,
                seed=build_seed,
            )

            if save_build and self.build_cache_path is not None:
                print(f"Saving MASCOTS build to {self.build_cache_path}")
                self._save_build(self.build_cache_path)

        if build_args:
            raise ValueError(f"Unsupported MASCOTS build args: {sorted(build_args)}")

        if warmup_numba:
            self._warmup_runtime(X_borf)

    def _save_build(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "borf": self.explainer.borf,
            "borf_pipeline": self.explainer.borf_pipeline,
            "attribution_method": self.explainer.attribution_method,
            "mapper": self.explainer.mapper,
            "mapper_info": self.explainer.mapper_info,
            "build_results": self.build_results,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)

    def _load_build(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.explainer.borf = payload["borf"]
        self.explainer.borf_pipeline = payload["borf_pipeline"]
        self.explainer.attribution_method = payload["attribution_method"]
        self.explainer.mapper = payload["mapper"]
        self.explainer.mapper_info = payload["mapper_info"]
        # SHAP DeepExplainer keeps module-level torch state that is not reliably
        # restored by pickle on Windows. Rebuild the explainer object over the
        # cached surrogate/background without retraining BoRF or the surrogate.
        self.explainer.attribution_method.build()
        return payload.get("build_results", {})

    def _warmup_runtime(self, X_borf):
        if len(X_borf) == 0:
            return

        warmup_sample = np.ascontiguousarray(X_borf[:1])
        print("[MASCOTS] Warming up prediction and BoRF kernels...", flush=True)

        try:
            # Warm the backend model path on a single sample.
            self._predict_proba_borf(warmup_sample)
            # Warm the BoRF/Numba transform path for the exact input rank used later.
            self.explainer.borf.transform(warmup_sample)
        except Exception as exc:
            print(f"[MASCOTS] Warmup skipped: {exc}", flush=True)

    def _to_borf_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        if X.ndim != 3:
            raise ValueError(f"Expected time series with shape (T, C) or (B, T, C), got {X.shape}")
        return np.swapaxes(X, 1, 2)

    @staticmethod
    def _to_tf_batch(X_borf):
        return np.swapaxes(np.asarray(X_borf), 1, 2)

    def _predict_proba_borf(self, X_borf):
        return self.model_wrapper.predict(self._to_tf_batch(X_borf))

    def _predict_cls_borf(self, X_borf):
        return np.argmax(self._predict_proba_borf(X_borf), axis=1)

    def _select_target(self, X_borf, desired_target):
        if desired_target is not None:
            return int(desired_target)
        proba = self._predict_proba_borf(X_borf)[0]
        return int(np.argsort(-proba)[1])

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        X_borf = self._to_borf_batch(x_orig)
        target = self._select_target(X_borf, desired_target)

        cf_args = {
            "swap_method": "scalar",
            "max_borf_changes": 20,
            "select_top_k": 5,
            "C": 0.5,
            "n_restarts": 1,
            "returns_meta": True,
            "seed": self.seed,
        }
        cf_args.update(self.counterfactual_args)

        cfs_borf, meta = self.explainer.counterfactual(
            X_borf,
            target_cls=target,
            **cf_args,
        )
        cfs_tf = self._to_tf_batch(cfs_borf)
        pred_proba = self.model_wrapper.predict(cfs_tf)
        pred_labels = np.argmax(pred_proba, axis=1)

        valid_idx = np.where(pred_labels == target)[0]
        if len(valid_idx) > 0:
            selected_idx = int(valid_idx[0])
        else:
            selected_idx = int(np.argmax(pred_proba[:, target]))

        return {
            "cf": np.expand_dims(cfs_tf[selected_idx], axis=0),
            "target": target,
            "meta": meta,
            "all_cfs": cfs_tf,
            "all_cf_pred_proba": pred_proba,
            "selected_idx": selected_idx,
            "build_results": self.build_results,
        }
