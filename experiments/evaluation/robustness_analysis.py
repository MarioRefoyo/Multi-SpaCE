"""Analyze Multi-SpaCE robustness from saved nearest-neighbor explanations.

Usage:
    Edit the configuration block below if needed, then run from the project root:
    python experiments/evaluation/robustness_analysis.py

The script reads the robustness generation outputs, computes L2 distance
summaries, and writes tables/plots without rerunning counterfactual generation.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "experiments/evaluation/.matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path("experiments/results")
MODEL_TO_EXPLAIN = "inceptiontime_noscaling"
EXPERIMENT_FAMILY = "multisubspace_v2_robustness"
DATASET = "SelfRegulationSCP1"  # Example: "ArticularyWordRecognition". Leave as None to analyze all discovered datasets.
EXPERIMENT_DIRS: list[Path] | None = None
OUTPUT_DIR = Path("experiments/evaluation/robustness_analysis_outputs")

EPSILON = 1e-8
SAVE_PDF = True
PLOT_DPI = 300
MAKE_PER_DATASET_SCATTERS = True


TABLES_DIRNAME = "tables"
FIGURES_DIRNAME = "figures"


def find_robustness_runs(
    results_root: Path = RESULTS_ROOT,
    model_to_explain: str = MODEL_TO_EXPLAIN,
    experiment_family: str = EXPERIMENT_FAMILY,
    dataset: str | None = DATASET,
) -> list[Path]:
    runs = []
    if dataset is None:
        dataset_dirs = sorted(results_root.iterdir()) if results_root.is_dir() else []
    else:
        dataset_dirs = [results_root / dataset]

    for dataset_dir in dataset_dirs:
        family_dir = dataset_dir / model_to_explain / experiment_family
        if not family_dir.is_dir():
            continue
        for run_dir in sorted(family_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "anchor_neighbor_relationships.csv").is_file() and (
                run_dir / "explained_instances.csv"
            ).is_file():
                runs.append(run_dir)
    return runs


def load_relationships(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "anchor_neighbor_relationships.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing relationship table: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "anchor_test_index", "neighbor_test_index", "neighbor_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    duplicated = df.duplicated(["dataset", "anchor_test_index", "neighbor_test_index", "neighbor_rank"]).sum()
    if duplicated:
        warnings.warn(f"{path} contains {duplicated} duplicated relationship rows; keeping first occurrence.")
        df = df.drop_duplicates(["dataset", "anchor_test_index", "neighbor_test_index", "neighbor_rank"])
    return df


def load_instances(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "explained_instances.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing instance table: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "test_index", "array_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    duplicated = df.duplicated(["dataset", "test_index"]).sum()
    if duplicated:
        warnings.warn(f"{path} contains {duplicated} duplicated instance rows; keeping first occurrence.")
        df = df.drop_duplicates(["dataset", "test_index"])
    return df


def resolve_saved_path(path_value: Any, run_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(run_dir / path)
        candidates.append(run_dir / path.name)
        candidates.append(run_dir / "arrays" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve saved array path: {path_value}")


def load_instance_arrays(instance_row: pd.Series, run_dir: Path) -> dict[str, np.ndarray]:
    array_path = resolve_saved_path(instance_row["array_path"], run_dir)
    data = np.load(array_path, allow_pickle=True)
    required = {"x", "x_cf"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{array_path} is missing arrays: {sorted(missing)}")
    return {"x": np.asarray(data["x"]), "x_cf": np.asarray(data["x_cf"])}


def compute_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a).reshape(-1) - np.asarray(b).reshape(-1), ord=2))


def build_pairwise_table(run_dir: Path, epsilon: float = EPSILON) -> tuple[pd.DataFrame, list[str]]:
    relationships = load_relationships(run_dir)
    instances = load_instances(run_dir)
    instance_lookup = {
        (str(row["dataset"]), int(row["test_index"])): row
        for _, row in instances.iterrows()
    }

    rows = []
    skipped = []
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for _, rel in tqdm(relationships.iterrows(), total=len(relationships), desc=f"Loading {run_dir.parent.parent.parent.name}"):
        dataset = str(rel["dataset"])
        anchor_idx = int(rel["anchor_test_index"])
        neighbor_idx = int(rel["neighbor_test_index"])
        anchor_key = (dataset, anchor_idx)
        neighbor_key = (dataset, neighbor_idx)
        try:
            if anchor_key not in cache:
                cache[anchor_key] = load_instance_arrays(instance_lookup[anchor_key], run_dir)
            if neighbor_key not in cache:
                cache[neighbor_key] = load_instance_arrays(instance_lookup[neighbor_key], run_dir)
            anchor_arrays = cache[anchor_key]
            neighbor_arrays = cache[neighbor_key]
            original_l2 = compute_l2(anchor_arrays["x"], neighbor_arrays["x"])
            cf_l2 = compute_l2(anchor_arrays["x_cf"], neighbor_arrays["x_cf"])
        except Exception as exc:
            skipped.append(f"{dataset} anchor={anchor_idx} neighbor={neighbor_idx}: {exc}")
            continue

        row = rel.to_dict()
        row.update(
            {
                "source_run_dir": str(run_dir),
                "original_l2_distance": original_l2,
                "cf_l2_distance": cf_l2,
                "cf_to_original_ratio": cf_l2 / (original_l2 + epsilon),
                "distance_difference": cf_l2 - original_l2,
                "relative_distance_difference": (cf_l2 - original_l2) / (original_l2 + epsilon),
                "cf_distance_leq_original_distance": bool(cf_l2 <= original_l2),
            }
        )
        rows.append(row)

    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        return pairwise_df, skipped
    pairwise_df = pairwise_df.replace([np.inf, -np.inf], np.nan)
    pairwise_df = pairwise_df.dropna(subset=["original_l2_distance", "cf_l2_distance"])
    return pairwise_df, skipped


def rank_preservation(group: pd.DataFrame) -> float:
    if group["neighbor_rank"].nunique() < 2 or group["cf_l2_distance"].nunique() < 2:
        return np.nan
    return float(group["neighbor_rank"].corr(group["cf_l2_distance"], method="spearman"))


def build_anchor_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()
    grouped = pairwise_df.groupby(["dataset", "anchor_test_index"], dropna=False)
    summary = grouped.agg(
        mean_original_l2_distance=("original_l2_distance", "mean"),
        mean_cf_l2_distance=("cf_l2_distance", "mean"),
        median_original_l2_distance=("original_l2_distance", "median"),
        median_cf_l2_distance=("cf_l2_distance", "median"),
        mean_cf_to_original_ratio=("cf_to_original_ratio", "mean"),
        median_cf_to_original_ratio=("cf_to_original_ratio", "median"),
        mean_distance_difference=("distance_difference", "mean"),
        median_distance_difference=("distance_difference", "median"),
        proportion_cf_distance_leq_original_distance=("cf_distance_leq_original_distance", "mean"),
        n_neighbors_available=("neighbor_test_index", "count"),
    ).reset_index()
    rank_rows = []
    for (dataset, anchor_idx), group in grouped:
        rank_rows.append(
            {
                "dataset": dataset,
                "anchor_test_index": anchor_idx,
                "cf_rank_preservation_spearman": rank_preservation(group),
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    return summary.merge(rank_df, on=["dataset", "anchor_test_index"], how="left")


def corr_or_nan(df: pd.DataFrame, method: str) -> float:
    if len(df) < 2:
        return np.nan
    if df["original_l2_distance"].nunique() < 2 or df["cf_l2_distance"].nunique() < 2:
        return np.nan
    return float(df["original_l2_distance"].corr(df["cf_l2_distance"], method=method))


def build_dataset_summary(pairwise_df: pd.DataFrame, anchor_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()
    rows = []
    for dataset, df in pairwise_df.groupby("dataset", dropna=False):
        anchors = anchor_df[anchor_df["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "neighbor_rank": "all",
                "n_anchors": int(anchors["anchor_test_index"].nunique()),
                "n_pairs": int(len(df)),
                "mean_original_l2_distance": float(df["original_l2_distance"].mean()),
                "mean_cf_l2_distance": float(df["cf_l2_distance"].mean()),
                "median_original_l2_distance": float(df["original_l2_distance"].median()),
                "median_cf_l2_distance": float(df["cf_l2_distance"].median()),
                "mean_cf_to_original_ratio": float(df["cf_to_original_ratio"].mean()),
                "median_cf_to_original_ratio": float(df["cf_to_original_ratio"].median()),
                "proportion_cf_distance_leq_original_distance": float(df["cf_distance_leq_original_distance"].mean()),
                "spearman_original_vs_cf_distance": corr_or_nan(df, "spearman"),
                "pearson_original_vs_cf_distance": corr_or_nan(df, "pearson"),
                "mean_cf_rank_preservation_spearman": float(anchors["cf_rank_preservation_spearman"].mean()),
                "median_cf_rank_preservation_spearman": float(anchors["cf_rank_preservation_spearman"].median()),
            }
        )
        for rank, rank_df in df.groupby("neighbor_rank", dropna=False):
            rows.append(
                {
                    "dataset": dataset,
                    "neighbor_rank": int(rank) if pd.notna(rank) else np.nan,
                    "n_anchors": int(rank_df["anchor_test_index"].nunique()),
                    "n_pairs": int(len(rank_df)),
                    "mean_original_l2_distance": float(rank_df["original_l2_distance"].mean()),
                    "mean_cf_l2_distance": float(rank_df["cf_l2_distance"].mean()),
                    "median_original_l2_distance": float(rank_df["original_l2_distance"].median()),
                    "median_cf_l2_distance": float(rank_df["cf_l2_distance"].median()),
                    "mean_cf_to_original_ratio": float(rank_df["cf_to_original_ratio"].mean()),
                    "median_cf_to_original_ratio": float(rank_df["cf_to_original_ratio"].median()),
                    "proportion_cf_distance_leq_original_distance": float(rank_df["cf_distance_leq_original_distance"].mean()),
                    "spearman_original_vs_cf_distance": corr_or_nan(rank_df, "spearman"),
                    "pearson_original_vs_cf_distance": corr_or_nan(rank_df, "pearson"),
                    "mean_cf_rank_preservation_spearman": np.nan,
                    "median_cf_rank_preservation_spearman": np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=PLOT_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def distance_long_df(pairwise_df: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    df = pairwise_df.copy()
    if normalized:
        medians = df.groupby("dataset")["original_l2_distance"].transform("median").replace(0, np.nan)
        df["original_l2_distance"] = df["original_l2_distance"] / medians
        df["cf_l2_distance"] = df["cf_l2_distance"] / medians
    return pd.concat(
        [
            df[["dataset", "neighbor_rank", "original_l2_distance"]].rename(
                columns={"original_l2_distance": "distance_value"}
            ).assign(distance_type="Original input distance"),
            df[["dataset", "neighbor_rank", "cf_l2_distance"]].rename(
                columns={"cf_l2_distance": "distance_value"}
            ).assign(distance_type="Counterfactual distance"),
        ],
        ignore_index=True,
    )


def make_distance_distribution_plots(pairwise_df: pd.DataFrame, figures_dir: Path) -> None:
    for normalized, stem in [
        (False, "distance_distribution_by_rank"),
        (True, "distance_distribution_by_rank_normalized"),
    ]:
        plot_df = distance_long_df(pairwise_df, normalized=normalized)
        g = sns.displot(
            data=plot_df,
            x="distance_value",
            hue="distance_type",
            col="neighbor_rank",
            kind="hist",
            kde=True,
            stat="density",
            common_norm=False,
            facet_kws={"sharex": False, "sharey": False},
            height=3.4,
            aspect=1.15,
        )
        g.set_axis_labels("Normalized L2 distance" if normalized else "L2 distance", "Density")
        g.set_titles("Neighbor rank {col_name}")
        g.fig.suptitle("Original vs Counterfactual Distances by Neighbor Rank", y=1.05)
        save_figure(g.fig, figures_dir, stem)


def add_diagonal(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    finite = pd.concat([x, y]).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    low = float(finite.min())
    high = float(finite.max())
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)


def make_scatter_plots(pairwise_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=pairwise_df,
        x="original_l2_distance",
        y="cf_l2_distance",
        hue="neighbor_rank",
        palette="viridis",
        alpha=0.75,
        ax=ax,
    )
    add_diagonal(ax, pairwise_df["original_l2_distance"], pairwise_df["cf_l2_distance"])
    ax.set_xlabel("Original L2 distance")
    ax.set_ylabel("Counterfactual L2 distance")
    ax.set_title("Original vs Counterfactual Distance")
    save_figure(fig, figures_dir, "scatter_original_vs_cf_distance")

    positive_df = pairwise_df[(pairwise_df["original_l2_distance"] > 0) & (pairwise_df["cf_l2_distance"] > 0)]
    if not positive_df.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(
            data=positive_df,
            x="original_l2_distance",
            y="cf_l2_distance",
            hue="neighbor_rank",
            palette="viridis",
            alpha=0.75,
            ax=ax,
        )
        add_diagonal(ax, positive_df["original_l2_distance"], positive_df["cf_l2_distance"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Original L2 distance")
        ax.set_ylabel("Counterfactual L2 distance")
        ax.set_title("Original vs Counterfactual Distance (log scale)")
        save_figure(fig, figures_dir, "scatter_original_vs_cf_distance_log")

    if MAKE_PER_DATASET_SCATTERS:
        per_dataset_dir = figures_dir / "per_dataset_scatter"
        for dataset, df in pairwise_df.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(
                data=df,
                x="original_l2_distance",
                y="cf_l2_distance",
                hue="neighbor_rank",
                palette="viridis",
                alpha=0.8,
                ax=ax,
            )
            add_diagonal(ax, df["original_l2_distance"], df["cf_l2_distance"])
            ax.set_xlabel("Original L2 distance")
            ax.set_ylabel("Counterfactual L2 distance")
            ax.set_title(str(dataset))
            save_figure(fig, per_dataset_dir, f"scatter_original_vs_cf_distance_{dataset}")


def make_ratio_plots(pairwise_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=pairwise_df, x="neighbor_rank", y="cf_to_original_ratio", ax=ax)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Neighbor rank")
    ax.set_ylabel("CF / original L2 distance ratio")
    ax.set_title("Counterfactual-to-Original Distance Ratio")
    save_figure(fig, figures_dir, "ratio_by_neighbor_rank")

    if pairwise_df["dataset"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(max(8, pairwise_df["dataset"].nunique() * 0.8), 5))
        sns.boxplot(data=pairwise_df, x="dataset", y="cf_to_original_ratio", hue="neighbor_rank", ax=ax)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("CF / original L2 distance ratio")
        ax.tick_params(axis="x", rotation=45)
        ax.set_title("Counterfactual-to-Original Distance Ratio by Dataset")
        save_figure(fig, figures_dir, "ratio_by_dataset_and_neighbor_rank")


def make_distance_difference_plots(pairwise_df: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=pairwise_df, x="neighbor_rank", y="distance_difference", ax=ax)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Neighbor rank")
    ax.set_ylabel("CF distance - original distance")
    ax.set_title("Distance Difference by Neighbor Rank")
    save_figure(fig, figures_dir, "distance_difference_by_neighbor_rank")


def make_rank_preservation_plot(anchor_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = anchor_df.dropna(subset=["cf_rank_preservation_spearman"]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_df["dataset"].nunique() > 1:
        sns.boxplot(data=plot_df, x="dataset", y="cf_rank_preservation_spearman", ax=ax)
        ax.tick_params(axis="x", rotation=45)
    else:
        sns.histplot(data=plot_df, x="cf_rank_preservation_spearman", kde=True, stat="density", ax=ax)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_ylabel("Spearman rank preservation")
    ax.set_title("Counterfactual Rank Preservation by Anchor")
    save_figure(fig, figures_dir, "rank_preservation_distribution")


def make_plots(pairwise_df: pd.DataFrame, anchor_df: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    make_distance_distribution_plots(pairwise_df, figures_dir)
    make_scatter_plots(pairwise_df, figures_dir)
    make_ratio_plots(pairwise_df, figures_dir)
    make_distance_difference_plots(pairwise_df, figures_dir)
    make_rank_preservation_plot(anchor_df, figures_dir)


def write_config(
    output_dir: Path,
    run_dirs: list[Path],
    pairwise_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    skipped: list[str],
) -> None:
    config = {
        "input_run_dirs": [str(path) for path in run_dirs],
        "dataset_filter": DATASET,
        "output_folder": str(output_dir),
        "epsilon": EPSILON,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets_analyzed": sorted(pairwise_df["dataset"].dropna().unique().tolist()) if not pairwise_df.empty else [],
        "n_anchors_loaded": int(anchor_df[["dataset", "anchor_test_index"]].drop_duplicates().shape[0]) if not anchor_df.empty else 0,
        "n_pairs_loaded": int(len(pairwise_df)),
        "n_pairs_skipped": int(len(skipped)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    if skipped:
        (output_dir / "skipped_pairs.txt").write_text("\n".join(skipped) + "\n")


def main() -> None:
    output_dir = OUTPUT_DIR
    tables_dir = output_dir / TABLES_DIRNAME
    figures_dir = output_dir / FIGURES_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = EXPERIMENT_DIRS if EXPERIMENT_DIRS is not None else find_robustness_runs(dataset=DATASET)
    run_dirs = [Path(path) for path in run_dirs]
    if not run_dirs:
        dataset_msg = "all datasets" if DATASET is None else f"dataset={DATASET}"
        raise FileNotFoundError(
            f"No robustness runs found for {dataset_msg}. Expected folders containing "
            "anchor_neighbor_relationships.csv and explained_instances.csv under "
            f"{RESULTS_ROOT}/*/{MODEL_TO_EXPLAIN}/{EXPERIMENT_FAMILY}/."
        )

    print(f"Found {len(run_dirs)} robustness run(s).")
    pairwise_frames = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        print("Loading:", run_dir)
        pairwise_df, skipped_run = build_pairwise_table(run_dir, epsilon=EPSILON)
        if not pairwise_df.empty:
            pairwise_frames.append(pairwise_df)
        skipped.extend(skipped_run)

    if not pairwise_frames:
        raise ValueError("No valid anchor-neighbor pairs could be loaded.")

    pairwise_df = pd.concat(pairwise_frames, ignore_index=True)
    anchor_df = build_anchor_summary(pairwise_df)
    dataset_df = build_dataset_summary(pairwise_df, anchor_df)

    write_dataframe(pairwise_df, tables_dir / "pairwise_robustness_distances.csv")
    write_dataframe(anchor_df, tables_dir / "anchor_level_robustness_summary.csv")
    write_dataframe(dataset_df, tables_dir / "dataset_level_robustness_summary.csv")
    write_config(output_dir, run_dirs, pairwise_df, anchor_df, skipped)

    make_plots(pairwise_df, anchor_df, figures_dir)

    print("Outputs written to:", output_dir)
    print(f"Pairs loaded: {len(pairwise_df)}")
    print(f"Anchors loaded: {anchor_df[['dataset', 'anchor_test_index']].drop_duplicates().shape[0]}")
    if skipped:
        print(f"Skipped pairs: {len(skipped)}. See {output_dir / 'skipped_pairs.txt'}")


if __name__ == "__main__":
    main()
