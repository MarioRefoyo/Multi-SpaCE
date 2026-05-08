"""Run the Multi-SpaCE ECG morphology analysis from saved CF data.

Usage:
    Edit the global variables below, then run:
    python experiments/evaluation/run_ecg_morphology_analysis.py

The script intentionally keeps all analysis functions in
ecg_morphology_analysis_utils.py. This file loads data, adapts it, runs the
same notebook analysis, and writes tables/plots to disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COUNTERFACTUALS_PICKLE = Path(
    "experiments/results/ptbxl/inceptiontime_noscaling/multisubspace_v2_ptbxl/v2_f3f682b91a67a3c7bd162bc42643e1ae656743cf/counterfactuals.pickle"
)
OUTPUT_DIR = Path("experiments/evaluation/ecg_morphology_outputs")

# Optional side data. Leave as None when those arrays are already stored in the
# counterfactual pickle.
CONTEXT_NPZ = None
X_TEST_PATH = None
Y_TEST_PATH = None
Y_PRED_TEST_PATH = None
NUNS_PATH = None
SUBSET_IDX_PATH = None

# ECG reference windows. These are fixed windows centered around the reference R.
SAMPLING_RATE = 100
DELINEATION_CHANNEL_IDX = 1
REFERENCE_WINDOW_MODE = "union"  # "union", "orig", "nun", or "intersection"
NEUROKIT_METHOD = "dwt"
USE_NEUROKIT = False
STRICT_DELINEATION = False
# The 175-point PTB-XL beat windows are not centered at n_time//2=87.
# Empirical inspection showed the main R/QRS-like event around index 75-78.
# We therefore use 76 as the fixed reference index for the R-centered windows.
FIXED_R_INDEX = 76

# Adapter and filtering.
# Options: "NORM_TO_MI", "MI_TO_NORM", "BOTH", "NONE"
ANALYSIS_DIRECTION = "BOTH"
MASK_TOLERANCE = 1e-6
CLASS_NAMES = {0: "NORM", 1: "MI"}  # Set to None to avoid label mapping.
APPLY_NORM_MI_FILTER = True
STRICT_NORM_MI_FILTER = False
FORCE_TRANSPOSE = False
STRICT_EXTRACTION = True
REQUIRE_NUN = True

# Plotting.
MAKE_PLOTS = True
EXAMPLE_INDEX = 0
EXAMPLE_CHANNEL = 0
AVERAGE_WAVEFORM_CHANNELS = [1]  # Lead II/channel index 1.
DELINEATION_EXAMPLE_INDICES = None  # Example: [0, 3, 7]. None uses the first examples.
N_DELINEATION_EXAMPLES = 6

from experiments.evaluation.ecg_morphology_analysis_utils import (
    AUXILIARY_CHANNELS,
    AnalysisConfig,
    CLINICAL_LEAD_CHANNELS,
    NEUROKIT_AVAILABLE,
    NEUROKIT_IMPORT_ERROR,
    V2_V5_CHANNELS,
    adapt_explanation_object,
    analyze_all_pairs_global,
    compute_narrative_indicators,
    make_result_sentences,
    make_directional_result_sentences,
    mask_region_heatmap_table,
    mask_overlap_all,
    get_centered_windows,
    plot_average_waveforms,
    plot_changed_point_concentration,
    plot_delineation_examples,
    plot_global_delta_distributions,
    plot_mask_overlap_summary,
    plot_mask_region_heatmaps,
    plot_one_pair_with_windows,
    plot_per_channel_medians,
    summarize_by_channel,
    summarize_directional_morphology,
    summarize_global_morphology,
    summarize_mask_overlap,
    summarize_object,
)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_array(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if len(data.files) != 1:
            raise ValueError(f"{path} contains {data.files}. Use CONTEXT_NPZ for multi-array files.")
        return data[data.files[0]]
    raise ValueError(f"Unsupported array file extension for {path}. Use .npy or single-array .npz.")


def load_context_npz(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def attach_direction(df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "instance_id" not in df.columns or "direction" in df.columns:
        return df
    if "instance_id" not in metadata_df.columns or "direction" not in metadata_df.columns:
        return df
    direction_map = metadata_df[["instance_id", "direction"]].drop_duplicates("instance_id")
    return df.merge(direction_map, on="instance_id", how="left")


def write_output_note(output_dir: Path) -> None:
    text = (
        "ECG morphology analysis notes\n"
        "=============================\n\n"
        "- The ECG regions are fixed R-centered descriptive windows, not clinical fiducial annotations.\n"
        "- Channels 0-11 are interpreted as standard PTB-XL ECG leads I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.\n"
        "- Channels 12-14 are auxiliary transformed channels and are not interpreted as standard ECG leads.\n"
        "- The analysis quantifies model explanation behavior, not clinical diagnostic validity.\n"
    )
    (output_dir / "README.txt").write_text(text)


def write_direction_outputs(
    direction: str,
    output_dir: Path,
    morphology_df: pd.DataFrame,
    channel_summary: pd.DataFrame,
    mask_overlap_df: pd.DataFrame,
    mask_region_heatmap_df: pd.DataFrame,
    narrative_indicators: pd.DataFrame,
) -> None:
    direction_dir = output_dir / direction
    direction_dir.mkdir(parents=True, exist_ok=True)
    direction_morphology = morphology_df[morphology_df["direction"] == direction] if "direction" in morphology_df else morphology_df
    direction_channel_summary = channel_summary[channel_summary["direction"] == direction] if "direction" in channel_summary else channel_summary
    direction_mask_overlap = mask_overlap_df[mask_overlap_df["direction"] == direction] if "direction" in mask_overlap_df else mask_overlap_df
    direction_mask_region = mask_region_heatmap_df[mask_region_heatmap_df["direction"] == direction] if "direction" in mask_region_heatmap_df else mask_region_heatmap_df
    direction_mask_summary = summarize_mask_overlap(direction_mask_overlap)
    direction_summary = summarize_directional_morphology(direction_morphology, direction)
    direction_indicators = narrative_indicators[narrative_indicators["direction"] == direction] if "direction" in narrative_indicators else narrative_indicators
    direction_sentences = make_directional_result_sentences(direction_summary, direction, narrative_indicators)

    write_dataframe(direction_summary, direction_dir / "global_summary.csv")
    write_dataframe(direction_channel_summary, direction_dir / "channel_summary.csv")
    write_dataframe(direction_mask_summary, direction_dir / "mask_overlap_summary.csv")
    write_dataframe(direction_mask_region, direction_dir / "mask_region_heatmap_df.csv")
    write_dataframe(direction_indicators, direction_dir / "narrative_indicators.csv")
    (direction_dir / "result_sentences.txt").write_text("\n".join(direction_sentences) + "\n")


def make_direction_plots(
    direction: str,
    output_dir: Path,
    x_orig_all: np.ndarray,
    x_cf_all: np.ndarray,
    x_nun_all: np.ndarray | None,
    metadata_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    mask_overlap_df: pd.DataFrame,
    mask_region_heatmap_df: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    if "direction" not in metadata_df.columns:
        return
    row_mask = metadata_df["direction"].to_numpy() == direction
    if not row_mask.any():
        return
    plot_dir = output_dir / direction / "plots"
    direction_morphology = morphology_df[morphology_df["direction"] == direction] if "direction" in morphology_df else morphology_df
    direction_mask_summary = summarize_mask_overlap(
        mask_overlap_df[mask_overlap_df["direction"] == direction] if "direction" in mask_overlap_df else mask_overlap_df
    )
    direction_mask_region = (
        mask_region_heatmap_df[mask_region_heatmap_df["direction"] == direction]
        if "direction" in mask_region_heatmap_df
        else mask_region_heatmap_df
    )
    plot_global_delta_distributions(direction_morphology, output_path=plot_dir / "global_delta_distributions.png")
    plot_mask_overlap_summary(direction_mask_summary, output_path=plot_dir / "mask_overlap_summary.png")
    plot_changed_point_concentration(direction_mask_summary, output_path=plot_dir / "changed_point_concentration.png")
    plot_mask_region_heatmaps(direction_mask_region, output_path=plot_dir / "mask_region_heatmaps.png")
    for metric in ["delta_qrs_energy", "delta_t_area_abs", "qrs_best_lag_abs"]:
        if metric in direction_morphology:
            plot_per_channel_medians(direction_morphology, metric, output_path=plot_dir / f"per_channel_{metric}.png")
    plot_average_waveforms(
        x_orig_all[row_mask],
        x_cf_all[row_mask],
        channels=[ch for ch in CLINICAL_LEAD_CHANNELS if ch < x_orig_all.shape[2]],
        output_path=plot_dir / "average_waveforms_clinical_leads.png",
        config=config,
        first_label=f"{direction} originals",
        second_label=f"{direction} CFs",
    )
    plot_average_waveforms(
        x_orig_all[row_mask],
        x_cf_all[row_mask],
        channels=[ch for ch in V2_V5_CHANNELS if ch < x_orig_all.shape[2]],
        output_path=plot_dir / "average_waveforms_v2_v5.png",
        config=config,
        first_label=f"{direction} originals",
        second_label=f"{direction} CFs",
    )
    auxiliary_channels = [ch for ch in AUXILIARY_CHANNELS if ch < x_orig_all.shape[2]]
    if auxiliary_channels:
        plot_average_waveforms(
            x_orig_all[row_mask],
            x_cf_all[row_mask],
            channels=auxiliary_channels,
            output_path=plot_dir / "average_waveforms_auxiliary.png",
            config=config,
            first_label=f"{direction} originals",
            second_label=f"{direction} CFs",
        )


def main() -> None:
    output_dir = OUTPUT_DIR / ANALYSIS_DIRECTION.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not COUNTERFACTUALS_PICKLE.exists():
        raise FileNotFoundError(
            f"COUNTERFACTUALS_PICKLE does not exist: {COUNTERFACTUALS_PICKLE}. "
            "Edit the global configuration block at the top of this script."
        )

    print("Loading counterfactuals:", COUNTERFACTUALS_PICKLE)
    explanation_obj = load_pickle(COUNTERFACTUALS_PICKLE)
    print("Loaded object summary:", summarize_object(explanation_obj))

    context = load_context_npz(CONTEXT_NPZ)
    optional_arrays = {
        "X_test": load_array(X_TEST_PATH),
        "y_test": load_array(Y_TEST_PATH),
        "y_pred_test": load_array(Y_PRED_TEST_PATH),
        "nuns": load_array(NUNS_PATH),
        "subset_idx": load_array(SUBSET_IDX_PATH),
    }
    context.update({key: value for key, value in optional_arrays.items() if value is not None})
    if context:
        print("Context arrays:", {key: getattr(value, "shape", None) for key, value in context.items()})

    config = AnalysisConfig(
        delineation_channel_idx=DELINEATION_CHANNEL_IDX,
        sampling_rate=SAMPLING_RATE,
        use_neurokit_delineation=USE_NEUROKIT,
        strict_delineation=STRICT_DELINEATION,
        strict_extraction=STRICT_EXTRACTION,
        fixed_r_index=FIXED_R_INDEX,
        neurokit_method=NEUROKIT_METHOD,
        reference_window_mode=REFERENCE_WINDOW_MODE,
        force_transpose=FORCE_TRANSPOSE,
        mask_tolerance=MASK_TOLERANCE,
        analysis_direction=ANALYSIS_DIRECTION,
        apply_norm_mi_filter=APPLY_NORM_MI_FILTER,
        strict_norm_mi_filter=STRICT_NORM_MI_FILTER,
        class_names=CLASS_NAMES,
    )

    print("NeuroKit available:", NEUROKIT_AVAILABLE)
    if not NEUROKIT_AVAILABLE:
        print("NeuroKit import error:", repr(NEUROKIT_IMPORT_ERROR))
    if USE_NEUROKIT and not NEUROKIT_AVAILABLE:
        raise RuntimeError("USE_NEUROKIT=True, but NeuroKit2 is not available in this environment.")
    print("Delineation channel:", config.delineation_channel_idx)
    print("Fixed R/reference index:", config.fixed_r_index)
    print("Reference windows for length 175:", get_centered_windows(175, config=config))
    print("Reference window mode:", config.reference_window_mode)
    print("Analysis direction:", config.analysis_direction)

    adapted = adapt_explanation_object(
        explanation_obj,
        context=context,
        source_object=str(COUNTERFACTUALS_PICKLE),
        config=config,
    )

    print("Filter status:", adapted.filter_status)
    if adapted.skipped_rows:
        print(f"Skipped {len(adapted.skipped_rows)} instances. First skipped rows: {adapted.skipped_rows[:10]}")
    print("x_orig_all:", adapted.x_orig_all.shape)
    print("x_cf_all:", adapted.x_cf_all.shape)
    print("masks_all:", adapted.masks_all.shape)
    print("x_nun_all:", None if adapted.x_nun_all is None else adapted.x_nun_all.shape)
    print("metadata_df:", adapted.metadata_df.shape)
    if REQUIRE_NUN and adapted.x_nun_all is None:
        raise RuntimeError("REQUIRE_NUN=True, but not every selected pair has a NUN signal.")

    morphology_df = analyze_all_pairs_global(
        adapted.x_orig_all,
        adapted.x_cf_all,
        x_nun_all=adapted.x_nun_all,
        instance_ids=adapted.instance_ids,
        config=config,
    )
    morphology_df = attach_direction(morphology_df, adapted.metadata_df)
    global_summary = summarize_global_morphology(morphology_df)
    channel_summary = summarize_by_channel(morphology_df)
    mask_overlap_df = mask_overlap_all(
        adapted.x_orig_all,
        adapted.x_cf_all,
        x_nun_all=adapted.x_nun_all,
        masks_all=adapted.masks_all,
        tolerance=config.mask_tolerance,
        instance_ids=adapted.instance_ids,
        config=config,
    )
    mask_overlap_df = attach_direction(mask_overlap_df, adapted.metadata_df)
    mask_region_heatmap_df = mask_region_heatmap_table(
        adapted.masks_all,
        x_orig_all=adapted.x_orig_all,
        x_nun_all=adapted.x_nun_all,
        instance_ids=adapted.instance_ids,
        config=config,
    )
    mask_region_heatmap_df = attach_direction(mask_region_heatmap_df, adapted.metadata_df)
    mask_overlap_summary = summarize_mask_overlap(mask_overlap_df)
    narrative_indicators = compute_narrative_indicators(morphology_df, mask_overlap_df, adapted.metadata_df)
    if ANALYSIS_DIRECTION == "BOTH" and "direction" in morphology_df:
        result_sentences = []
        for direction in ["NORM_TO_MI", "MI_TO_NORM"]:
            if (morphology_df["direction"] == direction).any():
                direction_summary = summarize_directional_morphology(morphology_df, direction)
                result_sentences.extend(make_directional_result_sentences(direction_summary, direction, narrative_indicators))
    elif ANALYSIS_DIRECTION in {"NORM_TO_MI", "MI_TO_NORM"}:
        result_sentences = make_directional_result_sentences(global_summary, ANALYSIS_DIRECTION, narrative_indicators)
    else:
        result_sentences = make_result_sentences(global_summary)

    write_dataframe(adapted.metadata_df, output_dir / "metadata_df.csv")
    write_dataframe(morphology_df, output_dir / "morphology_df.csv")
    write_dataframe(global_summary, output_dir / "global_summary.csv")
    write_dataframe(channel_summary, output_dir / "channel_summary.csv")
    write_dataframe(mask_overlap_df, output_dir / "mask_overlap_df.csv")
    write_dataframe(mask_overlap_summary, output_dir / "mask_overlap_summary.csv")
    write_dataframe(mask_region_heatmap_df, output_dir / "mask_region_heatmap_df.csv")
    write_dataframe(narrative_indicators, output_dir / "narrative_indicators.csv")
    (output_dir / "result_sentences.txt").write_text("\n".join(result_sentences) + "\n")
    write_output_note(output_dir)
    if ANALYSIS_DIRECTION == "BOTH":
        for direction in ["NORM_TO_MI", "MI_TO_NORM"]:
            if "direction" in morphology_df and (morphology_df["direction"] == direction).any():
                write_direction_outputs(
                    direction,
                    output_dir,
                    morphology_df,
                    channel_summary,
                    mask_overlap_df,
                    mask_region_heatmap_df,
                    narrative_indicators,
                )
    np.savez_compressed(
        output_dir / "adapted_arrays.npz",
        x_orig_all=adapted.x_orig_all,
        x_cf_all=adapted.x_cf_all,
        masks_all=adapted.masks_all,
        x_nun_all=np.asarray([]) if adapted.x_nun_all is None else adapted.x_nun_all,
        instance_ids=adapted.instance_ids,
        candidate_ids=adapted.candidate_ids,
    )

    print("Global morphology summary:")
    print(global_summary.to_string(index=False))
    print("Mask overlap summary:")
    print(mask_overlap_summary.to_string(index=False))
    print("Narrative indicators:")
    print(narrative_indicators.to_string(index=False))
    print("Manuscript-style result sentences:")
    for sentence in result_sentences:
        print(sentence)

    if MAKE_PLOTS:
        plot_dir = output_dir / "plots"
        plot_delineation_examples(
            adapted.x_orig_all,
            x_nun_all=adapted.x_nun_all,
            instance_ids=adapted.instance_ids,
            example_indices=DELINEATION_EXAMPLE_INDICES,
            max_examples=N_DELINEATION_EXAMPLES,
            output_path=plot_dir / "delineation_examples.png",
            config=config,
        )
        plot_global_delta_distributions(morphology_df, output_path=plot_dir / "global_delta_distributions.png")
        plot_mask_overlap_summary(mask_overlap_summary, output_path=plot_dir / "mask_overlap_summary.png")
        plot_changed_point_concentration(mask_overlap_summary, output_path=plot_dir / "changed_point_concentration.png")
        plot_mask_region_heatmaps(mask_region_heatmap_df, output_path=plot_dir / "mask_region_heatmaps.png")
        plot_average_waveforms(
            adapted.x_orig_all,
            adapted.x_cf_all,
            channels=AVERAGE_WAVEFORM_CHANNELS,
            output_path=plot_dir / "average_waveforms.png",
            config=config,
            first_label="Originals",
            second_label="Counterfactuals",
        )
        for metric in ["delta_q_amp", "delta_st_q_contrast", "delta_t_peak_abs", "delta_qrs_energy", "delta_t_area_abs", "qrs_best_lag_abs"]:
            if metric in morphology_df:
                plot_per_channel_medians(morphology_df, metric, output_path=plot_dir / f"per_channel_{metric}.png")
        if ANALYSIS_DIRECTION == "BOTH":
            for direction in ["NORM_TO_MI", "MI_TO_NORM"]:
                make_direction_plots(
                    direction,
                    output_dir,
                    adapted.x_orig_all,
                    adapted.x_cf_all,
                    adapted.x_nun_all,
                    adapted.metadata_df,
                    morphology_df,
                    mask_overlap_df,
                    mask_region_heatmap_df,
                    config,
                )
        example_index = min(max(EXAMPLE_INDEX, 0), adapted.x_orig_all.shape[0] - 1)
        plot_one_pair_with_windows(
            adapted.x_orig_all[example_index],
            adapted.x_cf_all[example_index],
            x_nun=None if adapted.x_nun_all is None else adapted.x_nun_all[example_index],
            channel=EXAMPLE_CHANNEL,
            title=f"Selected pair {example_index}",
            output_path=plot_dir / f"pair_{example_index}_channel_{EXAMPLE_CHANNEL}.png",
            config=config,
        )

    print("Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
