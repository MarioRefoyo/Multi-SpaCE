from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from cf_dashboard_utils import (
        available_numeric_columns,
        build_result_file_path,
        candidate_table,
        discover_candidate_files,
        enrich_instance_with_evaluation_metrics,
        load_result_file,
        find_nearby_params,
        list_experiment_options,
        list_family_experiments,
        list_result_datasets,
        list_result_subfolders,
        normalize_loaded_object,
        weighted_candidate,
    )
except ImportError:
    from dashboards.cf_dashboard_utils import (
        available_numeric_columns,
        build_result_file_path,
        candidate_table,
        discover_candidate_files,
        enrich_instance_with_evaluation_metrics,
        load_result_file,
        find_nearby_params,
        list_experiment_options,
        list_family_experiments,
        list_result_datasets,
        list_result_subfolders,
        normalize_loaded_object,
        weighted_candidate,
    )


st.set_page_config(page_title="Counterfactual Dashboard", layout="wide")


if hasattr(st, "cache_data"):
    cache_data_compat = st.cache_data
else:
    cache_data_compat = st.cache


def clear_cache(func) -> None:
    if hasattr(func, "clear"):
        func.clear()
    elif hasattr(st, "caching"):
        st.caching.clear_cache()


@cache_data_compat(show_spinner=False)
def cached_discovery(root: str, limit: int) -> list[str]:
    return [str(path) for path in discover_candidate_files(root, limit=limit)]


@cache_data_compat(show_spinner=False)
def cached_load(path: str, mtime_ns: int, size_bytes: int) -> dict:
    raw = load_result_file(path)
    params = find_nearby_params(path)
    return normalize_loaded_object(raw, path, params=params, load_companion=True)


def file_signature(path: str | Path) -> tuple[str, int, int]:
    source_path = Path(path)
    stat = source_path.stat()
    return str(source_path), int(stat.st_mtime_ns), int(stat.st_size)


def main() -> None:
    st.title("Multi-SpaCE Counterfactual Inspector")

    with st.sidebar:
        st.header("Source")
        results_root = st.text_input("Results root", value="experiments/results")
        if st.button("Refresh selectors"):
            clear_cache(cached_discovery)
            clear_cache(cached_load)
        source_path = render_hierarchical_selector(results_root)

        with st.expander("Manual path fallback", expanded=False):
            limit = st.number_input("Fallback discovery limit", min_value=20, max_value=5000, value=500, step=50)
            discovered = cached_discovery(results_root, int(limit))
            selected_discovered = st.selectbox(
                "Discovered files",
                options=[""] + discovered,
                format_func=lambda path: "Paste a path below" if path == "" else path,
            )
            pasted_path = st.text_input("Result file or folder path", value=selected_discovered)
            source_path = source_path or resolve_source_path(pasted_path)

    if source_path is None:
        st.info("Select or paste a supported result file. Folders are searched for likely result files.")
        return

    try:
        normalized = cached_load(*file_signature(source_path))
    except Exception as exc:
        st.error(f"Could not load `{source_path}`.")
        st.exception(exc)
        return

    render_dataset_summary(normalized)

    instances = normalized.get("instances", [])
    if not instances:
        st.warning("No inspectable instances were detected in this file.")
        render_debug(normalized)
        return

    instance = select_instance(instances)
    if instance is None:
        return
    with st.spinner("Calculating model, NUN, and AE scores for the selected instance..."):
        instance = enrich_instance_with_evaluation_metrics(instance, normalized["source_path"])

    table = candidate_table(instance)
    selected_candidate_id = select_candidate(table)
    if not table.empty:
        selected_candidate_id = st.selectbox(
            "Displayed candidate",
            options=table["candidate_id"].astype(int).tolist(),
            index=max(0, table["candidate_id"].astype(int).tolist().index(selected_candidate_id)),
        )
    selected_candidate = get_candidate(instance, selected_candidate_id)

    render_instance_metadata(instance, selected_candidate, table)
    render_pareto(table, selected_candidate_id)
    render_time_series(instance, selected_candidate)
    render_tables(instance, table)
    render_debug(normalized, instance)


def resolve_source_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = discover_candidate_files(path, limit=100)
        preferred = [candidate for candidate in candidates if candidate.name in {"counterfactuals.pickle", "counterfactuals.pkl"}]
        return (preferred or candidates or [None])[0]
    return path


def render_hierarchical_selector(results_root: str) -> Path | None:
    datasets = list_result_datasets(results_root)
    if not datasets:
        st.info("No datasets were found under the selected results root.")
        return None

    dataset = st.selectbox("Dataset", options=datasets)
    subfolders = list_result_subfolders(dataset, results_root)
    if not subfolders:
        st.info("No result subfolders were found for this dataset.")
        return None

    subfolder = st.selectbox("Experiment folder", options=subfolders)
    options = list_experiment_options(dataset, subfolder, results_root)

    direct_experiments = options["direct_experiments"]
    families = options["families"]

    if direct_experiments and families:
        mode = st.radio("Layout", ["Direct experiments", "Experiment family"], horizontal=True)
    elif direct_experiments:
        mode = "Direct experiments"
    elif families:
        mode = "Experiment family"
    else:
        st.info("No experiment runs with `counterfactuals.pickle` were found in this folder.")
        return None

    if mode == "Direct experiments":
        experiment = st.selectbox("Experiment", options=direct_experiments)
        source_path = build_result_file_path(dataset, subfolder, experiment, results_root=results_root)
    else:
        family = st.selectbox("Experiment family", options=families)
        experiments = list_family_experiments(dataset, subfolder, family, results_root)
        if not experiments:
            st.info("No experiment runs were found inside this family.")
            return None
        experiment = st.selectbox("Experiment", options=experiments)
        source_path = build_result_file_path(dataset, subfolder, experiment, family=family, results_root=results_root)

    st.caption(str(source_path))
    if not source_path.is_file():
        st.warning("The selected experiment path does not contain `counterfactuals.pickle`.")
        return None
    return source_path


def render_dataset_summary(normalized: dict) -> None:
    metadata = normalized.get("metadata", {})
    cols = st.columns(4)
    cols[0].metric("Instances", len(normalized.get("instances", [])))
    cols[1].metric("Dataset", normalized.get("dataset_name") or "unknown")
    cols[2].metric("Method", normalized.get("method_name") or "unknown")
    cols[3].metric("Source", Path(normalized.get("source_path", "")).name)

    with st.expander("Detected source metadata", expanded=False):
        st.json(safe_json(metadata))


def select_instance(instances: list[dict]) -> dict | None:
    labels = [instance_label(instance, i) for i, instance in enumerate(instances)]
    selected = st.sidebar.selectbox("Original instance", options=list(range(len(instances))), format_func=lambda i: labels[i])
    return instances[selected]


def select_candidate(table: pd.DataFrame) -> int | None:
    if table.empty:
        st.warning("No Pareto-front candidates were detected for this instance.")
        return None

    candidate_ids = table["candidate_id"].astype(int).tolist()
    st.sidebar.header("Candidate")
    selection_mode = st.sidebar.radio("Selection mode", ["Manual", "Weighted utility"], horizontal=True)

    if selection_mode == "Manual":
        return st.sidebar.selectbox("Candidate index", options=candidate_ids)

    numeric_cols = available_numeric_columns(table)
    selected_metrics = st.sidebar.multiselect("Utility objectives", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))])
    weights = {}
    for metric in selected_metrics:
        weights[metric] = st.sidebar.number_input(f"Weight: {metric}", value=1.0, step=0.1)
    maximize = st.sidebar.checkbox("Maximize weighted utility", value=False)
    selected = weighted_candidate(table, weights, maximize=maximize)
    if selected is None:
        st.sidebar.warning("No numeric utility could be calculated.")
        return candidate_ids[0]
    st.sidebar.caption(f"Selected candidate: {selected}")
    return selected


def render_instance_metadata(instance: dict, candidate: dict | None, table: pd.DataFrame) -> None:
    st.subheader("Instance")
    cols = st.columns(8)
    cols[0].metric("Instance", stringify(instance.get("instance_id")))
    cols[1].metric("y_true", stringify(instance.get("y_true")))
    cols[2].metric("predicted_orig", stringify(instance.get("predicted_orig")))
    cols[3].metric("target/NUN", stringify(instance.get("target_class")))
    cols[4].metric("NUN index", stringify(instance.get("nun_idx")))
    cols[5].metric("Candidates", len(instance.get("pareto", [])))
    cols[6].metric("selected_cf", stringify(candidate.get("candidate_id") if candidate else None))
    cols[7].metric("predicted_cf", stringify(candidate.get("predicted_cf") if candidate else None))

    if candidate and candidate.get("validity") is not None:
        st.caption(f"Selected validity: {candidate.get('validity')}")
    if table.empty:
        return


def render_pareto(table: pd.DataFrame, selected_candidate_id: int | None) -> None:
    st.subheader("Pareto Front")
    numeric_cols = available_numeric_columns(table)
    if len(numeric_cols) < 3:
        st.info("At least three numeric objective or metric columns are needed for a 3D Pareto plot.")
        return

    axis_cols = st.columns(3)
    default_x, default_y, default_z = default_pareto_axes(numeric_cols)
    x_col = axis_cols[0].selectbox("X axis", numeric_cols, index=numeric_cols.index(default_x))
    y_col = axis_cols[1].selectbox("Y axis", numeric_cols, index=numeric_cols.index(default_y))
    z_col = axis_cols[2].selectbox("Z axis", numeric_cols, index=numeric_cols.index(default_z))

    plot_df = table.copy()
    for col in [x_col, y_col, z_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col, z_col])
    if plot_df.empty:
        st.warning("Selected axes do not contain plottable numeric values.")
        return

    colors = np.where(plot_df["candidate_id"] == selected_candidate_id, "selected", "candidate")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=plot_df[x_col],
            y=plot_df[y_col],
            z=plot_df[z_col],
            mode="markers",
            marker={
                "size": np.where(plot_df["candidate_id"] == selected_candidate_id, 8, 5),
                "color": np.where(colors == "selected", "#d62728", "#1f77b4"),
                "opacity": 0.85,
            },
            text=[f"candidate_id={int(row.candidate_id)}" for row in plot_df.itertuples()],
            hovertemplate="%{text}<br>"
            + f"{x_col}=%{{x}}<br>"
            + f"{y_col}=%{{y}}<br>"
            + f"{z_col}=%{{z}}<extra></extra>",
        )
    )
    fig.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "b": 0, "t": 20},
        scene={"xaxis_title": x_col, "yaxis_title": y_col, "zaxis_title": z_col},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use the candidate_id in the hover text for reliable manual selection in the sidebar.")


def render_time_series(instance: dict, candidate: dict | None) -> None:
    st.subheader("Counterfactual Overlay")
    if candidate is None:
        return

    x_orig = instance.get("x_orig")
    x_cf = candidate.get("x_cf")
    x_nun = instance.get("x_nun")
    mask = candidate.get("mask")

    if x_orig is None or x_cf is None:
        st.info("This result does not contain enough array data to plot x_orig and x_cf.")
        return

    x_orig = ensure_2d(x_orig)
    x_cf = ensure_2d(x_cf)
    x_nun = ensure_2d(x_nun) if x_nun is not None else None
    mask = ensure_2d(mask) if mask is not None else infer_mask(x_orig, x_cf)

    if x_orig.shape != x_cf.shape:
        st.warning(f"Shape mismatch: x_orig {x_orig.shape}, x_cf {x_cf.shape}.")
        return

    size_pct = st.slider("Overlay size", min_value=40, max_value=100, value=70, step=5)
    fig = make_overlay_figure(x_orig, x_cf, x_nun, mask, size_scale=size_pct / 100)
    if size_pct == 100:
        st.plotly_chart(fig, use_container_width=True)
        return

    side_pad = max(1.0, (100 - size_pct) / 2)
    _, center, _ = st.columns([side_pad, float(size_pct), side_pad])
    with center:
        st.plotly_chart(fig, use_container_width=True)


def make_overlay_figure(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    x_nun: np.ndarray | None,
    mask: np.ndarray | None,
    size_scale: float = 1.0,
) -> go.Figure:
    n_channels = x_orig.shape[1]
    fig = make_subplots(rows=n_channels, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    time = np.arange(x_orig.shape[0])
    for channel in range(n_channels):
        row = channel + 1
        fig.add_trace(
            go.Scatter(
                x=time,
                y=x_orig[:, channel],
                mode="lines",
                name="x_orig",
                line={"color": "#1f77b4"},
                legendgroup="x_orig",
                showlegend=channel == 0,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=x_cf[:, channel],
                mode="lines",
                name="x_cf",
                line={"color": "#d62728"},
                legendgroup="x_cf",
                showlegend=channel == 0,
            ),
            row=row,
            col=1,
        )
        if x_nun is not None and x_nun.shape == x_orig.shape:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=x_nun[:, channel],
                    mode="lines",
                    name="x_nun",
                    line={"color": "#2ca02c", "dash": "dot"},
                    legendgroup="x_nun",
                    showlegend=channel == 0,
                ),
                row=row,
                col=1,
            )
        if mask is not None and mask.shape == x_orig.shape:
            add_changed_regions(fig, mask[:, channel], row=row)
        fig.update_yaxes(title_text=f"ch {channel}", row=row, col=1)

    base_height = max(320, 180 * n_channels)
    fig.update_layout(
        height=max(220, int(base_height * size_scale)),
        margin={"l": 0, "r": 10, "t": 20, "b": 0},
        xaxis_title="time",
    )
    return fig


def add_changed_regions(fig: go.Figure, mask: np.ndarray, row: int) -> None:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return
    starts = np.where(np.diff(mask.astype(int), prepend=0) == 1)[0]
    ends = np.where(np.diff(mask.astype(int), append=0) == -1)[0]
    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=int(start),
            x1=int(end),
            fillcolor="rgba(214, 39, 40, 0.12)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )


def render_tables(instance: dict, table: pd.DataFrame) -> None:
    st.subheader("Candidate Table")
    if table.empty:
        st.info("No candidate table is available.")
        return
    try:
        st.dataframe(table, width=None)
    except TypeError:
        st.write(table)


def render_debug(normalized: dict, instance: dict | None = None) -> None:
    with st.expander("Debug / raw structure", expanded=False):
        st.write("Normalized keys:", list(normalized.keys()))
        if instance is not None:
            st.write("Instance metadata")
            st.json(safe_json(instance.get("metadata", {})))
            st.write("Array shapes")
            st.json(
                {
                    "x_orig": shape_of(instance.get("x_orig")),
                    "x_nun": shape_of(instance.get("x_nun")),
                    "pareto": len(instance.get("pareto", [])),
                }
            )
        raw = normalized.get("raw")
        st.write("Raw type:", type(raw).__name__)
        if isinstance(raw, dict):
            st.write("Raw keys:", list(raw.keys())[:100])
        elif isinstance(raw, list):
            st.write("Raw length:", len(raw))
            if raw:
                st.write("First item type:", type(raw[0]).__name__)
                if isinstance(raw[0], dict):
                    st.write("First item keys:", list(raw[0].keys())[:100])


def get_candidate(instance: dict, candidate_id: int | None) -> dict | None:
    for candidate in instance.get("pareto", []):
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return instance.get("pareto", [None])[0]


def instance_label(instance: dict, position: int) -> str:
    instance_id = instance.get("instance_id", position)
    candidates = len(instance.get("pareto", []))
    y_true = instance.get("y_true")
    return f"{position}: id={instance_id}, y_true={stringify(y_true)}, candidates={candidates}"


def ensure_2d(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def infer_mask(x_orig: np.ndarray, x_cf: np.ndarray) -> np.ndarray:
    return (np.abs(x_orig - x_cf) > 1e-12).astype(int)


def default_pareto_axes(numeric_cols: list[str]) -> tuple[str, str, str]:
    preferred = [
        "sparsity",
        "subsequences",
        "subsequences %",
        "AE_OS",
        "AE_IOS",
        "proximity",
        "L2",
        "L1",
        "NoS",
        "contiguity",
        "validity",
        "times",
    ]
    chosen = [col for col in preferred if col in numeric_cols]
    chosen.extend(col for col in numeric_cols if col not in chosen)
    return chosen[0], chosen[1], chosen[2]


def stringify(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return str(value)


def shape_of(value) -> str | None:
    return str(np.asarray(value).shape) if value is not None else None


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value[:200]]
    if isinstance(value, tuple):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    main()
