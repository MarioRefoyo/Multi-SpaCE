from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Literal, get_args


def plot_counterfactuals(
    obs: NDArray[np.float64],
    counterfactuals: NDArray[np.float64],
    target_cls: int,
    predicted_cls: NDArray[np.int64],
) -> Figure:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    n_counterfactuals = counterfactuals.shape[0]
    correct_obs = (predicted_cls == target_cls).astype(int).sum()

    cmap = cm.get_cmap("inferno", correct_obs * 2)  # matter of taste

    steps = np.arange(obs.shape[2])

    c_cnt = 0

    for i in range(n_counterfactuals):
        if predicted_cls[i] == target_cls:
            color = cmap(c_cnt)
            ax.plot(steps, counterfactuals[i, 0, :], color=color, alpha=0.5)
            c_cnt += 1

    ax.plot(steps, obs[0, 0, :], color="blue")
    ax.set_title(
        f"Generated counterfactuals of class {target_cls} ({correct_obs}/{n_counterfactuals})"
    )

    ax.set_xlabel("timestamps")
    ax.set_ylabel("")

    return fig


def plot_metrics(
    df: pd.DataFrame,
    metric_groups: list[str] = ["sparsity", "diversity"],
    ncols: int = 3,
) -> Figure:

    pattern = ""
    for group in metric_groups:
        pattern += f"{group}|"
    pattern = pattern[:-1]

    metric_cols = df.columns[df.columns.str.contains(pattern)]

    nrows = (metric_cols.shape[0] // ncols) + (
        1 if metric_cols.shape[0] % ncols != 0 else 0
    )

    fig, ax = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows))

    ax = ax.flatten()

    hue_col = None if "config" not in df.columns else "config"

    for i in range(metric_cols.shape[0]):
        c_ax = ax[i]
        sns.boxplot(
            data=df,
            y=metric_cols[i],
            hue=hue_col,
            ax=c_ax,
            legend=False if i != 0 else True,
            showfliers=False,
            palette=sns.color_palette("tab20"),
        )
        xlabel = metric_cols[i].replace("-", " ")
        c_ax.set_xlabel(xlabel)
        c_ax.set_ylabel("")
        # if c_ax.get_legend():
        # c_ax.get_legend().remove()

    for i in range(metric_cols.shape[0], nrows * ncols):
        ax[i].set_visible(False)

    fig.suptitle("metrics of performance")
    ax[0].legend(bbox_to_anchor=(-1, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    return fig


def plot_cls_success_rate(df: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    df = df.copy()
    df = df.sort_values(["original_cls", "target_cls"])

    df["conversion"] = df.apply(
        lambda row: f"{row['original_cls']:1} -> {row['target_cls']:1}", axis=1
    )
    all_df = df.copy()
    all_df["conversion"] = "ALL"
    df = pd.concat([df, all_df])

    hue_col = None if "config" not in df.columns else "config"

    sns.boxplot(
        data=df,
        x="conversion",
        y="success_rate",
        hue=hue_col,
        ax=ax,
        showfliers=False,
        palette=sns.color_palette("tab20"),
    )
    ax.get_legend().remove()
    ax.set_title("Success rate on flipping the class")

    ax.set_xlabel("type of flip")
    ax.set_ylabel("sucess rate")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    return fig


PlotIncrementalModes = Literal["single", "multiple"]


def plot_incremental_counterfactual(
    obs: NDArray[np.float64],
    cnt_steps: list[NDArray[np.float64]],
    cnt_shapes: list[NDArray[np.int64]],
    mode: PlotIncrementalModes = "single",
) -> Figure:

    if len(obs.shape) != 1:
        assert obs.shape[1] == 1
        obs = obs.copy().reshape(-1)

    assert mode in get_args(PlotIncrementalModes)

    if mode == "single":
        fig = _plot_incremental_counterfactual_single(
            obs, cnt_steps, cnt_shapes
        )
    elif mode == "multiple":
        fig = _plot_incremental_counterfactual_multiple(
            obs, cnt_steps, cnt_shapes
        )

    return fig


def _plot_incremental_counterfactual_single(
    obs: NDArray[np.float64],
    cnt_steps: list[NDArray[np.float64]],
    cnt_shapes: list[NDArray[np.int64]],
) -> Figure:
    pallete = sns.color_palette("icefire", n_colors=len(cnt_steps) + 1)
    steps = np.arange(obs.shape[0])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    for idx, (cnt_step, shape) in enumerate(
        (zip(cnt_steps[::-1], cnt_shapes[::-1]))
    ):
        ax.plot(steps, cnt_step[0, 0, :], c=pallete[idx], label=f"{shape}")

    ax.plot(steps, obs, c=pallete[-1], label="original")
    ax.legend(bbox_to_anchor=(1, 1))

    return fig


def _plot_incremental_counterfactual_multiple(
    obs: NDArray[np.float64],
    cnt_steps: list[NDArray[np.float64]],
    cnt_shapes: list[NDArray[np.int64]],
    org_pred: NDArray[np.float64],
    cnt_preds: NDArray[np.float64],
    ncols: int = 3,
) -> Figure:

    bounds = obs.min(), obs.max()

    nrows = (len(cnt_steps) + 1) // ncols
    steps = np.arange(obs.shape[0])

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 3 * nrows))
    ax = ax.flatten()

    for idx in range(len(cnt_steps)):
        ax[idx].plot(
            steps, cnt_steps[idx][0, 0, :], c="r", label=f"{cnt_shapes[idx]}"
        )
        if idx == 0:
            ax[idx].plot(steps, obs, c="b")
            pred_change = cnt_preds[idx] - org_pred
        else:
            ax[idx].plot(steps, cnt_steps[idx - 1][0, 0, :], c="b")
            pred_change = cnt_preds[idx] - cnt_preds[idx - 1]

        change_formatted = pred_change.tolist()
        change_formatted = [float(f"{el:.3f}") for el in pred_change]

        ax[idx].set_title(change_formatted)

        ax[idx].legend(loc="upper left")
        ax[idx].set_ylim((bounds[0], bounds[1]))

    for idx in range(len(cnt_steps), nrows * ncols):
        fig.delaxes(ax[idx])

    fig.tight_layout()

    return fig
