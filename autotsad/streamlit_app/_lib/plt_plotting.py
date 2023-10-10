from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Optional, Sequence, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler

from autotsad.system.execution.aggregation import aggregate_scores, normalize_scores
from autotsad.util import mask_to_slices

FIGURE_WIDTH = 7.5
FIGURE_DPI = 500
FONT_SIZE = 14
SAVE_PATH = Path.cwd() / "paper-plots"

NAME_MAPPING = {
    # normalization strategies
    # "minmax": "Min-Max",
    # "gaussian": "Gaussian",
    # don't show normalization strategies
    "minmax": "",
    "gaussian": "",
    # ranking strategies
    "training-quality": "TQ",
    "training-result": "TR",
    "kmedoids-clustering": "KM",
    "affinity-propagation-clustering": "AP",
    "greedy-euclidean": "Greedy ESD",
    "greedy-annotation-overlap": "Greedy AOD",
    "mmq-euclidean": "MMQ ESD",
    "mmq-annotation-overlap": "MMQ AOD",
    "aggregated-minimum-influence": "RRA",
    # aggregation strategies
    "custom": "MoT",
    "max": "Max",
    # baselines
    "best-algo": "Oracle",
    "mean-algo": "Random Algorithm",
    "k-Means (TimeEval)": "k-Means",
    "SAND (TimeEval)": "SAND",
    # AutoTSAD variants
    "top-1": "RRA Top-1 Method",
    # basic methods
    "stomp": "STOMP",
    "kmeans": "k-Means",
    "subsequence_knn": "Sub-KNN",
    "subsequence_lof": "Sub-LOF",
    "subsequence_if": "Sub-IF",
    "grammarviz": "GrammarViz",
    "torsk": "Torsk",
    "dwt_mlead": "DWT-MLEAD",
    # runtime traces
    "Computing all combinations": "Ranking & Aggregation",
    "Algorithm Execution": "Algorithm Instance Execution",
    "Selecting best performers": "Algorithm Instance Selection",
    # misc
    "timeeval": "TimeEval",
}


def adjust_names(name: str, title: bool = False) -> str:
    for k, v in NAME_MAPPING.items():
        name = name.replace(k, v)
    name = name.replace("_", " ")
    if title:
        name = name.title()
    return name


class MarkerMap(list):
    def __call__(self, key: int) -> str:
        return self[key]


cm = matplotlib.colormaps["inferno"].resampled(9)
mm = MarkerMap(["o", "s", "*", "+", "x", "|", "v", "^", "D"])
method_type_colors = {"Baseline": cm(2), "AutoTSAD": cm(6), "AutoTSAD Top-1": cm(4), "AutoTSAD Ensemble": cm(5)}


def _save_figure(save_files_path: Optional[Path], name: str, fig: Figure, legend: Optional[Legend] = None) -> None:
    if save_files_path is not None:
        if legend is not None:
            fig.savefig(save_files_path / f"{name}.pdf", bbox_extra_artists=[legend], bbox_inches="tight")
            fig.savefig(save_files_path / f"{name}.png", bbox_extra_artists=[legend], bbox_inches="tight")
        else:
            fig.savefig(save_files_path / f"{name}.pdf", bbox_inches="tight")
            fig.savefig(save_files_path / f"{name}.png", bbox_inches="tight")


@st.cache_data(max_entries=2, show_spinner=False, persist=False)
def plot_quality(df_quality: pd.DataFrame, method_order: Sequence[str],
                 save_files_path: Optional[Path] = SAVE_PATH, name: str = "quality_comparison") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.5), dpi=FIGURE_DPI)
    ax = plt.gca()
    for m in df_quality["Method"].unique():
        df = df_quality[df_quality["Method"] == m]
        # filter out datasets for which the method failed (NaNs)
        df = df[~df["range_pr_auc"].isna()]
        method_type = df["Method Type"].iloc[0]
        color = method_type_colors[method_type]
        label = adjust_names(m)
        if "AutoTSAD" in method_type:
            # put AutoTSAD in front of all method names
            label = "AutoTSAD " + label
        plt.boxplot(df["range_pr_auc"].values,
                    patch_artist=True, vert=False, meanline=True, showfliers=False, showmeans=True, widths=0.8,
                    whis=(0., 100.),  # whiskers at min/max
                    boxprops=dict(color=color, linewidth=1, facecolor=color, alpha=0.5),
                    whiskerprops=dict(color=color, linewidth=1),
                    capprops=dict(color=color, linewidth=1),
                    medianprops=dict(color=color, linewidth=2),
                    meanprops=dict(color="black", linewidth=2),
                    labels=[label], positions=[method_order[::-1].index(m)],
                    )
    ax.set(axisbelow=True)
    ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Range-PR-AUC")
    ax.yaxis.set_tick_params(length=0)
    spines = ax.spines
    spines["top"].set_visible(False)
    spines["left"].set_visible(False)
    spines["right"].set_visible(False)

    # build legend
    lines = [
        Line2D([0, 0], [1, 0], color="black", linewidth=2, linestyle="-"),
        Line2D([0, 0], [1, 0], color="black", linewidth=2, linestyle="--"),
        Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["Baseline"]),
    ]
    labels = [
        "Median",
        "Mean",
        "Baselines",
    ]
    if "AutoTSAD" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD"])
        )
        labels.append("AutoTSAD Rankings")
    if "AutoTSAD Top-1" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD Top-1"])
        )
        labels.append("AutoTSAD Top-1")
    if "AutoTSAD Ensemble" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD Ensemble"])
        )
        labels.append("AutoTSAD Rank Aggregations")

    legend = fig.legend(
        lines, labels,
        loc="center",
        ncol=2 if len(lines) < 5 else 3,
        bbox_to_anchor=(0.4, 1.05),
        borderaxespad=0.,
    )
    _save_figure(save_files_path, name, fig, legend)
    return fig


@st.cache_data(max_entries=2, show_spinner=False, persist=False)
def plot_runtime_traces(df_runtime_trace: pd.DataFrame, group_traces: Optional[Dict[str, Sequence[str]]] = None,
                        save_files_path: Optional[Path] = SAVE_PATH, name: str = "runtime") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE
    trace_datasets = df_runtime_trace["dataset_name"].unique().tolist()

    fig, axs = plt.subplots(1, len(trace_datasets), sharey="row", figsize=(FIGURE_WIDTH, 3), dpi=FIGURE_DPI)
    for i, d in enumerate(trace_datasets):
        axs[i].set_title(f"({chr(65 + i)})")

        for n_jobs in df_runtime_trace.loc[df_runtime_trace["dataset_name"] == d, "n_jobs"].unique():
            df = df_runtime_trace[(df_runtime_trace["dataset_name"] == d) & (df_runtime_trace["n_jobs"] == n_jobs)]
            if group_traces is not None:
                for group_name in group_traces:
                    traces = group_traces[group_name]
                    mask = df["trace_name"].isin(traces)
                    runtime = df[mask]["runtime"].sum()
                    position = df[mask]["position"].min()
                    df_group = pd.DataFrame([{
                        "runtime": runtime,
                        "trace_name": group_name,
                        "dataset_name": d,
                        "n_jobs": n_jobs,
                        "position": position,
                    }])
                    df = pd.concat([df_group, df[~mask]], ignore_index=True)

            df["trace_name"] = df["trace_name"].str.split("-%-").str[-1].apply(adjust_names)
            df = df.sort_values("position")

            for j in np.arange(df.shape[0]):
                row = df.iloc[j]
                if j == 0:
                    b = None
                else:
                    b = df.iloc[:j]["runtime"].sum()
                axs[i].bar(x=f"{n_jobs}", height=row["runtime"], bottom=b, color=cm(j + 2), label=row["trace_name"])

    for i in range(len(axs)):
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[i].set(axisbelow=True)
        axs[i].xaxis.set_tick_params(length=0)
        axs[i].set_xlabel("Parallelism")
        axs[i].yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        spines = axs[i].spines
        spines["top"].set_visible(False)
        spines["bottom"].set_visible(False)
        spines["left"].set_visible(True)
        spines["right"].set_visible(False)
        if i == 0:
            axs[i].set_ylabel("Runtime (s)")

    # build legend
    exclude_traces = [t for traces in group_traces.values() for t in traces]
    n = len([t for t in df_runtime_trace["trace_name"].unique() if t not in exclude_traces]) + len(group_traces)
    lines, labels = axs[0].get_legend_handles_labels()
    lines = lines[:n][::-1]
    labels = labels[:n][::-1]
    legend = fig.legend(
        lines, labels,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(0.905, 0.5),
        borderaxespad=0.,
    )

    _save_figure(save_files_path, name, fig, legend)
    return fig


@st.cache_data(max_entries=2, show_spinner=False, persist=False)
def plot_optimization_scaling(df_optimization_variance: pd.DataFrame,
                              truncate_labels_to: Optional[int] = None,
                              save_files_path: Optional[Path] = SAVE_PATH,
                              name: str = "optimization-variance") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2), dpi=FIGURE_DPI)
    axs = [fig.gca()]

    for i, dataset in enumerate(df_optimization_variance["dataset"].unique()):
        df = df_optimization_variance[df_optimization_variance["dataset"] == dataset]
        label = dataset
        if truncate_labels_to is not None and len(label) > truncate_labels_to:
            label = label[:truncate_labels_to-3] + "..."
        axs[0].plot(df["max_trials"], df["mean"], color=cm(i + 1), marker=mm(i+1), label=label)
        axs[0].fill_between(df["max_trials"],
                            df["mean"] - df["std"],
                            df["mean"] + df["std"],
                            color=cm(i + 1), alpha=0.2)
    axs[0].set(axisbelow=True)
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("Range-PR-AUC")
    axs[0].set_xticks(sorted(df_optimization_variance["max_trials"].unique()))
    axs[0].xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    axs[0].set_xlabel("Max Trials")

    spines = axs[0].spines
    spines["top"].set_visible(False)
    spines["bottom"].set_visible(True)
    spines["left"].set_visible(True)
    spines["right"].set_visible(False)

    legend = axs[0].legend(loc="lower center", ncol=1, bbox_to_anchor=(0.5, 1.1), borderaxespad=0.)
    _save_figure(save_files_path, name, fig, legend)
    return fig


@st.cache_data(max_entries=2, show_spinner=False, persist=False)
def plot_optimization_improvement(df_improvements: pd.DataFrame,
                                  save_files_path: Optional[Path] = SAVE_PATH,
                                  name: str = "optimization-improvement") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE

    fig = plt.figure(figsize=(7.5, df_improvements.shape[1]*0.5), dpi=500)
    axs = [plt.gca()]
    for i, method in enumerate(df_improvements.columns[::-1]):
        color = method_type_colors["AutoTSAD"]
        if method.startswith("aggregated"):
            color = method_type_colors["AutoTSAD Ensemble"]
        label = adjust_names(method)
        axs[0].boxplot(df_improvements[method].values,
                       patch_artist=True, vert=False, meanline=True, showfliers=False, showmeans=True, widths=0.8,
                       whis=(0., 100.),  # whiskers at min/max
                       boxprops=dict(color=color, linewidth=1, facecolor=color, alpha=0.5),
                       whiskerprops=dict(color=color, linewidth=1),
                       capprops=dict(color=color, linewidth=1),
                       medianprops=dict(color=color, linewidth=2),
                       meanprops=dict(color="black", linewidth=2),
                       labels=[label], positions=[i],
                       )
    axs[0].set(axisbelow=True)
    axs[0].xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    axs[0].set_xlim(-1, 1)
    axs[0].set_xlabel("$\\Delta$ Range-PR-AUC of AutoTSAD")
    # axs[0].set_yticklabels([])
    axs[0].yaxis.set_tick_params(length=0)

    spines = axs[0].spines
    spines["top"].set_visible(False)
    spines["bottom"].set_visible(True)
    spines["left"].set_visible(False)
    spines["right"].set_visible(False)

    _save_figure(save_files_path, name, fig)
    return fig


@st.cache_data(max_entries=3, show_spinner=False, persist=False)
def plot_ensembling_strategies(df_quality: pd.DataFrame, method_order: Sequence[str],
                               save_files_path: Optional[Path] = SAVE_PATH,
                               name: str = "ensembling_comparison") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.5), dpi=FIGURE_DPI)
    ax = plt.gca()
    for m in df_quality["Method"].unique():
        df = df_quality[df_quality["Method"] == m]
        # filter out datasets for which the method failed (NaNs)
        df = df[~df["range_pr_auc"].isna()]
        method_type = df["Method Type"].iloc[0]
        color = method_type_colors[method_type]
        label = adjust_names(m)
        if "AutoTSAD" in method_type:
            # put AutoTSAD in front of all method names
            label = "AutoTSAD " + label
        plt.boxplot(df["range_pr_auc"].values,
                    patch_artist=True, vert=True, meanline=True, showfliers=False, showmeans=True, widths=0.8,
                    whis=(0., 100.),  # whiskers at min/max
                    boxprops=dict(color=color, linewidth=1, facecolor=color, alpha=0.5),
                    whiskerprops=dict(color=color, linewidth=1),
                    capprops=dict(color=color, linewidth=1),
                    medianprops=dict(color=color, linewidth=2),
                    meanprops=dict(color="black", linewidth=2),
                    labels=[label], positions=[method_order[::-1].index(m)],
                    )
    ax.set(axisbelow=True)
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Range-PR-AUC")
    ax.xaxis.set_tick_params(length=0)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=50, ha="right")
    spines = ax.spines
    spines["top"].set_visible(False)
    spines["bottom"].set_visible(False)
    spines["right"].set_visible(False)

    # build legend
    lines = [
        Line2D([0, 0], [1, 0], color="black", linewidth=2, linestyle="-"),
        Line2D([0, 0], [1, 0], color="black", linewidth=2, linestyle="--"),
    ]
    labels = [
        "Median",
        "Mean",
    ]
    if "Baseline" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["Baseline"])
        )
        labels.append("Baselines")
    if "AutoTSAD" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD"])
        )
        labels.append("AutoTSAD Rankings")
    if "AutoTSAD Top-1" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD Top-1"])
        )
        labels.append("AutoTSAD Top-1")
    if "AutoTSAD Ensemble" in df_quality["Method Type"].unique():
        lines.append(
            Rectangle((0, 0), 1, 0.5, linestyle="-", fill=True, alpha=0.5, linewidth=1, color=method_type_colors["AutoTSAD Ensemble"])
        )
        labels.append("AutoTSAD Rank Aggregations")

    legend = fig.legend(
        lines, labels,
        loc="center",
        ncol=2 if len(lines) < 5 else 3,
        bbox_to_anchor=(0.4, 1.06),
        borderaxespad=0.,
    )
    _save_figure(save_files_path, name, fig, legend)
    return fig


@st.cache_data(max_entries=2, show_spinner=False, persist=False)
def plot_ranking_comparison(data, ranking1, ranking2,
                            quality_left: float, quality_right: float,
                            method_left: str, method_right: str,
                            show_anomalies: bool = False,
                            show_algo_quality: bool = False,
                            show_algo_params: bool = False,
                            save_files_path: Optional[Path] = SAVE_PATH,
                            name: str = "ranking_example") -> Figure:
    plt.rcParams["font.size"] = FONT_SIZE

    algorithms = sorted(list(set(ranking1[0]["algorithm"].unique()) | set(ranking2[0]["algorithm"].unique())))
    algo_color_map = dict(zip(algorithms, cycle(cm.colors[2:-1])))
    ts_color = cm.colors[0]
    combined_score_color = cm.colors[1]
    anomaly_highlight_color = cm.colors[-2]
    anomaly_highlight_color_alpha = 0.6

    start_idx = 0
    end_idx = data.shape[0]
    ranking_entries = max(ranking1[0].shape[0], ranking2[0].shape[0])

    def _plot_single_ranking(axs: np.ndarray, col: int, results: pd.DataFrame, scores: pd.DataFrame, method: str) -> None:
        rmethod, nmethod, amethod = method.split("_")
        scores = scores[(scores["time"] >= start_idx) & (scores["time"] <= end_idx)]

        for i in range(1, len(results) + 1):
            rank, scoring_id, algo, params = results.loc[i - 1, :]
            c = algo_color_map[algo]
            label = adjust_names(algo)
            if show_algo_params:
                label += f" {params}"
            # if show_algo_quality:
            #     label += f" ({quality:.0%})"

            scoring = scores[scores["algorithm_scoring_id"] == scoring_id]
            axs[i, col].plot(index, scoring["score"].values, label=label, color=c)
            if col == 0:
                axs[i, col].text(-0.05, 0.5, label, transform=axs[i, col].transAxes, ha="right", va="center", color=c)
            else:
                axs[i, col].text(1.05, 0.5, label, transform=axs[i, col].transAxes, ha="left", va="center", color=c)

        # compute combined scores
        scores = scores.pivot(index="time", columns="algorithm_scoring_id", values="score").values
        scores = normalize_scores(scores, normalization_method=nmethod)
        combined_score = aggregate_scores(scores, agg_method=amethod)
        axs[0, col].plot(index, combined_score, label="Combined score", color=combined_score_color)

    fig, axs = plt.subplots(1 + ranking_entries, 2, sharex="col", figsize=(FIGURE_WIDTH, 4), dpi=FIGURE_DPI,
                            gridspec_kw={"height_ratios": [2, *[1 for _ in range(ranking_entries)]]})

    # plot data
    anomalies = mask_to_slices(data["is_anomaly"].values)
    data = data[(data["time"] >= start_idx) & (data["time"] <= end_idx)]
    index = data["time"].values
    data = data["value"].values
    data = MinMaxScaler().fit_transform(data.reshape(-1, 1)).ravel()

    axs[0, 0].set_title(adjust_names(method_left), loc="right")
    axs[0, 0].plot(index, data, color=ts_color, alpha=0.4, label="Timeseries")
    axs[0, 1].set_title(adjust_names(method_right), loc="left")
    axs[0, 1].plot(index, data, color=ts_color, alpha=0.4, label="Timeseries")
    if show_anomalies:
        # mark anomalies
        y_min = data.min()
        y_max = data.max()
        for begin, end in anomalies:
            width = end - begin
            if width < 2:
                width += 2
                begin -= 1
            axs[0, 0].add_patch(Rectangle((begin, y_min), width, y_max - y_min,
                                          color=anomaly_highlight_color,
                                          alpha=anomaly_highlight_color_alpha))
            axs[0, 1].add_patch(Rectangle((begin, y_min), width, y_max - y_min,
                                          color=anomaly_highlight_color,
                                          alpha=anomaly_highlight_color_alpha))
    if show_algo_quality:
        axs[0, 0].text(-0.05, 0.5, f"Range-PR-AUC\n{quality_left:.2%}", transform=axs[0, 0].transAxes, ha="right", va="center")
        axs[0, 1].text(1.05, 0.5, f"Range-PR-AUC\n{quality_right:.2%}", transform=axs[0, 1].transAxes, ha="left", va="center")

    # display styling
    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[0, 0].set_xlim(index[0] - 5, index[-1] + 5)
    axs[0, 1].set_xlim(index[0] - 5, index[-1] + 5)

    for i in range(1, axs.shape[0] - 1):
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
    for j in range(0, axs.shape[1]):
        axs[-1, j].set(axisbelow=True)
        axs[-1, j].set_yticks([])
        axs[-1, j].yaxis.set_tick_params(length=0)
        spines = axs[-1, j].spines
        spines["top"].set_visible(False)
        spines["left"].set_visible(False)
        spines["right"].set_visible(False)

    # plot rankings
    _plot_single_ranking(axs, 0, ranking1[0], ranking1[1], method_left)
    _plot_single_ranking(axs, 1, ranking2[0], ranking2[1], method_right)
    _save_figure(save_files_path, name, fig)
    return fig
