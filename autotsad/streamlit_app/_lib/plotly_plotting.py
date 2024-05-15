from itertools import cycle
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from autotsad.config import ALGORITHMS, SCORE_NORMALIZATION_METHODS, SCORE_AGGREGATION_METHODS, \
    ALGORITHM_SELECTION_METHODS
from autotsad.system.execution.aggregation import normalize_scores, aggregate_scores
from autotsad.util import mask_to_slices

_baselines = ["best-algo", "mean-algo", "k-Means (TimeEval)", "SAND (TimeEval)", "cae-ensemble"]
_ranking_methods = sorted(_baselines + [
    f"{rm}-{nm}-{am}"
    for rm in (list(ALGORITHM_SELECTION_METHODS) + ["aggregated-minimum-influence", "aggregated-robust-borda"])
    for nm in SCORE_NORMALIZATION_METHODS
    for am in SCORE_AGGREGATION_METHODS
])

ranking_method_color_map = dict(zip(_ranking_methods, cycle(px.colors.qualitative.Dark24)))
algo_color_map = dict(zip(ALGORITHMS, cycle(px.colors.qualitative.Vivid)))


def add_baseline_aggregated_scores(fig: go.Figure, df_ranking: pd.DataFrame, df_scores: pd.DataFrame,
                                   baseline_name: str, row=None, col=None) -> None:
    # construct label
    label = baseline_name
    if baseline_name == "best-algo":
        label = f"{baseline_name} ({df_ranking.loc[0, 'algorithm']})"

    # plot combined score
    scoring_ids = df_ranking["algorithm_scoring_id"].unique()
    df_scores = df_scores[df_scores["algorithm_scoring_id"].isin(scoring_ids)]
    scores = df_scores.pivot(index="time", columns="algorithm_scoring_id", values="score").values
    scores = normalize_scores(scores, normalization_method="minmax")
    combined_score = aggregate_scores(scores, agg_method="custom")
    fig.add_trace(
        go.Scatter(x=df_scores["time"], y=combined_score, mode="lines", name=label,
                   line_color=ranking_method_color_map[baseline_name]),
        row=row,
        col=col
    )


def add_dataset(fig: go.Figure, df_dataset: pd.DataFrame, row=None, col=None) -> None:
    fig.add_trace(
        go.Scatter(x=df_dataset["time"], y=df_dataset["value"], mode="lines", name="Dataset"),
        row=row,
        col=col
    )
    anomalies = mask_to_slices(df_dataset["is_anomaly"].values)
    for b, e in anomalies:
        # to see the anomalies better:
        b = max(0, b - 1)
        e = min(len(df_dataset), e + 1)
        fig.add_vrect(b, e, fillcolor="red", opacity=0.25, line_width=0, row=row, col=col)


@st.cache_data
def plot_dataset(name: str, df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_dataset(fig, df)
    fig.update_layout(title=name, xaxis_title="time", yaxis_title="value")
    return fig


@st.cache_data
def plot_aggregated_scores(df_dataset: pd.DataFrame, df_combined_scores: pd.DataFrame,
                           ranking_method: str, normalization_method: Optional[str] = None,
                           aggregation_method: Optional[str] = None, title: str = "") -> go.Figure:
    if normalization_method is not None and aggregation_method is not None:
        name = f"{ranking_method}-{normalization_method}-{aggregation_method}"
    else:
        name = ranking_method
    fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=0.02)
    add_dataset(fig, df_dataset, row=1, col=1)

    # plot combined score
    fig.add_trace(
        go.Scatter(x=df_combined_scores["time"], y=df_combined_scores["score"], mode="lines", name=name,
                   line_color=ranking_method_color_map[name]),
        row=2,
        col=1
    )
    fig.update_layout(
        title=title or name,
        xaxis_title="time",
        yaxis_title="value",
        height=400,
        legend=dict(yanchor="top", y=-.1, xanchor="center", x=0.5),
    )
    return fig


@st.cache_data
def plot_ranking(df_ranking: pd.DataFrame, df_scores: pd.DataFrame, df_dataset: Optional[pd.DataFrame] = None,
                 title: str = "",
                 normalization_method: Optional[str] = None) -> go.Figure:
    df_ranking = df_ranking.reset_index(drop=True)
    include_dataset = int(df_dataset is not None)
    fig = make_subplots(df_ranking.shape[0] + include_dataset, 1,
                        # subplot_titles=df_ranking["rank"].tolist(),
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    if include_dataset:
        add_dataset(fig, df_dataset, row=1, col=1)

    for i, (rank, scoring_id, algorithm, hyper_params) in df_ranking.iterrows():
        df_scoring = df_scores[df_scores["algorithm_scoring_id"] == scoring_id]
        if normalization_method:
            scores = normalize_scores(df_scoring["score"].values.reshape(-1, 1),
                                      normalization_method=normalization_method).ravel()
        else:
            scores = df_scoring["score"].values
        fig.add_trace(
            go.Scatter(x=df_scoring["time"], y=scores, mode="lines", name=f"{algorithm} {hyper_params}",
                       line_color=algo_color_map[algorithm]),
            row=i + 1 + include_dataset,
            col=1
        )
    fig.update_layout(
        legend=dict(yanchor="top", y=-.05, xanchor="center", x=0.5),
        height=100 + 100 * df_ranking.shape[0],
        title=title
    )
    return fig
