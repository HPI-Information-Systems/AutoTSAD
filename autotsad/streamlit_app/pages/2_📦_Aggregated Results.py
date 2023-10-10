from typing import List

import _lib.preamble  # noqa: F401
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from _lib.plotly_plotting import plot_aggregated_scores, plot_ranking, ranking_method_color_map, algo_color_map  # noqa: F401
from _lib.toggle import ToggleButton  # noqa: F401
from _lib.autotsad_filter import create_autotsad_filters  # noqa: F401
from streamlit.delta_generator import DeltaGenerator

from autotsad.database.autotsad_connection import AutoTSADConnection

st.set_page_config(
    page_title="Aggregated Results",
    page_icon="📦",
    layout="wide",
)

conn = st.experimental_connection("autotsad", type=AutoTSADConnection)


@st.cache_data
def plot_average_results(df: pd.DataFrame, method_order: List[str]) -> go.Figure:
    fig = px.violin(df, y="range_pr_auc", x="Method", color="Method Type", violinmode="overlay")
    fig.update_traces(meanline_visible=True)
    fig.update_yaxes(title="Range PR AUC", range=[-0.1, 1.1])
    fig.update_xaxes(categoryorder="array", categoryarray=method_order, tickangle=45)
    return fig


@st.cache_data
def plot_aggregated_results(df: pd.DataFrame, method_order: List[str], only_show_top: bool = False,
                            only_show_when_all: bool = False, agg: str = "mean") -> go.Figure:
    if only_show_top:
        baseline_methods = df[df["Method Type"] == "Baseline"]["Method"].unique().tolist()
        ranking_methods = [m for m in method_order if m not in baseline_methods]
        methods = [m for m in method_order if m in ranking_methods[:1] or m in baseline_methods]
        df = df[df["Method"].isin(methods)]

    df = df.groupby(["Dataset", "Method", "Method Type"]).agg(agg).reset_index()

    if only_show_when_all:
        df_tmp = df.groupby("Dataset")[["Method"]].nunique().reset_index()
        datasets = df_tmp.loc[df_tmp["Method"] == df_tmp["Method"].max(), "Dataset"].tolist()
        df = df[df["Dataset"].isin(datasets)]

    def sort_key(x):
        if x.name == "Dataset":
            return x
        else:
            return x.apply(method_order.index)

    df = df.sort_values(["Dataset", "Method"], key=sort_key)
    df["Dataset"] = df["Dataset"].str[:50]
    fig = px.bar(df, y="Dataset", x="range_pr_auc", color="Method", barmode="group",
                 color_discrete_map=ranking_method_color_map,
                 orientation="h",
                 height=5*df["Method"].nunique()*df["Dataset"].nunique())
    fig.update_xaxes(title="Range PR AUC", range=[-0.1, 1.1])
    return fig


# sidebar with page global filters
with st.sidebar:
    filters = create_autotsad_filters("aggregated-sidebar", conn, use_columns=False)

    st.write("Baselines")
    baselines = st.multiselect("Baseline", options=conn.list_baselines(), key="filter_form_baselines",
                               help="Select the baselines to include in the results. If empty, all baselines are "
                                    "included.")
    filters["baseline"] = baselines

# main content
st.write(f"# AutoTSAD Results")

available_methods = conn.list_available_methods()
results = conn.all_aggregated_results(filters)
# filter out results that are not meaningful (we don't describe them in the paper)
results = results[~results["Method"].str.endswith("_mean")]
results = results[~results["Method"].str.endswith("minmax_max")]
results = results[~results["Method"].str.startswith("interchange")]
results = results[~results["Method"].str.startswith("training-coverage")]
method_order = results.groupby("Method")[["range_pr_auc"]].mean() \
    .reset_index() \
    .sort_values("range_pr_auc", ascending=False)["Method"] \
    .tolist()

st.write("## Average performance of each algorithm")
st.plotly_chart(plot_average_results(results, method_order), use_container_width=True)

st.write("## Range PR AUC per Dataset")
only_show_top = st.checkbox("Only show top AutoTSAD ranking method", value=True)
only_show_when_all = st.checkbox("Only show datasets, for which all results exist", value=True)
aggregation = st.selectbox("Aggregation", ["mean", "median", "max", "min"])
st.plotly_chart(plot_aggregated_results(results, method_order, only_show_top, only_show_when_all, aggregation),
                use_container_width=True)

st.write("## Compare rankings")

datasets = conn.list_datasets()
datasets = pd.concat([pd.DataFrame([["", ""]], columns=["name", "collection"]), datasets], axis=0).set_index("name")
datasets = datasets.sort_index()
dataset = st.selectbox("Select Dataset", datasets.index,
                       format_func=lambda x: f"{datasets.loc[x, 'collection']} {x}")
if filters["baseline"]:
    methods = available_methods[
        (available_methods["Method Type"] == "Baseline") & (available_methods["Method"].isin(filters["baseline"]))
        | (available_methods["Method Type"].str.startswith("AutoTSAD"))
    ]
else:
    methods = available_methods
methods1: List[str] = [""] + (methods["Method Type"].str.cat(methods["Method"], sep=" - ")).tolist()
methods2 = methods1.copy()

left, right = st.columns(2)
ranking_method1 = left.selectbox("Select ranking method 1", methods1)
ranking_method2 = right.selectbox("Select ranking method 2", methods2)
metric = "range_pr_auc"

if ranking_method1 != "" and ranking_method2 != "" and ranking_method1 != ranking_method2:
    df_dataset = conn.load_dataset(dataset)


    def fill_column(rm: str, col: DeltaGenerator) -> None:
        method_type, method = rm.split(" - ")
        parts = method.split("_")
        if len(parts) == 3:
            rmethod, nmethod, amethod = parts
        else:
            rmethod, nmethod, amethod = parts[0], None, None

        quality = conn.method_quality(dataset, rmethod, nmethod, amethod, method_type, filters, metric)
        if quality is None or np.isnan(quality):
            col.warning(f"{rm} did not produce any results for the selected dataset and filters!")
            return

        col.metric(label=metric, value=quality)
        if rmethod == "mean-algo":
            return

        df_combined_scores = conn.load_aggregated_scoring_for(dataset, rmethod, nmethod, amethod, filters)
        col.plotly_chart(plot_aggregated_scores(df_dataset, df_combined_scores,
                                                ranking_method=rmethod,
                                                normalization_method=nmethod,
                                                aggregation_method=amethod), use_container_width=True)

        if rmethod not in ["k-Means", "SAND", "k-Means (TimeEval)", "SAND (TimeEval)", "best-algo"]:
            expander_bt = ToggleButton(col, "rankings", key=f"expander-{rm}", use_container_width=True)
            container = col.empty()
            if expander_bt.state:
                df_ranking, df_scores = conn.load_ranking_results(dataset, rmethod, nmethod, amethod, method_type, filters)
                container.plotly_chart(plot_ranking(df_ranking, df_scores,
                                       title=rm, normalization_method=nmethod), use_container_width=True)


    fill_column(ranking_method1, left)
    fill_column(ranking_method2, right)
