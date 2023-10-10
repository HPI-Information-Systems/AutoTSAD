from typing import List, Any, Dict

import _lib.preamble  # noqa: F401
import pandas as pd
import numpy as np
import streamlit as st
from _lib.plt_plotting import (  # noqa: F401
    plot_quality, plot_runtime_traces, plot_optimization_scaling, plot_optimization_improvement,  # noqa: F401
    plot_ensembling_strategies, plot_ranking_comparison  # noqa: F401
)
from timeeval.metrics import RangePrAUC, RangeRocAUC

from autotsad.database.autotsad_connection import AutoTSADConnection
from autotsad.system.execution.aggregation import aggregate_scores

st.set_page_config(
    page_title="Super Ensemble",
    page_icon="🦸",
    layout="wide",
)

conn = st.experimental_connection("autotsad", type=AutoTSADConnection)
filters = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["26a8aa1409ae87761d9d405ff0e49f9e"],
    "baseline": ["best-algo", "k-Means (TimeEval)", "SAND (TimeEval)", "mean-algo"],
}
metric_options = ["range_pr_auc", "range_roc_auc"]
aggregation_options = ["mean", "median", "min"]
sort_metric = "range_pr_auc"
sort_aggregation = "mean"

# load data
available_methods = conn.list_available_methods()
method_options = available_methods[available_methods["Method Type"] != "Baseline"]["Method"].tolist()
results = conn.all_aggregated_results(filters, only_paper_datasets=True, only_paper_methods=True,
                                      exclude_super_ensembles=False)


############################################################
st.write("# Top-k mean aggregation")
st.write("Can we avoid the need to select the best ensembling method by using a top-k mean aggregation on the rank "
         "aggregated ensembling methods?")

# present info for best method selection

df_tmp = results[results["Method Type"] != "Baseline"].copy()
method_order = df_tmp.groupby("Method")[[sort_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(sort_metric, ascending=False) \
    .reset_index(drop=True)
method_order = method_order["Method"].tolist()
fig = plot_ensembling_strategies(df_tmp, method_order)


sorted_super_ensembling_methods = results[results["Method Type"] == "AutoTSAD Ensemble"]\
    .groupby("Method")[[sort_metric]]\
    .agg(sort_aggregation) \
    .reset_index() \
    .sort_values(sort_metric, ascending=False)["Method"].tolist()
top_super_methods = []
top_super_ensembling_methods_text = "Top super ensembling methods:\n"
for i, method in enumerate(sorted_super_ensembling_methods[:6]):
    top_super_ensembling_methods_text += f"{i + 1}. {method}\n"
    top_super_methods.append(method)

c0, c1 = st.columns(2)
with c0:
    st.write(top_super_ensembling_methods_text)
with c1:
    st.write(fig)


# select best methods
def clear() -> None:
    st.session_state["best-methods-select"] = []


def use_top_super_methods() -> None:
    st.session_state["best-methods-select"] = top_super_methods


c0, c1 = st.columns(2)
selected_methods = c0.multiselect("Select methods", method_options, key="best-methods-select")
c1.button("Use top super ensembling methods", on_click=use_top_super_methods, use_container_width=True)
c1.button("Clear", on_click=clear, use_container_width=True)

if len(selected_methods) == 0:
    st.stop()


# load rankings and compute super ranking
range_pr_auc_metric = RangePrAUC(buffer_size=100)
range_roc_auc_metric = RangeRocAUC(buffer_size=100)
super_method_name = "Aggregated Super Ensemble"
super_method_type = "AutoTSAD Ensemble"
datasets = results[results["Method"].isin(selected_methods)]["Dataset"].unique()


@st.cache_data(persist=True, max_entries=5)
def compute_super_ranking_results(selected_methods: List[str], filters: Dict[str, List[Any]]) -> pd.DataFrame:
    df_super = pd.DataFrame(index=datasets, columns=results.columns[1:])
    df_super.index.name = "Dataset"
    df_super["Method"] = super_method_name
    df_super["Method Type"] = super_method_type

    bar = st.progress(0.0, "Computing super ranking results...")
    for pi, dataset in enumerate(df_super.index):
        dataset_collection, dataset_name = dataset.split(" ")
        # load dataset
        df_dataset = conn.load_dataset(dataset_name)

        # load combined scores of each ranking method
        combined_scores = np.full((df_dataset.shape[0], len(selected_methods)), fill_value=np.nan, dtype=np.float_)
        for i, ranking_method in enumerate(selected_methods):
            tmp = results.loc[(results["Dataset"] == dataset) & (results["Method"] == ranking_method), "range_pr_auc"].isna()
            if len(tmp) == 0 or tmp.item():
                st.warning(f"{dataset_name} {ranking_method}: skipping!")
                continue

            rmethod, nmethod, amethod = ranking_method.split("_")
            df_combined_score = conn.load_aggregated_scoring_for(
                dataset=dataset_name, rmethod=rmethod, nmethod=nmethod, amethod=amethod, filters=filters
            )
            combined_scores[:, i] = df_combined_score["score"].values

        # compute super ranking (by mean)
        super_scoring = np.nanmean(combined_scores, axis=1)

        # score super ranking
        rprauc = range_pr_auc_metric(df_dataset["is_anomaly"].values, super_scoring)
        rrocauc = range_roc_auc_metric(df_dataset["is_anomaly"].values, super_scoring)
        df_super.loc[dataset, "range_pr_auc"] = rprauc
        df_super.loc[dataset, "range_roc_auc"] = rrocauc
        bar.progress(pi/len(df_super.index), "Computing super ranking results...")
    bar.progress(1.0, "Computing super ranking results...")
    return df_super.reset_index()


df_super = compute_super_ranking_results(selected_methods, filters)
st.write(df_super)

df_top1 = conn.compute_top1_baseline(filters, only_paper_datasets=True)
df_quality = pd.concat([results, df_super, df_top1], axis=0, ignore_index=True)
method_order = df_quality.groupby("Method")[[sort_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(sort_metric, ascending=False) \
    .reset_index(drop=True)
method_order = method_order["Method"].tolist()
fig = plot_ensembling_strategies(df_quality, method_order)
st.write(fig)

default_filters = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],  # default ensemble
}
df_default_super = compute_super_ranking_results(selected_methods, default_filters)

df_improvement = df_super.set_index(["Dataset", "Method Type", "Method"]).sort_index()
df_tmp = df_default_super.set_index(["Dataset", "Method Type", "Method"]).sort_index()
st.write(df_tmp)
df_improvement -= df_tmp
st.write(df_improvement)
fig = plot_optimization_improvement(df_improvement,
                                    save_files_path=None,
                                    name="super-ensemble-improvement")
st.write(fig)
