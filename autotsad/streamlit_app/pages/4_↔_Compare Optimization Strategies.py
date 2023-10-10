from typing import List, Dict, Any

import _lib.preamble  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st
from _lib.plt_plotting import plot_optimization_improvement  # noqa: F401

from autotsad.database.autotsad_connection import AutoTSADConnection

st.set_page_config(
    page_title="Optimization Strategies",
    page_icon="↔",
    layout="wide",
)

conn = st.experimental_connection("autotsad", type=AutoTSADConnection)

best_ensemble_method = "aggregated-minimum-influence_gaussian_custom"
default_filters = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
}
timeeval_filters = {
    "autotsad_version": ["0.2.2-timeeval"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
}
timeeval_optimized_filters = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["26a8aa1409ae87761d9d405ff0e49f9e"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
}
bad_filters = {
    "autotsad_version": ["0.2.2-bad"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
}
bad_optimizated_filters = {
    "autotsad_version": ["0.2.2-bad"],
    "config_id": ["26a8aa1409ae87761d9d405ff0e49f9e"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
}


st.write("# Compare AutoTSAD Optimization Strategies")


def load_improvements(filters_default: Dict[str, List[Any]], filters_optimized: Dict[str, List[Any]]) -> pd.DataFrame:
    default_filter_clauses = conn._build_autotsad_filter_clauses(filters_default)
    optimized_filter_clauses = conn._build_autotsad_filter_clauses(filters_optimized)

    sub_query = """select distinct d.collection || ' ' || d.name as "dataset", e.range_pr_auc
                   from autotsad_execution e, dataset d, configuration c
                   where e.dataset_id = d.hexhash
                     and e.config_id = c.id"""
    query = f"""select a.dataset, a.range_pr_auc - b.range_pr_auc as "improvement"
                from ({sub_query} and d.paper = True {optimized_filter_clauses}) a inner join
                     ({sub_query} and d.paper = True {default_filter_clauses}) b
                     on a.dataset = b.dataset
                order by dataset;"""
    # print(query)
    return conn.query(query)


df_improvement = load_improvements(default_filters, timeeval_optimized_filters)
df_improvement = df_improvement.rename(columns={"improvement": "original (default → timeeval-optim)"})
df_improvement["timeeval → timeeval-optim"] = load_improvements(timeeval_filters, timeeval_optimized_filters)["improvement"]
df_improvement["bad → bad-optim"] = load_improvements(bad_filters, bad_optimizated_filters)["improvement"]
df_improvement["timeeval → bad-optim"] = load_improvements(timeeval_filters, bad_optimizated_filters)["improvement"]
df_improvement["bad → timeeval"] = load_improvements(bad_filters, timeeval_filters)["improvement"]
df_improvement["default → timeeval"] = load_improvements(default_filters, timeeval_filters)["improvement"]
df_improvement = df_improvement.set_index("dataset").sort_index()
df_improvement = df_improvement.astype(np.float_)

# remove non-yet ready results
df_improvement = df_improvement.dropna(axis=1, thresh=0.7)

st.write("Improvement of the different optimization strategies compared to others:")
st.dataframe(df_improvement.style.background_gradient(cmap="inferno", low=0.0, high=1.0, vmin=-1.0, vmax=1.0), use_container_width=True)
df_improvement = df_improvement.fillna(.0)

fig = plot_optimization_improvement(df_improvement, save_files_path=None, name="test")
st.write(fig)
