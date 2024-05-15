import _lib.preamble  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st
from _lib.autotsad_filter import create_autotsad_filters  # noqa: F401
from _lib.plotly_plotting import plot_aggregated_scores, add_baseline_aggregated_scores, plot_ranking  # noqa: F401

from autotsad.config import ALGORITHM_SELECTION_METHODS, SCORE_NORMALIZATION_METHODS, SCORE_AGGREGATION_METHODS
from autotsad.database.autotsad_connection import AutoTSADConnection
from autotsad.system.execution.aggregation import normalize_scores, aggregate_scores

st.set_page_config(
    page_title="Algorithm Scoring",
    page_icon="📈",  # 💯, 📈, 📉, 🗠
    layout="wide",
)

conn = st.connection("autotsad", type=AutoTSADConnection)

st.write("# AutoTSAD Results")

filters = create_autotsad_filters("autotsad-results", conn=conn)

st.write("## Select AutoTSAD execution")
c0, c1, c2, c3 = st.columns(4)
datasets = conn.list_datasets(filters)
datasets = pd.concat([
    pd.DataFrame([["", ""]], columns=["name", "collection"]),
    datasets
], axis=0).set_index("name")
dataset = c0.selectbox("Dataset", datasets.index,
                       format_func=lambda x: f"{datasets.loc[x, 'collection']} {x}")
ranking_method = c1.selectbox("Ranking method", filters["ranking_method"] or ALGORITHM_SELECTION_METHODS)
normalization_method = c2.selectbox("Normalization method", filters["normalization_method"] or SCORE_NORMALIZATION_METHODS)
aggregation_method = c3.selectbox("Aggregation method", filters["aggregation_method"] or SCORE_AGGREGATION_METHODS)

if dataset == "":
    st.error("No executions found. Please select a dataset.")
    st.stop()

df_exec = conn.query(f"""select e.id, e.autotsad_version, c.description as "config", e.ranking_method,
        e.normalization_method, e.aggregation_method, e.range_pr_auc as "Range-PR-AUC", e.runtime, e.experiment_id
    from autotsad_execution e, dataset d, configuration c
    where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and d.name = '{dataset}'
        and e.ranking_method = '{ranking_method}'
        and e.normalization_method = '{normalization_method}'
        and e.aggregation_method = '{aggregation_method}' {conn._build_autotsad_filter_clauses(filters)};""")
df_exec["runtime"] = df_exec["runtime"].astype(np.float_)

if df_exec.shape[0] == 0:
    st.error("No executions found. Please change the filters.")
    st.stop()

elif df_exec.shape[0] > 1:
    st.warning("Multiple executions found... please select one!")
    col1, col2 = st.columns(2)
    col1.write("Available executions:")
    col1.write(df_exec)
    df_tmp = df_exec.set_index("id")
    execution_id = col2.selectbox("Select execution", df_tmp.index,
                                  format_func=lambda i: f"{i} - {df_tmp.loc[i, 'autotsad_version']} {df_tmp.loc[i, 'config'][:25]}...")
    df_exec = df_exec[df_exec["id"] == execution_id]

execution = df_exec.iloc[0, :]

baseline = st.selectbox("Compare to baseline", ["None"] + sorted(conn.list_baselines()))
baseline_execution = None
if baseline != "None":
    df_tmp = conn.query(f"""select b.id, b.name, b.range_pr_auc as "Range-PR-AUC", b.runtime
        from baseline_execution b, dataset d
        where b.dataset_id = d.hexhash
            and d.name = '{dataset}'
            and b.name = '{baseline}';""")
    if df_tmp.shape[0] > 0:
        baseline_execution = df_tmp.iloc[0, :]
        if np.isnan(baseline_execution["Range-PR-AUC"]):
            st.warning(f"Baseline {baseline} did not produce any results for the selected dataset!")
            baseline_execution = None
    else:
        st.warning(f"Baseline {baseline} not found!")

st.write("## Results")
col1, col2 = st.columns(2)
delta = None
if baseline_execution is not None:
    delta = np.round(execution["Range-PR-AUC"] - baseline_execution["Range-PR-AUC"], 4)
col1.metric("Range-PR-AUC", np.round(execution["Range-PR-AUC"], 4), delta=delta)
delta = None
if baseline_execution is not None and baseline_execution["runtime"] is not None:
    delta = np.round(execution["runtime"] - baseline_execution["runtime"])
col2.metric("Runtime (s)", np.round(execution["runtime"]), delta=delta, delta_color="inverse")

# load data for plots
name = f"{dataset} - {ranking_method}-{normalization_method}-{aggregation_method}"
df_ranking = conn.load_ranking_results(dataset, rmethod=execution, method_type="AutoTSAD", filters=filters)
df_dataset = conn.load_dataset(dataset)

# compute combined scores
scores = df_ranking[1].pivot(index="time", columns="algorithm_scoring_id", values="score").values
scores = normalize_scores(scores, normalization_method=normalization_method)
combined_score = aggregate_scores(scores, agg_method=aggregation_method)
df_combined_scores = pd.DataFrame({"time": df_ranking[1]["time"].unique(), "score": combined_score})

st.write("## Combined anomaly score")
fig = plot_aggregated_scores(df_dataset, df_combined_scores, ranking_method, normalization_method, aggregation_method, title=name)
if baseline_execution is not None and baseline != "mean-algo":
    df_baseline_ranking = conn.load_ranking_results(dataset, rmethod=baseline, method_type="Baseline", filters=filters)
    add_baseline_aggregated_scores(fig, df_baseline_ranking[0], df_baseline_ranking[1], baseline, row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.write("## Algorithm ranking")
st.plotly_chart(plot_ranking(df_ranking[0], df_ranking[1], df_dataset, title=name, normalization_method=normalization_method), use_container_width=True)
