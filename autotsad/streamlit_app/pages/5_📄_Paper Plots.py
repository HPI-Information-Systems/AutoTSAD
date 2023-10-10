from copy import deepcopy
from functools import reduce
from typing import Dict, List, Any

import _lib.preamble  # noqa: F401
import numpy as np
import pandas as pd
import streamlit as st
from _lib.plt_plotting import (  # noqa: F401
    plot_quality, plot_runtime_traces, plot_optimization_scaling, plot_optimization_improvement,  # noqa: F401
    plot_ensembling_strategies, plot_ranking_comparison  # noqa: F401
)

from autotsad.database.autotsad_connection import AutoTSADConnection

st.set_page_config(
    page_title="Paper Plots",
    page_icon="📄",
    layout="wide",
)

conn = st.experimental_connection("autotsad", type=AutoTSADConnection)
filters = {
    "autotsad_version": ["0.2.2-timeeval"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "baseline": ["best-algo", "k-Means (TimeEval)", "SAND (TimeEval)", "mean-algo"]
}
metric_options = ["range_pr_auc", "range_roc_auc"]
aggregation_options = ["mean", "median", "min"]
sort_metric = "range_pr_auc"
sort_aggregation = "median"

best_method = "training-quality_gaussian_custom"
best_ensemble_method = "aggregated-minimum-influence_gaussian_custom"

traces = ["autotsad-%-Base TS generation", "autotsad-%-Cleaning", "autotsad-%-Limiting",
          "autotsad-%-Anomaly injection", "autotsad-%-Optimization-%-Sensitivity analysis",
          "autotsad-%-Optimization-%-Hyperparams opt.", "autotsad-%-Optimization-%-Selecting best performers",
          "autotsad-%-Execution-%-Algorithm Execution", "autotsad-%-Execution-%-Computing all combinations"]
scaling_datasets = ["-69_2_0.02_15", "022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4"]
scaling_configs = ["paper v1 - n_jobs=1", "paper v1 - n_jobs=5", "paper v1 - n_jobs=10", "paper v1", "paper v1 - n_jobs=40"]
datagen_traces = ["autotsad-%-Base TS generation", "autotsad-%-Cleaning", "autotsad-%-Limiting", "autotsad-%-Anomaly injection"]
optimization_experiment_datasets = [
    "KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727",  # TSB-UAD-synthetic
    "A4Benchmark-13",  # WebscopeS5
    # "c69a50cf-ee03-3bd7-831e-407d36c7ee91",  # IOPS (too large)
]
optimization_experiment_configs = [
    "paper v1",
    "paper v1 optimization / variance%",
    "paper v1 - default ensemble%",
]
ranking_example_dataset = "A4Benchmark-2"
ranking_example_method = "affinity-propagation-clustering_gaussian_max"

# load data
available_methods = conn.list_available_methods()["Method"].tolist()
results = conn.all_aggregated_results(filters,
                                      only_paper_datasets=True,
                                      only_paper_methods=True,
                                      exclude_super_ensembles=False)

best_methods = []
best_ranking_methods_text = "Best methods:\n"
for metric in metric_options:
    for agg in aggregation_options:
        method = results[results["Method Type"] == "AutoTSAD"].groupby("Method")[[metric]].agg(agg) \
            .reset_index() \
            .sort_values(metric, ascending=False)["Method"] \
            .tolist()[0]
        best_methods.append(method)
        best_ranking_methods_text += f"- best {agg} in {metric}: {method}\n"

# main content
st.write(f"# AutoTSAD Results for the Paper")
with st.expander("Fixed configuration"):
    c0, c1 = st.columns(2)
    with c0:
        st.write("Global configuration for the paper (fixed):")
        configuration = deepcopy(filters)
        del configuration["baseline"]
        configuration["only_paper_datasets"] = True
        st.write(configuration)
        st.write(f"""
            Notes:

            - excluding _minmax_ normalization combined with _max_ aggregation
            - excluding _mean_ normalization strategy
            - excluding all _interchange_ and _training-coverage_ ranking strategies
            - excluding all _aggregated-robust-borda_ ranking strategies

            AutoTSAD configuration for final output:

            - Metric: {sort_metric}
            - Best ensembling method: {best_ensemble_method}

            Optimization experiment configuration:

            - Dataset: {', '.join(optimization_experiment_datasets)}
            - Configs: {', '.join(optimization_experiment_configs)}
            - Method: {best_ensemble_method} (best method)

            Ranking example:

            - Dataset: {ranking_example_dataset}
            - Method: {best_method} (best method)
            - Comparison method: {ranking_example_method}
        """)

    with c1:
        st.write("Runtime configuration:")
        st.write({
            "traces": traces,
            "datasets": scaling_datasets,
            "additional_configs": scaling_configs,
            "group_traces": datagen_traces,
            "only_paper_datasets": True,
        })

st.write("## Datasets")

df_datasets = conn.dataset_collections_for_paper()
df_datasets = df_datasets.set_index("collection")
st.dataframe(df_datasets.T, use_container_width=True)

st.write("## Quality Comparison")

# include max quality of best methods in the plot
# df_quality = results[(results["Method Type"] == "Baseline") | (results["Method"] == best_method)]
# df_max = results[results["Method"].isin(best_methods)].copy()
# df_max = df_max.groupby("Dataset")[["range_pr_auc"]].max()
# df_max["Method"] = "max-best-ensembling-methods"
# df_max["Method Type"] = "AutoTSAD"
# df_max = df_max.reset_index()
# df_quality = pd.concat([df_quality, df_max], ignore_index=True)

df_quality = results[results["Method Type"] != "AutoTSAD"].copy()
method_order = df_quality.groupby("Method")[[sort_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(sort_metric, ascending=False)["Method"] \
    .tolist()

df_autotsad_results = df_quality[df_quality["Method"] == best_ensemble_method]
st.write(f"Unprocessed datasets by AutoTSAD: {(df_autotsad_results[sort_metric].fillna(0.) <= 0.0001).sum()}")
st.write(f"Unprocessed datasets by SAND: {(df_quality[df_quality['Method'] == 'SAND (TimeEval)'][sort_metric].fillna(0.) <= 0.0001).sum()}")

threshold = 0.3
st.write(f"Datasets where AutoTSAD has a {sort_metric} < {threshold}:")
st.write(df_autotsad_results[df_autotsad_results[sort_metric] < threshold].sort_values(sort_metric))

fig = plot_quality(df_quality, method_order)
st.write(fig)

st.write("Results on the individual datasets:")
df_table = df_quality.pivot(index="Dataset", columns="Method", values="range_pr_auc")
df_table = df_table.sort_index()
df_table = df_table[method_order]
st.dataframe(df_table.style.background_gradient(cmap="inferno", low=0.0, high=1.0, vmin=0.0, vmax=1.0), use_container_width=True)
st.warning("Note: Unfortunately, SAND could not process many of the datasets.")

st.write("## Runtime")

# scaling_configs = ["paper v1 - default ensemble (no optimization, seed=1)"]
df_runtime_trace = conn.load_runtime_traces(traces, scaling_datasets, scaling_configs)
# done by database:
# df_runtime_trace = df_runtime_trace.sort_values(["dataset_name", "n_jobs", "position"])
trace_datasets = df_runtime_trace["dataset_name"].unique().tolist()
optim_filters = deepcopy(filters)
optim_filters["autotsad_version"] = ["0.2.1"]
optim_filters["config_id"] = ["26a8aa1409ae87761d9d405ff0e49f9e"]
df_runtime_baselines = conn.load_mean_runtime(optim_filters=optim_filters, default_filters=filters, only_paper_datasets=True)
df_runtime_baselines = df_runtime_baselines.sort_values("mean_runtime")

st.write("Mean runtime of baselines and AutoTSAD on all datasets (in seconds):")
st.write(df_runtime_baselines.set_index("name").T)

fig = plot_runtime_traces(df_runtime_trace, group_traces={
    "Data Generation Module": datagen_traces,
    "Opt. Seeding & HPO & Pruning": ["autotsad-%-Optimization-%-Sensitivity analysis", "autotsad-%-Optimization-%-Hyperparams opt."],
})
st.write(fig)

dataset_mapping = {chr(65+i): d for i, d in enumerate(trace_datasets)}
st.write("Datasets:", dataset_mapping)

runtime_filters = deepcopy(filters)
runtime_filters["config_id"] = ["e0b0b7602b803e4e4b9110edcd55440b"]
runtime_filters["autotsad_version"] = ["0.2.2-timeeval"]
df_runtime = conn.load_runtimes(runtime_filters, only_paper_datasets=True)
df_runtime["factor"] = df_runtime["autotsad_runtime"] / df_runtime["algorithm_execution_runtime"]

st.write("Runtime of single-threaded AutoTSAD (without HPO) in relation to the algorithm execution step:")
if runtime_filters["autotsad_version"] != ["0.2.2-timeeval"]:
    st.error(":warning: Change AutoTSAD version to '0.2.2-timeeval' to get the correct results!")
st.write(df_runtime[["autotsad_runtime", "algorithm_execution_runtime", "factor"]].mean())

###### tmp traces for single-threaded runtime
trace_list = ",".join([f"'{t}'" for t in traces])
dataset_name_list = ",".join([f"'{d}'" for d in scaling_datasets])
config_name_list = ",".join([f"'{c}'" for c in ["paper v1 - default ensemble (no optimization, seed=1)"]])

query = f"""select a.name as "dataset_name", t.trace_name,
                   t.position, t.duration_ns / 1e9 as "runtime"
            from runtime_trace t,
                 (select distinct e.experiment_id, d.name
                  from dataset d, configuration c, autotsad_execution e
                  where e.dataset_id = d.hexhash
                    and e.config_id = c.id
                    and c.description in ({config_name_list})
                    and d.name in ({dataset_name_list})
                    and d.paper = True) a
            where t.experiment_id = a.experiment_id
              and trace_name in ({trace_list})
              and trace_type = 'END'
            order by a.name, t.position
            ;"""
df_st_traces = conn.query(query)
df_st_traces["n_jobs"] = 1

fig = plot_runtime_traces(df_st_traces, group_traces={
    "Data Generation Module": datagen_traces,
    "Opt. Seeding & HPO & Pruning": ["autotsad-%-Optimization-%-Sensitivity analysis", "autotsad-%-Optimization-%-Hyperparams opt."],
})
st.write(fig)
######

st.write("## Training Data Generation Strategies")
st.info("currently not available (not in paper)")

st.write("## Optimization Strategies")

# optimization variance experiment
filters_optimization = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]]
}
df_optimization_variance = conn.load_optimization_variance_results(
    filters_optimization,
    datasets=optimization_experiment_datasets,
    config_names=optimization_experiment_configs,
)

df_optimization_variance = df_optimization_variance.groupby(["collection", "dataset", "max_trials"])[["range_pr_auc"]].agg(["mean", "std", "count"]).reset_index()
df_optimization_variance.columns = ["collection", "dataset", "max_trials", "mean", "std", "n_seeds"]
df_optimization_variance["dataset"] = df_optimization_variance["collection"] + " " + df_optimization_variance["dataset"]
df_optimization_variance = df_optimization_variance.drop(columns=["collection"])
df_optimization_variance = df_optimization_variance.sort_values(["dataset", "max_trials"])

st.write(f"Performance and variance of {best_ensemble_method} (with optimization starting from TimeEval-hyperparameters) over different number of trials:")
c0, c1 = st.columns(2)
with c0:
    st.write(df_optimization_variance[["dataset", "max_trials", "n_seeds"]])
with c1:
    fig = plot_optimization_scaling(df_optimization_variance, truncate_labels_to=45)
    st.write(fig)

# optimization improvements compared to different default hyperparameter baselines:
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


df_improvement = load_improvements(timeeval_filters, timeeval_optimized_filters)
df_improvement = df_improvement.rename(columns={"improvement": "timeeval → timeeval-optim"})
df_improvement["naive → naive-optim"] = load_improvements(bad_filters, bad_optimizated_filters)["improvement"]
df_improvement["timeeval → naive-optim"] = load_improvements(timeeval_filters, bad_optimizated_filters)["improvement"]
df_improvement = df_improvement.set_index("dataset").sort_index()
df_improvement = df_improvement.astype(np.float_)
df_improvement = df_improvement.fillna(.0)

text = f"Improvement of {best_ensemble_method} for different ensembles:\n"
for c in df_improvement.columns:
    text += f"- {c}: {df_improvement[c].mean():.4f} (median: {df_improvement[c].median():.4f})\n"
st.write(text)
fig = plot_optimization_improvement(df_improvement)
st.write(fig)

st.write("## Ensembling Strategies")

# df_tmp = results[(results["Method"].isin(best_methods)) | (results["Method Type"] == "Baseline")].copy()
top1_results = conn.compute_top1_baseline(filters, only_paper_datasets=True)
top1_mean = top1_results[[sort_metric]].agg(sort_aggregation).item()
df_tmp = pd.concat([results, top1_results], ignore_index=True)
df_tmp = df_tmp[df_tmp["Method Type"] != "Baseline"]
# df_tmp = df_tmp[~((df_tmp["Method Type"] == "AutoTSAD Ensemble") & (df_tmp["Method"] != best_ensemble_method))]
method_order = df_tmp.groupby("Method")[[sort_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(sort_metric, ascending=False) \
    .reset_index(drop=True)
method_order = method_order["Method"].tolist()

c0, c1 = st.columns(2)
with c0:
    st.write(best_ranking_methods_text)
    st.write(df_tmp.groupby("Method")[[sort_metric]].agg({sort_metric: ["mean", "median"]}))

with c1:
    fig = plot_ensembling_strategies(df_tmp, method_order)
    st.write(fig)


st.write("Example rankings, where another ranking is better than our top-ranking:")
st.write(f"Dataset: {ranking_example_dataset}")

# load both rankings
# best_method = best_ensemble_method
df_dataset = conn.load_dataset(ranking_example_dataset)
quality1 = conn.method_quality(
    dataset=ranking_example_dataset,
    rmethod=best_ensemble_method.split("_")[0],
    nmethod=best_ensemble_method.split("_")[1],
    amethod=best_ensemble_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters,
)
df_ranking1 = conn.load_ranking_results(
    dataset=ranking_example_dataset,
    rmethod=best_ensemble_method.split("_")[0],
    nmethod=best_ensemble_method.split("_")[1],
    amethod=best_ensemble_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters
)
quality2 = conn.method_quality(
    dataset=ranking_example_dataset,
    rmethod=ranking_example_method.split("_")[0],
    nmethod=ranking_example_method.split("_")[1],
    amethod=ranking_example_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters
)
df_ranking2 = conn.load_ranking_results(
    dataset=ranking_example_dataset,
    rmethod=ranking_example_method.split("_")[0],
    nmethod=ranking_example_method.split("_")[1],
    amethod=ranking_example_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters
)
c0, c1 = st.columns(2)
c0.metric(f"Range-PR-AUC {best_method.replace('_', ' ')}", quality1)
c1.metric(f"Range-PR-AUC {ranking_example_method.replace('_', ' ')}", quality2, delta=quality2-quality1)
fig = plot_ranking_comparison(df_dataset, df_ranking1, df_ranking2,
                              quality_left=quality1,
                              quality_right=quality2,
                              method_left=best_ensemble_method,
                              method_right=ranking_example_method,
                              show_anomalies=True, show_algo_quality=True)
st.write(fig)
