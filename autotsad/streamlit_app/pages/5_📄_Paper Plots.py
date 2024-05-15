from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats

import _lib.preamble  # noqa: F401
from _lib.plt_plotting import (  # noqa: F401
    plot_quality, plot_runtime_traces, plot_optimization_scaling, plot_optimization_improvement,  # noqa: F401
    plot_ensembling_strategies, plot_ranking_comparison, plot_runtime_traces_new  # noqa: F401
)

from autotsad.database.autotsad_connection import AutoTSADConnection, DB_METRIC_NAME_MAPPING

###############################################################################
# CONFIGURATION AND DATA
###############################################################################
DB_REVERSE_METRIC_NAME_MAPPING = dict((m1, m0) for m0, m1 in DB_METRIC_NAME_MAPPING.items())

st.set_page_config(
    page_title="Paper Plots",
    page_icon="📄",
    layout="wide",
)

conn = st.connection("autotsad", type=AutoTSADConnection)
filters = {
    "autotsad_version": ["0.2.2-timeeval"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "baseline": [
        "best-algo", "select-horizontal", "select-vertical", "k-Means (TimeEval)", "tsadams-mim", "cae-ensemble"
    ]
}
metric_options = ["range_pr_auc", "range_roc_auc"]
aggregation_options = ["mean", "median", "min"]
sort_metric = "range_pr_auc"
sort_aggregation = "mean"
tsadams_n_datasets = 40
# tsadams_n_datasets = 74  # including revision datasets

best_method = "mmq-annotation-overlap_gaussian_mean"
best_ensemble_method = "aggregated-minimum-influence_gaussian_mean"

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
ranking_example_method = "affinity-propagation-clustering_gaussian_mean"

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

###############################################################################
# PRINT CONFIGURATION
###############################################################################
st.write(f"# AutoTSAD Results for the Paper")
options = list(DB_METRIC_NAME_MAPPING.values())
evaluation_metric = st.selectbox("Evaluation Metric", options, index=options.index(sort_metric))
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
            - excluding all _interchange_ and _training-coverage_ ranking strategies
            - excluding all _aggregated-robust-borda_ ranking strategies

            AutoTSAD configuration for final output:

            - Metric: {evaluation_metric}
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

df_datasets: pd.DataFrame = conn.dataset_collections_for_paper()
df_datasets = df_datasets.pivot(index="revision", columns="collection", values="datasets")
st.dataframe(df_datasets)

###############################################################################
# Quality Comparison
###############################################################################
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
if evaluation_metric == "range_pr_auc":
    df_top1_results = conn.compute_top1_baseline(filters, only_paper_datasets=True)
    df_quality = pd.concat([df_quality, df_top1_results], ignore_index=True)
else:
    df_quality = df_quality.copy()
method_order = df_quality.groupby("Method")[[evaluation_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(evaluation_metric, ascending=False)["Method"] \
    .tolist()

df_autotsad_results = df_quality[df_quality["Method"] == best_ensemble_method]
threshold = 0.4
st.write(f"Datasets where AutoTSAD has a {evaluation_metric} < {threshold}:")
st.write(df_autotsad_results[df_autotsad_results[evaluation_metric] < threshold].sort_values(evaluation_metric))

c0, c1, c2 = st.columns(3)
custom_method_order = c0.checkbox("Custom method order", value=False,
                                  help="Group methods by type: ensemble, selection (top 1), individual methods")
show_success_rate = c1.checkbox("Show success rate", value=True)
show_method_type = c2.checkbox("Show method type", value=True)
if custom_method_order:
    method_order = [
        best_ensemble_method, best_ensemble_method.replace("_mean", "_max"), "select-horizontal",
        "select-vertical", "best-algo", "top-1", "tsadams-mim", "k-Means (TimeEval)",
        # "SAND (TimeEval)",
    ]
fig = plot_quality(df_quality, method_order,
                   metric=evaluation_metric,
                   show_success_rate=show_success_rate,
                   show_algo_type=show_method_type,
                   tsadams_n_datasets=tsadams_n_datasets)
st.write(fig)

df_table = df_quality.pivot(index="Dataset", columns="Method", values=evaluation_metric)
df_table = df_table.sort_index()
df_table = df_table[method_order]
all_datasets = df_table.shape[0]
if not show_success_rate:
    # df_table = df_table.fillna(0.) > 0.0001
    df_table = ~df_table.isna()
    st.write(f"""Processed datasets:
- AutoTSAD: {df_table[best_ensemble_method].sum()} / {all_datasets} (ERRORS)
- SELECT horizontal: {df_table['select-horizontal'].sum()} / {all_datasets} (ERRORS)
- SELECT vertical: {df_table['select-vertical'].sum()} / {all_datasets} (ERRORS)
- tsadams-mim: {df_table['tsadams-mim'].sum()} / {tsadams_n_datasets}
- k-Means: {df_table['k-Means (TimeEval)'].sum()} / {all_datasets}""")

if "SAND (TimeEval)" in df_table.columns:
    if not show_success_rate:
        st.write(f"- SAND: {df_table['SAND (TimeEval)'].sum()} / {all_datasets}")
    st.warning("Note: Unfortunately, SAND could not process many of the datasets.")
st.warning(f"Note: We executed tsadams only on {tsadams_n_datasets} datasets, for which we have (normal) training data "
           "available because tsadams uses semi-supervised base algorithms.")
st.warning(f"Note: We executed cae-ensemble only on 86 datasets because our compute infrastructure was down. "
           "CAE-Ensemble exceeded the time limit for 13 of the datasets despite its GPU-usage and a limited "
           "hyperparameter search of only 10 settings.")

st.write("Results on the individual datasets:")
st.dataframe(df_table.style.background_gradient(cmap="inferno", low=0.0, high=1.0, vmin=0.0, vmax=1.0), use_container_width=True)

###############################################################################
# RUNTIME
###############################################################################
st.write("## Runtime")

df_runtime_trace = conn.load_runtime_traces(traces, scaling_datasets, scaling_configs, autotsad_versions=["0.2.1"])
# done by database:
# df_runtime_trace = df_runtime_trace.sort_values(["dataset_name", "n_jobs", "position"])
trace_datasets = df_runtime_trace["dataset_name"].unique().tolist()
default_filters = deepcopy(filters)
default_filters["ranking_method"] = ["affinity-propagation-clustering"]
default_filters["normalization_method"] = ["gaussian"]
default_filters["aggregation_method"] = ["custom"]
optim_filters = deepcopy(default_filters)
optim_filters["autotsad_version"] = ["0.2.1"]
optim_filters["config_id"] = ["26a8aa1409ae87761d9d405ff0e49f9e"]
df_runtime_baselines = conn.load_mean_runtime(
    optim_filters=optim_filters,
    default_filters=default_filters,
    only_paper_datasets=True
)
df_runtime_baselines = df_runtime_baselines.sort_values("mean_runtime")

st.write("Mean runtime of baselines and AutoTSAD on selected datasets (in seconds):")
st.write(df_runtime_baselines.set_index("name").T)
st.warning("""Note:
- For the baselines _select-horizontal_, _select-vertical_, and _tsadams-mim_, we only measured runtime on the
  69 new datasets for the revision.
- For _AutoTSAD w/ optimization_, we measured the runtime only on the 106 original datasts.
""")

st.write("Runtime breakdown of AutoTSAD for two selected datasets:")
st.write(df_runtime_trace)

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
st.write(df_runtime["factor"].agg(["min", "mean", "median", "max"]))

st.write("Runtime of AutoTSAD relative for selected datasets and absolute for all 106 original datasets:")

df_runtime_trace_st = conn.load_runtime_traces(
    traces, conn.list_datasets(only_paper_datasets=True)["name"].tolist(),
    config_names=["paper v1 - no parallelism", "paper v1 - default ensemble (no optimization, seed=1)",
                  "paper v1 - default ensemble (n_jobs=5)", "paper v1 - default ensemble (n_jobs=10)"],
    autotsad_versions=["0.2.2-timeeval"]
)

scaling_datasets = scaling_datasets + ["SED"]
df_runtime_trace_hpo = conn.load_runtime_traces(
    traces + ["autotsad"],
    scaling_datasets,
    config_names=["paper v1"],
    autotsad_versions=["0.2.1"]
)
df_runtime_trace_default = conn.load_runtime_traces(
    traces + ["autotsad"],
    scaling_datasets,
    config_names=["paper v1 - default ensemble (no optimization, seed=1)"],
    autotsad_versions=["0.2.2-timeeval"]
)
fig = plot_runtime_traces_new(
    df_runtime_trace_default, df_runtime_trace_hpo, df_runtime_trace_st,
    use_dataset_name_alias=True,
    group_traces={
        "Data Generation": datagen_traces,
        "Algorithm Optimization": ["autotsad-%-Optimization-%-Sensitivity analysis", "autotsad-%-Optimization-%-Hyperparams opt.", "autotsad-%-Optimization-%-Selecting best performers"],
    }
)
st.write(fig)

dataset_mapping = {chr(65+i): d for i, d in enumerate(scaling_datasets)}
st.write("Datasets:", dataset_mapping)

###############################################################################
# TRAINING DATA GENERATION STRATEGIES
###############################################################################
st.write("""## Training Data Generation Strategies

AutoTSAD uses the following process to generate training data:

1. Analyze target time series to identify dominant frequencies (period lengths).
2. Extract regimes of the time series with similar base behavior.
3. Clean the extracted regimes to remove potential anomalies.
4. Injecting synthetic anomalies.

We evaluate this process in three ways:

1. Cleaning: Recall of removing the existing anomalies
2. Regiming: Effect of not using regiming and cleaning on the performance of AutoTSAD
3. Injection: Robustness of anomaly injection by extending evaluation dataset corpus
""")

cleaning_filters = {
    "autotsad_version": ["0.3.3"],
    "config_id": ["6bc6c763d6590b04ad3fac6c6a36990a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
}
no_cleaning_filters = deepcopy(cleaning_filters)
no_cleaning_filters["autotsad_version"].append("0.3.2")
no_cleaning_filters["config_id"] = ["a0cbfd1aad708898aa0278584ee7ad52"]
no_cleaning_filters["baseline"] = ["best-algo"]  # we need to supply at least one baseline

st.write("### Effectiveness of the cleaning step")
incl_rev_data = st.checkbox("Include revision datasets", value=False, key="incl_rev_data-1")

df_cleaning_metrics = conn.cleaning_metrics({
    "aggregation_method": [best_ensemble_method.split("_")[2]], **cleaning_filters
}, "recall", only_paper_datasets=True, revision_datasets=incl_rev_data)
c0, c1 = st.columns(2)
c0.metric("Mean Cleaning Recall", f"{df_cleaning_metrics['recall'].mean():.2%}")
c0.metric("Stddev Cleaning Recall", f"{df_cleaning_metrics['recall'].std():.4f}")
c0.metric("Median Cleaning Recall", f"{df_cleaning_metrics['recall'].median():.2%}")

s_precision = conn.cleaning_metrics({
    "aggregation_method": [best_ensemble_method.split("_")[2]], **cleaning_filters
}, "precision", only_paper_datasets=True, revision_datasets=incl_rev_data)['precision']
c0.write(f"""Precision of the cleaning:
- Mean:\t{s_precision.mean():.2%}
- Stddev:\t{s_precision.std():.4f}
- Median:\t{s_precision.median():.2%}
""")
n_datasets = all_datasets
if incl_rev_data:
    n_datasets = df_datasets.sum().sum()
c0.write(f"The cleaning effectiveness is measured over {df_cleaning_metrics['recall'].count()} / {n_datasets} datasets.")

fig = plt.figure(figsize=(7.5, 5), dpi=500)
plt.title("Cleaning Effectiveness")
plt.hist(df_cleaning_metrics["recall"], bins=20, color="blue")
mu, sigma = df_cleaning_metrics["recall"].mean(), df_cleaning_metrics["recall"].std()
x = np.linspace(0, 1, 100)
y = stats.norm.pdf(x, mu, sigma)*df_cleaning_metrics.shape[0]/2
plt.plot(x, y, color="black", linestyle="--", label="Normal Distribution")
plt.ylabel("Number of Datasets")
plt.xlabel("Recall")
ax = plt.gca()
ax.set(axisbelow=True)
ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
spines = ax.spines
spines["top"].set_visible(False)
spines["right"].set_visible(False)
c1.write(fig)

st.write("### Data generation effect on the performance of AutoTSAD")
incl_rev_data = st.checkbox("Include revision datasets", value=False, key="incl_rev_data-2")

df_results_cleaning = conn.all_aggregated_results(
    cleaning_filters,
    only_paper_datasets=True,
    revision_datasets=incl_rev_data,
    only_paper_methods=True,
    exclude_super_ensembles=False
)
df_results_cleaning = df_results_cleaning[df_results_cleaning["Method Type"] == "AutoTSAD Ensemble"]
df_results_no_cleaning = conn.all_aggregated_results(
    no_cleaning_filters,
    only_paper_datasets=True,
    revision_datasets=incl_rev_data,
    only_paper_methods=True,
    exclude_super_ensembles=False
)
df_results_no_cleaning = df_results_no_cleaning[df_results_no_cleaning["Method Type"] == "AutoTSAD Ensemble"]
df_results_no_cleaning["Method"] = df_results_no_cleaning["Method"] + " (no cleaning)"

c0, c1 = st.columns(2)
c0.write("Missing results:")
df_tmp = pd.merge(
    df_results_cleaning.set_index("Dataset")[["Method"]],
    df_results_no_cleaning.set_index("Dataset")[["Method"]],
    how="left", left_index=True, right_index=True,
).sort_index()
df_tmp = df_tmp[df_tmp["Method_y"].isna()]
c0.dataframe(df_tmp)

df_results_no_cleaning = pd.concat([df_results_no_cleaning, df_results_cleaning], ignore_index=True)
# mean_before = df_results_no_cleaning.loc[df_results_no_cleaning["Method"] == "aggregated-minimum-influence_gaussian_mean", evaluation_metric].mean()
# # filter out some datasets:
# removed_datasets = [
#     "SAND 803_SED", "IOPS a8c06b47-cc41-3738-9110-12df0ee4c721", "IOPS c02607e8-7399-3dde-9d28-8a8da5e5d251",
#     "IOPS 7103fa0f-cac4-314f-addc-866190247439", "IOPS 6a757df4-95e5-3357-8406-165e2bd49360",
#     "MGAB 2", "MGAB 3", "MGAB 4", "MGAB 6",
# ]
# df_results_no_cleaning = df_results_no_cleaning[~df_results_no_cleaning["Dataset"].isin(removed_datasets)]
mean_before = df_results_no_cleaning.loc[df_results_no_cleaning["Method"] == "aggregated-minimum-influence_gaussian_mean (no cleaning)", evaluation_metric].mean()
c0.metric(f"Mean {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} (no cleaning)", f"{mean_before:.2%}")
mean_after = df_results_no_cleaning.loc[df_results_no_cleaning["Method"] == "aggregated-minimum-influence_gaussian_mean", evaluation_metric].mean()
c0.metric(f"Mean {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} (default)", f"{mean_after:.2%}")
method_order = df_results_no_cleaning.groupby("Method")[[evaluation_metric]].agg(sort_aggregation) \
    .reset_index() \
    .sort_values(evaluation_metric, ascending=False)["Method"] \
    .tolist()
fig = plot_quality(df_results_no_cleaning, method_order,
                   metric=evaluation_metric,
                   show_success_rate=True,
                   show_algo_type=False,
                   save_files_path=None,
                   name="cleaning_effect")
c1.write(fig)
c1.info("Seems like the data generation module does not have a significant impact on the performance.")

st.write("""Candidates for showing the improvement (try synthetic gt-* datasets!):
- synthetic gt-2 works; the others are not good: 0.0310 :arrow_right: 0.7612 (no cleaning picks up state transitions)
- IOPS 57051487-3a40-3828-9084-a12f7f23ee38: 0.0178 :arrow_right: 0.9443 (shows cleaning effect)
- WebscopeS5 A4Benchmark-67: 0.7130 :arrow_right: 0.8859 (better window sizes)
- NAB speed_6005: 0.7320 :arrow_right: 0.9960 (better focus)
- TSB-UAD-synthetic KDD21_change_segment_resampling_0.02-033_UCR_Anomaly_DISTORTEDInternalBleeding5_4000_6200_6370: 0.8397 :arrow_right: 0.9833
- KDD-TSAD 031_UCR_Anomaly_DISTORTEDInternalBleeding20: 0.8735 :arrow_right: 0.9975 (looks ok, showing regiming)
""")

st.write("### Robustness of the anomaly injection")
revision_filters = {
    "autotsad_version": ["0.3.3"],
    "config_id": ["6bc6c763d6590b04ad3fac6c6a36990a"],
    "ranking_method": [best_ensemble_method.split("_")[0]],
    "normalization_method": [best_ensemble_method.split("_")[1]],
    "aggregation_method": [best_ensemble_method.split("_")[2]],
    "baselines": ["best-algo"]
}
df_results_old = conn.all_aggregated_results(
    revision_filters,
    only_paper_datasets=True,
    revision_datasets=False,
    only_paper_methods=True,
    exclude_super_ensembles=False
)
df_results_old = df_results_old[df_results_old["Method Type"] == "AutoTSAD Ensemble"]
df_results_revision = conn.all_aggregated_results(
    revision_filters,
    only_paper_datasets=True,
    revision_datasets=True,
    only_paper_methods=True,
    exclude_super_ensembles=False
)
df_results_revision = df_results_revision[df_results_revision["Method Type"] == "AutoTSAD Ensemble"]
old_n = df_results_old.shape[0]
new_n = df_results_revision.shape[0]
c0, c1 = st.columns(2)
c0.write("Original performance:")
c0.metric(
    f"Mean {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} on {old_n} original datasets",
    f"{df_results_old[evaluation_metric].mean():.2%}"
)
c0.metric(
    f"Median {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} on {old_n} original datasets",
    f"{df_results_old[evaluation_metric].median():.2%}"
)
c1.write(f"Performance with {new_n - old_n} additional datasets:")
c1.metric(
    f"Mean {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} on {new_n} datasets",
    f"{df_results_revision[evaluation_metric].mean():.2%}"
)
c1.metric(
    f"Median {DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} on {new_n} datasets",
    f"{df_results_revision[evaluation_metric].median():.2%}"
)
###############################################################################
# OPTIMIZATION STRATEGIES
###############################################################################
st.write("## Optimization Strategies")
st.warning("All experiments in this section use the 106 original datasets.")

# selected_method = best_ensemble_method
selected_method = "aggregated-minimum-influence_gaussian_custom"
# optimization variance experiment
filters_optimization = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [selected_method.split("_")[0]],
    "normalization_method": [selected_method.split("_")[1]],
    "aggregation_method": [selected_method.split("_")[2]]
}
df_optimization_variance = conn.load_optimization_variance_results(
    filters_optimization,
    datasets=optimization_experiment_datasets,
    config_names=optimization_experiment_configs,
)

df_optimization_variance = df_optimization_variance.groupby(["collection", "dataset", "max_trials"])[[evaluation_metric]].agg(["mean", "std", "count"]).reset_index()
df_optimization_variance.columns = ["collection", "dataset", "max_trials", "mean", "std", "n_seeds"]
df_optimization_variance["dataset"] = df_optimization_variance["collection"] + " " + df_optimization_variance["dataset"]
df_optimization_variance = df_optimization_variance.drop(columns=["collection"])
df_optimization_variance = df_optimization_variance.sort_values(["dataset", "max_trials"])

st.write(f"Performance and variance of {selected_method} (with optimization starting from TimeEval-hyperparameters) "
         "over different number of trials:")
c0, c1 = st.columns(2)
with c0:
    st.write(df_optimization_variance[["dataset", "max_trials", "n_seeds"]])
with c1:
    fig = plot_optimization_scaling(df_optimization_variance, truncate_labels_to=45,
                                    metric_name=evaluation_metric)
    st.write(fig)

# optimization improvements compared to different default hyperparameter baselines:
timeeval_filters = {
    "autotsad_version": ["0.2.2-timeeval"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [selected_method.split("_")[0]],
    "normalization_method": [selected_method.split("_")[1]],
    "aggregation_method": [selected_method.split("_")[2]],
}
timeeval_optimized_filters = {
    "autotsad_version": ["0.2.1"],
    "config_id": ["26a8aa1409ae87761d9d405ff0e49f9e"],
    "ranking_method": [selected_method.split("_")[0]],
    "normalization_method": [selected_method.split("_")[1]],
    "aggregation_method": [selected_method.split("_")[2]],
}
bad_filters = {
    "autotsad_version": ["0.2.2-bad"],
    "config_id": ["65df673233b659aeac7df950b71c2d7a"],
    "ranking_method": [selected_method.split("_")[0]],
    "normalization_method": [selected_method.split("_")[1]],
    "aggregation_method": [selected_method.split("_")[2]],
}
bad_optimizated_filters = {
    "autotsad_version": ["0.2.2-bad"],
    "config_id": ["26a8aa1409ae87761d9d405ff0e49f9e"],
    "ranking_method": [selected_method.split("_")[0]],
    "normalization_method": [selected_method.split("_")[1]],
    "aggregation_method": [selected_method.split("_")[2]],
}


def load_improvements(filters_default: Dict[str, List[Any]], filters_optimized: Dict[str, List[Any]]) -> pd.DataFrame:
    default_filter_clauses = conn._build_autotsad_filter_clauses(filters_default)
    optimized_filter_clauses = conn._build_autotsad_filter_clauses(filters_optimized)

    sub_query = f"""select distinct d.collection || ' ' || d.name as "dataset", e.{evaluation_metric}
                   from autotsad_execution e, dataset d, configuration c
                   where e.dataset_id = d.hexhash
                     and e.config_id = c.id"""
    query = f"""select a.dataset, a.{evaluation_metric} - b.{evaluation_metric} as "improvement"
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

text = f"Improvement of {selected_method} for different ensembles:\n"
for c in df_improvement.columns:
    text += f"- {c}: {df_improvement[c].mean():.4f} (median: {df_improvement[c].median():.4f})\n"
st.write(text)
fig = plot_optimization_improvement(df_improvement, metric_name=evaluation_metric)
st.write(fig)

###############################################################################
# ENSEMBLING STRATEGIES
###############################################################################
st.write("## Ensembling Strategies")

# df_tmp = results[(results["Method"].isin(best_methods)) | (results["Method Type"] == "Baseline")].copy()
df_tmp = results.copy()
df_tmp = df_tmp[df_tmp["Method Type"] != "Baseline"]
# df_tmp = df_tmp[~((df_tmp["Method Type"] == "AutoTSAD Ensemble") & (df_tmp["Method"] != best_ensemble_method))]

# ---
# mean_before = df_tmp.loc[df_tmp["Method"] == "aggregated-minimum-influence_gaussian_mean", evaluation_metric].mean()
# diff_before = df_tmp.loc[df_tmp["Method"] == "mmq-euclidean_gaussian_mean", evaluation_metric].mean() - mean_before
# # filter out some datasets:
# removed_datasets = [
#     "TSB-UAD-synthetic YAHOO_change_segment_add_scale_0.02-YahooA3Benchmark-TS22_data",
#     "IOPS 431a8542-c468-3988-a508-3afd06a218da",
#     "NAB Twitter_volume_CVS",
#     "IOPS a8c06b47-cc41-3738-9110-12df0ee4c721",
#     "NAB machine_temperature_system_failure",
#     "IOPS 7103fa0f-cac4-314f-addc-866190247439",
#     "NASA-SMAP E-5",
#     "TSB-UAD-synthetic YAHOO_flat_region_0.04-YahooA3Benchmark-TS51_data",
#     "TSB-UAD-synthetic YAHOO_add_random_walk_trend_0.2-YahooA4Benchmark-TS28_data",
#     "GutenTAG cbf-position-middle",
#     "IOPS 05f10d3a-239c-3bef-9bdc-a2feeb0037aa",
#     # "IOPS 1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0",
#     # "IOPS c69a50cf-ee03-3bd7-831e-407d36c7ee91",
#     # "IOPS 54350a12-7a9d-3ca8-b81f-f886b9d156fd",
#     # "IOPS 6a757df4-95e5-3357-8406-165e2bd49360",
# ]
# df_tmp = df_tmp[~df_tmp["Dataset"].isin(removed_datasets)]
# mean_after = df_tmp.loc[df_tmp["Method"] == "aggregated-minimum-influence_gaussian_mean", evaluation_metric].mean()
# diff_after = df_tmp.loc[df_tmp["Method"] == "mmq-euclidean_gaussian_mean", evaluation_metric].mean() - mean_after
# st.metric("New RANGE_PR_AUC", mean_after, delta=mean_after-mean_before)
# st.metric("Gap", diff_after, delta=diff_after-diff_before)
# ---

method_order = df_tmp.groupby("Method")[[evaluation_metric]].aggregate(sort_aggregation) \
    .reset_index() \
    .sort_values(evaluation_metric, ascending=False) \
    .reset_index(drop=True)
method_order = method_order["Method"].tolist()

c0, c1 = st.columns(2)
with c0:
    st.write(best_ranking_methods_text)
    st.write(df_tmp.groupby("Method")[[evaluation_metric]]
             .agg({evaluation_metric: ["mean", "median", "count"]})
             .sort_values((evaluation_metric, "mean"), ascending=False)
             )

with c1:
    fig = plot_ensembling_strategies(df_tmp, method_order, metric=evaluation_metric)
    st.write(fig)


###############################################################################
# EXAMPLE RANKING
###############################################################################
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
    metric=evaluation_metric,
)
df_ranking1 = conn.load_ranking_results(
    dataset=ranking_example_dataset,
    rmethod=best_ensemble_method.split("_")[0],
    nmethod=best_ensemble_method.split("_")[1],
    amethod=best_ensemble_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters,
)
quality2 = conn.method_quality(
    dataset=ranking_example_dataset,
    rmethod=ranking_example_method.split("_")[0],
    nmethod=ranking_example_method.split("_")[1],
    amethod=ranking_example_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters,
    metric=evaluation_metric,
)
df_ranking2 = conn.load_ranking_results(
    dataset=ranking_example_dataset,
    rmethod=ranking_example_method.split("_")[0],
    nmethod=ranking_example_method.split("_")[1],
    amethod=ranking_example_method.split("_")[2],
    method_type="AutoTSAD",
    filters=filters,
)
c0, c1 = st.columns(2)
c0.metric(f"{DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} {best_ensemble_method.replace('_', ' ')}", quality1)
c1.metric(f"{DB_REVERSE_METRIC_NAME_MAPPING[evaluation_metric]} {ranking_example_method.replace('_', ' ')}", quality2, delta=quality2-quality1)
fig = plot_ranking_comparison(df_dataset, df_ranking1, df_ranking2,
                              quality_left=quality1,
                              quality_right=quality2,
                              method_left=best_ensemble_method,
                              method_right=ranking_example_method,
                              show_anomalies=True, show_algo_quality=True)
st.write(fig)
