import json
import logging
from itertools import combinations
from typing import Tuple, Union, Dict, Any

import optuna
import pandas as pd
from optuna import Study
from optuna.study import StudySummary
from optuna.trial import TrialState

from .optuna_auto_storage import OptunaStorageReference
from ..hyperparameters import ParamSetting, param_setting_binned_list_intersection
from ..timer import Timers
from ...config import config
from ...dataset import TrainingDatasetCollection, TrainingTSDataset

log = logging.getLogger("autotsad.optimization.consolidation")


def main(train_collection: TrainingDatasetCollection,
         consider_dataset_characteristics: bool = True,
         key: str = "none") -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_consolidation_cache_path = config.general.cache_dir() / f"dataset-consolidation-{key}.csv"
    config.general.cache_dir().mkdir(parents=True, exist_ok=True)

    if dataset_consolidation_cache_path.exists():
        log.info("Loading consolidation information from cache...")
        results = pd.read_csv(dataset_consolidation_cache_path)
    else:
        results = _perform_consolidation(train_collection, consider_dataset_characteristics)
        results.to_csv(dataset_consolidation_cache_path, index=False)

    # print("\nParams that consolidate datasets:")
    # print(results.set_index(["algorithm", "params"]).sort_index())

    if not results.empty:
        return _select_representative(results)
    else:
        log.info("No consolidations necessary!")
        return (
            pd.DataFrame(columns=["algorithm", "params", "dataset"]),
            pd.DataFrame(columns=["algorithm", "dataset", "proxy"])
        )


def _perform_consolidation(train_collection: TrainingDatasetCollection,
                           consider_dataset_characteristics: bool = True) -> pd.DataFrame:
    # load all trial results and get the parameters according to the selection strategy
    log.info("Loading studies and candidate trials ...")
    Timers.start(["Consolidating", "Loading studies and trials"])
    candidate_trials = {}
    with OptunaStorageReference.from_config(config) as storage:
        studies = optuna.get_all_study_summaries(storage)
        for algorithm in config.optimization.algorithms:
            algo_studies = [s for s in studies if _study_has_tags(s, algorithm=algorithm, status="active")]
            candidate_trials[algorithm] = {}
            log.debug(f"Selecting candidate trials for {algorithm} "
                      f"(strategy: {config.consolidation.param_selection_strategy})")
            for study in algo_studies:
                trials = storage.get_all_trials(study._study_id, states=(TrialState.COMPLETE,))
                if config.consolidation.param_selection_strategy == "best":
                    eps = config.consolidation.param_selection_best_quality_epsilon
                    trials = [t for t in trials if abs(t.value - study.best_trial.value) <= eps]
                elif config.consolidation.param_selection_strategy == "threshold":
                    threshold = config.consolidation.param_selection_quality_threshold
                    trials = [t for t in trials if t.value >= threshold]
                candidate_trials[algorithm][study.user_attrs["dataset"]] = trials
    Timers.stop("Loading studies and trials")

    log.info("Consolidating trials ...")
    Timers.start("Combining trials")
    results = []
    for algorithm in config.optimization.algorithms:
        trials = candidate_trials[algorithm]
        if config.consolidation.plot:
            import networkx as nx
            G = nx.MultiGraph(label=algorithm)
            G.add_nodes_from(trials.keys())

        log.debug(f"{algorithm}: computing pairwise shared hyperparameter settings")
        consolidations = {}
        for dataset, other_dataset in combinations(trials, 2):
            d1 = train_collection.find(dataset)
            params1 = [ParamSetting(t.params) for t in trials[dataset]]
            params2 = [ParamSetting(t.params) for t in trials[other_dataset]]
            # shared_params = param_setting_list_intersection(params1, params2)
            shared_params = param_setting_binned_list_intersection(params1, params2)
            if len(shared_params) >= 1:
                if consider_dataset_characteristics:
                    d2 = train_collection.find(other_dataset)
                    if _datasets_are_not_similar(d1, d2, config.consolidation.dataset_similarity_threshold):
                        continue

                for params in shared_params:
                    if config.consolidation.plot:
                        G.add_edge(dataset, other_dataset, params=params)

                    if params not in consolidations:
                        consolidations[params] = set()
                    consolidations[params].add(dataset)
                    consolidations[params].add(other_dataset)

        # remove dataset consolidations that are fully covered by others
        flagged_removal = set()
        for p1, p2 in combinations(consolidations, 2):
            l1 = consolidations[p1]
            l2 = consolidations[p2]
            if l1.issubset(l2):
                flagged_removal.add(p1)
            elif l2.issubset(l1):
                flagged_removal.add(p2)

        for r in flagged_removal:
            log.debug(f"{algorithm}: removing {r} because it is covered by another param setting")
            del consolidations[r]

        log.debug(f"{algorithm}: collecting hyperparameter metrics (quality, n_trials)")
        for params in consolidations:
            for d in consolidations[params]:
                study = [s for s in studies if _study_has_tags(s, algorithm=algorithm, dataset=d, status="active")][0]
                trial = [t for t in trials[d] if ParamSetting(t.params).bin == params.bin][0]
                n_trials = study.n_trials
                results.append({
                    "algorithm": algorithm,
                    "params": json.dumps(dict(params)),
                    "dataset": d,
                    "quality": trial.value,
                    "n_trials": n_trials
                })

        if config.consolidation.plot:
            _plot_consolidations(consolidations, G)
    results = pd.DataFrame(results)
    Timers.stop("Combining trials")
    Timers.stop("Consolidating")
    return results


def _study_has_tags(study: Union[Study, StudySummary], **tags) -> bool:
    for t in tags:
        if t not in study.user_attrs or study.user_attrs[t] != tags[t]:
            return False
    return True


def _datasets_are_not_similar(d1: TrainingTSDataset, d2: TrainingTSDataset, threshold: float) -> bool:
    ddims1 = d1.opt_dims
    ddims2 = d2.opt_dims
    sim_score = 0
    for k in ddims1:
        sim_score += ddims1[k] == ddims2[k]
    max_sim_score = len(ddims1)
    return sim_score < threshold * max_sim_score


def _plot_consolidations(consolidations: Dict[ParamSetting, Any], graph: Any) -> None:
    # plot algorithm graph
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    graph_plot_params = {
        "node_size": 100, "alpha": 0.6
    }
    cmap = plt.colormaps.get_cmap("tab20")
    G: nx.Graph = graph
    # pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G)

    plt.figure()
    plt.title(G.graph["label"])
    nx.draw_networkx_nodes(G, pos=pos, **graph_plot_params, label="Datasets")
    nx.draw_networkx_labels(G, pos=pos, font_size=8)
    for i, params in enumerate(consolidations):
        edges = [e for e in G.edges.data("params") if e[2].bin == params.bin]
        color = np.full((len(edges), 1, 3), fill_value=cmap.colors[i % len(cmap.colors)], dtype=np.float_)
        nx.draw_networkx_edges(G, pos=pos, **graph_plot_params,
                               edgelist=edges,
                               edge_color=color,
                               label=str(params))
    plt.legend()
    plt.show()


def _select_representative(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Performing consolidation by selecting a representative dataset...")
    Timers.start("Apply consolidation")
    strategy = config.consolidation.dataset_selection_strategy
    log.debug(f"selection strategy = {strategy}")
    if strategy == "best":
        sort_cols = ["quality", "n_trials", "dataset"]
        sort_asc = [False, False, False]

    elif strategy == "worst":
        sort_cols = ["quality", "n_trials", "dataset"]
        sort_asc = [True, False, False]

    elif strategy == "fastest":
        sort_cols = ["n_trials", "quality", "dataset"]
        sort_asc = [False, False, False]

    else:
        raise ValueError(f"Unknown dataset selection strategy '{strategy}'!")

    representatives = results \
        .sort_values(sort_cols, ascending=sort_asc) \
        .groupby(["algorithm", "params"])[["dataset"]] \
        .agg({"dataset": "first"}) \
        .reset_index(drop=False)
    log.info(f"Selected representatives {representatives.values.tolist()}")

    studies_to_deactivate = results[["algorithm", "params", "dataset"]]
    studies_to_deactivate = pd.merge(studies_to_deactivate, representatives,
                                     on=["algorithm", "params"], how="left", suffixes=("", "_repr"))
    studies_to_deactivate = studies_to_deactivate[["algorithm", "dataset", "dataset_repr"]]
    studies_to_deactivate.columns = ["algorithm", "dataset", "proxy"]

    # studies_to_deactivate = results[["algorithm", "dataset"]]
    studies_to_deactivate = studies_to_deactivate[~(
            studies_to_deactivate["algorithm"].isin(representatives["algorithm"])
            & studies_to_deactivate["dataset"].isin(representatives["dataset"])
    )]
    log.debug(f"{len(studies_to_deactivate)} studies to deactivate")
    Timers.stop("Apply consolidation")
    return representatives, studies_to_deactivate
