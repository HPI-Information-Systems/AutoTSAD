from typing import Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from timeeval.metrics.thresholding import ThresholdingStrategy, SigmaThresholding
from timeeval.utils.hash_dict import hash_dict

from .aggregation import load_scores, algorithm_instances
from ...config import config
from ...rra import minimum_influence

RANKING_METHODS_FOR_RRA = (
    # "training-coverage",
    "training-quality",
    "training-result",
    "affinity-propagation-clustering",
    "kmedoids-clustering",
    "greedy-euclidean", "greedy-annotation-overlap",
    "mmq-euclidean", "mmq-annotation-overlap",
    # "interchange-euclidean", "interchange-annotation-overlap"
)


def annotation_overlap_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Jaccard distance"""
    x = np.array(x, dtype=np.bool_)
    y = np.array(y, dtype=np.bool_)
    assert x.shape == y.shape

    if np.sum(x) == 0 or np.sum(y) == 0:
        return 0

    return 1 - np.sum(x & y) / np.sum(x | y)


def _get_metric(strategy_name: str) -> str:
    return "-".join(strategy_name.lower().replace("_", "-").split("-")[1:])


def _annotation_overlap_distances(x: np.ndarray, thresholding: ThresholdingStrategy) -> np.ndarray:
    annotations = np.empty((x.shape[0], x.shape[1]), dtype=np.bool_)
    for i in np.arange(x.shape[1]):
        s = x[:, i]
        annotations[:, i] = thresholding.fit_transform(s, s)
    distances = pairwise_distances(annotations.T, metric=annotation_overlap_distance)
    return distances


def _score_distance_matrix(scores: Optional[pd.DataFrame] = None,
                           algorithm_instances: Optional[Sequence[str]] = None,
                           distance_metric: str = "euclidean") -> pd.DataFrame:
    if (scores is None and algorithm_instances is None):
        raise ValueError("Either 'scores' or 'algorithm_instances' must be given!")

    if algorithm_instances is not None:
        if scores is None:
            scores = load_scores(algorithm_instances,
                                 dataset_id=config.general.cache_key,
                                 base_scores_path=config.general.tmp_path / "scores")
        else:
            scores = scores[algorithm_instances]

    distance_metric = distance_metric.lower().replace("_", "-")
    if distance_metric == "euclidean":
        distances = euclidean_distances(scores.T)

    # DTW is way too slow!
    # elif metric == "dtw":
    #     from tslearn.metrics import dtw
    #     distances = pairwise_distances(scores.T, metric=dtw)

    elif distance_metric == "inv-cosine":
        distances = 1 - cosine_similarity(scores.T)

    elif distance_metric == "annotation-overlap":
        distances = _annotation_overlap_distances(scores.values, SigmaThresholding(factor=2))

    else:
        raise ValueError(f"Metric '{distance_metric}' is not supported! Choose between 'euclidean', "
                         f"'annotation_overlap', and 'inv_cosine'.")

    distances = pd.DataFrame(distances, index=scores.columns, columns=scores.columns)
    return distances


def best_training_coverage(df_results: pd.DataFrame) -> pd.DataFrame:
    return df_results.sort_values("no_datasets", ascending=False)


def best_training_quality(df_results: pd.DataFrame) -> pd.DataFrame:
    return df_results.sort_values("mean_train_quality", ascending=False)


def best_training(df_results: pd.DataFrame) -> pd.DataFrame:
    df_results["q"] = np.mean(MinMaxScaler().fit_transform(df_results[["no_datasets", "mean_train_quality"]]), axis=1)
    df_results = df_results.sort_values("q", ascending=False)
    # df_results = df_results.drop(columns="q")
    return df_results


def affinity_propagation_clustering(df_results: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    cls = AffinityPropagation(random_state=42)
    cls.fit(scores.T)
    cluster_centers = scores.columns[cls.cluster_centers_indices_]
    results = df_results[algorithm_instances(df_results).isin(cluster_centers)]

    return results.sort_values(["no_datasets", "mean_train_quality"], ascending=False)


def kmedoids_clustering(df_results: pd.DataFrame, scores: pd.DataFrame,
                        k: int = config.general.max_algorithm_instances) -> pd.DataFrame:
    cls = KMedoids(
        n_clusters=k, metric="euclidean", method="pam", init="k-medoids++", random_state=42
    )
    cls.fit(scores.T)
    cluster_centers = scores.columns[cls.medoid_indices_]
    results = df_results[algorithm_instances(df_results).isin(cluster_centers)]

    return results.sort_values(["no_datasets", "mean_train_quality"], ascending=False)


def greedy_farthest_distance(df_results: pd.DataFrame, scores: pd.DataFrame,
                             metric: str = "euclidean",
                             max_instances: int = config.general.max_algorithm_instances) -> pd.DataFrame:
    """Greedy selection of scorings with farthest distance to each other similar to farthest point sampling (FPS) in
    point cloud analysis (https://arxiv.org/pdf/2208.08795.pdf, https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-562.pdf)
    or image sampling (https://ieeexplore.ieee.org/document/623193).
    """
    results = df_results.copy()
    results = results.sort_values("no_datasets", ascending=False)
    results["name"] = algorithm_instances(results)
    distances = _score_distance_matrix(scores, distance_metric=metric)

    # select first instance based on training data coverage
    first_instance = results.iloc[0, :]
    first_instance_name = first_instance["algorithm"] + "-" + hash_dict(first_instance["params"])
    selected_instances = [first_instance_name]
    distances_tmp = distances.drop(index=first_instance_name)

    # select next instance based on most different scoring (max mean distance to algo candidates in selected set)
    for i in range(1, max_instances):
        idx = np.argmax(distances_tmp[selected_instances].mean(axis=1))
        s = distances_tmp.index[idx]
        selected_instances.append(s)
        distances_tmp = distances_tmp.drop(index=s)

    results = results[results["name"].isin(selected_instances)]
    results = results.sort_values("name", ascending=True, key=lambda x: x.map(selected_instances.index))
    results = results.drop(columns=["name"])
    return results


def maximal_marginal_quality(df_results: pd.DataFrame, scores: pd.DataFrame, distance: str = "euclidean",
                             quality: str = "mean_train_quality", lambda_: float = 0.3,
                             max_instances: int = config.general.max_algorithm_instances) -> pd.DataFrame:
    """Maximal marginal quality algorithm instance selection is inspired by the Maximal Marginal Relevance (MMR)
    criterion [1]_ from the information retrieval area.

    References
    ----------
    [1] Jaime Carbonell and Jade Goldstein (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents
        and Producing Summaries. In Proceedings of the International Conference on Research and Development in
        Information Retrieval (SIGIR). 335–336. https://doi.org/10.1145/290941.291025.
    """
    results = df_results.copy()
    results["name"] = algorithm_instances(results)
    quality_metric = quality.lower().replace("-", "_")
    results["relevance"] = results[quality_metric]
    distances = _score_distance_matrix(scores, distance_metric=distance).values

    # rescale distances and convert to similarity measure
    # distances = np.exp(-distances)
    # distances = 1 / (1e-10 + distances)
    kernel_sigma = 10
    kernel = lambda x: np.exp(-x**2/(2*kernel_sigma**2))  # https://stats.stackexchange.com/a/158308
    distances = kernel(distances)
    distances = pd.DataFrame(distances, index=scores.columns, columns=scores.columns)

    selected_instances = pd.DataFrame()
    results = results.set_index("name")
    for i in range(max_instances):
        s_mmr_score = pd.Series(
            lambda_ * results["relevance"] -
            (
                (1 - lambda_) * distances.loc[results.index, selected_instances.index].max(axis=1)
                if not selected_instances.empty
                else 0
            ),
            name="mmr_score"
        )
        # df_tmp = pd.DataFrame({"name": results.index,
        #                        "mmr_score": s_mmr_score,
        #                        "relevance": results["relevance"],
        #                        "similarity":
        #                            distances.loc[results.index, selected_instances.index].max(axis=1)
        #                            if not selected_instances.empty
        #                            else 0
        #                        })
        # df_tmp["w_relevance"] = lambda_ * df_tmp["relevance"]
        # df_tmp["w_similarity"] = (1 - lambda_) * df_tmp["similarity"]
        # df_tmp = df_tmp.sort_values("mmr_score", ascending=False)
        # print(df_tmp)
        # print(df_tmp.describe())
        idx = s_mmr_score.argmax()
        instance = pd.DataFrame([results.iloc[idx, :]])
        selected_instances = pd.concat([selected_instances, instance], ignore_index=False)
        results = results.drop(index=instance.index)

    selected_instances = selected_instances.reset_index(drop=True).drop(columns=["relevance"])
    # print(selected_instances)
    return selected_instances


def interchange_ranking(df_results: pd.DataFrame,
                        scores: pd.DataFrame,
                        relevance: str = "coverage",
                        distance_metric: str = "euclidean",
                        max_instances: int = config.general.max_algorithm_instances,
                        upper_bound: Optional[float] = None,
                        top_k_lower_bound: Optional[int] = None) -> pd.DataFrame:
    """Instance selection based on interchange algorithm (Algo. 4, [1]_).

    First, we select the top-k best algorithm instances, and then we replace the candidate with the lowest average
    distance (diversity contribution) with a non-selected candidate with higher distance and above a certain quality
    threshold.

    References
    ----------
    [1] Cong Yu, Laks Lakshmanan, and Sihem Amer-Yahia (2009). It takes variety to make a world: diversification in
        recommender systems. In Proceedings of the International Conference on Extending Database Technology (EDBT).
        368–378. https://doi.org/10.1145/1516360.1516404.
    """
    if upper_bound is None and top_k_lower_bound is None:
        raise ValueError("Either upper_bound or top_k_lower_bound must be set!")

    results = df_results.copy()
    results["name"] = algorithm_instances(results)
    results = results.set_index("name")

    def get_rel_score(item: pd.Series) -> float:
        if relevance == "coverage":
            return item["no_datasets"].astype(np.float_)
        elif relevance == "quality":
            return item["mean_train_quality"]
        elif relevance == "combined":
            # return np.mean(
            #     MinMaxScaler().fit_transform(
            #         np.array([item["no_datasets"], item["mean_train_quality"]], dtype=np.float_)
            #     ),
            #     axis=1
            # )
            return item["q"]

    # sort according to relevance criterion:
    relevance = relevance.lower().replace("_", "-")
    if relevance == "coverage":
        results = best_training_coverage(results)
    elif relevance == "quality":
        results = best_training_quality(results)
    elif relevance == "combined":
        results = best_training(results)

    selected_instances = results.iloc[:max_instances, :]
    # print("Base ranking")
    # print(selected_instances)
    # print()

    distances = _score_distance_matrix(
        scores=scores, algorithm_instances=selected_instances.index, distance_metric=distance_metric
    )
    end_idx = min(top_k_lower_bound, results.shape[0]) or results.shape[0]
    for next_candidate_idx in range(max_instances, end_idx):
        candidate = results.iloc[next_candidate_idx, :]

        # find item with the lowest diversity (avg. distance)
        idx = np.argmin(distances[selected_instances.index].mean(axis=1))
        s = distances.index[idx]

        # print(f"Swapping {selected_instances.loc[s, :].name} ({idx}) with {candidate.name} ({next_candidate_idx})?")
        # print(f"{get_rel_score(selected_instances.loc[s, :])=} - {get_rel_score(candidate)=} > {upper_bound=}")

        # if quality drop is too high, stop
        if upper_bound is not None and get_rel_score(selected_instances.loc[s, :]) - get_rel_score(candidate) > upper_bound:
            break

        # compute distances when swapped
        candidate_set: pd.Index = selected_instances.index.drop(s)
        candidate_set = candidate_set[::-1].insert(0, candidate.name)[::-1]  # workaround to add item at the end
        candidate_distances = _score_distance_matrix(
            scores=scores, algorithm_instances=list(candidate_set), distance_metric=distance_metric
        )
        # print("distances when swapped:")
        # print(pd.DataFrame({
        #     "current": distances.mean(axis=1).values,
        #     "when swapped": candidate_distances.mean(axis=1).values,
        # }))
        # print()
        if candidate_distances[candidate.name].mean() > distances.iloc[idx, :].mean():
            # print("--- swap ---")
            # candidate improves diversity --> switch
            selected_instances = selected_instances.drop(index=s)
            selected_instances = pd.concat([selected_instances, pd.DataFrame([candidate])], axis=0)
            distances = candidate_distances
        # else:
    #         print("--- no swap ---")
    # print("--- hit quality bound ---")

    return selected_instances


def rank_aggregation_mim(df_results: pd.DataFrame,
                         scores: Optional[pd.DataFrame] = None,
                         max_instances: int = config.general.max_algorithm_instances) -> pd.DataFrame:
    """Robust rank aggregation method using minimum influence metric (MIM) [1]_.

    We use a selected set of our ranking methods to compute individual rankings and then aggregate them using the MIM.
    See [1] for details. Implementation copied and modified from
    `https://github.com/mononitogoswami/tsad-model-selection/blob/master/src/tsadams/model_selection/rank_aggregation.py`_.

    References
    ----------
    [1] Goswami, Mononito, Cristian Challu, Laurent Callot, Lenon Minorics, and Andrey Kan.
        "Unsupervised Model Selection for Time-Series Anomaly Detection."
        In Proceedings of the International Conference on Learning Representations (ICLR), 2023.
        http://arxiv.org/abs/2210.01078.
    """
    instances = algorithm_instances(df_results)

    # compute individual rankings
    ranks = pd.DataFrame(len(instances), columns=RANKING_METHODS_FOR_RRA, index=instances, dtype=np.int_)
    for ranking_method in RANKING_METHODS_FOR_RRA:
        maxi = df_results.shape[0]
        if ranking_method in ["kmedoids-clustering", "affinity-propagation-clustering"]:
            maxi = max_instances
        df = select_algorithm_instances(df_results, scores, selection_method=ranking_method, max_instances=maxi)
        ranked_instances = algorithm_instances(df.reset_index(drop=True))
        ranks.loc[ranked_instances.values, ranking_method] = ranked_instances.index

    # compute aggregated ranking
    aggregated_ranks = minimum_influence(ranks.values.T, aggregation_type="borda")
    selected_instances = pd.Series(aggregated_ranks, index=instances, dtype=np.int_)

    selected_results = df_results[algorithm_instances(df_results).isin(selected_instances.index)].copy()
    selected_results["name"] = algorithm_instances(selected_results)
    selected_results["rank"] = selected_results["name"].map(selected_instances)
    selected_results = selected_results.sort_values("rank")
    selected_results = selected_results.drop(columns=["name", "rank"])
    return selected_results


def select_algorithm_instances(df_results: pd.DataFrame,
                               scores: Optional[pd.DataFrame] = None,
                               selection_method: str = config.general.algorithm_selection_method,
                               max_instances: int = config.general.max_algorithm_instances) -> pd.DataFrame:
    method = selection_method.lower().replace("_", "-")
    max_instances = min(max_instances, df_results.shape[0])

    if method == "training-coverage":
        df_results = best_training_coverage(df_results)
    elif method == "training-quality":
        df_results = best_training_quality(df_results)
    elif method == "training-result":
        df_results = best_training(df_results)
    else:
        # now all methods that require scores
        if scores is None:
            scores = load_scores(algorithm_instances(df_results),
                                 dataset_id=config.general.cache_key,
                                 base_scores_path=config.general.tmp_path / "scores")

        if method == "affinity-propagation-clustering":
            df_results = affinity_propagation_clustering(df_results, scores)
        elif method == "kmedoids-clustering":
            df_results = kmedoids_clustering(df_results, scores, k=max_instances)
        elif method.startswith("greedy"):
            df_results = greedy_farthest_distance(
                df_results, scores, metric=_get_metric(method), max_instances=max_instances
            )
        elif method.startswith("mmq"):
            df_results = maximal_marginal_quality(df_results, scores, distance=_get_metric(method),
                                                  quality="mean_train_quality", lambda_=0.5,
                                                  max_instances=max_instances)
        elif method.startswith("interchange"):
            df_results = interchange_ranking(df_results, scores, distance_metric=_get_metric(method),
                                             max_instances=max_instances, relevance="coverage", top_k_lower_bound=20,
                                             upper_bound=3)
            # df_results = interchange_ranking(df_results, scores, distance_metric=_get_metric(method),
            #                                  relevance="quality", top_k_lower_bound=30, upper_bound=0.025)
        elif method == "aggregated-minimum-influence":
            df_results = rank_aggregation_mim(df_results, scores, max_instances=max_instances)
        else:
            raise ValueError(f"Selection method '{method}' is not supported!")

    return df_results[:max_instances]
