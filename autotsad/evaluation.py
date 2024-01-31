from pathlib import Path
from typing import Dict, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, euclidean_distances
from timeeval import Metric
from timeeval.metrics import RangePrecision, RangeRecall, RangeFScore, RangePrAUC, RangeRocAUC, PrecisionAtK
from timeeval.metrics.thresholding import ThresholdingStrategy, PercentileThresholding, SigmaThresholding, \
    TopKRangesThresholding
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .config import METRIC_MAPPING
from .dataset import TestDataset
from .system.execution.aggregation import aggregate_scores, load_scores, algorithm_instances
from .system.execution.algo_selection import _annotation_overlap_distances
from .util import mask_to_slices


def _hit_percentage(labels: np.ndarray, predictions: np.ndarray, buffer_size: int = 0) -> float:
    anomaly_ranges = mask_to_slices(labels)
    hit_percentage = 0
    for b, e in anomaly_ranges:
        hit_percentage += np.any(predictions[b-buffer_size:e+buffer_size])
    hit_percentage /= anomaly_ranges.shape[0]
    return hit_percentage


def _variety(labels: np.ndarray, scores: np.ndarray, similarity_measure: str = "annotation-overlap") -> float:
    similarity_measure = similarity_measure.lower().replace("_", "-")
    if similarity_measure == "euclidean":
        similarity = euclidean_distances(scores.T)
        mean_distance = similarity[np.triu_indices_from(similarity, k=1)].mean()
        variety = 1 - 1 / (1e-10 + mean_distance)
    elif similarity_measure == "annotation-overlap":
        similarity = _annotation_overlap_distances(scores, SigmaThresholding(factor=2))
        mean_distance = similarity[np.triu_indices_from(similarity, k=1)].mean()
        variety = 1 - mean_distance
    # does not make a lot of sense despite the nice values (0-1)
    # similarity = cosine_similarity(scores.T)
    # variety = similarity[np.triu_indices_from(similarity, k=1)].mean()
    else:
        raise ValueError(f"Unknown similarity measure '{similarity_measure}'!")

    return variety


def _quality(anomaly_range_scores: np.ndarray) -> float:
    return anomaly_range_scores.max(axis=0, initial=0).mean()


def _variance(anomaly_range_scores: np.ndarray) -> float:
    return anomaly_range_scores.std(axis=0, ddof=1).mean()  # use unbiased estimator (similar to pandas default)


def _dcg(anomaly_range_scores: np.ndarray) -> float:
    algo_instance_relevance = anomaly_range_scores.mean(axis=1)
    dcg = np.sum(algo_instance_relevance / np.log2(np.arange(len(algo_instance_relevance))+2))
    return dcg  # type: ignore


def _precision_at_k(labels: np.ndarray, scores: np.ndarray, buffer_size: int = 50) -> float:
    # FIXME: how to deal with multiple scorings?

    # 1. using aggregated scores:
    # scores = MinMaxScaler().fit_transform(scores)
    # scores = scores.max(axis=1)
    # thresholding = TopKRangesThresholding()
    # predictions = thresholding.fit_transform(labels, scores).astype(np.bool_)

    # 2. each algorithm scoring gets their own thresholding:
    predictions = np.empty_like(scores, dtype=np.bool_)
    for i in range(scores.shape[1]):
        predictions[:, i] = TopKRangesThresholding().fit_transform(labels, scores[:, i])
    predictions = np.bitwise_or.reduce(predictions, axis=1, dtype=np.bool_)

    # 3. other way?

    # compute quality metric
    precision = _hit_percentage(labels, predictions, buffer_size)
    return precision


def compute_metrics_old(labels: np.ndarray, scores: np.ndarray,
                    thresholding_strategy: ThresholdingStrategy = PercentileThresholding(),
                    buffer_size: int = 10,
                    metrics: Sequence[str] = ("quality", "precision@k", "variety")) -> Dict[str, float]:
    if "SC-score" in metrics:
        metrics = list(metrics)
        metrics.append("quality")
        metrics.append("variety")

    results = {}
    if "precision@k" in metrics:
        results["precision@k"] = _precision_at_k(labels, scores, buffer_size)
    if "variety" in metrics:
        results["variety"] = _variety(labels, scores)

    if any(m in metrics for m in (
            "accuracy", "precision", "recall", "f1", "roc_auc", "range_precision", "range_recall", "range_f1", "hits"
    )):
        predictions = np.empty_like(scores, dtype=np.bool_)
        for i in range(scores.shape[1]):
            predictions[:, i] = thresholding_strategy.fit_transform(labels, scores[:, i])
        # aggregate predictions of all algorithms (to maximize recall)
        predictions = np.bitwise_or.reduce(predictions, axis=1, dtype=np.bool_)
        predictions = predictions.astype(np.int_)  # temporary fix for TimeEval metrics!

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(labels, predictions)
        if "precision" in metrics:
            results["precision"] = precision_score(labels, predictions)
        if "recall" in metrics:
            results["recall"] = recall_score(labels, predictions)
        if "f1" in metrics:
            results["f1"] = f1_score(labels, predictions)
        if "roc_auc" in metrics:
            results["roc_auc"] = roc_auc_score(labels, predictions)
        if "range_precision" in metrics:
            results["range_precision"] = RangePrecision(alpha=0, cardinality="reciprocal",
                                                        bias="flat")(labels, predictions)
        if "range_recall" in metrics:
            results["range_recall"] = RangeRecall(alpha=0.5, cardinality="reciprocal",
                                                  bias="flat")(labels, predictions)
        if "range_f1" in metrics:
            results["range_f1"] = RangeFScore(beta=0.5, p_alpha=0, r_alpha=0.5, cardinality="reciprocal",
                                              p_bias="flat", r_bias="flat")(labels, predictions)
        if "hits" in metrics:
            results["hits"] = _hit_percentage(labels, predictions, buffer_size)

    if any(m in metrics for m in ("quality", "variance", "dcg")):
        anomaly_ranges = mask_to_slices(labels)
        anomaly_range_scores = np.empty((scores.shape[1], len(anomaly_ranges)), dtype=np.float_)
        for i, (b, e) in enumerate(anomaly_ranges):
            b = max(0, b-buffer_size)
            e = min(len(labels), e+buffer_size)
            for j in np.arange(scores.shape[1]):
                s = scores[:, j]
                max_anomaly_score = s[b:e].max()
                anomaly_range_scores[j, i] = max_anomaly_score

        if "quality" in metrics:
            results["quality"] = _quality(anomaly_range_scores)
        if "variance" in metrics:
            results["variance"] = _variance(anomaly_range_scores)
        if "dcg" in metrics:
            results["dcg"] = _dcg(anomaly_range_scores)

    if "SC-score" in metrics:
        # final metric: harmonic mean of quality and diversity metric
        quality = results["quality"]
        diversity = results["variety"]
        results["SC-score"] = (2 * quality * diversity) / (quality + diversity)
    return results


def compute_metrics(y_label: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, n_jobs: int = 1) -> Dict[str, float]:
    y_pred = y_pred.astype(np.int_)

    def _compute(metric_name: str) -> Tuple[str, float]:
        metric: Metric = METRIC_MAPPING[metric_name]
        if metric.supports_continuous_scorings():
            score = metric(y_label, y_score)
        else:
            score = metric(y_label, y_pred)
        return metric_name, score

    with tqdm_joblib(tqdm(desc="Computing metrics", total=len(METRIC_MAPPING))):
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_compute)(metric_name) for metric_name in METRIC_MAPPING
        )
    results = dict(results)
    return results


def evaluate_result(dataset: TestDataset,
                    results: pd.DataFrame,
                    base_scores_path: Path,
                    normalization_method: str = "minmax",
                    combination_method: str = "custom",
                    k: int = 6,) -> Dict[str, float]:
    results = results[:k]
    scores = load_scores(algorithm_instances(results),
                         dataset_id=dataset.hexhash,
                         base_scores_path=base_scores_path,
                         normalization_method=normalization_method)
    combined_score = aggregate_scores(scores.values, combination_method)
    return compute_metrics(dataset.label, combined_score, combined_score > 0)


def evaluate_individual_results(dataset: TestDataset,
                                results: pd.DataFrame,
                                base_scores_path: Path,) -> Dict[str, Dict[str, float]]:
    scores = load_scores(algorithm_instances(results),
                         dataset_id=dataset.hexhash,
                         base_scores_path=base_scores_path,
                         normalization_method=None)

    metrics = {}
    for a in scores.columns:
        s = scores[a]
        pred = SigmaThresholding(2).fit_transform(dataset.label, s)
        metrics["-".join(a.split("-")[:-1])] = compute_metrics(dataset.label, s, pred)
    return metrics
