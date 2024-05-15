from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .remote_funcs import _cleaning_job
from ..timer import Timers
from ...config import config
from ...dataset import TrainingDatasetCollection, BaseTSDataset, TrainDataset
from ...util import mask_to_slices, invert_slices


def execute_algorithms(dataset_collection: TrainingDatasetCollection) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    log = logging.getLogger("autotsad.data_gen.cleaning")
    Timers.start("exec_algos")

    # prepare algorithms
    jobs: List[Tuple[str, TrainDataset, Dict[str, Any]]] = []
    for d in dataset_collection:
        jobs.append(("subsequence_lof", d, {"window_size": d.period_size}))
        jobs.append(("subsequence_lof", d, {"window_size": d.period_size // 2}))
        jobs.append(("subsequence_if", d, {"window_size": d.period_size}))
        jobs.append(("subsequence_if", d, {"window_size": d.period_size // 2}))
        jobs.append(("subsequence_knn", d, {"window_size": d.period_size}))
        jobs.append(("subsequence_knn", d, {"window_size": d.period_size // 2}))
        jobs.append(("dwt_mlead", d, {}))
        jobs.append(("grammarviz", d, {"anomaly_window_size": d.period_size}))

    # run algorithms in parallel
    log.info(f"Running algorithms on datasets in parallel ({len(jobs)} / {joblib.effective_n_jobs(config.general.n_jobs)}) "
             f"- however, this could take a while!")
    with tqdm_joblib(tqdm(desc=f"Running cleaning algorithms on datasets",
                          total=len(jobs),
                          disable=not config.general.progress)):
        results = joblib.Parallel(n_jobs=min(config.general.n_jobs, len(jobs)))(
            joblib.delayed(_cleaning_job)(algo, dataset, params, config.general) for algo, dataset, params in jobs
        )

    # organize and return results
    dataset_names, algos, scores = zip(*results)
    algo_names = {}
    algo_scores = {}
    for name in np.unique(dataset_names):
        idx = np.argwhere(np.array(dataset_names) == name).reshape(-1)
        algo_names[name] = np.array(algos, dtype=str)[idx]
        algo_scores[name] = np.stack(np.array(scores, dtype=np.object_)[idx], axis=-1).astype(np.float_).T

    Timers.stop("exec_algos")
    return algo_names, algo_scores


def anomaly_predictions(scores: np.ndarray, algos: np.ndarray, d: BaseTSDataset) -> np.ndarray:
    log = logging.getLogger("autotsad.data_gen.cleaning")
    prediction_list = []
    # thresholds = []
    for s, name in zip(scores, algos):
        log.debug(f"Processing algo {name} with scores {s.shape}")

        threshold = np.nanpercentile(s,
                                     q=config.data_gen.anom_filter_scoring_threshold_percentile,
                                     method="median_unbiased")
        predictions = s <= threshold

        if config.data_gen_plotting.cleaning_algo_scores:
            plt.figure()
            plt.title(f"{name} on {d.name}")
            plt.plot(s, label="scoring")
            plt.hlines([s.mean()], 0, d.length, color="green", label="mean")
            mean_limit = s.min() + (s.max() - s.min()) * config.data_gen.ANOM_FILTER_BAD_SCORING_MEAN_LIMIT
            plt.hlines([mean_limit], 0, d.length, linestyles="dashed", color="black", label="mean limit")
            plt.hlines([threshold], 0, d.length, color="red", label="threshold")
            plt.legend()

        # HEURISTIC 0: if scores contain NaNs or Infs, ignore the results
        if np.any(np.isnan(s) | np.isinf(s) | np.isneginf(s)):
            log.info(f"Found Inf or NaN in scoring of algo {name}, ignoring!")
            prediction_list.append(np.ones_like(s, dtype=np.bool_))
            # thresholds.append(np.inf)
            continue

        # HEURISTIC 1: ignore results from confused algorithms
        if s.mean() > s.min() + (s.max() - s.min()) * config.data_gen.ANOM_FILTER_BAD_SCORING_MEAN_LIMIT:
            log.info(f"Mean is above it's upper limit for algo {name}, ignoring!")
            prediction_list.append(np.ones_like(s, dtype=np.bool_))
            # thresholds.append(s.max() + 0.1)
            continue

        # HEURISTIC 2: periodic predictions --> increase threshold and try again
        distinct_values = np.unique(s)
        if len(distinct_values) == 1:
            log.warning(f"Only one distinct value in scoring of algo {name}, ignoring!")
            prediction_list.append(np.ones_like(s, dtype=np.bool_))
            # thresholds.append(distinct_values[0])
            continue

        while np.ceil(
                s.shape[0] / d.period_size * config.data_gen.ANOM_FILTER_PERIODIC_SCORING_PERCENTAGE
        ) < np.sum(np.diff(np.r_[0, predictions, 0]) == 1):
            log.debug(f"{name} predicted too many regions to remove (threshold={threshold}) "
                      f"--> looking for more relaxed threshold (next threshold {threshold})")
            current_threshold_idx = np.argmin(np.abs(distinct_values - threshold))
            threshold = distinct_values[min(current_threshold_idx + 1, distinct_values.shape[0])]
            predictions = s <= threshold

        prediction_list.append(predictions)
        # thresholds.append(threshold)
    return np.array(prediction_list, dtype=np.bool_)


def clean_base_timeseries(dataset_collection: TrainingDatasetCollection, use_gt: bool = False) -> TrainingDatasetCollection:
    log = logging.getLogger("autotsad.data_gen.cleaning")
    algo_names_cache_path = config.general.cache_dir() / "dataset-algo-names.pkl"
    algo_scores_cache_path = config.general.cache_dir() / "dataset-algo-scores.pkl"
    Timers.start("Cleaning")

    if not use_gt:
        # run TSAD algorithms on base TS to get anomaly scores
        if algo_names_cache_path.exists() and algo_scores_cache_path.exists():
            Timers.start("Loading cached results")
            algo_names = joblib.load(algo_names_cache_path)
            algo_scores = joblib.load(algo_scores_cache_path)
            Timers.stop("Loading cached results")

        else:
            algo_names, algo_scores = execute_algorithms(dataset_collection)
            joblib.dump(algo_names, algo_names_cache_path)
            joblib.dump(algo_scores, algo_scores_cache_path)
            log.debug("Saved results to disk")

    # compute threshold, binary predictions, and remove all vaguely anomalous regions
    Timers.start("Remove anomalies")
    for d in dataset_collection:
        if not isinstance(d, BaseTSDataset):  # mainly to make type checker happy
            continue
        log.info(f"Removing anomalies from dataset {d.name}")

        if use_gt:
            ########################################
            # using groundtruth to cut out real anomalies
            log.error("Computing cutout using ground truth!")
            label = dataset_collection.test_data.label
            m = d.reverse_index_mapping
            indices = np.array(list(m.keys()))
            indices.sort()
            cutout = np.zeros_like(d.label, dtype=np.bool_)
            log.debug(f"Calculating ground truth for {d.name}, indices={indices}")
            for b, e in mask_to_slices(label):
                errors = 0
                try:
                    begin = m[b]
                except KeyError:
                    # find next valid index
                    next_b_i = np.nonzero(indices > b)[0]
                    if len(next_b_i) == 0:
                        log.debug(f"Skipped because end of TS reached ({b=} > {indices[-1]=})")
                        continue
                    begin = m[indices[next_b_i][0]]
                    errors += 1

                try:
                    end = m[e-1]+1
                except KeyError:
                    # find previous valid index
                    next_e_i = np.nonzero(indices < e)[0]
                    if len(next_e_i) == 0:
                        log.debug(f"Skipped because begin of TS reached ({e=} < {indices[0]=})")
                        continue
                    end = m[indices[next_e_i][-1]-1]+1
                    errors += 1

                if errors >= 2:
                    log.debug(f"Skipped because anomaly not within dataset")
                    continue
                log.debug(f"Mapped anomaly [{b}:{e}] to [{begin}:{end}]")
                cutout[begin:end] = 1
            cut_slices = mask_to_slices(cutout)
            ########################################
        else:
            scores = algo_scores[d.name]
            algos = algo_names[d.name]
            # for i in range(len(scores)):
            #     scores[i] = MinMaxScaler().fit_transform(scores[i].reshape(-1, 1)).reshape(-1)
            prediction_list = anomaly_predictions(scores, algos, d)

            log.info("Computing cutout")
            votes = np.sum(np.array(prediction_list), axis=0)
            vote_threshold = len(prediction_list) * config.data_gen.anom_filter_voting_threshold
            cutout = votes < vote_threshold
            cut_slices = mask_to_slices(cutout)
            log.debug(
                f"{len(prediction_list)} of {len(algos)} algorithms voted with threshold {vote_threshold}: "
                f"Cutting out {len(cut_slices)} regions."
            )

        # add too small, kept subsequences to the anomaly cutouts
        inv_slices = invert_slices(cut_slices, first_last=(0, d.length))
        fix_slices = inv_slices[np.abs(inv_slices[:, 1] - inv_slices[:, 0]) < d.period_size]
        for s, e in fix_slices:
            log.debug(f"Adding encased region {[s, e]} to cutout")
            cutout[s:e] = True
        # recompute slices
        cut_slices = mask_to_slices(cutout)

        if config.data_gen_plotting.cutouts:
            fig, axs = plt.subplots(3, 2, sharex="col")
            axs[0, 0].set_title("original TS")
            original_data = dataset_collection.test_data.data
            axs[0, 0].plot(original_data, label="TS")

            axs[1, 0].set_title("masked TS")
            data = original_data.copy()
            data[~d.mask] = np.nan
            axs[1, 0].plot(data, label="training TS")
            idx_mapping = d.index_mapping
            for b, e in cut_slices:
                begin = idx_mapping[b]
                end = idx_mapping[e-1]+1
                axs[1, 0].add_patch(Rectangle(
                    (begin, original_data.min()-original_data.std()),
                    end-begin,
                    original_data.max()-original_data.min()+2*original_data.std(),
                    edgecolor="orange", facecolor="yellow", alpha=0.5
                ))

            d.plot(ax=axs[1, 1], cuts=True)
            for b, e in cut_slices:
                axs[1, 1].add_patch(Rectangle(
                    (b, d.data.min()-d.data.std()),
                    e-b,
                    d.data.max()-d.data.min()+2*d.data.std(),
                    edgecolor="orange", facecolor="yellow", alpha=0.5
                ))

        # remove cutouts from datasets:
        d.remove_slices(cut_slices)

        if config.data_gen_plotting.cutouts:
            axs[2, 0].set_title("masked TS - removed anomalies")
            data = original_data.copy()
            data[~d.mask] = np.nan
            axs[2, 0].plot(data, label="training TS")

            d.plot(ax=axs[2, 1], cuts=True)

            axs[0, 0].legend()
            axs[1, 0].legend()
            axs[2, 0].legend()
            axs[1, 1].legend()
            axs[2, 1].legend()

    log.debug(f"Removed anomalies from {len(dataset_collection)} datasets.")
    Timers.stop("Cleaning")
    print("Finished cleaning datasets.")

    if config.data_gen_plotting:
        plt.show()

    return dataset_collection
