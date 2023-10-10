from __future__ import annotations

import logging
from itertools import combinations
from typing import Tuple, Set

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from ..timer import Timers
from ...config import config
from ...dataset import TrainingDatasetCollection, TrainDataset


def _get_logger() -> logging.Logger:
    return logging.getLogger("autotsad.data_gen.regiming.pruning")


def prune_regimes_with_overlap(train_collection: TrainingDatasetCollection,
                               generated_masks: np.ndarray,
                               periods: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Regime overlap pruning:

    Check if one regime mask is (mostly) overlapping with another, then remove the one with more coverage because
    smaller mask approx. equals more specialized base TS. We agree on overlap if the masks share at least
    REGIME_OVERLAP_PRUNING_THRESHOLD * 100% of the original dataset's points.
    """
    log = _get_logger()
    remove_idcs = set()
    for i, j in combinations(np.arange(generated_masks.shape[0]), 2):
        if i in remove_idcs or j in remove_idcs:
            continue

        mask_overlap = np.sum(generated_masks[i] & generated_masks[j]) / generated_masks.shape[1]
        if mask_overlap > config.data_gen.REGIME_OVERLAP_PRUNING_THRESHOLD:
            if np.sum(generated_masks[i]) < np.sum(generated_masks[j]):
                remove_idx = j
            else:
                remove_idx = i
            log.info(f"Removing regime #{remove_idx} due to overlap between #{i} and #{j} "
                     f"({mask_overlap:.2%} > {config.data_gen.REGIME_OVERLAP_PRUNING_THRESHOLD:.2%})")
            remove_idcs.add(remove_idx)

            if config.data_gen_plotting.regime_overlap_pruning:
                fig, axs = plt.subplots(3, 1, sharex="col")
                axs[0].set_title(f"Regime overlap pruning for regimes #{i} and #{j}")
                train_collection.test_data.plot(ax=axs[0])
                axs[0].legend()
                axs[1].plot(generated_masks[i] & generated_masks[j], label="overlap")
                axs[1].legend()
                axs[2].plot(generated_masks[i],
                            label=f"regime #{i} ps={periods[i]} {'(keep)' if i != remove_idx else ''}")
                axs[2].plot(generated_masks[j],
                            label=f"regime #{j} ps={periods[i]} {'(keep)' if j != remove_idx else ''}")
                axs[2].legend()
                plt.show()

    log.debug(f"BEFORE regime overlap pruning: {generated_masks.shape[0]} regimes with periods: {periods}")
    generated_masks = np.delete(generated_masks, list(remove_idcs), axis=0)
    periods = np.delete(periods, list(remove_idcs), axis=0)
    log.debug(f"AFTER regime overlap pruning: {generated_masks.shape[0]} regimes with periods: {periods}")
    return generated_masks, periods


def filter_similar_timeseries(train_collection: TrainingDatasetCollection) -> None:
    """Dataset similarity pruning:

    Check if two timeseries are similar (i.e. their distance is below a threshold), then remove the one whose period
    size is less frequent in the dataset.
    """
    log = _get_logger()
    Timers.start("Filtering similar timeseries")

    length_groups = {}
    period_size_groups = {}
    for d in train_collection.datasets:
        length_groups.setdefault(d.length, []).append(d)
        period_size_groups.setdefault(d.period_size, []).append(d)

    for length, group in length_groups.items():
        data = np.vstack([d.data for d in group])
        data = MinMaxScaler().fit_transform(data.T).T  # FIXME: shouldn't we rather use StandardScaler?
        dists = euclidean_distances(data)
        threshold = config.data_gen.DATASET_PRUNING_SIMILARITY_THRESHOLD * length
        log.debug(f"Distance matrix for length {length} (sim. {threshold=:.2f}):\n{dists}")
        dists = dists < threshold

        to_remove: Set[TrainDataset] = set()
        i = 0
        while i < dists.shape[0]:
            sim_idxs = np.arange(dists.shape[0])[dists[i, :]]
            if len(sim_idxs) > 1:
                log.debug(f"Found {len(sim_idxs)} similar timeseries of length {length}: {sim_idxs}")
                ps_cardinalities = np.array([len(period_size_groups[group[i].period_size]) for i in sim_idxs])
                idx_to_remove = sim_idxs[np.argsort(ps_cardinalities)[-(len(sim_idxs) - 1):]]

                for idx in idx_to_remove:
                    d = group[idx]
                    log.info(f"Removing redundant timeseries {d.name} (length={d.length}, period_size={d.period_size})")
                    if d in period_size_groups[d.period_size]:
                        period_size_groups[d.period_size].remove(d)
                    to_remove.add(d.name)
                remaining_dists_idxs = [i for i in np.arange(dists.shape[0]) if i not in idx_to_remove]
                dists = dists[remaining_dists_idxs, :][:, remaining_dists_idxs]
            i += 1
        # actually remove the datasets from the collection
        train_collection.datasets = [d for d in train_collection.datasets if d.name not in to_remove]
    Timers.stop("Filtering similar timeseries")
