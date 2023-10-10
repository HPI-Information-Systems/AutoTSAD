from __future__ import annotations

import logging
import multiprocessing
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ...config import config, GeneralSection, DataGenerationSection, DataGenerationPlottingSection
from ...dataset import TrainingDatasetCollection, BaseTSDataset, Dataset
from ...util import slice_lengths, mask_to_slices, slices_to_mask


def _plot_sampled_regime(dataset: Dataset, masks: np.ndarray) -> None:
    n_samples = masks.shape[0]
    length = np.sum(masks[0], axis=0)
    fig, axs = plt.subplots(2, 1, sharex="col")
    axs[0].set_title(f"Extracting {n_samples} samples of size {length} of the single regime")
    axs[0].plot(dataset.data, label="timeseries")
    y0, y1, = axs[0].get_ylim()
    for i in np.arange(n_samples):
        idx = np.argmax(masks[i])
        axs[0].add_patch(
            Rectangle((idx, y0), length, y1 - y0, edgecolor="black", facecolor="lightgray",
                      alpha=0.5)
        )
    axs[1].plot(np.bitwise_or.reduce(masks, axis=0), label="mask")
    axs[0].legend()
    axs[1].legend()
    plt.show()


class TimeseriesLimiter:
    def __init__(self,
                 general_config: GeneralSection = config.general,
                 data_gen_config: DataGenerationSection = config.data_gen,
                 plotting_config: DataGenerationPlottingSection = config.data_gen_plotting,
                 ) -> None:
        self._general_config = general_config
        self._data_gen_config = data_gen_config
        self._plotting_config = plotting_config
        self._plot = self._plotting_config.truncated_timeseries
        self._log = logging.getLogger(f"autotsad.data_gen.{self.__class__.__name__}")
        process = multiprocessing.current_process()
        if process._parent_pid is None:
            from ..timer import Timers

            self._timers = Timers
        else:
            from ..timer import TimersMngr

            self._timers = TimersMngr(f"Limiting-{process.pid}", general_config=general_config)

    def _find_subsequence_with_least_cuts(self, dataset: BaseTSDataset, desired_length: int) -> int:
        self._timers.start(f"Find least cuts-{dataset.name}")
        n_cuts = len(dataset.cut_points())
        cuts = np.r_[0, dataset.cut_points(), dataset.length].astype(np.int_)
        cut_indices = np.arange(0, n_cuts + 2)
        # print(f"{cuts=}")
        for allowed_cuts in range(n_cuts + 1):
            space = np.roll(cuts, -(allowed_cuts + 1)) - cuts
            # print(f"{space=}")
            idx = cut_indices[space >= desired_length]
            if len(idx) > 0:
                middle = int(np.ceil((len(idx) - 1) / 2))
                subsequence_idx = cuts[idx[middle]]
                self._log.debug(f"... found idx={subsequence_idx} with {allowed_cuts} cuts.")
                self._timers.stop(f"Find least cuts-{dataset.name}")
                return subsequence_idx

    def _get_length_limit(self, period_size: int, soft: bool = False) -> int:
        length_limit = max(
            self._general_config.TRAINING_TIMESERIES_LENGTH,
            period_size * self._general_config.TRAINING_TIMESERIES_MIN_NO_PERIODS
        )
        # length_limit = 1000
        if soft:
            length_limit = int(length_limit * 1.5)
        return length_limit

    def enforce_length_limit(self, dataset: BaseTSDataset) -> None:
        self._timers.start(f"Limiting-{dataset.name}")
        length_limit = self._get_length_limit(dataset.period_size)

        # find the best subsequence ordering of desired length (with the least amount of cuts/jumps)
        if dataset.length > length_limit:
            self._log.debug(f"Base TS {dataset.name} (length={dataset.length}) exceeds length limit of {length_limit}, "
                            "truncating ...")
            slices = np.c_[np.r_[0, dataset.cut_points()], np.r_[dataset.cut_points(), dataset.length]]
            slices = self._select_slices(slices, length_limit)
            mask = slices_to_mask(slices, dataset.length)
            remove_slices = mask_to_slices(~mask)
        else:
            self._log.debug(f"Base TS {dataset.name} (length={dataset.length}) is within length limit of "
                            f"{length_limit}.")
            slices = np.array([[0, dataset.length]], dtype=np.int_)
            remove_slices = np.array([], dtype=np.int_)

        if self._plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].set_title(f"Selected subsequence of maximum length={length_limit}")
            dataset.plot(cuts=True, ax=axs[0])
            y0, y1 = axs[0].get_ylim()
            height = y1 - y0
            for b, e in slices:
                axs[0].add_patch(
                    Rectangle((b, y0), e-b, height,
                              edgecolor="black",
                              facecolor="lightgray",
                              alpha=0.5)
                )
            axs[0].legend()

        # remove everything else
        dataset.remove_slices(remove_slices)
        self._timers.stop(f"Limiting-{dataset.name}")

        if self._plot:
            axs[1].set_title("Truncated timeseries")
            dataset.plot(cuts=True, ax=axs[1])
            axs[1].legend()
            plt.show()

    def enforce_length_limit_old(self, dataset: BaseTSDataset) -> None:
        self._timers.start(f"Limiting-{dataset.name}")
        length_limit = self._get_length_limit(dataset.period_size)

        # find the best subsequence of desired length (with the least amount of cuts)
        if dataset.length > length_limit:
            self._log.debug(f"Base TS {dataset.name} (length={dataset.length}) exceeds length limit of {length_limit}, "
                            "truncating ...")
            idx = self._find_subsequence_with_least_cuts(dataset, length_limit)
        else:
            self._log.debug(f"Base TS {dataset.name} (length={dataset.length}) is within length limit of "
                            f"{length_limit}.")
            idx = 0

        if self._plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].set_title(f"Selected subsequence of maximum length={length_limit}")
            dataset.plot(cuts=True, ax=axs[0])
            y0, y1 = axs[0].get_ylim()
            height = y1 - y0
            axs[0].add_patch(
                Rectangle((idx, y0), min(length_limit, dataset.length), height,
                          edgecolor="black",
                          facecolor="lightgray",
                          alpha=0.5)
            )
            axs[0].legend()

        # remove everything else
        remove_slices = np.array([[idx + min(length_limit, dataset.length) - 1, dataset.length]])
        if idx != 0:
            start_slice = [[0, idx]]
            remove_slices = np.concatenate([start_slice, remove_slices], axis=0)
        dataset.remove_slices(remove_slices)
        self._timers.stop(f"Limiting-{dataset.name}")

        if self._plot:
            axs[1].set_title("Truncated timeseries")
            dataset.plot(cuts=True, ax=axs[1])
            axs[1].legend()
            plt.show()

    def extract_sampled_regime(self, collection: TrainingDatasetCollection,
                               period_size: int,
                               in_place: bool = True) -> Optional[np.ndarray]:
        """Implements simple random sampling as base behavior extraction strategy.

        If the dataset is smaller than the desired length, the whole dataset is used.
        If we can only extract all desired samples by sampling uniformly from the whole dataset, we do so.
        """
        max_samples: int = self._data_gen_config.regime_max_sampling_number
        max_overlap: float = self._data_gen_config.regime_max_sampling_overlap
        plot: bool = self._plotting_config.subsequence_sampling
        rng = np.random.default_rng(self._general_config.seed)

        self._timers.start("Extracting sample")
        dataset = collection.test_data
        desired_length = self._get_length_limit(period_size, soft=True)
        if max_samples < 1:
            self._log.warning(f"{max_samples=} must be greater than 0; assuming it is 1!")
            max_samples = 1

        if dataset.length < desired_length:
            self._log.info(f"...dataset is smaller than maximum length, using whole dataset of length={dataset.length}")

            # take everything
            masks = np.ones((1, dataset.length), dtype=np.bool_)
            self._timers.stop("Extracting sample")

            if plot:
                _plot_sampled_regime(dataset, masks)

        else:
            n_samples = min(
                max_samples,
                int((dataset.length - desired_length * max_overlap) / ((1 - max_overlap) * desired_length))
            )
            self._log.debug(f"... extracting {n_samples} samples")
            if n_samples == 1:
                # sample a single random subsequence
                random_idx = rng.integers(low=0, high=dataset.length - desired_length, endpoint=True)
                start_idxs = np.array([random_idx], dtype=np.int_)
            elif (n_samples + 1) * desired_length - n_samples * (desired_length * max_overlap) > dataset.length:
                self._log.debug("... too tight for random sampling, extracting uniformly")
                # sample uniformly
                if n_samples == 2:
                    # sample two evenly spaced subsequences
                    spacing = dataset.length // 4
                    start_idxs = np.array([spacing - desired_length//2, 3*spacing - desired_length//2], dtype=np.int_)
                else:
                    # evenly space subsequences from both ends
                    start_idxs = np.linspace(0, dataset.length - desired_length, n_samples, endpoint=True, dtype=np.int_)
            else:
                # sample randomly (greedy)
                start_idxs = np.zeros(n_samples, dtype=np.int_)
                i = 0
                n_tries = 0
                while i < n_samples and n_tries < 500:
                    n_tries += 1
                    idx = rng.integers(low=0, high=dataset.length - desired_length, endpoint=True)
                    if not (
                            np.any(  # check if begin is in any of the already selected slices
                                (start_idxs[:i] < idx) &
                                (idx <= start_idxs[:i] + desired_length * (1 - max_overlap))
                            ) or
                            np.any(  # check if end is in any of the already selected slices
                                (start_idxs[:i] + desired_length * max_overlap <= idx + desired_length) &
                                (idx + desired_length < start_idxs[:i] + desired_length)
                            )
                    ):
                        start_idxs[i] = idx
                        i += 1

                if i < n_samples:
                    self._log.warning(f"Could not find valid positions for all samples, only found {i} samples!")
                    n_samples = i
                    start_idxs = start_idxs[:i]

            masks = np.zeros((n_samples, dataset.length), dtype=np.bool_)
            for i, idx in enumerate(start_idxs):
                masks[i, idx:idx + desired_length] = True
            self._timers.stop("Extracting sample")

            if plot:
                _plot_sampled_regime(dataset, masks)

        if in_place:
            for i in range(masks.shape[0]):
                collection.add_base_ts(mask=masks[i, :], period_size=period_size)
        else:
            return masks

    def _select_slices(self, slices: np.ndarray, desired_length: int) -> np.ndarray:
        # sort slices according to their length and add them incrementally until desired length is exceeded
        sls = slice_lengths(slices)
        sorted_idxs = np.argsort(sls)[::-1]
        # sum up lengths until desired length is exceeded (+ next slice because we want to be larger)
        indices = np.nonzero(np.cumsum(sls[sorted_idxs]) < desired_length)[0]
        if indices.shape[0] == 0:  # if first slice is already larger than desired length, use it
            indices = [0]
        elif indices.shape[0] == sorted_idxs.shape[0]:  # if we need all slices, use all
            pass
        else:  # if we selected a specific number of slices, we need to add one to exceed the desired length
            indices = np.r_[indices, indices[-1] + 1]
        # print(f"{sorted_idxs=}")
        # print(f"{indices=}")
        indices = sorted_idxs[indices]
        self._log.debug(f"Removing {slices.shape[0] - indices.shape[0]} excess small slices; "
                        f"remaining length={int(np.cumsum(sls[indices])[-1])}.")
        slices = slices[indices]
        # cap the last slice to the desired length
        # FIXME: guard against too small slices
        length_reduction = int(np.cumsum(sls[indices])[-1] - desired_length)
        slices[-1, 1] -= length_reduction
        self._log.debug(f"Reducing last slice by {length_reduction} to reach desired length={desired_length}")
        sls = slice_lengths(slices)
        self._log.debug(f"Selected slices={slices.tolist()} and their lengths={sls.tolist()}")
        return slices


def limit_base_timeseries(collection: TrainingDatasetCollection) -> TrainingDatasetCollection:
    from ..timer import Timers

    limiter = TimeseriesLimiter()
    Timers.start("Limiting")
    for d in collection:
        assert isinstance(d, BaseTSDataset)
        limiter.enforce_length_limit(d)
    Timers.stop("Limiting")
    return collection
