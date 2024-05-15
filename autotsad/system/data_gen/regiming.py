from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import stumpy
from matplotlib import pyplot as plt

from .limiting import TimeseriesLimiter
from ...config import config, DataGenerationPlottingSection, DataGenerationSection, GeneralSection
from ...dataset import TrainingDatasetCollection
from ...util import mask_to_slices, invert_slices, slice_lengths

color_vec = ['orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'yellow', 'black']
tiny = np.finfo(np.float_).tiny


def find_best_k(areas: np.ndarray, period: int,
                data_gen_config: DataGenerationSection = config.data_gen,
                plot_profile_area: bool = config.data_gen_plotting.profile_area) -> int:
    k_range = np.arange(1, data_gen_config.snippets_max_no + 1)
    if areas.min() < tiny:
        best_k = k_range[np.argmin(areas)]
    else:
        df_aud = pd.DataFrame({
            "area": areas,
            "area_change": np.r_[np.nan, areas[:-1] / areas[1:] - 1.0],
            "k": k_range
        })
        if df_aud["area_change"].max() < data_gen_config.SNIPPETS_PROFILE_AREA_CHANGE_THRESHOLD:
            best_k = 1
        else:
            best_k = df_aud.loc[df_aud["area_change"].argmax(), "k"]

        if plot_profile_area:
            fig, axs = plt.subplots(2, 1, sharex="col")
            axs[0].set_title(f"Profile area for snippets of length {period}")
            axs[0].bar(df_aud.index, df_aud["area"], tick_label=df_aud["k"], label="area")
            axs[1].set_title(f"Change in profile area")
            axs[1].bar(df_aud.index, df_aud["area_change"], tick_label=df_aud["k"], label="area change")
            axs[1].vlines([best_k - .5], 0, df_aud["area_change"].max(), label="Threshold", color="orange")
            axs[1].set_xlabel("k")

    return best_k


def extract_regimes(profiles: np.ndarray, p_idx: np.ndarray, period_size: int,
                    data_gen_config: DataGenerationSection = config.data_gen,
                    plog: logging.Logger = logging.getLogger("autotsad.data_gen.regiming")) -> Tuple[np.ndarray, np.ndarray]:
    regimes = []
    masks = np.zeros_like(profiles, dtype=np.bool_)
    total_min = np.min(profiles[p_idx], axis=0)
    # add a small epsilon to prevent quickly shifting between multiple good matching snippets
    masks[p_idx, :] = profiles[p_idx, :] <= total_min + 1e-6
    slices = {}
    for i in p_idx:
        slices[i] = mask_to_slices(masks[i])

    # add small skipped subsequences to the same regime
    for i in p_idx:
        candidate_slices = slices[i]
        # only consider strong (large enough slices) to compute skipped regions
        plog.debug(f"snippet {i}: all slices: {candidate_slices}")
        strong_slices = candidate_slices[
            slice_lengths(candidate_slices) > data_gen_config.REGIME_STRONG_PERCENTAGE * period_size
        ]
        plog.debug(f"snippet {i}: strong slices: {strong_slices}")

        inv_slices = invert_slices(strong_slices)
        plog.debug(f"inverse slices {i}: {inv_slices}")

        fix_slices = inv_slices[
            slice_lengths(inv_slices, absolute=True) < data_gen_config.REGIME_CONSOLIDATION_PERCENTAGE * period_size
        ]
        plog.debug(f"fix slices {i}: {fix_slices}")

        for s, e in fix_slices:
            plog.debug(f"Adding skipped region {[s, e]} to snippet #{i}")
            masks[i, s:e] = True

    # # recompute slices
    # for i in range(best_k):
    #     slices[i] = mask_to_slices(masks[i])
    #
    # # remove small slices that are fully contained in another slice of another snippet (modifies masks)
    # for i in range(best_k):
    #     candidate_slices = slices[i]
    #     mask = slice_lengths(candidate_slices) <= data_gen_config.REGIME_STRONG_PERCENTAGE * period_size
    #     candidate_slices_idx = np.where(mask)[0]
    #     candidate_slices = candidate_slices[mask]
    #     if candidate_slices.shape[0] == 0:
    #         continue
    #     for j in range(best_k):
    #         if i == j:
    #             continue
    #         cs_mask = np.zeros_like(candidate_slices_idx, dtype=np.bool_)
    #         for b, e in slices[j]:
    #             cs_mask |= (b <= candidate_slices[:, 0]) & (candidate_slices[:, 1] <= e)
    #         for ib, ie in candidate_slices[cs_mask]:
    #             log.debug(f"Removing encased slice {[ib, ie]} from #{i} (within #{j})")
    #             masks[i, ib:ie] = False
    #         slices_mask = np.ones(slices[i].shape[0], dtype=np.bool_)
    #         slices_mask[candidate_slices_idx[cs_mask]] = False
    #         slices[i] = slices[i][slices_mask]
    #         candidate_slices = candidate_slices[~cs_mask]
    #         candidate_slices_idx = candidate_slices_idx[~cs_mask]

    # limiter = TimeseriesLengthLimiter()
    for i in p_idx:
        # recompute slices
        slices = mask_to_slices(masks[i])

        # only consider strong slices
        mask = slice_lengths(slices) > data_gen_config.REGIME_STRONG_PERCENTAGE * period_size
        if mask.sum() > 0:
            plog.debug(f"Removing {slices.shape[0] - mask.sum()} too short slices from #{i}")
            slices = slices[mask]

        # check if we have many cuts/regime changes (--> small slices), then ignore this snippet and recompute
        median_periods_per_slice = np.median(slice_lengths(slices) // period_size)
        plog.debug(f"Median number of periods per slices: {median_periods_per_slice}")
        if median_periods_per_slice < data_gen_config.REGIME_MIN_PERIODS_PER_SLICE_FOR_VALID_SNIPPET:
            plog.warning(f"Regimes are too short for snippet #{i}: {median_periods_per_slice} < "
                         f"{data_gen_config.REGIME_MIN_PERIODS_PER_SLICE_FOR_VALID_SNIPPET}, rerunning regiming w/o it!")
            return extract_regimes(profiles, np.delete(p_idx, np.where(p_idx == i)), period_size, data_gen_config, plog)
            # return extract_regimes(np.delete(profiles, i, axis=0), best_k-1, period_size)

        # # select the largest slices until we exceed the desired training dataset length
        # desired_length = limiter._get_length_limit(period_size, soft=True)
        # slices = limiter.select_slices(slices, desired_length=desired_length)

        # adjust slice indices based on sub-window-size
        slices += int(data_gen_config.SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE * period_size)
        regimes.append(np.c_[np.repeat(i, slices.shape[0]), slices])

    # include borders
    # sort_indices = regimes[:, 1].argsort(axis=0)
    # regimes[sort_indices[0], 1] = 0
    # regimes[sort_indices[-1], 2] = train_collection.test_data.length - 1

    return np.vstack(regimes), p_idx


def get_regime_masks(train_collection: TrainingDatasetCollection,
                     period_size: int,
                     general_config: GeneralSection = config.general,
                     data_gen_config: DataGenerationSection = config.data_gen,
                     data_gen_plot_config: DataGenerationPlottingSection = config.data_gen_plotting,
                     plog: logging.Logger = logging.getLogger("autotsad.data_gen.regiming")) -> np.ndarray:
    plog.info(f"Processing period {period_size}")
    if period_size > data_gen_config.AUTOPERIOD_MAX_PERIOD_LENGTH:
        plog.warning(
            f"Period={period_size} too large (> MAX_PERIOD_LENGTH={data_gen_config.AUTOPERIOD_MAX_PERIOD_LENGTH}), ignoring")
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)
    if period_size < 3 / data_gen_config.SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE:
        plog.warning(f"Period={period_size} too small for proper window size, ignoring")
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)
    if period_size >= 0.1 * train_collection.test_data.length:
        plog.warning(f"Period={period_size} too large (>= 10% of TS length) for proper window size, ignoring")
        return np.empty((0, train_collection.test_data.length), dtype=np.bool_)

    # Timers.start(f"Stumpy-{period_size}")
    plog.info("  running snippet detection (stumpy)")
    _snippets, indices, profiles, _fractions, areas, _regimes = stumpy.snippets(
        T=train_collection.test_data.data,
        m=period_size,
        k=data_gen_config.snippets_max_no,
        percentage=data_gen_config.SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE
    )
    best_k = find_best_k(areas,
                         period=period_size,
                         data_gen_config=data_gen_config,
                         plot_profile_area=data_gen_plot_config.profile_area)
    plog.debug(f"  best k={best_k}")
    # Timers.stop(f"Stumpy-{period_size}")

    if best_k == 1:
        plog.info(f"Only one snippet found! Extracting max random subsequences as base TSs!")
        return TimeseriesLimiter(
            general_config=general_config,
            data_gen_config=data_gen_config,
            plotting_config=data_gen_plot_config,
        ).extract_sampled_regime(train_collection, period_size=period_size, in_place=False)

    # Timers.start(f"Regime extraction-{period_size}")
    plog.info("  extracting regimes...")
    # extract snippet regimes
    old_best_k = best_k
    # use only the best k snippets
    p_idx = np.arange(profiles.shape[0])[:best_k]
    regimes, p_idx = extract_regimes(profiles, p_idx, period_size, data_gen_config=data_gen_config, plog=plog)
    best_k = p_idx.shape[0]
    if best_k != old_best_k:
        plog.info(f"  number of snippets changed from {old_best_k} to {best_k}!")

    # create base time series
    masks = np.zeros((best_k, train_collection.test_data.length), dtype=np.bool_)
    for i, snippet_idx in enumerate(p_idx):
        slices_of_indices = regimes[np.where(regimes[:, 0] == snippet_idx)][:, 1:]
        for start_idx, stop_idx in slices_of_indices:
            masks[i, start_idx:stop_idx] = True
    # Timers.stop(f"Regime extraction-{period_size}")

    # now follows all plotting code:
    if data_gen_plot_config.snippets:
        plt.figure()
        plt.title(f"Detected snippets of length {period_size}")
        y = train_collection.test_data.data
        plt.plot()
        for i in range(data_gen_config.snippets_max_no):
            idx = indices[i]
            # print(f"the starting index of the snippet #{i} with length m={m} is: {idx}")
            plt.plot(range(idx, idx + period_size), y[idx: idx + period_size], c=color_vec[i], lw=2,
                     label=f"snippet #{i}{' (ignored)' if i not in p_idx else ''}")
        plt.legend()

    if data_gen_plot_config.snippet_regimes:
        plt.figure()
        plt.title(f"Regimes for snippets of length {period_size}")
        y = train_collection.test_data.data
        plt.plot(y, c="k", lw=1, label="original TS")
        for snippet_idx in p_idx:
            slices_of_indices = regimes[np.where(regimes[:, 0] == snippet_idx)][:, 1:]
            xx = np.full_like(y, fill_value=np.nan)
            for per_slice in slices_of_indices:
                start_idx = per_slice[0]
                stop_idx = per_slice[1]
                xx[start_idx:stop_idx] = y[start_idx:stop_idx] + snippet_idx + 1  # + (i+1)*10
            plt.plot(xx, c=color_vec[snippet_idx], lw=2, label=f"snippet #{snippet_idx}")
        plt.legend(loc="upper right", fontsize=15)

    if data_gen_plot_config.snippet_profiles:
        plt.figure()
        plt.title(f"Distance profiles of length {period_size}")
        for i in np.arange(profiles.shape[0]):
            plt.plot(profiles[i], c=color_vec[i], lw=2,
                     label=f"snippet #{i} {' (ignored)' if i not in p_idx else ''}")
        plt.legend(loc="center right", fontsize=15)

    if data_gen_plot_config:
        plt.show()

    return masks
