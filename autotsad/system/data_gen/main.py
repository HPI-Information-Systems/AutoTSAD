from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
from matplotlib import pyplot as plt
from periodicity_detection.methods.autoperiod import Autoperiod
from timeeval.utils.tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from .cleaning import clean_base_timeseries
from .injecting import generate_anomaly_time_series
from .limiting import limit_base_timeseries, TimeseriesLimiter
from .pruning import prune_regimes_with_overlap, filter_similar_timeseries
from .regiming import get_regime_masks
from ..logging import LOGGING_Q
from ..timer import Timers
from ...config import config
from ...dataset import TrainingDatasetCollection, TestDataset
from ...util import getsize


def _get_logger() -> logging.Logger:
    return logging.getLogger("autotsad.data_gen")


def analyze_dataset(testdataset: TestDataset) -> List[int]:
    autoperiod = Autoperiod(
        use_number_peaks_fallback=True,
        detrend=True,
        random_state=42,
        return_multi=config.data_gen.autoperiod_max_periods,
        plot=config.data_gen_plotting.autoperiod,
        acf_hill_steepness=1e6,
    )
    periods = autoperiod(testdataset.data)
    if config.data_gen_plotting.autoperiod:
        plt.show()

    _get_logger().info(f"Dataset characteristics:\nInput shape={testdataset.shape}\nPeriods={periods}")
    return periods


def generate_base_ts_collection(testdataset: TestDataset) -> TrainingDatasetCollection:
    log = _get_logger()
    Timers.start("Base TS generation")
    Timers.start("Dataset analysis")
    periods = analyze_dataset(testdataset)

    # # smooth data with gaussian filter
    # data = testdataset.data
    # kernel_size = 41
    # kernel = stats.norm.pdf(np.linspace(-5, 5, kernel_size), loc=0, scale=1)
    # kernel /= np.sum(kernel)
    # data = np.convolve(data, kernel, mode="valid")
    # padding_size = (kernel_size - 1) // 2
    # data = np.pad(data, (padding_size, padding_size), mode="edge")
    #
    # print(f"Periods of original dataset: {periods}")
    # autoperiod = Autoperiod(
    #     use_number_peaks_fallback=True,
    #     detrend=True,
    #     random_state=42,
    #     return_multi=AUTOPERIOD_MAX_PERIODS
    # )
    # periods = autoperiod(data)
    # print(f"Periods of smoothed dataset: {periods}")
    #
    # plt.figure()
    # plt.plot(testdataset.data, lw=1, color="blue", label="original")
    # plt.plot(data, lw=2, color="green", label="smoothed")
    # plt.show()
    # testdataset = TestDataset(data, testdataset.labels)

    train_collection = TrainingDatasetCollection.from_base_timeseries(testdataset)
    Timers.stop("Dataset analysis")
    Timers.start("Dataset segmentation")

    # segment base time series based on prominent behavior per period size
    split_into_regimes(train_collection, periods)

    if len(train_collection) == 0:
        print("No proper period_size found! Extracting samples with period_size 100 as base TSs!")
        log.warning(f"No proper period size found! Just extracting {config.data_gen.regime_max_sampling_number} "
                    f"sampled subsequences with period size 100 as base TSs.")
        TimeseriesLimiter().extract_sampled_regime(train_collection, period_size=100, in_place=True)

    if config.data_gen.enable_dataset_pruning:
        # limit number of similar base time series
        filter_similar_timeseries(train_collection)

    log.debug("Extracted base TSs:")
    log.debug("\tws\tshape\tcut points")
    for i, base_ts in enumerate(train_collection):
        log.debug(f"{i}, {base_ts.period_size}, {base_ts.shape}, {base_ts.cut_points()}")

    Timers.stop("Dataset segmentation")
    Timers.stop("Base TS generation")

    if config.data_gen_plotting.base_ts:
        fig, axs = plt.subplots(train_collection.size, 1, squeeze=False)
        for i, ts in enumerate(train_collection):
            axs[i, 0].set_title(f"Base TS with default window_size {ts.period_size}")
            ts.plot(axs[i, 0], cuts=True)
            axs[i, 0].legend()
        plt.show()

    return train_collection


def split_into_regimes(train_collection: TrainingDatasetCollection, period_sizes: Sequence[int]) -> TrainingDatasetCollection:
    # capture variables from config
    base_cache_dir = config.general.cache_dir()
    general_config = config.general
    data_gen_config = config.data_gen
    data_gen_plot_config = config.data_gen_plotting

    # capture queue object for subprocesses
    q = LOGGING_Q

    def call_get_regime_masks(train_collection: TrainingDatasetCollection, period_size: int) -> Tuple[np.ndarray, np.ndarray]:
        import logging
        from autotsad.system.logging import setup_process_logging

        with setup_process_logging(q) as pid:
            process_log = logging.getLogger(f"autotsad.data_gen.regiming-{pid}")

            masks_cache_path = base_cache_dir / f"masks-{period_size}.pkl"
            if masks_cache_path.exists():
                process_log.info(f"Loading masks for period size {period_size} from cache")
                masks: np.ndarray = joblib.load(masks_cache_path)
            else:
                process_log.info(f"Computing masks for period size {period_size}")
                masks = get_regime_masks(train_collection, period_size, general_config, data_gen_config,
                                         data_gen_plot_config, plog=process_log)
                joblib.dump(masks, masks_cache_path)
            periods = np.repeat(period_size, masks.shape[0])
            return masks, periods

    Timers.start("Regime extraction")
    with tqdm_joblib(tqdm(desc="Extracting regimes", total=len(period_sizes), disable=not config.general.progress)):
        results = joblib.Parallel(n_jobs=min(config.general.n_jobs, len(period_sizes)))(
            joblib.delayed(call_get_regime_masks)(train_collection, period_size) for period_size in period_sizes
        )
    generated_masks, periods = zip(*results)
    generated_masks = np.concatenate(generated_masks, axis=0)
    periods = np.concatenate(periods, axis=0)
    Timers.stop("Regime extraction")

    if config.data_gen.enable_regime_overlap_pruning and len(generated_masks) > 1:
        Timers.start("Regime overlap pruning")
        generated_masks, periods = prune_regimes_with_overlap(train_collection, generated_masks, periods)
        Timers.stop("Regime overlap pruning")

    for i in np.arange(generated_masks.shape[0]):
        train_collection.add_base_ts(generated_masks[i], period_size=periods[i])

    return train_collection


def main(testdataset: TestDataset, use_gt_for_cleaning: bool = False) -> TrainingDatasetCollection:
    print("\n############################################################")
    print("#             STEP 1: Training data generation             #")
    print("############################################################")
    base_ts_cache_path = config.general.cache_dir() / "base-ts-collection.pkl"
    cleaned_base_ts_cache_path = config.general.cache_dir() / "cleaned-base-ts-collection.pkl"
    limited_base_ts_cache_path = config.general.cache_dir() / "limited-base-ts-collection.pkl"
    train_ts_cache_path = config.general.cache_dir() / "train-ts-collection.pkl"
    config.general.cache_dir().mkdir(parents=True, exist_ok=True)

    if limited_base_ts_cache_path.exists() or cleaned_base_ts_cache_path.exists() or train_ts_cache_path.exists():
        pass
    elif base_ts_cache_path.exists():
        train_collection = TrainingDatasetCollection.load(base_ts_cache_path)
    else:
        print("\n# Generating base time series")
        print("###########################")
        train_collection = generate_base_ts_collection(testdataset)
        train_collection.save(base_ts_cache_path)

    if limited_base_ts_cache_path.exists() or train_ts_cache_path.exists():
        pass
    elif cleaned_base_ts_cache_path.exists():
        train_collection = TrainingDatasetCollection.load(cleaned_base_ts_cache_path)
    else:
        print("\n# Cleaning base time series")
        print("###########################")
        train_collection = clean_base_timeseries(train_collection, use_gt=use_gt_for_cleaning)
        train_collection.save(cleaned_base_ts_cache_path)

    if train_ts_cache_path.exists():
        pass
    elif limited_base_ts_cache_path.exists():
        train_collection = TrainingDatasetCollection.load(limited_base_ts_cache_path)
    else:
        print("\n# Limiting base time series")
        train_collection = limit_base_timeseries(train_collection)
        train_collection.save(limited_base_ts_cache_path)

    if train_ts_cache_path.exists():
        print("\n# Loading training time series collection")
        print("###########################")
        train_collection = TrainingDatasetCollection.load(train_ts_cache_path)
    else:
        print("\n# Injecting anomalies")
        print("###########################")
        train_collection = generate_anomaly_time_series(train_collection)
        train_collection.save(train_ts_cache_path)

    _get_logger().info(
        f"Training dataset collection ({len(train_collection)} datasets) size: {getsize(train_collection)}"
    )
    print("Finished generating training data.\n")
    return train_collection


if __name__ == '__main__':
    testdataset = TestDataset.from_file(
        Path("/home/projects/akita/data/univariate-anomaly-test-cases/cbf-type-mean/test.csv")
    )
    main(testdataset)
