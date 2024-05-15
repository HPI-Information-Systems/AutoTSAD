from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .remote_funcs import _perform_anomaly_injection
from ..anomaly import Injection
from ..logging import LOGGING_Q
from ..timer import Timers
from ...config import config
from ...dataset import BaseTSDataset, TrainingDatasetCollection, TrainingTSDataset
from ...util import getsize


def generate_anomaly_time_series(base_ts_collection: TrainingDatasetCollection) -> TrainingDatasetCollection:
    log = logging.getLogger("autotsad.data_gen.injecting")
    new_datasets = TrainingDatasetCollection(base_ts_collection.test_data)
    rng = np.random.default_rng(config.general.seed)
    Timers.start("Anomaly injection")

    # prepare training dataset directory
    base_ts_path = config.general.tmp_path / "training-time-series"
    base_ts_path.mkdir(parents=True, exist_ok=True)

    for d in base_ts_collection:
        if not isinstance(d, BaseTSDataset):  # mainly to make type checker happy
            continue
        log.debug(f"Generating anomalies for {d.name}")

        # generate injections
        injections = []
        # single anomaly of different length
        for anomaly_type in config.anomaly.allowed_anomaly_types:
            for length in config.anomaly.anomaly_lengths(d.period_size, d.length):
                injection = Injection(
                    anomaly_types=[anomaly_type],
                    n_anomalies=1,
                    length=length,
                    random_state=rng.integers(1e10),
                    anomaly_config=config.anomaly,
                )
                injections.append(injection)

        # multiple anomalies of same kind (random length)
        if config.anomaly.generate_multiple_same:
            for anomaly_type in config.anomaly.allowed_anomaly_types:
                for n in config.anomaly.number_of_anomalies_per_dataset:
                    if n == 1:
                        continue
                    length_options = config.anomaly.anomaly_lengths(d.period_size, d.length)
                    length = int(rng.choice(length_options))
                    injection = Injection(
                        anomaly_types=[anomaly_type],
                        n_anomalies=n,
                        length=length,
                        random_state=rng.integers(1e10),
                        anomaly_config=config.anomaly,
                    )
                    injections.append(injection)

        # multiple different kind of anomaly
        if config.anomaly.generate_multiple_different:
            for types in [c for i in config.anomaly.number_of_different_anomalies
                          for c in combinations(config.anomaly.allowed_anomaly_types, i)]:
                for n in config.anomaly.number_of_anomalies_per_dataset:
                    if n == 1:
                        continue
                    length_options = config.anomaly.anomaly_lengths(d.period_size, d.length)
                    length = int(rng.choice(length_options))
                    injection = Injection(
                        anomaly_types=list(types),
                        n_anomalies=max(len(types), n),
                        length=length,
                        random_state=rng.integers(1e10),
                        anomaly_config=config.anomaly,
                    )
                    injections.append(injection)

        log.info(f"Performing {len(injections)}/{joblib.effective_n_jobs(config.general.n_jobs)} injections")
        plot_injections = config.data_gen_plotting.injected_anomaly and config.general.n_jobs == 1
        # capture queue object for subprocesses
        q = LOGGING_Q

        def _call_perform_anomaly_injection(d: BaseTSDataset, i: Injection, p: Path) -> Optional[TrainingTSDataset]:
            import logging
            from autotsad.system.logging import setup_process_logging

            with setup_process_logging(q) as pid:
                process_log = logging.getLogger(f"autotsad.data_gen.injecting-{pid}")
                return _perform_anomaly_injection(d, i, p, process_log, plot_injections)

        # results = joblib.Parallel(n_jobs=min(config.general.n_jobs, len(injections)))(
        results = joblib.Parallel(n_jobs=1)(
            joblib.delayed(_call_perform_anomaly_injection)(d, injection, base_ts_path)
            for injection in injections
        )
        for dataset in results:
            if dataset is not None:
                new_datasets.append(dataset)

        if config.anomaly.same_anomalies_for_all_base_ts:
            # re-initialize RNG
            rng = np.random.default_rng(config.general.seed)

    Timers.stop("Anomaly injection")
    print("Finished injecting anomalies.")
    log.debug(f"Base TS Collection ({len(base_ts_collection)} datasets): {getsize(base_ts_collection)}")
    log.debug(f"New TS Collection ({len(new_datasets)} datasets): {getsize(new_datasets)}")
    return new_datasets
