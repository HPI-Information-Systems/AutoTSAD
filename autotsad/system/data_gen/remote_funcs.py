import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from timeeval.utils.hash_dict import hash_dict

from autotsad.config import GeneralSection
from autotsad.dataset import TrainDataset, BaseTSDataset, TrainingTSDataset
from autotsad.system.anomaly import Injection
from autotsad.system.pynisher import pynish_func, PynisherException
from autotsad.tsad_algorithms.interop import params_from_dict, exec_algo


def _cleaning_job(algorithm: str,
                  dataset: TrainDataset,
                  params: Dict[str, Any],
                  general_config: GeneralSection) -> Tuple[str, str, np.ndarray]:
    algo_name = f"{algorithm}-{hash_dict(params)}"
    tmp_path = general_config.cache_dir() / "cleaning-scores" / f"{dataset.name}-{algo_name}.csv"
    if not tmp_path.parent.exists():
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

    def objective(data: np.ndarray) -> None:
        p = params_from_dict(params, algorithm)
        scores = exec_algo(data, algorithm, p, ignore_cuts=False)
        np.savetxt(str(tmp_path), scores, delimiter=",")

    result: np.ndarray = np.full(dataset.length, fill_value=np.nan, dtype=np.float_)
    try:
        if not tmp_path.exists():
            pynish_func(objective,
                        **general_config.adjusted_training_limits(dataset.length),
                        name=f"CLEAN: {dataset.name}-{algo_name}")(dataset.data)
        result = np.genfromtxt(str(tmp_path), delimiter=",")
    except PynisherException as e:
        print(f"ERROR: {algorithm} failed on {dataset.name} with params {params} because {repr(e)}")

    return dataset.name, algo_name, result


def _perform_anomaly_injection(dataset: BaseTSDataset,
                               injection: Injection,
                               base_ts_path: Path,
                               process_log: logging.Logger,
                               plot: bool = False,
                               ) -> Optional[TrainingTSDataset]:
    new_dataset = dataset.inject_anomalies(injection)
    c = injection.anomaly_config

    if np.sum(new_dataset.label) == 0:
        process_log.warning(
            f"Skipping {new_dataset.name} ({injection.n_anomalies} x {'|'.join(injection.anomaly_types)}) because no "
            f"anomalies were injected!")
        return None
    if c.skip_dataset_less_than_desired_anomalies and len(new_dataset.annotations) < injection.n_anomalies:
        process_log.warning(
            f"Skipping {new_dataset.name} ({injection.n_anomalies} x {'|'.join(injection.anomaly_types)}) "
            f"because not all anomalies could be injected")
        return None
    if c.skip_dataset_over_contamination_threshold and new_dataset.contamination > c.contamination_threshold:
        process_log.warning(
            f"Skipping {new_dataset.name} because contamination is too high ({new_dataset.contamination:02.0%} "
            f"> {c.contamination_threshold:02.0%})")
        return None

    new_dataset.to_csv(base_ts_path)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, sharex="col")
        axs[0].set_title(dataset.name)
        dataset.plot(ax=axs[0], cuts=True)
        axs[1].set_title(new_dataset.name)
        new_dataset.plot(ax=axs[1], cuts=True)
        axs[2].plot(new_dataset.label, label=f"label (contamination={new_dataset.contamination:.2f})", color="orange")
        for a in new_dataset.annotations:
            axs[2].annotate(a.text, (a.display_idx, 1), color="black", ha="center", va="top")
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        plt.show()

    return new_dataset
