from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from optuna import Trial
from scipy.sparse import csc_matrix, hstack
from timeeval.adapters.docker import AlgorithmInterface
from timeeval.data_types import ExecutionType

from ...dataset import Dataset

supports_sliding_window_view = False
supports_tumbling_window_view = False
supports_nan_view = False
supports_custom_cut_handling = True


@dataclass
class CustomParameters:
    anomaly_window_size: int = 10
    paa_transform_size: int = 2
    alphabet_size: int = 3
    normalization_threshold: float = 0.1
    random_state: int = 42

    @staticmethod
    def from_trial(trial: Trial) -> CustomParameters:
        return CustomParameters(
            anomaly_window_size=trial.suggest_int("anomaly_window_size", 10, 1000),
            paa_transform_size=trial.suggest_int("paa_transform_size", 2, 6),
            alphabet_size=trial.suggest_int("alphabet_size", 3, 6),
            normalization_threshold=trial.suggest_float("normalization_threshold", 0.0001, 0.5, log=True),
        )

    @staticmethod
    def timeeval(period_size: int) -> CustomParameters:
        return CustomParameters(
            anomaly_window_size=period_size if period_size > 1 else 100,
            paa_transform_size=5,
            alphabet_size=6,
        )


def post_grammarviz(algorithm_parameter: Union[np.ndarray, Path]) -> np.ndarray:
    if isinstance(algorithm_parameter, np.ndarray):
        results = pd.DataFrame(algorithm_parameter, columns=["index", "score", "length"])
        results = results.set_index("index")
    else:
        results = pd.read_csv(algorithm_parameter, header=None, index_col=0, names=["index", "score", "length"])
    anomalies = results[results["score"] > .0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        length = int(row["length"])
        tmp = np.zeros(len(results))
        tmp[idx:idx + length] = np.repeat([row["score"]], repeats=length)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx:idx + length] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1  # type: ignore


def main(data: Union[np.ndarray, Dataset],
         params: CustomParameters = CustomParameters(),
         cut_mode: bool = False,
         postprocess: bool = True,
         ) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        data_file = tmp_dir / "data.csv"
        window_file = tmp_dir / "data.windows.csv"
        result_file = tmp_dir / "result.csv"

        if cut_mode:
            # windowing:
            view = data.sliding_window_view(params.anomaly_window_size)
            pd.DataFrame(view.data).to_csv(window_file, header=True, index=False)
            # print(f"Saved windowed data to {window_file}")

            # extract timeseries data
            data = data.data

        pd.DataFrame(data).to_csv(data_file, header=True, index=True)
        # print(f"Saved data to {data_file}")

        # print(f"Expecting result in {result_file}")
        grammarviz_jar = Path(__file__).parent.resolve() / "grammarviz.jar"
        if not grammarviz_jar.exists():
            raise RuntimeError("Could not locate grammarviz.jar!")

        # print(f"Located grammarviz.jar at {grammarviz_jar}")
        param_dict = asdict(params)
        param_dict["window_input"] = cut_mode
        interface = AlgorithmInterface(
            dataInput=data_file,
            dataOutput=result_file,
            modelInput=tmp_dir / "model.pkl",
            modelOutput=tmp_dir / "model.pkl",
            executionType=ExecutionType.EXECUTE,
            customParameters=param_dict,
        )
        process = subprocess.run(["java", "-jar", grammarviz_jar, interface.to_json_string()],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True)

        if process.returncode != 0:
            print(f"Error while running grammarviz: {process.returncode}, process output:")
            print("=====================================\n")
            print(process.stdout)
            print(process.stderr)
            print("=====================================\n")
            raise RuntimeError(f"Error while running grammarviz: {process.returncode}")
        else:
            scores = np.genfromtxt(result_file, delimiter=",")
            # print(f"Received scores of shape {scores.shape}, postprocessing...")
            if postprocess:
                scores = post_grammarviz(scores)
            # print(f"Returning scores of shape {scores.shape}")
    return scores
