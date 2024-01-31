import argparse
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List

import gutenTAG.api as gt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from timeeval import DatasetManager
from timeeval.datasets import DatasetAnalyzer, DatasetRecord
from tqdm import tqdm

sys.path.append(".")

from autotsad.dataset import TestDataset


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the custom synthetic datasets for AutoTSAD.")
    parser.add_argument("data_path", type=Path, help="Path to the target folder. If it not exists, it"
                                                     "is created!")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the generated datasets before saving them.")
    parser.add_argument("--save", action="store_true",
                        help="Save the generated datasets in the target folder.")
    return parser.parse_args(args)


def white_noise(length: int, variance: float, rng: np.random.Generator = np.random.default_rng()) -> np.ndarray:
    return rng.normal(0, variance, length)


def generate_synthetic_data(i: int = 0, with_anomaly: bool = True, plot: bool = False, save: bool = True,
                            datasets_path: Path = Path("data"),
                            random_state: Optional[int] = None) -> Union[Tuple[Path, TestDataset], TestDataset]:
    n = 1000
    f = 2

    if i == 0:
        rng = np.random.default_rng(random_state)
        data = np.r_[
            gt.ecg(rng=rng, length=5 * n, frequency=f / 3, ecg_sim_method="ecgsyn") - 0.25,
            gt.ecg(rng=np.random.default_rng(random_state+1), length=5 * n, frequency=f / 4, ecg_sim_method="ecgsyn",
                   amplitude=1.5),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)

        if with_anomaly:
            # transition is anomalous
            labels[5000:5080] = True

            # anomaly 1
            shift_by = -.15
            data[3058:3258] += np.r_[
                np.linspace(0, shift_by, 50),
                np.full(100, shift_by),
                np.linspace(shift_by, 0, 50)
            ]
            labels[3058:3258] = True

            # anomaly 2
            start, end = 6785, 6816
            first_point = data[start - 1]
            last_point = data[end + 1]
            mirror_axis = np.linspace(first_point, last_point, end - start)
            data[start:end] = (mirror_axis - data[start:end]) + mirror_axis
            labels[start:end] = True

        # add a bit of noise
        data += rng.normal(scale=0.01, size=data.shape)

    elif i == 1:
        path = Path("../data/global-temperature-mean-monthly.csv")
        data = (pd.read_csv(path, skiprows=1, index_col=None)
                .iloc[:-2, 1]
                .str.strip()
                .astype(np.float_)
                .values)
        labels = np.zeros_like(data, dtype=np.bool_)

    elif i == 2:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f, width=0),
            gt.square(length=n, frequency=f),
            gt.ecg(rng=np.random.default_rng(random_state), length=n, frequency=f / 2),
        ]
        data = np.tile(data, reps=3)
        if with_anomaly:
            data = np.r_[data[:3000], gt.mls(rng=np.random.default_rng(random_state + 1), length=50), data[3000:]]
            labels = np.zeros(data.shape[0], dtype=np.bool_)
            labels[3000:3100] = 1
        else:
            labels = np.zeros(data.shape[0], dtype=np.bool_)

    elif i == 3:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=np.random.default_rng(random_state), length=n, frequency=f, ecg_sim_method="ecgsyn"),
            gt.ecg(rng=np.random.default_rng(random_state), length=n, frequency=f, ecg_sim_method="ecgsyn"),

            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=np.random.default_rng(random_state), length=n, frequency=f, ecg_sim_method="ecgsyn"),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
        ]
        if with_anomaly:
            data = np.r_[data[:3*n], gt.mls(rng=np.random.default_rng(random_state + 1), length=100), data[3*n:]]
            labels = np.zeros(data.shape[0], dtype=np.bool_)
            labels[3000:3100] = 1
        else:
            labels = np.zeros(data.shape[0], dtype=np.bool_)

    elif i == 4:
        rng = np.random.default_rng(random_state)
        data = np.r_[
            gt.dirichlet(length=n, frequency=f, amplitude=.55),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.5),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.6),
            gt.sawtooth(length=n, frequency=f / 2, width=0.05),
            gt.ecg(rng=np.random.default_rng(random_state+1), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.2,
            gt.sawtooth(length=n, frequency=f / 2, width=0, amplitude=0.8),
            gt.dirichlet(length=n, frequency=f, amplitude=.45),
            gt.ecg(rng=np.random.default_rng(random_state+2), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.sawtooth(length=n, frequency=f / 2, width=0.1, amplitude=1.2),
            gt.dirichlet(length=n, frequency=f, amplitude=.65),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        # labels[2300:2400] = 1  # existing variation in ECG signal

        if with_anomaly:
            data[3650] = data[3650] - 0.5
            labels[3650] = 1

            data[6200:6400] = data[6200:6400] + norm.pdf(np.linspace(-5, 5, 200), loc=0, scale=1)
            labels[6200:6400] = 1

            data[10200:10300] = data[10200:10300] + (rng.random(100) - 0.5) / 3
            labels[10200:10300] = 1

    else:
        raise ValueError(f"Invalid dataset index {i}.")

    name = f"gt-{i}" if with_anomaly else f"gt-{i}.train"
    dataset = TestDataset(data, labels, name)

    if plot:
        fig, axs = plt.subplots(2, 1, sharex="col")
        axs[0].plot(dataset.data, label=dataset.name)
        axs[1].plot(dataset.label, label="label", color="orange")
        axs[0].legend()
        axs[1].legend()
        plt.show()

    if save:
        dataset_path = dataset.to_csv(datasets_path / "synthetic")
        return dataset_path, dataset

    return dataset


def main(sys_args: List[str]) -> None:
    args = parse_args(sys_args)
    data_path = args.data_path
    plot = args.plot
    save = args.save
    collection_name = "autotsad-synthetic"
    dmgr = DatasetManager(data_path / "synthetic", create_if_missing=True)

    bar = tqdm((0, 2, 3, 4), desc="Generating synthetic datasets")
    for i in bar:
        bar.write(f"Generating and analyzing synthetic dataset {i} (testing TS)")
        dataset_path, testdataset = generate_synthetic_data(i=i, plot=plot, save=save, datasets_path=data_path, random_state=1)
        name = testdataset.name
        del testdataset
        meta = DatasetAnalyzer((collection_name, name), dataset_path=dataset_path, is_train=False)
        meta.save_to_json(data_path / "synthetic" / f"{name}.meta.json", overwrite=True)
        metadata = meta.metadata

        bar.write(f"Generating and analyzing synthetic dataset {i} (training TS)")
        dataset_path, _ = generate_synthetic_data(i=i, plot=plot, save=save, datasets_path=data_path, random_state=5, with_anomaly=False)
        meta = DatasetAnalyzer((collection_name, name), dataset_path=dataset_path, is_train=True)
        meta.save_to_json(data_path / "synthetic" / f"{name}.meta.json", overwrite=False)

        dmgr.add_dataset(DatasetRecord(
            collection_name=collection_name,
            dataset_name=name,
            train_path=f"{name}.train.csv",
            test_path=f"{name}.csv",
            dataset_type="synthetic",
            datetime_index=False,
            split_at=None,
            train_type="semi-supervised",
            train_is_normal=True,
            input_type="univariate",
            length=metadata.length,
            dimensions=metadata.dimensions,
            contamination=metadata.contamination,
            num_anomalies=metadata.num_anomalies,
            min_anomaly_length=metadata.anomaly_length.min,
            median_anomaly_length=metadata.anomaly_length.median,
            max_anomaly_length=metadata.anomaly_length.max,
            mean=metadata.mean,
            stddev=metadata.stddev,
            trend=metadata.trend,
            stationarity=metadata.stationarity.name.lower(),
            period_size=None,
        ))
        if save:
            dmgr.save()
    print(dmgr.df())


if __name__ == '__main__':
    main(sys.argv[1:])
