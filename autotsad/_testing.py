import hashlib
from pathlib import Path
from typing import Union, Tuple

import gutenTAG.api as gt
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from nx_config import fill_config_from_path

from .config import config
from .dataset import TestDataset
from .system.hyperparameters import ParamSetting, param_setting_list_intersection
from .system.main import autotsad


def generate_synthetic_data(i: int = 0, plot: bool = False, save: bool = True) -> Union[Tuple[Path, TestDataset], TestDataset]:
    n = 1000
    f = 2

    if i == 0:
        rng = np.random.default_rng(42)
        data = np.r_[
            gt.ecg(rng=rng, length=5 * n, frequency=f / 3, ecg_sim_method="ecgsyn") - 0.25,
            gt.ecg(rng=np.random.default_rng(43), length=5 * n, frequency=f / 4, ecg_sim_method="ecgsyn",
                   amplitude=1.5),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)

        # transition is anomalous
        labels[5000:5080] = True

        # anomaly 1
        shift_by = -.15
        data[3058:3258] += np.r_[np.linspace(0, shift_by, 50), np.full(100, shift_by), np.linspace(shift_by, 0, 50)]
        labels[3058:3258] = True

        # anomaly 2
        start, end = 6785, 6816
        first_point = data[start - 1]
        last_point = data[end + 1]
        mirror_axis = np.linspace(first_point, last_point, end - start)
        data[start:end] = (mirror_axis - data[start:end]) + mirror_axis
        labels[start:end] = True

        # add a bit of noise
        data += np.random.normal(scale=0.01, size=data.shape)

        testdataset = TestDataset(data, labels, f"gt-{i}")

    elif i == 1:
        path = Path("../data/global-temperature-mean-monthly.csv")
        with path.open("rb") as fh:
            digest = hashlib.md5(fh.read()).hexdigest()
        data = (pd.read_csv(path, skiprows=1, index_col=None)
                .iloc[:-2, 1]
                .str.strip()
                .astype(np.float_)
                .values)
        labels = np.zeros_like(data, dtype=np.bool_)
        testdataset = TestDataset(data, labels, digest)

    elif i == 2:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f, width=0),
            gt.square(length=n, frequency=f),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f / 2),
        ]
        data = np.tile(data, reps=3)
        data = np.r_[data[:3000], gt.mls(rng=np.random.default_rng(2), length=50), data[3000:]]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        labels[3000:3100] = 1
        testdataset = TestDataset(data, labels, f"gt-{i}")

    elif i == 3:
        data = np.r_[
            gt.sawtooth(length=n, frequency=f / 2, width=0),
                # gt.square(length=n, frequency=f),
                # gt.dirichlet(length=n, frequency=f, amplitude=.5),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),

                # gt.square(length=n, frequency=f),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),
            gt.mls(rng=np.random.default_rng(2), length=100),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=np.random.default_rng(1), length=n, frequency=f, ecg_sim_method="ecgsyn"),
                # gt.square(length=n, frequency=f),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        labels[3000:3100] = 1
        testdataset = TestDataset(data, labels, f"gt-{i}")

    elif i == 4:
        rng = np.random.default_rng(1)
        data = np.r_[
            gt.dirichlet(length=n, frequency=f, amplitude=.55),
            gt.sawtooth(length=n, frequency=f / 2, width=0),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.5),
            gt.ecg(rng=rng, length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.dirichlet(length=n, frequency=f, amplitude=.6),
            gt.sawtooth(length=n, frequency=f / 2, width=0.05),
            gt.ecg(rng=np.random.default_rng(2), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.2,
            gt.sawtooth(length=n, frequency=f / 2, width=0, amplitude=0.8),
            gt.dirichlet(length=n, frequency=f, amplitude=.45),
            gt.ecg(rng=np.random.default_rng(3), length=n, frequency=f, ecg_sim_method="ecgsyn") - 0.25,
            gt.sawtooth(length=n, frequency=f / 2, width=0.1, amplitude=1.2),
            gt.dirichlet(length=n, frequency=f, amplitude=.65),
        ]
        labels = np.zeros(data.shape[0], dtype=np.bool_)
        # labels[2300:2400] = 1  # existing variation in ECG signal

        data[3650] = data[3650] - 0.5
        labels[3650] = 1

        data[6200:6400] = data[6200:6400] + scipy.stats.norm.pdf(np.linspace(-5, 5, 200), loc=0, scale=1)
        labels[6200:6400] = 1

        data[10200:10300] = data[10200:10300] + (rng.random(100) - 0.5) / 3
        labels[10200:10300] = 1
        testdataset = TestDataset(data, labels, f"gt-{i}")
    else:
        raise ValueError(f"Invalid dataset index {i}.")

    if plot:
        fig, axs = plt.subplots(2, 1, sharex="col")
        axs[0].plot(testdataset.data, label="TS")
        axs[1].plot(testdataset.label, label="label", color="orange")
        axs[0].legend()
        axs[1].legend()
        plt.show()

    if save:
        dataset_path = testdataset.to_csv(Path("data").resolve() / "synthetic")
        return dataset_path, testdataset

    return testdataset


def main():
    dataset_path, data = generate_synthetic_data(2, plot=False, save=True)
    # dataset_path = Path("../data/univariate-anomaly-test-cases/cbf-type-mean/test.csv")
    # dataset_path = Path("../data/sand-data/processed/timeeval/806.csv")
    # dataset_path = Path("../data/benchmark-data/data-processed/univariate/NASA-SMAP/D-8.test.csv")
    # dataset_path = Path("../data/benchmark-data/data-processed/univariate/WebscopeS5/A2Benchmark-22.test.csv")
    # data = TestDataset.from_file(dataset_path)
    # data.data = data.data[:50000]
    # data.label = data.label[:50000]

    fill_config_from_path(config, path=Path("testing.yaml"), env_prefix="AUTOTSAD")
    autotsad(dataset_path, testdataset=data, use_gt_for_cleaning=False)


def test_param_setting():
    params = {"window_size": 1, "hidden_layers": [10, 100, 20], "alpha": 0.1}
    ps = []
    ps.append(ParamSetting(params))
    ps.append(ParamSetting({"hidden_layers": (10, 100, 20), "window_size": 1, "alpha": 0.1}))
    params.update({"alpha": 0.100005})
    ps.append(ParamSetting(params))
    params.update({"window_size": 1.0})
    ps.append(ParamSetting(params))
    params.update({"test": {"bla": True, "blub": {1, 2, 5}}})
    ps.append(ParamSetting(params))
    params.update({"test": {"blub": {5, 2, 1}, "bla": True}})
    ps.append(ParamSetting(params))
    ps.append(ParamSetting({"hidden_layers": (10, 100, 20), "window_size": 1, "alpha": 0.1000005}))

    print("testing hashs")
    hs = []
    for p in ps:
        h = hash(p)
        print(h, p, sep="\t")
        hs.append(h)
    assert hs[0] == hs[1]
    assert hs[2] == hs[3]
    assert hs[4] == hs[5]

    print("creating set")
    for s in set(ps):
        print(s)
    assert len(set(ps)) == len(ps) - 1

    l1 = ps[:3]
    l2 = ps[3:]

    print("testing intersection")
    print(l1)
    print(l2)
    intersection = param_setting_list_intersection(l1, l2)
    print("intersection", intersection)
    assert len(intersection) == 1


if __name__ == '__main__':
    main()
    # test_param_setting()
