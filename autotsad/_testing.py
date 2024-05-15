from pathlib import Path

from nx_config import fill_config_from_path

from .config import config
from .dataset import TestDataset
from .system.hyperparameters import ParamSetting, param_setting_list_intersection
from .system.main import autotsad


def main():
    dataset_path = Path.cwd() / "data" / "synthetic" / "gt-2.csv"
    # dataset_path = Path("../data/univariate-anomaly-test-cases/cbf-type-mean/test.csv")
    # dataset_path = Path("../data/sand-data/processed/timeeval/806.csv")
    # dataset_path = Path("../data/benchmark-data/data-processed/univariate/NASA-SMAP/D-8.test.csv")
    # dataset_path = Path("../data/benchmark-data/data-processed/univariate/WebscopeS5/A2Benchmark-22.test.csv")
    # dataset_path = Path("../data/benchmark-data/data-processed/univariate/TODS-synthetic/collective_global_0.05.test.csv")
    data = TestDataset.from_file(dataset_path)
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
