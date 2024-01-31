import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from timeeval import MultiDatasetManager


def selected_datasets() -> List[Tuple[str, str]]:
    df_datasets = pd.read_csv("data/baseline-results/results-oracle-baselines.csv")
    benchmark_datasets = df_datasets[["collection_name", "dataset_name"]].values.tolist()
    datasets = [(x, y) for x, y in benchmark_datasets]
    return datasets


def main():
    target_path = Path("data") / "autotsad-data"
    dmgr = MultiDatasetManager([
        Path.cwd().parent / "data" / "benchmark-data" / "data-processed",
        Path.cwd().parent / "data" / "univariate-anomaly-test-cases",
        Path.cwd().parent / "data" / "sand-data" / "processed" / "timeeval",
        # Path.cwd() / "data" / "synthetic",
    ])

    selected_dataset_list = selected_datasets()
    print(f"Selected {len(selected_dataset_list)} datasets")

    df_index = pd.DataFrame(selected_dataset_list, columns=["collection_name", "dataset_name"])
    # fix GutenTAG names
    mask = df_index["collection_name"] == "GutenTAG"
    df_index.loc[mask, "collection_name"] = "univariate-anomaly-test-cases"
    df_index.loc[mask, "dataset_name"] = df_index.loc[mask, "dataset_name"].str.cat(others=["semi-supervised"]*np.sum(mask), sep=".")
    # end fix GutenTAG names
    df_index = df_index.set_index(["collection_name", "dataset_name"])

    df = dmgr.df()
    df = pd.merge(df_index, df, left_index=True, right_index=True, how="left")
    print(f"Retrieved metadata for {df.shape[0]} datasets")

    # adapt GutenTAG names
    df = df.reset_index()
    gt_mask = df["collection_name"] == "univariate-anomaly-test-cases"
    df.loc[gt_mask, "collection_name"] = "GutenTAG"
    df.loc[gt_mask, "train_path"] = "GutenTAG/" + df.loc[gt_mask, "train_path"].str.replace("/", ".")
    df.loc[gt_mask, "test_path"] = "GutenTAG/" + df.loc[gt_mask, "test_path"].str.replace("/", ".")

    # add source information and adapt folder structure
    df["source"] = "TimeEval"
    df.loc[df["collection_name"] == "SAND", "source"] = "SAND"

    df["test_path"] = df["source"] + "/" + df["test_path"].str.replace("univariate/", "")
    df["train_path"] = df["source"] + "/" + df["train_path"].str.replace("univariate/", "")
    df = df.set_index(["collection_name", "dataset_name"])
    print(df[["test_path", "source", "train_path"]])

    df_tmp = df[df["train_type"] == "semi-supervised"]
    semi_sup_datasets = df_tmp.index.tolist()
    print(f"Found {len(semi_sup_datasets)} semi-supervised datasets:")
    for c, d in semi_sup_datasets:
        print(f"\t{c} {d}")

    # copy datasets to new location within this repository (with complete datasets.csv) for all paper-datasets
    for dataset_id, s in df.iterrows():
        collection_name, dataset_name = dataset_id
        old_dataset_id = ("univariate-anomaly-test-cases" if collection_name == "GutenTAG" else collection_name, dataset_name)
        print(f"Processing {dataset_id}")
        source_test_path = dmgr.get_dataset_path(old_dataset_id, train=False)
        target_test_path = target_path / s["test_path"]
        print(f"  Copying test TS   ({source_test_path} -> {target_test_path})")
        target_test_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_test_path, target_test_path)
        if s["train_type"] == "semi-supervised":
            source_train_path = dmgr.get_dataset_path(old_dataset_id, train=True)
            target_train_path = target_path / s["train_path"]
            print(f"  Copying train TS  ({source_train_path} -> {target_train_path})")
            shutil.copy2(source_train_path, target_train_path)

        meta_name = f"{dataset_name}.metadata.json"
        source_meta_path = source_test_path.with_name(meta_name)
        target_meta_path = target_test_path.with_name(meta_name)
        if source_meta_path.exists():
            print(f"  Copying metadata  ({source_meta_path} -> {target_meta_path})")
            shutil.copy2(source_meta_path, target_meta_path)
    df.to_csv(target_path / "datasets.csv", index=True)


if __name__ == '__main__':
    main()
