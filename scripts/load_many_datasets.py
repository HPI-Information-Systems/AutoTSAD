import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.append(".")

from autotsad.database.database import Database
from autotsad.database.load_dataset import load_dataset


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load multiple datasets into DB.")
    parser.add_argument("--db-host", type=str, default="172.17.17.32:5432",
                        help="Database hostname (and port)")
    parser.add_argument("datasets_path", type=Path,
                        help="Path to a file storing the dataset paths (CSV schema: name, collection, dataset_path, "
                             "paper).")
    return parser.parse_args(args)


def create_db_url(args: argparse.Namespace) -> str:
    db_user = "autotsad"
    db_pw = "holistic-tsad2023"
    db_database_name = "akita"

    return f"postgresql+psycopg2://{db_user}:{db_pw}@{args.db_host}/{db_database_name}"


def main(sys_args: List[str]) -> None:
    args = parse_args(sys_args)
    db = Database(create_db_url(args))

    df = pd.read_csv(args.datasets_path)
    if "dataset_path" not in df.columns:
        raise ValueError("The dataset_path column is missing, but required!")

    # validate paths:
    df["dataset_path"] = df["dataset_path"].apply(lambda p: Path(p).resolve())
    mask = df["dataset_path"].apply(lambda p: not p.exists() or not p.is_file()).astype(np.bool_)
    noncomplying_paths = df.loc[mask, "dataset_path"].unique()
    noncomplying_paths = "\n  ".join([str(p) for p in noncomplying_paths])
    if len(noncomplying_paths) > 0:
        raise ValueError(f"The following paths are invalid:\n  {noncomplying_paths}")

    existing_datasets = []
    for _, row in df.iterrows():
        dataset_path = row["dataset_path"]
        name = row.get("name", None)
        collection = row.get("collection", None)
        paper = row.get("paper", False)

        print("\n###############################################")
        print(f"Processing {dataset_path}\n")
        try:
            load_dataset(db, dataset_path, name, collection, paper)
        except ValueError as e:
            if "already exists" in str(e):
                existing_datasets.append(row)
            else:
                raise

    if len(existing_datasets) > 0:
        print("\n###############################################")
        print("The following datasets already exist in the database:")
        for row in existing_datasets:
            print(f"  {row.get('collection')} {row.get('name')} ({row['dataset_path']})")


if __name__ == '__main__':
    main(sys.argv[1:])
