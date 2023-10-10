import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, insert

from .database import Database


def load_dataset(db: Database, dataset_path: Path, name: Optional[str] = None, collection: Optional[str] = None,
                 paper: bool = False) -> None:
    path = Path(dataset_path).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Path to the dataset ({path}) is invalid!")

    print(f"Reading dataset from {dataset_path}")
    with dataset_path.open("rb") as fh:
        hexhash = hashlib.md5(fh.read()).hexdigest()
    if name is not None:
        dataset_name = name
    else:
        dataset_name = dataset_path.stem
        if dataset_name == "test":  # for GutenTAG datasets, the containing folder is the dataset name
            dataset_name = dataset_path.parent.stem
        if dataset_name.endswith(".test"):
            dataset_name = dataset_name[:-5]
        if dataset_name.startswith("KPI-"):  # for IOPS datasets, strip the prefix
            dataset_name = dataset_name[4:]

    with db.begin() as conn:
        res = conn.execute(select(db.dataset_table).where(db.dataset_table.c.hexhash == hexhash)).first()
        if res:
            raise ValueError("Dataset already exists in the database!")

    df = pd.read_csv(dataset_path)
    print(f"  dataset collection: {collection}")
    print(f"  dataset name: {dataset_name}")
    print(f"  dataset hash: {hexhash}")
    print(f"  dataset length: {df.shape[0]}")
    print(f"  dataset has paper flag: {paper}")

    df_target = df[["timestamp", "is_anomaly"]].copy()
    df_target.columns = ["time", "is_anomaly"]
    df_target.insert(1, "value", df.iloc[:, 1])
    df_target.insert(1, "dataset_id", hexhash)
    try:
        df_target["time"] = df_target["time"].astype(np.int_)
    except ValueError:
        # use numerical index instead
        df_target["time"] = np.arange(df_target.shape[0])
    df_target["value"] = df_target["value"].astype(np.float_)
    df_target["is_anomaly"] = df_target["is_anomaly"].astype(np.bool_)
    print(df_target)

    print("uploading dataset to the database...")
    with db.begin() as conn:
        conn.execute(
            insert(db.dataset_table),
            {"hexhash": hexhash, "name": dataset_name, "collection": collection, "paper": paper}
        )
        df_target.to_sql(con=conn, **db.timeseries_table_meta, if_exists="append", index=False)
    print("...successfully loaded dataset into the database!")
