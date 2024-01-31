from __future__ import annotations

import argparse
from typing import Iterator

from sqlalchemy import create_engine as create_pg_engine, Engine, MetaData, Table, Connection


def engine_url_from_args(args: argparse.Namespace) -> str:
    db_user = args.db_user
    db_pw = args.db_password
    db_host = args.db_host
    db_database_name = args.db_database_name

    return f"postgresql+psycopg2://{db_user}:{db_pw}@{db_host}/{db_database_name}"


def create_engine(url: str, isolation_level: str = "SERIALIZABLE") -> Engine:
    return create_pg_engine(
        url,
        isolation_level=isolation_level,
        # echo=True,
        future=True,
        pool_pre_ping=True,
    )


class Database:
    schema = "autotsad"
    configuration_table_meta = {"name": "configuration", "schema": schema}
    dataset_table_meta = {"name": "dataset", "schema": schema}
    timeseries_table_meta = {"name": "timeseries", "schema": schema}
    experiment_table_meta = {"name": "experiment", "schema": schema}
    algorithm_scoring_table_meta = {"name": "algorithm_scoring", "schema": schema}
    scoring_table_meta = {"name": "scoring", "schema": schema}
    algorithm_execution_table_meta = {"name": "autotsad_algorithm_execution", "schema": schema}
    ranking_table_meta = {"name": "algorithm_ranking", "schema": schema}
    ranking_entry_table_meta = {"name": "algorithm_ranking_entry", "schema": schema}
    autotsad_execution_table_meta = {"name": "autotsad_execution", "schema": schema}
    baseline_execution_table_meta = {"name": "baseline_execution", "schema": schema}
    runtime_trace_table_meta = {"name": "runtime_trace", "schema": schema}

    @staticmethod
    def create_engine(url: str, isolation_level: str = "SERIALIZABLE") -> Engine:
        return create_engine(url, isolation_level)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Database":
        return Database(engine_url_from_args(args))

    def __init__(self, db_url: str, isolation_level: str = "SERIALIZABLE") -> None:
        self.url = db_url
        self.engine = create_engine(db_url, isolation_level)

        metadata_obj = MetaData()
        self.configuration_table = Table("configuration", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.dataset_table = Table("dataset", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.timeseries_table = Table("timeseries", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.experiment_table = Table("experiment", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.algorithm_scoring_table = Table("algorithm_scoring", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.scoring_table = Table("scoring", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.algorithm_execution_table = Table("autotsad_algorithm_execution", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.ranking_table = Table("algorithm_ranking", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.ranking_entry_table = Table("algorithm_ranking_entry", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.autotsad_execution_table = Table("autotsad_execution", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.baseline_execution_table = Table("baseline_execution", metadata_obj, autoload_with=self.engine, schema=self.schema)
        self.runtime_trace_table = Table("runtime_trace", metadata_obj, autoload_with=self.engine, schema=self.schema)

    def begin(self) -> Iterator[Connection]:
        return self.engine.begin()

    def load_test_dataset(self, dataset_id: str) -> "TestDataset":
        import pandas as pd
        from autotsad.dataset import TestDataset
        from sqlalchemy import select

        with self.begin() as conn:
            df = pd.read_sql(select(
                self.timeseries_table.c.time, self.timeseries_table.c.value, self.timeseries_table.c.is_anomaly
            ).where(self.timeseries_table.c.dataset_id == dataset_id), conn)
            if len(df) == 0:
                raise ValueError(f"Dataset with ID {dataset_id} does not exist in the database!")
            dataset_name = conn.execute(
                select(self.dataset_table.c.name).where(self.dataset_table.c.hexhash == dataset_id)
            ).first()[0]

        df = df.sort_values("time")
        df = df.rename(columns={"time": "timestamp"})
        return TestDataset.from_df(df, hexhash=dataset_id, name=dataset_name)
