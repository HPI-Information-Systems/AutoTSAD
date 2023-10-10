import os
import warnings
from importlib import import_module
from pathlib import Path
from typing import Union, Tuple, List, Any, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine, MetaData, Table, select
from timeeval import TrainingType


base_folder = Path("/home/sebastian/Documents/projects/akita")
root_data_folder = base_folder / "data"
root_results_folder = base_folder / "results"


def load_scores(rel_path: Union[str, os.PathLike]) -> np.ndarray:
    path = root_results_folder / rel_path
    return pd.read_csv(path, header=None).iloc[:, 0].values


def load_dataset(rel_path: Union[str, os.PathLike]) -> pd.DataFrame:
    path = root_data_folder / rel_path
    return pd.read_csv(path)


class Database:
    def __init__(self, password: str, user: str = "akita", host: str = "172.17.17.32:5432", database: str = "akita"):
        self.engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}/{database}",
            isolation_level="SERIALIZABLE",
            # echo=True,
            future=True,
        )
        metadata_obj = MetaData()
        self.algorithm_table = Table("algorithm", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.collection_table = Table("dataset_collection", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.dataset_table = Table("dataset", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.timeseries_table = Table("timeseries", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.gutentag_meta_table = Table("gutentag_meta", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.run_table = Table("timeeval_run", metadata_obj, autoload_with=self.engine, schema="tsad")
        self.experiment_table = Table("timeeval_experiment", metadata_obj, autoload_with=self.engine, schema="tsad")

    def load_timeseries_path(self, dataset_id: Tuple[str, str], train: bool = False) -> Path:
        collection, name = dataset_id
        with self.engine.begin() as conn:
            paths = conn.execute(
                select(self.timeseries_table.c.path)
                .where(self.dataset_table.c.collection == collection,
                       self.dataset_table.c.name == name,
                       self.dataset_table.c.train_timeseries == self.timeseries_table.c.id
                       if train else
                       self.dataset_table.c.test_timeseries == self.timeseries_table.c.id))
        return paths.scalar_one()

    def load_timeseries(self, dataset_id: Optional[Tuple[str, str]] = None, train: bool = False,
                        ts_id: Optional[int] = None) -> pd.DataFrame:
        if dataset_id is not None:
            path = self.load_timeseries_path(dataset_id, train=train)
        elif ts_id is not None:
            with self.engine.begin() as conn:
                paths = conn.execute(
                    select(self.timeseries_table.c.path)
                    .where(self.timeseries_table.c.id == ts_id)
                )
                path = paths.scalar_one()
        else:
            raise ValueError("Either dataset_id or ts_id have to be specified!")
        return load_dataset(path)

    def load_scoring(self,
                     algorithm: Optional[str] = None,
                     dataset_id: Optional[Tuple[str, str]] = None,
                     exp_id: Optional[int] = None,
                     hyper_params_id: Optional[str] = None) -> np.ndarray:
        if exp_id is not None:
            with self.engine.begin() as conn:
                result = conn.execute(select(self.experiment_table.c.scoring_path)
                                      .where(self.experiment_table.c.id == exp_id))
                scoring_path = result.scalar_one()
        elif algorithm is not None and dataset_id is not None:
            collection_name, dataset_name = dataset_id
            with self.engine.begin() as conn:
                if hyper_params_id is None:
                    result = conn.execute(
                        select(self.algorithm_table.c.selected_param_id)
                        .where(self.algorithm_table.c.id == algorithm))
                    hyper_params_id = result.scalar_one()
                result = conn.execute(
                    select(self.experiment_table.c.scoring_path)
                    .where(self.experiment_table.c.algorithm == algorithm,
                           self.experiment_table.c.collection == collection_name,
                           self.experiment_table.c.dataset == dataset_name,
                           self.experiment_table.c.hyper_params_id == hyper_params_id))
                scoring_path = result.scalar_one()
        else:
            raise ValueError("Either algorithm and dataset_id must be specified or exp_id!")
        return load_scores(scoring_path)

    def plot_scores(self,
                    algorithm: Union[List[Tuple[str, str]], List[str], Tuple[str, str], str],
                    dataset_id: Tuple[str, str],
                    use_plotly: bool = False,
                    metric: str = "roc_auc",
                    **kwargs) -> Any:
        if not isinstance(algorithm, list):
            algorithm = [algorithm]
        if len(algorithm) > 0 and not isinstance(algorithm[0], tuple):
            algorithms: List[Tuple[str, str]] = [(a, "") for a in algorithm]
        else:
            algorithms = algorithm

        # deconstruct dataset ID
        collection_name, dataset_name = dataset_id

        # load dataset details
        with self.engine.begin() as conn:
            df_dataset_meta = pd.read_sql_query(
                select(self.dataset_table.c.input_dimensionality, self.timeseries_table.c.path)
                .where(self.dataset_table.c.test_timeseries == self.timeseries_table.c.id,
                       self.dataset_table.c.collection == collection_name,
                       self.dataset_table.c.name == dataset_name),
                con=conn
            )
        if len(df_dataset_meta) != 1:
            raise ValueError(f"Dataset {dataset_id} not found in DB!")
        dataset_meta = df_dataset_meta.iloc[0]
        dataset_dim = dataset_meta["input_dimensionality"].lower()

        df_scores = load_dataset(dataset_meta["path"])
        algo_names = [a for a, p in algorithms]
        param_ids = np.unique([p for a, p in algorithms if p]).tolist()
        show_param_id = bool(param_ids)
        with self.engine.begin() as conn:
            df_algo_meta = pd.read_sql_query(
                select(self.algorithm_table.c.id, self.experiment_table.c.hyper_params_id,
                       self.algorithm_table.c.display_name, self.experiment_table.c[metric],
                       self.experiment_table.c.scoring_path)
                .where(self.experiment_table.c.algorithm.in_(algo_names),
                       self.experiment_table.c.collection == collection_name,
                       self.experiment_table.c.dataset == dataset_name,
                       self.experiment_table.c.algorithm == self.algorithm_table.c.id,
                       self.experiment_table.c.hyper_params_id == self.algorithm_table.c.selected_param_id
                       if not param_ids
                       else self.experiment_table.c.hyper_params_id.in_(param_ids)),
                con=conn)
        df_algo_meta = df_algo_meta.groupby(["id", "hyper_params_id", "display_name"]).agg({metric: "max", "scoring_path": "first"})
        df_algo_meta = df_algo_meta.reset_index(drop=False).set_index(["id", "hyper_params_id"])

        for algo_param_tuple in algorithms:
            algo, param_id = algo_param_tuple
            if show_param_id and algo_param_tuple not in df_algo_meta.index:
                warnings.warn(f"No results found! Probably {algo} was not executed with parameters {param_id} on {dataset_id}.")
                continue
            elif algo not in df_algo_meta.index.get_level_values(0):
                warnings.warn(f"No results found! Probably {algo} was not executed on {dataset_id}.")
                continue

        from collections import defaultdict
        defaultdict()

        # fix param ids
        algorithms = []
        for algo_param_tuple in list(df_algo_meta.index.values):
            try:
                df_scores[algo_param_tuple] = load_scores(df_algo_meta.loc[algo_param_tuple, "scoring_path"])
                algorithms.append(algo_param_tuple)
            except (ValueError, FileNotFoundError):
                algo, param_id = algo_param_tuple
                warnings.warn(f"No anomaly scores found! Probably {algo} was not executed with params {param_id} or failed on {dataset_id}.")
                continue

        if use_plotly:
            return self._plot_scores_plotly(algorithms, df_algo_meta, df_scores, dataset_dim, dataset_id, metric=metric, show_param_id=show_param_id, **kwargs)
        else:
            return self._plot_scores_plt(algorithms, df_algo_meta, df_scores, dataset_dim, dataset_id, metric=metric, show_param_id=show_param_id, **kwargs)

    @staticmethod
    def _plot_scores_plotly(
            algorithms: List[Tuple[str, str]],
            df_algo_meta: pd.DataFrame,
            df_scores: pd.DataFrame,
            dataset_dim: str,
            dataset_id: Tuple[str, str],
            metric: str = "roc_auc", show_param_id: bool = False, **kwargs) -> Any:
        import plotly.offline as py
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create plot
        fig = make_subplots(2, 1)
        if dataset_dim == "multivariate":
            dim_cols = [c for c in df_scores.columns if c not in ["index", "timestamp", "is_anomaly", *algorithms]]
            for col in dim_cols:
                fig.add_trace(go.Scatter(x=df_scores.index, y=df_scores[col], name=col), 1, 1)
        else:
            fig.add_trace(go.Scatter(x=df_scores.index, y=df_scores.iloc[:, 1], name="timeseries"), 1, 1)
        fig.add_trace(go.Scatter(x=df_scores.index, y=df_scores["is_anomaly"], name="label"), 2, 1)

        for algo_param_tuple in algorithms:
            name = df_algo_meta.loc[algo_param_tuple, "display_name"]
            if show_param_id:
                name += f" ({algo_param_tuple[1]})"
            score = df_algo_meta.loc[algo_param_tuple, metric]
            fig.add_trace(go.Scatter(x=df_scores.index, y=df_scores[algo_param_tuple], name=f"{name} {metric}={score:.4f}"), 2, 1)
        fig.update_xaxes(matches="x")
        fig.update_layout(
            title=f"Results on {dataset_id}",
            height=400
        )
        return py.iplot(fig)

    @staticmethod
    def _plot_scores_plt(
            algorithms: List[Tuple[str, str]],
            df_algo_meta: pd.DataFrame,
            df_scores: pd.DataFrame,
            dataset_dim: str,
            dataset_id: Tuple[str, str],
            metric: str = "roc_auc", close_figure: bool = False, show_param_id: bool = False, **kwargs) -> Any:
        # Create plot
        fig, axs = plt.subplots(2, 1, sharex="col", figsize=(20, 8))
        if dataset_dim == "multivariate":
            for col in [c for c in df_scores.columns if c not in ["index", "timestamp", "is_anomaly", *algorithms]]:
                axs[0].plot(df_scores.index, df_scores[col], label=col)
        else:
            axs[0].plot(df_scores.index, df_scores.iloc[:, 1], label="timeseries")
        axs[1].plot(df_scores.index, df_scores["is_anomaly"], label="label")

        for algo_param_tuple in algorithms:
            name = df_algo_meta.loc[algo_param_tuple, "display_name"]
            if show_param_id:
                name += f" ({algo_param_tuple[1]})"
            score = df_algo_meta.loc[algo_param_tuple, metric]
            axs[1].plot(df_scores.index, df_scores[algo_param_tuple], label=f"{name} {metric}={score:.4f}")
        axs[0].legend()
        axs[1].legend()
        fig.suptitle(f"Results on {dataset_id}")
        fig.tight_layout()
        if close_figure:
            plt.close()
        return fig

    def execute_algorithm(self, algo_name: str, dataset: Union[Tuple[str, str], Path], params: Dict[str, Any] = {}) -> np.ndarray:
        algo_func = getattr(import_module(f"timeeval_experiments.algorithms"), algo_name)
        algo = algo_func(skip_pull=False)
        print(f"Loaded algorithm {algo.name}")

        results_path = Path("results")
        if not results_path.exists():
            print("Preparing results path")
            results_path.mkdir(parents=True, exist_ok=True)

        if isinstance(dataset, Path):
            dataset_path = dataset.resolve()
        else:
            path = self.load_timeseries_path(dataset)
            dataset_path = root_data_folder / path

        if algo.training_type != TrainingType.UNSUPERVISED:
            if isinstance(dataset, Path):
                raise ValueError("Only unsupervised algorithms can be executed on fixed dataset paths!")
            path = self.load_timeseries_path(dataset, train=True)
            train_dataset_path = root_data_folder / path
            algo.train(train_dataset_path, {"hyper_params": params})
        print("Dataset:", dataset_path)
        print("Parameters:", params)
        result = algo.execute(dataset_path, {"hyper_params": params})
        if algo.postprocess:
            result = algo.postprocess(result, {"hyper_params": params})
        result = MinMaxScaler().fit_transform(result.reshape(-1, 1)).reshape(-1)
        return result


if __name__ == '__main__':
    db = Database("")
    # df = db.load_timeseries(dataset_id=("multivariate-test-cases", "channels-2-rw-platform-100.unsupervised"))
    # print(df)
    # scores = db.load_scoring(
    #     algorithm="multi_subsequence_lof",
    #     dataset_id=("multivariate-test-cases", "channels-2-rw-platform-100.unsupervised"),
    #     hyper_params_id="23ce3d734fcd9eb9549d66c3d450d897")
    # # scores = db.load_scoring(exp_id=34933)
    # print(scores)
    #
    # db.plot_scores(
    #     # [("multi_subsequence_lof", "6b07d8c5097379c04ddbd70911ad1881"), ("multi_subsequence_lof", "23ce3d734fcd9eb9549d66c3d450d897")],
    #     "kmeans",
    #     ("multivariate-test-cases", "channels-2-ecg-extremum-1.unsupervised"),
    #     use_plotly=False
    # )
    # plt.show()

    res = db.execute_algorithm("multi_subsequence_lof", ("multivariate-test-cases", "channels-2-ecg-extremum-1.unsupervised"), {
        "window_size": 100
    })
    print(res)
