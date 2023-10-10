from typing import Tuple, Optional, Literal, List, Dict, Any, Generator, Union

import pandas as pd
from sqlalchemy import select, text, Connection, Select
from streamlit import cache_data
from streamlit.connections import ExperimentalBaseConnection

from .database import Database

AUTOTSAD_SUPER_ENSEMBLES = ("aggregated-minimum-influence", "aggregated-robust-borda")
AUTOTSAD_SUPER_ENSEMBLES_SQL = ','.join(f"'{m}'" for m in AUTOTSAD_SUPER_ENSEMBLES)


def _if_filter(filters: Dict[str, List[Any]], name: str) -> Generator[List[Any], None, None]:
    items = filters.get(name, [])
    if len(items) > 0:
        yield items


class AutoTSADConnection(ExperimentalBaseConnection[Database]):
    def _connect(self, **kwargs) -> Database:
        self._persist = self._secrets["persist_cache"]
        self._ttl = self._secrets["ttl"] if not self._persist else None

        if "url" in kwargs:
            url = kwargs.pop("url")
        else:
            url = self._secrets["url"]
        return Database(url)

    def query(self, query: str) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=10)
        def _query(query: str) -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _query(query)

    def list_datasets(self, filters: Optional[Dict[str, List[Any]]] = None) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _list_datasets(filters: Optional[Dict[str, List[Any]]]) -> pd.DataFrame:
            if filters:
                with self._instance.begin() as conn:
                    query = select(self._instance.dataset_table.c.name, self._instance.dataset_table.c.collection).where(
                        self._instance.dataset_table.c.hexhash == self._instance.autotsad_execution_table.c.dataset_id
                    )
                    query = self._add_autotsad_filters(query, filters)
                    query = query.order_by(self._instance.dataset_table.c.collection, self._instance.dataset_table.c.name)
                    query = query.distinct()
                    return pd.read_sql(query, conn)
            else:
                with self._instance.begin() as conn:
                    return pd.read_sql(
                        select(self._instance.dataset_table.c.name, self._instance.dataset_table.c.collection), conn
                    )

        return _list_datasets(filters)

    def dataset_collections_for_paper(self) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=1)
        def _dataset_collections_for_paper() -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(
                    text("""select collection, count(*) as "datasets"
                            from dataset
                            where paper = True
                            group by collection
                            order by count(*) desc;"""),
                    conn
                )

        return _dataset_collections_for_paper()

    def list_autotsad_versions(self) -> List[str]:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=1)
        def _list_autotsad_versions() -> List[str]:
            with self._instance.begin() as conn:
                results = conn.execute(text("SELECT DISTINCT autotsad_version FROM autotsad_execution")).fetchall()
                if len(results) == 1:
                    return results[0]
                else:
                    return [r[0] for r in results]

        return _list_autotsad_versions()

    def list_autotsad_configs(self) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist)
        def _list_autotsad_configs() -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(
                    select(self._instance.configuration_table.c.id, self._instance.configuration_table.c.description),
                    conn
                )

        return _list_autotsad_configs()

    def list_experiments(self) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=1)
        def _list_experiments() -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(select(self._instance.experiment_table), conn)

        return _list_experiments()

    def list_baselines(self) -> List[str]:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=1)
        def _list_baselines() -> List[str]:
            with self._instance.begin() as conn:
                results = conn.execute(select(self._instance.baseline_execution_table.c.name).distinct()).fetchall()
                if len(results) == 1:
                    return results[0]
                else:
                    return [r[0] for r in results]

        return _list_baselines()

    def load_dataset(self, name: str) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=100)
        def _load_dataset(name: str) -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(
                    select(self._instance.timeseries_table.c.time, self._instance.timeseries_table.c.value,
                           self._instance.timeseries_table.c.is_anomaly)
                    .where(self._instance.timeseries_table.c.dataset_id == self._instance.dataset_table.c.hexhash,
                           self._instance.dataset_table.c.name == name)
                    .order_by(self._instance.timeseries_table.c.time), conn)

        return _load_dataset(name)

    def load_scoring(self, scoring_id: int) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=100)
        def _load_scoring(scoring_id: int) -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(
                    select(self._instance.scoring_table.c.time, self._instance.scoring_table.c.score)
                    .where(
                        self._instance.scoring_table.c.algorithm_scoring_id == self._instance.algorithm_scoring_table.c.id,
                        self._instance.algorithm_scoring_table.c.id == scoring_id)
                    .order_by(self._instance.scoring_table.c.time), conn)

        return _load_scoring(scoring_id)

    def list_available_methods(self) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=1)
        def _list_available_methods() -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(text(
                    f"""select "Method", case when ranking_method in ({AUTOTSAD_SUPER_ENSEMBLES_SQL})
                                    then 'AutoTSAD Ensemble'
                                    else 'AutoTSAD'
                                end as "Method Type"
                            from (select distinct ranking_method || '_' || normalization_method || '_' || aggregation_method as "Method", ranking_method
                                  from autotsad_execution
                            ) as autotsad
                        union
                        select "Method", 'Baseline' as "Method Type"
                            from (select distinct name as "Method" from baseline_execution) as baseline
                        order by "Method Type", "Method"
                    """
                ), conn)

        return _list_available_methods()

    def all_aggregated_results(self, filters: Dict[str, List[Any]],
                               only_paper_datasets: bool = False,
                               only_paper_methods: bool = False,
                               exclude_super_ensembles: bool = False) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _all_aggregated_results(filters: Dict[str, List[Any]], only_paper_datasets: bool = False) -> pd.DataFrame:
            with self._instance.begin() as conn:
                autotsad_filter_clauses = self._build_autotsad_filter_clauses(filters)

                baseline_filter_clauses = ""
                for baselines in _if_filter(filters, "baseline"):
                    baselines = [f"'{b}'" for b in baselines]
                    baseline_filter_clauses += f" and e.name in ({','.join(baselines)})"

                if only_paper_datasets:
                    autotsad_filter_clauses += " and d.paper = True"
                    baseline_filter_clauses += " and d.paper = True"

                return pd.read_sql(text(
                    f"""select "Dataset", "Method", range_pr_auc, range_roc_auc,
                        case when ranking_method in ({AUTOTSAD_SUPER_ENSEMBLES_SQL})
                            then 'AutoTSAD Ensemble'
                            else 'AutoTSAD'
                        end as "Method Type"
                        from (select concat(d.collection, ' ', d.name) as "Dataset",
                                     concat(e.ranking_method, '_', e.normalization_method, '_', e.aggregation_method) as "Method",
                                     -- e.ranking_method as "Method",
                                     e.range_pr_auc,
                                     e.range_roc_auc,
                                     e.ranking_method
                              from autotsad_execution e, dataset d, experiment x
                              where e.dataset_id = d.hexhash and e.experiment_id = x.id
                              and x.description != 'paper v1 - optimization / variance' -- avoid duplicates
                              {autotsad_filter_clauses}) as autotsad
                        union
                        select "Dataset", "Method", range_pr_auc, range_roc_auc, 'Baseline' as "Method Type"
                        from (select concat(d.collection, ' ', d.name) as "Dataset", e.name as "Method", e.range_pr_auc, e.range_roc_auc
                              from baseline_execution e,
                                   dataset d
                              where e.dataset_id = d.hexhash {baseline_filter_clauses}) as baseline
                        order by "Dataset", range_pr_auc desc
                    """), conn)

        df = _all_aggregated_results(filters, only_paper_datasets)
        if only_paper_methods:
            # filter out results that are not meaningful (we don't describe them in the paper)
            df = df[~df["Method"].str.endswith("_mean")]
            df = df[~df["Method"].str.contains("minmax")]
            df = df[~df["Method"].str.startswith("interchange")]
            df = df[~df["Method"].str.startswith("training-coverage")]
            df = df[~df["Method"].str.startswith("aggregated-robust-borda")]
        if exclude_super_ensembles:
            df = df[~df["Method Type"] == "AutoTSAD Ensemble"]
        return df

    def _resolve_method_type(self, method: str, conn: Connection) -> str:
        res = conn.execute(select(self._instance.autotsad_execution_table.c.ranking_method)
                           .where(self._instance.autotsad_execution_table.c.ranking_method == method)
                           ).first()
        if method in AUTOTSAD_SUPER_ENSEMBLES:
            return "AutoTSAD Ensemble"
        elif res and len(res) > 0:
            return "AutoTSAD"
        else:
            return "Baseline"

    def _add_autotsad_filters(self, query: Select, filters: Dict[str, List[Any]]) -> Select:
        for autotsad_versions in _if_filter(filters, "autotsad_version"):
            query = query.where(self._instance.autotsad_execution_table.c.autotsad_version.in_(autotsad_versions))
        for config_ids in _if_filter(filters, "config_id"):
            query = query.where(self._instance.autotsad_execution_table.c.config_id.in_(config_ids))
        for experiment_ids in _if_filter(filters, "experiment_id"):
            query = query.where(self._instance.autotsad_execution_table.c.experiment_id.in_(experiment_ids))
        for ranking_methods in _if_filter(filters, "ranking_method"):
            query = query.where(self._instance.autotsad_execution_table.c.ranking_method.in_(ranking_methods))
        for normalization_methods in _if_filter(filters, "normalization_method"):
            query = query.where(self._instance.autotsad_execution_table.c.normalization_method.in_(normalization_methods))
        for aggregation_methods in _if_filter(filters, "aggregation_method"):
            query = query.where(self._instance.autotsad_execution_table.c.aggregation_method.in_(aggregation_methods))
        return query

    def _build_autotsad_filter_clauses(self, filters: Dict[str, List[Any]], execution_table_name: str = "e") -> str:
        e = execution_table_name
        autotsad_filter_clauses = ""
        for experiment_ids in _if_filter(filters, "experiment_id"):
            autotsad_filter_clauses += f" and {e}.experiment_id in ({','.join(str(e) for e in experiment_ids)})"
        for autotsad_versions in _if_filter(filters, "autotsad_version"):
            autotsad_versions = [f"'{v}'" for v in autotsad_versions]
            autotsad_filter_clauses += f" and {e}.autotsad_version in ({','.join(autotsad_versions)})"
        for config_ids in _if_filter(filters, "config_id"):
            config_ids = [f"'{c}'" for c in config_ids]
            autotsad_filter_clauses += f" and {e}.config_id in ({','.join(config_ids)})"
        for ranking_methods in _if_filter(filters, "ranking_method"):
            ranking_methods = [f"'{rm}'" for rm in ranking_methods]
            autotsad_filter_clauses += f" and {e}.ranking_method in ({','.join(ranking_methods)})"
        for normalization_methods in _if_filter(filters, "normalization_method"):
            normalization_methods = [f"'{nm}'" for nm in normalization_methods]
            autotsad_filter_clauses += f" and {e}.normalization_method in ({','.join(normalization_methods)})"
        for aggregation_methods in _if_filter(filters, "aggregation_method"):
            aggregation_methods = [f"'{am}'" for am in aggregation_methods]
            autotsad_filter_clauses += f" and {e}.aggregation_method in ({','.join(aggregation_methods)})"
        return autotsad_filter_clauses

    def method_quality(self, dataset: str, rmethod: str, nmethod: Optional[str] = None, amethod: Optional[str] = None,
                       method_type: Optional[str] = None, filters: Dict[str, List[Any]] = {},
                       metric: Literal["range_pr_auc","range_roc_auc","precision_at_k","precision","recall"] = "range_pr_auc") -> float:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=200)
        def _method_quality(dataset: str, rmethod: str, nmethod: Optional[str] = None, amethod: Optional[str] = None,
                            method_type: Optional[str] = None, filters: Dict[str, List[Any]] = {},
                            metric: str = "range_pr_auc") -> float:
            with self._instance.begin() as conn:
                if method_type is None:
                    method_type = self._resolve_method_type(rmethod, conn)

                if method_type.startswith("AutoTSAD"):
                    query = select(text(metric)).where(
                        self._instance.autotsad_execution_table.c.ranking_method == rmethod,
                        self._instance.autotsad_execution_table.c.normalization_method == nmethod,
                        self._instance.autotsad_execution_table.c.aggregation_method == amethod,
                        self._instance.autotsad_execution_table.c.dataset_id == self._instance.dataset_table.c.hexhash,
                        self._instance.dataset_table.c.name == dataset
                    )
                    query = self._add_autotsad_filters(query, filters)
                    res = conn.execute(query).first()
                else:
                    res = conn.execute(select(text(metric)).where(
                        self._instance.baseline_execution_table.c.name == rmethod,
                        self._instance.baseline_execution_table.c.dataset_id == self._instance.dataset_table.c.hexhash,
                        self._instance.dataset_table.c.name == dataset)
                    ).first()

                if res and len(res) > 0:
                    return res[0]
                else:
                    raise ValueError(f"Method {rmethod} {nmethod} {amethod} not found for dataset {dataset}. Are you "
                                     f"sure the method was executed on the dataset?")
        return _method_quality(dataset, rmethod, nmethod, amethod, method_type, filters, metric)

    def load_ranking_results(self, dataset: str, rmethod: Union[str, pd.Series], nmethod: Optional[str] = None,
                             amethod: Optional[str] = None, method_type: Optional[str] = None,
                             filters: Dict[str, List[Any]] = {}) -> Tuple[pd.DataFrame, pd.DataFrame]:

        execution_id = None
        if isinstance(rmethod, pd.Series):
            execution_id = int(rmethod["id"])
            nmethod = rmethod["normalization_method"]
            amethod = rmethod["aggregation_method"]
            rmethod = rmethod["ranking_method"]

        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=100)
        def _load_ranking_results(dataset: str, rmethod: str, nmethod: Optional[str], amethod: Optional[str],
                                  method_type: Optional[str], execution_id: Optional[int],
                                  filters: Dict[str, List[Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
            with self._instance.begin() as conn:
                if method_type is None:
                    method_type = self._resolve_method_type(rmethod, conn)

                if method_type.startswith("AutoTSAD"):
                    query = select(
                        self._instance.ranking_entry_table.c.rank,
                        self._instance.ranking_entry_table.c.algorithm_scoring_id,
                        self._instance.algorithm_scoring_table.c.algorithm,
                        self._instance.algorithm_scoring_table.c.hyper_params
                    ).where(
                        self._instance.autotsad_execution_table.c.dataset_id == self._instance.dataset_table.c.hexhash,
                        self._instance.autotsad_execution_table.c.algorithm_ranking_id == self._instance.ranking_table.c.id,
                        self._instance.ranking_table.c.id == self._instance.ranking_entry_table.c.ranking_id,
                        self._instance.ranking_entry_table.c.algorithm_scoring_id == self._instance.algorithm_scoring_table.c.id,
                        self._instance.dataset_table.c.name == dataset,
                        self._instance.autotsad_execution_table.c.ranking_method == rmethod,
                    ).order_by(self._instance.ranking_entry_table.c.rank)
                    if nmethod is not None:
                        query = query.where(self._instance.autotsad_execution_table.c.normalization_method == nmethod)
                    if amethod is not None:
                        query = query.where(self._instance.autotsad_execution_table.c.aggregation_method == amethod)
                    if execution_id is not None:
                        query = query.where(self._instance.autotsad_execution_table.c.id == execution_id)
                    query = self._add_autotsad_filters(query, filters)
                    df_ranking = pd.read_sql(query, conn)
                else:
                    query = text(
                        f"""select x.rank, x.algorithm_scoring_id, s.algorithm, s.hyper_params
                            from (select coalesce(re.rank, 1)                                      as "rank",
                                         coalesce(re.algorithm_scoring_id, e.algorithm_scoring_id) as "algorithm_scoring_id"
                                  from (select b.algorithm_scoring_id, b.algorithm_ranking_id
                                        from baseline_execution b,
                                             dataset d
                                        where b.dataset_id = d.hexhash
                                          and d.name = '{dataset}'
                                          and b.name = '{rmethod}') e
                                           left outer join algorithm_ranking r on e.algorithm_ranking_id = r.id
                                           left outer join algorithm_ranking_entry re on re.ranking_id = r.id) x,
                                 algorithm_scoring s
                            where x.algorithm_scoring_id = s.id
                            order by x.rank
                        """
                    )
                    df_ranking = pd.read_sql(query, conn)

                ids = df_ranking["algorithm_scoring_id"].unique().tolist()
                df_scorings = pd.read_sql(
                    select(self._instance.scoring_table)
                    .where(self._instance.scoring_table.c.algorithm_scoring_id.in_(ids))
                    .order_by(self._instance.scoring_table.c.algorithm_scoring_id, self._instance.scoring_table.c.time)
                    , conn)
                return df_ranking, df_scorings

        return _load_ranking_results(dataset, rmethod, nmethod, amethod, method_type, execution_id, filters)

    def load_aggregated_scoring(self, aggregated_scoring_id: int) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=100)
        def _load_aggregated_scoring(aggregated_scoring_id: int) -> pd.DataFrame:
            with self._instance.begin() as conn:
                return pd.read_sql(
                    select(self._instance.aggregated_scoring_scores_table.c.time,
                           self._instance.aggregated_scoring_scores_table.c.score)
                    .where(self._instance.aggregated_scoring_scores_table.c.aggregated_scoring_id == aggregated_scoring_id)
                    .order_by(self._instance.aggregated_scoring_scores_table.c.time),
                    conn
                )

        return _load_aggregated_scoring(aggregated_scoring_id)

    def load_aggregated_scoring_for(self, dataset: str, rmethod: str, nmethod: str, amethod: str,
                                    filters: Dict[str, List[Any]] = {}) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _load_aggregated_scoring_for(dataset: str, rmethod: str, nmethod: str, amethod: str, filters: Dict[str, List[Any]]) -> pd.DataFrame:
            with self._instance.begin() as conn:
                query = (
                    select(self._instance.aggregated_scoring_scores_table.c.time, self._instance.aggregated_scoring_scores_table.c.score)
                    .where(
                        self._instance.aggregated_scoring_scores_table.c.aggregated_scoring_id == self._instance.autotsad_execution_table.c.aggregated_scoring_id,
                        self._instance.autotsad_execution_table.c.dataset_id == self._instance.dataset_table.c.hexhash,
                        self._instance.autotsad_execution_table.c.experiment_id == self._instance.experiment_table.c.id,
                        self._instance.experiment_table.c.description != "paper v1 - optimization / variance",
                        self._instance.autotsad_execution_table.c.ranking_method == rmethod,
                        self._instance.autotsad_execution_table.c.normalization_method == nmethod,
                        self._instance.autotsad_execution_table.c.aggregation_method == amethod,
                        self._instance.dataset_table.c.name == dataset
                    )
                    .order_by(self._instance.aggregated_scoring_scores_table.c.time))
                query = self._add_autotsad_filters(query, filters)
                return pd.read_sql(query, conn)

        return _load_aggregated_scoring_for(dataset, rmethod, nmethod, amethod, filters)

    def load_runtime_traces(self, traces: List[str], datasets: List[str], config_names: List[str]) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _load_runtime_traces(traces: List[str], datasets: List[str], config_names: List[str]) -> pd.DataFrame:
            trace_list = ",".join([f"'{t}'" for t in traces])
            dataset_name_list = ",".join([f"'{d}'" for d in datasets])
            config_name_list = ",".join([f"'{c}'" for c in config_names])
            query = f"""select a.name as "dataset_name", a."n_jobs", t.trace_name,
                               t.position, t.duration_ns / 1e9 as "runtime"
                        from runtime_trace t,
                             (select distinct e.experiment_id, d.name,
                                     (c.config #>> '{{general, n_jobs}}')::integer as "n_jobs"
                              from experiment x, dataset d, configuration c, autotsad_execution e
                              where e.dataset_id = d.hexhash
                                and e.config_id = c.id
                                and e.experiment_id = x.id
                                and x.description in ('paper v1 - quality', 'paper v1 - scaling')
                                and c.description in ({config_name_list})
                                and d.name in ({dataset_name_list})) a
                        where t.experiment_id = a.experiment_id
                          and trace_name in ({trace_list})
                          and trace_type = 'END'
                        order by a.name, a."n_jobs", t.position
                        ;"""
            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _load_runtime_traces(traces, datasets, config_names)

    def load_mean_runtime(self, optim_filters: Dict[str, List[Any]], default_filters: Dict[str, List[Any]], only_paper_datasets: bool = True) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _load_mean_runtime(optim_filters: Dict[str, List[Any]], default_filters: Dict[str, List[Any]], only_paper_datasets: bool) -> pd.DataFrame:
            default_filter_clauses = self._build_autotsad_filter_clauses(default_filters)
            optim_filter_clauses = self._build_autotsad_filter_clauses(optim_filters)

            baseline_filter_clauses = ""
            for baselines in list(_if_filter(default_filters, "baseline")) + list(_if_filter(optim_filters, "baseline")):
                baselines = [f"'{b}'" for b in baselines]
                baseline_filter_clauses += f" and b.name in ({','.join(baselines)})"

            if only_paper_datasets:
                default_filter_clauses += " and d.paper = True"
                optim_filter_clauses += " and d.paper = True"
                baseline_filter_clauses += " and d.paper = True"

            query = f"""select b.name, avg(runtime) as "mean_runtime"
                        from dataset d, baseline_execution b
                        where d.hexhash = b.dataset_id and b.runtime is not null {baseline_filter_clauses}
                        group by b.name
                        union
                        select 'AutoTSAD w/ optimization', avg(runtime) as "mean_runtime"
                        from dataset d, autotsad_execution e, experiment x, configuration c
                        where d.hexhash = e.dataset_id
                            and e.experiment_id = x.id
                            and e.config_id = c.id
                            and e.runtime is not null {optim_filter_clauses}
                        union
                        select 'AutoTSAD w/o optimization', avg(runtime) as "mean_runtime"
                        from dataset d, autotsad_execution e, experiment x, configuration c
                        where d.hexhash = e.dataset_id
                            and e.experiment_id = x.id
                            and e.config_id = c.id
                            and e.runtime is not null {default_filter_clauses};"""
            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _load_mean_runtime(optim_filters, default_filters, only_paper_datasets)

    def load_runtimes(self, filters: Dict[str, List[Any]], only_paper_datasets: bool = True) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=3)
        def _load_runtimes(filters: Dict[str, List[Any]], only_paper_datasets: bool) -> pd.DataFrame:
            autotsad_filter_clauses = self._build_autotsad_filter_clauses(filters)

            if only_paper_datasets:
                autotsad_filter_clauses += " and d.paper = True"

            query = f"""select a.name as "dataset_name",
                               a.runtime as "autotsad_runtime",
                               t.duration_ns / 1e9 as "algorithm_execution_runtime"
                        from runtime_trace t,
                            (select distinct e.experiment_id, d.name, e.runtime
                             from dataset d, autotsad_execution e
                             where e.dataset_id = d.hexhash {autotsad_filter_clauses}) a
                        where t.experiment_id = a.experiment_id
                            and trace_name = 'autotsad-%-Execution-%-Algorithm Execution'
                            and trace_type = 'END';"""
            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _load_runtimes(filters, only_paper_datasets)

    def load_optimization_variance_results(self, filters: Dict[str, List[Any]], datasets: List[str],
                                           config_names: List[str]) -> pd.DataFrame:
        filters.pop("config_id", None)
        filters.pop("experiment_id", None)

        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _load_optimization_variance_results(filters: Dict[str, List[Any]], datasets: List[str], config_names: List[str]) -> pd.DataFrame:
            autotsad_filter_clauses = self._build_autotsad_filter_clauses(filters, execution_table_name="e")

            dataset_name_list = ",".join([f"'{d}'" for d in datasets])
            config_clauses = " or ".join([f"c.description like '{c}'" for c in config_names])
            query = f"""select collection, dataset,
                               case when optimization_disabled then 0 else max_trials end as "max_trials",
                               seed,
                               range_pr_auc,
                               runtime
                        from (select d.collection, d.name as "dataset",
                                     (c.config #>> '{{general, seed}}')::integer                      as "seed",
                                     (c.config #>> '{{optimization, disabled}}')::boolean             as "optimization_disabled",
                                     (c.config #>> '{{optimization, max_trails_per_study}}')::integer as "max_trials",
                                     e.range_pr_auc, e.runtime
                              from autotsad_execution e, configuration c, dataset d, experiment x
                              where e.config_id = c.id
                                and e.dataset_id = d.hexhash
                                and e.experiment_id = x.id
                                and d.name in ({dataset_name_list})
                                and d.paper = True
                                and ({config_clauses})
                                and x.description like 'paper v1 - optimization / variance%' {autotsad_filter_clauses}) a
                        order by max_trials, seed;"""
            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _load_optimization_variance_results(filters, datasets, config_names)

    def load_optimization_improvements(self, filters: Dict[str, List[Any]], only_paper_datasets: bool = True) -> pd.DataFrame:
        filters.pop("config_id", None)
        # filters.pop("experiment_id", None)

        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _load_optimization_improvements(filters: Dict[str, List[Any]], only_paper_datasets: bool) -> pd.DataFrame:
            autotsad_filter_clauses = self._build_autotsad_filter_clauses(filters)

            if only_paper_datasets:
                autotsad_filter_clauses += " and d.paper = True"
            sub_query = """select distinct d.collection || ' ' || d.name as "dataset", e.range_pr_auc
                           from autotsad_execution e, dataset d, configuration c
                           where e.dataset_id = d.hexhash
                             and e.config_id = c.id"""
            query = f"""select a.dataset, a.range_pr_auc - b.range_pr_auc as "improvement"
                        from ({sub_query} and c.description = 'paper v1' {autotsad_filter_clauses}) a inner join
                             ({sub_query} and c.description = 'paper v1 - default ensemble (no optimization, seed=1)' {autotsad_filter_clauses}) b
                             on a.dataset = b.dataset
                        order by dataset;"""

            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _load_optimization_improvements(filters, only_paper_datasets)

    def compute_top1_baseline(self, filters: Dict[str, List[Any]], only_paper_datasets: bool = True) -> pd.DataFrame:
        @cache_data(ttl=self._ttl, persist=self._persist, max_entries=5)
        def _compute_top1_baseline(filters: Dict[str, List[Any]], only_paper_datasets: bool) -> pd.DataFrame:
            autotsad_filter_clauses = self._build_autotsad_filter_clauses(filters)

            if only_paper_datasets:
                autotsad_filter_clauses += " and d.paper = True"

            query = f"""select d.collection || ' ' || d.name as "dataset",
                               'AutoTSAD Top-1' as "Method Type", 'top-1' as "Method",
                               s.range_pr_auc, s.range_roc_auc
                        from autotsad_execution e, dataset d, configuration c, algorithm_ranking r, algorithm_ranking_entry re, algorithm_scoring s
                        where e.dataset_id = d.hexhash
                            and e.config_id = c.id
                            and e.algorithm_ranking_id = r.id
                            and r.id = re.ranking_id
                            and re.algorithm_scoring_id = s.id
                            and e.ranking_method = 'aggregated-minimum-influence'
                            and e.normalization_method = 'gaussian'
                            and e.aggregation_method = 'custom'  -- does not really matter but removes duplicates
                            and re.rank = 1 {autotsad_filter_clauses};"""

            with self._instance.begin() as conn:
                return pd.read_sql(text(query), conn)

        return _compute_top1_baseline(filters, only_paper_datasets)
