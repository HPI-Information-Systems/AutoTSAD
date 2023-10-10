-- dataset stuff
select t.*
    from timeseries t, dataset d
    where d.hexhash = t.dataset_id
        and d.name = '803'
    order by t.time
    limit 100;

select d.name, count(*), avg(t.value), stddev(t.value)
    from timeseries t, dataset d
    where d.hexhash = t.dataset_id
    group by d.name
    order by count(*) desc;

select count(*) from timeseries t;

select dataset_id, count(*)
    from timeseries
    where time between 10000 and 20000
    group by dataset_id;

-- scoring stuff
select id, dataset_id, algorithm, hyper_params_id
     from algorithm_scoring
     where id not in (select distinct algorithm_scoring_id from scoring)
     and range_pr_auc is not null;

select dataset_id, count(*)
    from algorithm_scoring
    group by dataset_id
    order by count(*) desc;

select count(distinct algorithm_scoring_id)
    from scoring;

select algorithm_scoring_id, count(*)
    from scoring
    group by algorithm_scoring_id;

select *
    from scoring
    where time between 10000 and 20000
    and algorithm_scoring_id = 336;

select s.*
    from scoring s, algorithm_scoring a, dataset d
    where s.algorithm_scoring_id = a.id
      and a.dataset_id = d.hexhash
        and d.name = '803';

select s.algorithm_scoring_id, count(*)
    from scoring s, algorithm_scoring a
    where s.algorithm_scoring_id = a.id
--         and a.experiment_id = 3
    group by s.algorithm_scoring_id;

select id, experiment_id from algorithm_scoring where experiment_id = 3;

-- size and counts
select
    (select count(*) from scoring) as "scoring",
    (select count(*) from timeseries) as "timeseries"
;

select
    pg_size_pretty(pg_total_relation_size('scoring')) as "scoring_table",
    pg_size_pretty(pg_total_relation_size('aggregated_scoring_scores')) as "aggregated_scoring_table",
    pg_size_pretty(pg_total_relation_size('timeseries')) as "timeseries_table",
    pg_size_pretty(hypertable_size('timeseries')) as "timeseries_hypertable",
    pg_size_pretty(pg_database_size('akita')) as "database";

SELECT * from chunks_detailed_size('timeseries');
--
select *
    from algorithm_ranking
    where id in (select distinct ranking_id
                     from algorithm_ranking_entry
                     order by ranking_id);

--
select ranking_method, trunc(avg(range_pr_auc)::numeric, 4), trunc(stddev(range_pr_auc)::numeric, 4)
    from autotsad_execution
    group by ranking_method
    order by avg(range_pr_auc) desc
;

insert into algorithm_ranking (experiment_id) values (Null) returning id;

select algorithm_scoring_id, count(*)
    from scoring s, algorithm_scoring a
    where s.algorithm_scoring_id = a.id
        and a.experiment_id = 9
    group by algorithm_scoring_id;

select * from scoring where algorithm_scoring_id = 336 and time between 10000 and 20000;

select algorithm_scoring_id, sum(score)
    from scoring
    where score > 1
    group by algorithm_scoring_id
;
select count(*) from scoring;

select
    a.hyper_params_id,
    (select hyper_params from algorithm_scoring where hyper_params_id = a.hyper_params_id limit 1) as hyper_params
from (select distinct hyper_params_id from algorithm_scoring) a;

select d.name,
       avg(a.range_pr_auc) filter (where a.ranking_method = 'interchange-annotation-overlap') as "interchange-annotation-overlap",
       avg(a.range_pr_auc) filter (where a.ranking_method = 'greedy-annotation-overlap') as "greedy-annotation-overlap",
       avg(a.range_pr_auc) filter (where a.ranking_method = 'greedy-euclidean') as "greedy-euclidean",
       avg(a.range_pr_auc) filter (where a.ranking_method = 'kmedoids-clustering') as "kmedoids-clustering",
       avg(a.range_pr_auc) filter (where a.ranking_method = 'affinity-propagation-clustering') as "affinity-propagation-clustering",
       avg(a.range_pr_auc) filter (where a.ranking_method = 'training-coverage') as "training-coverage",
       avg(b.range_pr_auc) filter (where b.name = 'default-baseline') as "default-baseline"
    from autotsad_execution a
        left outer join baseline_execution b on a.dataset_id = b.dataset_id, dataset d
    where a.dataset_id = d.hexhash
    group by d.name
    order by length(d.name)
;

select *
    from algorithm_scoring
    where id = 5000
;

-- get maximum window sizes
select dataset, algorithm, max(window_size), avg(window_size)::int, min(window_size)
from (select d.name as dataset,
             s.algorithm,
             coalesce(
                         s.hyper_params ->> 'window_size',
                         s.hyper_params ->> 'anomaly_window_size'
                 )::int as window_size
      from algorithm_scoring as s,
           dataset as d
      where s.dataset_id = d.hexhash
        and s.range_pr_auc is not null
        and s.algorithm not in ('k-Means', 'SAND')
        and not (s.hyper_params::jsonb @> '{}'::jsonb and '{}'::jsonb @> s.hyper_params::jsonb)
--   and d.name = '803_805_806_820'
     ) as a
group by dataset, algorithm
order by max(window_size) desc nulls last
;

-- get maximum window sizes within transaction
begin transaction;
drop table if exists "#window_sizes";
create temporary table "#window_sizes" as
    select d.name     as dataset,
           s.algorithm,
           coalesce(
                       s.hyper_params ->> 'window_size',
                       s.hyper_params ->> 'anomaly_window_size'
               )::int as window_size
    from algorithm_scoring as s,
         dataset as d
    where s.dataset_id = d.hexhash
      and s.range_pr_auc is not null
      and s.algorithm not in ('k-Means', 'SAND')
      and not (s.hyper_params::jsonb @> '{}'::jsonb and '{}'::jsonb @> s.hyper_params::jsonb)
;
select b.dataset,
       b.algorithm,
       b.window_size,
       trunc(a.max_ws_instances::numeric / b.instances::numeric, 2) as ratio,
       b.instances as all_instances
from (select dataset,
             algorithm,
             window_size,
             max(window_size) over (partition by dataset, algorithm)   as max_window_size,
             count(window_size) over (partition by dataset, algorithm) as instances
      from "#window_sizes") as b,
     (select dataset, algorithm, window_size, count(*) as max_ws_instances
      from "#window_sizes"
      group by dataset, algorithm, window_size) as a
where a.dataset = b.dataset
  and a.algorithm = b.algorithm
  and a.window_size = b.window_size
  and b.max_window_size = b.window_size
order by b.max_window_size desc nulls last
;
commit transaction;
-- end of maximum window sizes within transaction

select s.algorithm, s.hyper_params, s.range_pr_auc
    from algorithm_scoring s, dataset d
    where s.dataset_id = d.hexhash
        and d.name = '803_820'
    order by s.range_pr_auc desc nulls last
;

select dataset_id, ranking_method, normalization_method, aggregation_method, range_pr_auc, runtime
from autotsad_execution
where experiment_id = 45
;
-- update autotsad_execution
-- set runtime = 44356.246112256
-- where experiment_id = 45
-- ;

select normalization_method, aggregation_method, avg(range_pr_auc), avg(range_roc_auc), avg(precision_at_k)
    from autotsad_execution
    group by normalization_method, aggregation_method
    order by avg(precision_at_k) desc
;

update autotsad_execution
    set config_id = 'a598b763ddb3c34e7ca95c11c0388810'
    where config_id in (
        '799a2285c09c56e1a7327859affe8b92'
    )
;

delete from configuration
where id in ('71dda25d2ce23833635e7dd3c7eb2e56', '25822f61f6e2e77d15fd0b5a5a659d04', '7d019f179c402213bd393760c03be02f',
             'c1532957dd69f383410b9c7ea7840c0f', 'a1f5c4617a73ab9ad0fa48df0d6aeb6f', '781bf8a17792afba1ca19b4e342857cd',
             'c714557559e65ba3135ca45f2ba6a6fc', '44465b2d8eb16d4703f1c60dbc6ab125', '6f9186c9bd1b8dc34968f13ba1b1b30d',
             '6f606dd8d27cc67c47539c4771e1cf1d', '61e47beb162829be45f039aaf833c1ee', '84f46f7376b66bd505f4f8c1b93dbf9e',
             '99fe24f6f1d260dcda077d0f5d52e935', '83bbcd80f9db08c007865647ff2a1237', 'bb8d11e0a461c8dd3dc5ca99adf79933',
             '1181009036dfc1d3595ab32ff25ec20f', '8f9f67bdd36c7191122fee4dbd9c797d', '624073149ef88e309602186ae0d57c5f',
             '0d61a6d9f9702d6f9cc4ddc46d077000', '23852ce8121b5acac060e3a501fd0b0e')
;

-- datasets with missing 'base-algo' baseline
select *
    from dataset
    where hexhash not in (select distinct dataset_id from baseline_execution b where b.name = 'best-algo')
;

-- datasets with missing 'SAND' baseline
select *
    from dataset
    where hexhash not in (select distinct dataset_id from baseline_execution e where e.name = 'SAND (TimeEval)')
;

select * from scoring
    where algorithm_scoring_id in
-- select *
--     from algorithm_scoring
--     where id in
-- delete from algorithm_scoring where id in
delete from baseline_execution where id in
    (select distinct b.id
                 from dataset d,
                      baseline_execution b
                 where d.collection = 'univariate-anomaly-test-cases'
                   and d.hexhash = b.dataset_id
                   and b.name = 'k-Means (TimeEval)')
;

select *
from baseline_execution
where coalesce(algorithm_scoring_id, algorithm_ranking_id) is not null
;

select s.algorithm, s.hyper_params, d.name
    from algorithm_scoring s, baseline_execution b, dataset d
    where s.id = b.algorithm_scoring_id and b.dataset_id = d.hexhash
    and b.name = 'best-algo'
    and d.name = 'Discords_marotta_valve_tek_17'
;

-- datasets with missing AutoTSAD results
select distinct d.collection, d.name
    from dataset d left outer join autotsad_execution e on d.hexhash = e.dataset_id
    where d.paper = True and e.ranking_method is null
    order by collection, name
;

select a.trace_name, a."mean_runtime", a."stddev_runtime"
from (select trace_name,
             avg(duration_ns / 1e9)    as "mean_runtime",
             stddev(duration_ns / 1e9) as "stddev_runtime"
      from runtime_trace t inner join experiment e on t.experiment_id = e.id
      where trace_name in ('autotsad-%-Base TS generation', 'autotsad-%-Cleaning', 'autotsad-%-Limiting',
                           'autotsad-%-Anomaly injection', 'autotsad-%-Optimization-%-Sensitivity analysis',
                           'autotsad-%-Optimization-%-Hyperparams opt.',
                           'autotsad-%-Optimization-%-Selecting best performers',
                           'autotsad-%-Execution-%-Algorithm Execution',
                           'autotsad-%-Execution-%-Computing all combinations')
        and trace_type = 'END'
        and e.description = 'paper v1 - quality'
      group by trace_name) a,
     (select trace_name, min(position) as position
      from runtime_trace
      where trace_type = 'START'
      group by trace_name
      order by min(position)) b
where a.trace_name = b.trace_name
order by b.position
;

select a.name as "dataset_name", a."n_jobs", t.trace_name, t.position, t.duration_ns / 1e9 as "runtime"
from runtime_trace t,
     (select distinct e.experiment_id, d.name, (c.config #>> '{general, n_jobs}')::integer as "n_jobs"
      from experiment x, dataset d, configuration c, autotsad_execution e
      where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and e.experiment_id = x.id
        and x.description in ('paper v1 - quality', 'paper v1 - scaling')
        and c.description in
            ('paper v1', 'paper v1 - n_jobs=10', 'paper v1 - n_jobs=5', 'paper v1 - n_jobs=1',
             'paper v1 - n_jobs=40')
        and d.name in ('-69_2_0.02_15', '022_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z4')) a
where t.experiment_id = a.experiment_id
  and trace_name in ('autotsad-%-Base TS generation', 'autotsad-%-Cleaning', 'autotsad-%-Limiting',
                     'autotsad-%-Anomaly injection', 'autotsad-%-Optimization-%-Sensitivity analysis',
                     'autotsad-%-Optimization-%-Hyperparams opt.',
                     'autotsad-%-Optimization-%-Selecting best performers',
                     'autotsad-%-Execution-%-Algorithm Execution',
                     'autotsad-%-Execution-%-Computing all combinations')
  and trace_type = 'END'
order by a.name, a."n_jobs", t.position
;

select b.name, avg(runtime) as "mean_runtime"
from dataset d, baseline_execution b
where d.hexhash = b.dataset_id
    and d.paper = True
    and b.runtime is not null
group by b.name
union
select 'AutoTSAD', avg(runtime) as "mean_runtime"
from dataset d, autotsad_execution e, experiment x, configuration c
where d.hexhash = e.dataset_id
    and d.paper = True
    and e.runtime is not null
    and e.experiment_id = x.id
    and x.description = 'paper v1 - quality'
    and e.config_id = c.id
    and c.description = 'paper v1'
;

select nullif(range_pr_auc, 'NaN') is null, count(*)
    from baseline_execution b, dataset d
    where b.dataset_id = d.hexhash
    and d.paper = True
    and b.name = 'SAND (TimeEval)'
group by nullif(range_pr_auc, 'NaN') is null
;

select a.collection, a.name, a.ranking_method, a.normalization_method, a.aggregation_method,
       b.range_pr_auc - a.range_pr_auc as "improvement"
from (select d.collection, d.name, e.ranking_method, e.normalization_method, e.aggregation_method, e.range_pr_auc
      from autotsad_execution e, dataset d
      where e.dataset_id = d.hexhash
        and d.paper = True
        and e.config_id = '65df673233b659aeac7df950b71c2d7a'
        and e.autotsad_version = '0.2.1') a
inner join (select d.collection, d.name, e.ranking_method, e.normalization_method, e.aggregation_method, e.range_pr_auc
            from autotsad_execution e, dataset d
            where e.dataset_id = d.hexhash
              and d.paper = True
              and e.config_id = '26a8aa1409ae87761d9d405ff0e49f9e'
              and e.autotsad_version = '0.2.1') b
on a.collection = b.collection and a.name = b.name and a.aggregation_method = b.aggregation_method
    and a.normalization_method = b.normalization_method and a.ranking_method = b.ranking_method
order by a.collection, a.name, a.range_pr_auc - b.range_pr_auc desc
;

select case when optimization_disabled then 0 else max_trials end as "max_trials",
       seed,
       range_pr_auc,
       runtime
from (select (c.config #>> '{general, seed}')::integer                      as "seed",
             (c.config #>> '{optimization, disabled}')::boolean             as "optimization_disabled",
             (c.config #>> '{optimization, max_trails_per_study}')::integer as "max_trials",
             e.range_pr_auc,
             e.runtime
      from autotsad_execution e, configuration c, dataset d, experiment x
      where e.config_id = c.id
        and e.dataset_id = d.hexhash
        and e.experiment_id = x.id
        and d.collection = 'TSB-UAD-synthetic'
        and d.name = 'KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727'
        and d.paper = True
        and c.description in ('paper v1', 'paper v1 optimization / variance', 'paper v1 - default ensemble (no optimization)')
        and e.autotsad_version = '0.2.1'
        and e.ranking_method = 'training-quality'
        and e.normalization_method = 'gaussian'
        and e.aggregation_method = 'custom'
        and x.description = 'paper v1 - optimization / variance') a
order by max_trials, seed
;

select a.dataset, a.range_pr_auc - b.range_pr_auc as "improvement", a.range_pr_auc as "optimized", b.range_pr_auc as "default"
from (select d.collection || ' ' || d.name as "dataset", e.range_pr_auc
      from autotsad_execution e, dataset d, configuration c
      where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and d.paper = True
        and e.autotsad_version = '0.2.1'
        and e.ranking_method = 'training-result'
        and e.normalization_method = 'gaussian'
        and e.aggregation_method = 'custom'
        and c.description = 'paper v1'
      order by d.collection, d.name) a inner join
     (select d.collection || ' ' || d.name as "dataset", e.range_pr_auc
      from autotsad_execution e, dataset d,configuration c
      where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and d.paper = True
        and e.autotsad_version = '0.2.1'
        and e.ranking_method = 'training-result'
        and e.normalization_method = 'gaussian'
        and e.aggregation_method = 'custom'
        and c.description = 'paper v1 - default ensemble (no optimization, seed=1)'
      order by d.collection, d.name) b on a.dataset = b.dataset
where a.range_pr_auc - b.range_pr_auc > 0.1
order by "default" asc
;

-- top-1 method for each dataset
select d.collection || ' ' || d.name as "dataset", 'Baseline' as "Method Type", 'top-1' as "Method", s.range_pr_auc, s.range_roc_auc
from autotsad_execution e, dataset d, configuration c, algorithm_ranking r, algorithm_ranking_entry re, algorithm_scoring s
where e.dataset_id = d.hexhash
    and e.config_id = c.id
    and e.algorithm_ranking_id = r.id
    and r.id = re.ranking_id
    and re.algorithm_scoring_id = s.id
    and d.paper = True
    and e.autotsad_version = '0.2.1'
    and e.ranking_method = 'training-quality'
    and e.normalization_method = 'gaussian'
    and e.aggregation_method = 'custom'  -- does not really matter but removes duplicates
    and c.description = 'paper v1'
    and re.rank = 1
;

-- find example for "other ranking than training-quality would have been better"
select a.collection, a.name, a.range_pr_auc, b.range_pr_auc, b.ranking_method, b.normalization_method, b.aggregation_method
from (select d.collection, d.name, e.range_pr_auc, e.range_roc_auc
      from autotsad_execution e,
           dataset d,
           configuration c
      where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and d.paper = True
        and e.autotsad_version = '0.2.1'
        and c.description = 'paper v1'
        and e.ranking_method = 'training-quality'
        and e.normalization_method = 'gaussian'
        and e.aggregation_method = 'custom'
        and e.range_pr_auc < 0.5
      order by e.range_pr_auc) a
         inner join (select d.collection, d.name, e.range_pr_auc, e.range_roc_auc, e.ranking_method, e.normalization_method, e.aggregation_method
                     from autotsad_execution e,
                          dataset d,
                          configuration c
                     where e.dataset_id = d.hexhash
                       and e.config_id = c.id
                       and d.paper = True
                       and e.autotsad_version = '0.2.1'
                       and c.description = 'paper v1'
                       and e.ranking_method in ('training-result', 'mmq-annotation-overlap', 'mmq-euclidean',
                                                'affinity-propagation-clustering', 'kmedoids-clustering')
                       and e.normalization_method = 'gaussian'
                       and e.aggregation_method in ('custom', 'max')
                       and e.range_pr_auc > 0.5
                     order by e.range_pr_auc) b on a.collection = b.collection and a.name = b.name
where a.name != 'ec2_network_in_257a54'
order by b.range_pr_auc - a.range_pr_auc desc
;

select seed, max_trails, optimization_disabled, description
from (
select (config #>> '{general, seed}')::integer as "seed",
       (config #>> '{optimization, max_trails_per_study}')::integer as "max_trails",
       (config #>> '{optimization, disabled}')::bool as "optimization_disabled",
       c.*
    from configuration c) a
where description like '%paper v1%'
order by optimization_disabled, max_trails, seed
;

select *
      from autotsad_execution e, dataset d, configuration c, experiment x
      where e.dataset_id = d.hexhash
        and e.config_id = c.id
        and e.experiment_id = x.id
        and d.paper = True
        and d.name = 'A4Benchmark-13'
--         and d.name = 'KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727'
        and e.autotsad_version = '0.2.1'
        and c.description = 'paper v1'
        and e.ranking_method = 'training-quality'
        and e.normalization_method = 'gaussian'
        and e.aggregation_method = 'custom'
;

select concat(d.collection, ' ', d.name)                                                as "Dataset",
       concat(e.ranking_method, '_', e.normalization_method, '_', e.aggregation_method) as "Method",
       e.range_pr_auc,
       e.range_roc_auc
from autotsad_execution e, dataset d
where e.dataset_id = d.hexhash
    and e.autotsad_version = '0.2.1'
    and e.config_id = '26a8aa1409ae87761d9d405ff0e49f9e'
    and d.paper = True
    and d.name = 'KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727'
order by concat(e.ranking_method, '_', e.normalization_method, '_', e.aggregation_method)
;

select algorithm_ranking_id, aggregated_scoring_id
from autotsad_execution
where ranking_method in ('aggregated-minimum-influence', 'aggregated-robust-borda')
  and (aggregated_scoring_id is null or algorithm_ranking_id is null)
;

select c.id, c.description, e.ranking_method
from configuration c left outer join (
    select config_id, ranking_method
    from autotsad_execution
    where ranking_method in ('aggregated-minimum-influence', 'aggregated-robust-borda')
) e on c.id = e.config_id
where c.description like 'paper v1%' and e.ranking_method is null
;

-- FIXME! those scoring are missing
select d.name, b.*
from baseline_execution b, dataset d
where b.name = 'k-Means (TimeEval)'
and b.algorithm_scoring_id is null
and b.dataset_id = d.hexhash
;

select d.name, s.*
from autotsad_execution e, configuration c, dataset d, algorithm_ranking_entry r, algorithm_scoring s
where e.config_id = c.id
    and e.dataset_id = d.hexhash
    and e.algorithm_ranking_id = r.ranking_id
    and r.algorithm_scoring_id = s.id
    and c.description like 'paper v1 - default%'
    and e.autotsad_version = '0.2.1'
    and e.ranking_method = 'training-quality'
    and e.normalization_method = 'gaussian'
    and e.aggregation_method = 'custom'
order by s.algorithm, s.hyper_params_id
;

select distinct description from experiment order by description;
