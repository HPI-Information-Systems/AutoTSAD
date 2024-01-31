create user autotsad with encrypted password 'holistic-tsad2023';

-- schema
create schema if not exists autotsad;

alter schema autotsad owner to autotsad;

set search_path = "autotsad";
set schema 'autotsad';

---------------------------------------
-- dataset tables
create table if not exists dataset
(
    hexhash    varchar not null
        constraint dataset_pk primary key,
    name       varchar,
    collection varchar,
    paper      boolean default false not null
);

alter table dataset
    owner to autotsad;

create table if not exists timeseries
(
    time       integer not null,
    dataset_id varchar not null
        constraint timeseries_dataset_id_fk
            references dataset
            on update cascade on delete cascade,
    value      double precision,
    is_anomaly boolean
);

alter table timeseries
    owner to autotsad;

select create_hypertable('timeseries', 'time',
                         chunk_time_interval => 86400,
                         if_not_exists => TRUE);

-- not compatible with compression:
alter table timeseries
    add constraint time_dataset_id_unqiue unique (time, dataset_id);

create index if not exists timeseries_dataset_id_time_idx
    on timeseries (dataset_id, time asc);

-- we don't need compression for now
-- alter table timeseries set (
--     timescaledb.compress,
--     timescaledb.compress_orderby = 'time',
--     timescaledb.compress_segmentby = 'dataset_id'
-- );
---------------------------------------

---------------------------------------
-- configuration table
create table if not exists configuration
(
    id          varchar not null
        constraint configuration_pk primary key,
    config      json    not null,
    description varchar
);

alter table configuration
    owner to autotsad;
---------------------------------------


---------------------------------------
-- experiment table
create table if not exists experiment
(
    id          serial not null
        constraint experiment_pk primary key,
    name        varchar not null,
    description varchar,
    date        timestamptz
);

alter table experiment
    owner to autotsad;

create sequence if not exists experiment_id_seq as integer;
alter sequence experiment_id_seq owner to autotsad;
alter sequence experiment_id_seq owned by experiment.id;
---------------------------------------

---------------------------------------
-- scoring tables
create table if not exists algorithm_scoring
(
    id              serial  not null
        constraint algorithm_scoring_pk primary key,
    dataset_id      varchar
        constraint timeseries_dataset_id_fk
            references dataset
            on update cascade on delete set null,
    experiment_id   integer
        constraint algorithm_scoring_experiment_id_fk
            references experiment
            on update cascade on delete restrict,
    algorithm       varchar not null,
    hyper_params_id varchar not null,
    hyper_params    json    not null,
    range_pr_auc    double precision,
    range_roc_auc   double precision,
    precision_at_k  double precision,
    runtime         double precision
);

alter table algorithm_scoring
    owner to autotsad;

create sequence if not exists algorithm_scoring_id_seq as integer;
alter sequence algorithm_scoring_id_seq owner to autotsad;
alter sequence algorithm_scoring_id_seq owned by algorithm_scoring.id;

create table if not exists scoring
(
    time                 integer not null,
    algorithm_scoring_id integer not null
        constraint scoring_algorithm_scoring_id_fk
            references algorithm_scoring
            on update cascade on delete cascade,
    score                double precision
);

alter table scoring
    owner to autotsad;

create index if not exists scoring_algorithm_scoring_id_time_idx
    on scoring (algorithm_scoring_id, time asc);

-- create table if not exists scoring
-- (
--     time                 integer not null,
--     algorithm_scoring_id integer not null
--         constraint scoring_algorithm_scoring_id_fk
--             references algorithm_scoring
--             on update cascade on delete cascade,
--     score                double precision
-- );
--
-- alter table scoring
--     owner to autotsad;
--
-- select create_hypertable('scoring', 'time',
--                          chunk_time_interval => 86400,
--                          if_not_exists => TRUE);
--
-- -- not compatible with compression:
-- alter table scoring
--     add constraint scoring_time_algorithm_scoring_id_unqiue unique (time, algorithm_scoring_id);
--
-- create index if not exists scoring_algorithm_scoring_id_time_idx
--     on scoring (algorithm_scoring_id, time asc);
---------------------------------------

---------------------------------------
-- result tables

-- table for algorithm rankings
create table if not exists algorithm_ranking (
    id serial not null
        constraint algorithm_ranking_pk primary key,
    experiment_id   integer
        constraint algorithm_ranking_experiment_id_fk
            references experiment
            on update cascade on delete restrict
);

alter table algorithm_ranking
    owner to autotsad;

create sequence if not exists algorithm_ranking_id_seq as integer;
alter sequence algorithm_ranking_id_seq owner to autotsad;
alter sequence algorithm_ranking_id_seq owned by algorithm_ranking.id;

create table if not exists algorithm_ranking_entry (
    ranking_id           integer not null
        constraint algorithm_ranking_entry_ranking_id_fk
            references algorithm_ranking
            on update cascade on delete cascade,
    rank                 integer not null,
    algorithm_scoring_id integer
        constraint algorithm_ranking_entry_scoring_id_fk
            references algorithm_scoring
            on update cascade on delete set null
);

alter table algorithm_ranking_entry
    owner to autotsad;

alter table algorithm_ranking_entry
    add constraint algorithm_ranking_entry_pk
        primary key (ranking_id, rank);

create index if not exists algorithm_ranking_entry_ranking_id_idx
    on algorithm_ranking_entry (ranking_id);

-- autotsad algorithm executions
create table if not exists autotsad_algorithm_execution
(
    id                    serial  not null
        constraint autotsad_algorithm_execution_pk primary key,
    algorithm_scoring_id  integer
        constraint autotsad_algorithm_execution_algorithm_scoring_id_fk
            references algorithm_scoring
            on update cascade on delete set null,
    experiment_id   integer
        constraint autotsad_algorithm_execution_experiment_id_fk
            references experiment
            on update cascade on delete restrict,
    mean_train_quality    double precision,
    no_datasets           integer,
    quality               double precision
);

alter table autotsad_algorithm_execution
    owner to autotsad;

create sequence if not exists autotsad_algorithm_execution_id_seq as integer;
alter sequence autotsad_algorithm_execution_id_seq owner to autotsad;
alter sequence autotsad_algorithm_execution_id_seq owned by autotsad_algorithm_execution.id;

-- result table for each autotsad execution and its metric scores
create table if not exists autotsad_execution
(
    id                   serial  not null
        constraint autotsad_execution_pk primary key,
    dataset_id           varchar not null
        constraint autotsad_execution_dataset_id_fk
            references dataset
            on update cascade on delete restrict,
    config_id            integer not null
        constraint autotsad_execution_config_id_fk
            references configuration
            on update cascade on delete restrict,
    experiment_id        integer
        constraint autotsad_execution_experiment_id_fk
            references experiment,
    autotsad_version     varchar not null,
    ranking_method       varchar not null,
    normalization_method varchar not null,
    aggregation_method   varchar not null,
    algorithm_ranking_id integer
        constraint autotsad_execution_algorithm_ranking_id_fk
            references algorithm_ranking
            on update cascade on delete set null,
    runtime              double precision,
    pr_auc               double precision,
    roc_auc              double precision,
    range_pr_auc         double precision,
    range_roc_auc        double precision,
    range_pr_vus         double precision,
    range_roc_vus        double precision,
    range_precision      double precision,
    range_recall         double precision,
    range_fscore         double precision,
    precision_at_k       double precision,
    precision            double precision,
    recall               double precision,
    fscore               double precision
);

alter table autotsad_execution
    owner to autotsad;

create sequence if not exists autotsad_execution_id_seq as integer;
alter sequence autotsad_execution_id_seq owner to autotsad;
alter sequence autotsad_execution_id_seq owned by autotsad_execution.id;

create index if not exists autotsad_execution_dataset_id_idx
    on autotsad_execution (dataset_id);

-- result table for the baseline executions
create table if not exists baseline_execution
(
    id              serial  not null
        constraint baseline_execution_pk primary key,
    dataset_id      varchar not null
        constraint baseline_execution_dataset_id_fk
            references dataset
            on update cascade on delete restrict,
    name            varchar not null,
    algorithm_ranking_id integer
        constraint baseline_execution_algorithm_ranking_id_fk
            references algorithm_ranking
            on update cascade on delete set null,
    algorithm_scoring_id integer
        constraint baseline_execution_algorithm_scoring_id_fk
            references algorithm_scoring
            on update cascade on delete set null,
    runtime         double precision,
    pr_auc          double precision,
    roc_auc         double precision,
    range_pr_auc    double precision,
    range_roc_auc   double precision,
    range_pr_vus    double precision,
    range_roc_vus   double precision,
    range_precision double precision,
    range_recall    double precision,
    range_fscore    double precision,
    precision_at_k  double precision,
    precision       double precision,
    recall          double precision,
    fscore          double precision
);

alter table baseline_execution
    owner to autotsad;

create sequence if not exists baseline_execution_id_seq as integer;
alter sequence baseline_execution_id_seq owner to autotsad;
alter sequence baseline_execution_id_seq owned by baseline_execution.id;

-- table for the runtime traces
create table if not exists runtime_trace
(
    experiment_id integer
        constraint runtime_trace_experiment_id_fk
            references experiment
            on update cascade on delete cascade,
    position      integer not null,
    trace_name    varchar not null,
    trace_type    varchar not null,
    begin_ns      bigint,
    end_ns        bigint,
    duration_ns   bigint
);

alter table runtime_trace
    owner to autotsad;
---------------------------------------
