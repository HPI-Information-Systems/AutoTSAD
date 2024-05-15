import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, FrozenSet, Tuple, List, Dict

import numpy as np
from nx_config import Config, ConfigSection, validate
from timeeval.metrics import (
    Metric,
    RangePrAUC,
    RangeRocAUC,
    PrAUC,
    RocAUC,
    RangePrecision,
    RangeRecall,
    PrecisionAtK,
    RangeFScore,
    RangePrVUS,
    RangeRocVUS,
    Precision,
    Recall,
    F1Score,
)
from timeeval.metrics.thresholding import NoThresholding


class ConfigurationError(Exception):
    def __init__(self, name: str, value: Any, message: str):
        super().__init__(f"Option '{name}={value}': {message}")


METRIC_MAPPING = {
    "PrAUC": PrAUC(),
    "RocAUC": RocAUC(),
    "RangePrAUC": RangePrAUC(buffer_size=100),
    "RangeRocAUC": RangeRocAUC(buffer_size=100),
    "RangePrVUS": RangePrVUS(),
    "RangeRocVUS": RangeRocVUS(),
    "RangePrecision": RangePrecision(),
    "RangeRecall": RangeRecall(),
    "RangeFScore": RangeFScore(),
    "Precision": Precision(NoThresholding()),
    "Recall": Recall(NoThresholding()),
    "FScore": F1Score(NoThresholding()),
    "PrecisionAtK": PrecisionAtK(),
}
ALGORITHMS: Tuple[str, ...] = (
    "subsequence_lof",
    "subsequence_knn",
    "subsequence_if",
    "stomp",
    "kmeans",
    "dwt_mlead",
    "torsk",
    "grammarviz",
)
ANOMALY_TYPES: Tuple[str, ...] = (
    "outlier",
    "compress",
    "stretch",
    "noise",
    "smoothing",
    "hmirror",
    "vmirror",
    "scale",
    "pattern",
)
SCORE_NORMALIZATION_METHODS: Tuple[str, ...] = ("minmax", "gaussian")
SCORE_AGGREGATION_METHODS: Tuple[str, ...] = ("custom", "max", "mean")
ALGORITHM_SELECTION_METHODS: Tuple[str, ...] = (
    "training-coverage",
    "training-quality",
    "training-result",
    "affinity-propagation-clustering",
    "kmedoids-clustering",
    "greedy-euclidean",
    "greedy-annotation-overlap",
    "mmq-euclidean",
    "mmq-annotation-overlap",
    "interchange-euclidean",
    "interchange-annotation-overlap",
    "aggregated-minimum-influence",
)

BASELINE_MAX_NAME = "best-algo"
BASELINE_MEAN_NAME = "mean-algo"
BASELINE_KMEANS_NAME = "k-Means (TimeEval)"
BASELINE_SAND_NAME = "SAND (TimeEval)"


class GeneralSection(ConfigSection):
    """General configuration options for the AutoTSAD system.

    Attributes
    ----------
    tmp_path :
        Folder for all temporary files and caches.
    result_path :
        Folder for the results.
    cache_key :
        Cache key to distinguish between multiple runs of AutoTSAD without overwriting the results. If None, the hash of
        the target dataset is used.
    logging_level :
        Logging level: 0 = off, 50 = critical, 40 = error, 30 = warning, 20 = info, 10 = debug.
    progress :
        Show progress bars.
    n_jobs :
        Controls the maximum parallelism of various steps in the pipeline.
    seed :
        Seed for the random number generator (e.g. for anomaly strengths and positions or for the optimization process).
        Set to ``None`` to use a random seed.
    max_algorithm_instances :
        Number of output scores to select for the final result and visualization.
    algorithm_selection_method :
        Algorithm instance selection method used to select the algorithm candidates (algorithm and its parameters) for
        displaying it to the user.
    score_normalization_method :
        Method used to normalize the scores of the individual algorithms to make them comparable.
    score_aggregation_method :
        Method used to combine the normalized scores of the individual algorithms to a single score. The 'custom'
        method applies a threshold to each individual scores, computes the difference of the score to the threshold,
        and just aggregates the differences > 0 via max().
    compute_all_combinations :
        Whether to compute and save all combinations of ``algorithm_selection_method``s,
        ``score_normalization_method``s, and ``score_combination_method``s. This is useful for the evaluation of
        different combinations of these methods. Usually, this should be set to ``False``.
    plot_final_scores :
        Whether to create a plot of the final scores on the target time series.
    TRAINING_TIMESERIES_LENGTH :
        Used to limit the length of generated training time series. Each training time series must be smaller than
        either the maximum length or the maximum number of periods (max(max_length), period_size*max_periods)).
    TRAINING_TIMESERIES_MIN_NO_PERIODS :
        Used to limit the length of generated training time series. Each training time series must be smaller than
        either the maximum length or the maximum number of periods (max(max_length), period_size*max_periods)).
    training_timeout_s :
        Timeout for optimizing an algorithm on a training dataset of minimum length (in seconds). If a training dataset
        is larger, the timeout is increased proportionally. A value of <= 0 disables the timeout.
    testing_timeout_s :
        Timeout for optimizing an algorithm on a testing dataset of minimum length (in seconds). A value of <= 0
        disables the timeout.
    memory_limit_mb :
        Maximum amount of main memory an algorithm is allowed to use (in megabytes). A value of <= 0 disables the limit.
    pynisher_resource_enforcer :
        Whether to use the pynisher resource enforcer to enforce the timeout and memory limit. If set to ``False``, the
        timeout and memory limits are not enforced.
    """

    tmp_path: Path = Path(tempfile.gettempdir()) / "autotsad"
    result_path: Path = Path("results-autotsad")
    TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_key: Optional[str] = None
    logging_level: int = logging.INFO
    use_timer: bool = True
    timer_logging_level: int = logging.INFO
    progress: bool = True
    n_jobs: int = -1

    seed: Optional[int] = 1

    max_algorithm_instances: int = 6
    algorithm_selection_method: str = "aggregated-minimum-influence"
    score_normalization_method: str = "gaussian"
    score_aggregation_method: str = "custom"
    plot_final_scores: bool = True
    compute_all_combinations: bool = False

    # length limiting
    TRAINING_TIMESERIES_LENGTH: int = 2000
    TRAINING_TIMESERIES_MIN_NO_PERIODS: int = 10

    # resource constraints
    training_timeout_s: int = 10 * 60  # 10 minutes
    testing_timeout_s: int = 1 * 3600  # 1 hour
    memory_limit_mb: int = 6 * 1024  # 6 GB
    pynisher_resource_enforcer: bool = True

    def cache_dir(self) -> Path:
        if not self.cache_key:
            path = self.tmp_path / "cache"
        else:
            path = self.tmp_path / "cache" / self.cache_key
        return path.resolve()

    def result_dir(self) -> Path:
        if not self.cache_key:
            path = self.result_path
        else:
            path = self.result_path / f"{self.TIMESTAMP}-{self.cache_key}"
        return path.resolve()

    def adjusted_training_limits(self, length: int) -> Dict[str, Any]:
        adjusted_timeout: Optional[int] = None
        if self.training_timeout_s > 0:
            factor = length / self.TRAINING_TIMESERIES_LENGTH
            adjusted_timeout = int(factor * self.training_timeout_s)

        memory_limit = self.memory_limit_mb if self.memory_limit_mb > 0 else None
        return {
            "enabled": self.pynisher_resource_enforcer,
            "time_limit": adjusted_timeout,
            "memory_limit": memory_limit,
        }

    def default_training_limits(self) -> Dict[str, Any]:
        time_limit = self.training_timeout_s if self.training_timeout_s > 0 else None
        memory_limit = self.memory_limit_mb if self.memory_limit_mb > 0 else None
        return {
            "enabled": self.pynisher_resource_enforcer,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }

    def default_testing_limits(self) -> Dict[str, Any]:
        time_limit = self.testing_timeout_s if self.testing_timeout_s > 0 else None
        memory_limit = self.memory_limit_mb if self.memory_limit_mb > 0 else None
        return {
            "enabled": self.pynisher_resource_enforcer,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }

    @validate
    def validate_options(self) -> None:
        # if not self.tmp_path.parent.exists():
        #     raise ConfigurationError("tmp_path", self.tmp_path,
        #                              f"Temp folder destination {self.tmp_path.parent} does not exist!")
        if self.tmp_path.exists() and not self.tmp_path.parent.is_dir():
            raise ConfigurationError(
                "tmp_path", self.tmp_path, "Temp path exists, but is not a folder!"
            )

        if self.logging_level < logging.NOTSET or self.logging_level > logging.CRITICAL:
            raise ConfigurationError(
                "logging_level", self.logging_level, "Out of range!"
            )

        if self.max_algorithm_instances < 1:
            raise ConfigurationError(
                "max_algorithm_instances",
                self.max_algorithm_instances,
                "A minimum of 1 algorithm instance is required!",
            )
        if self.algorithm_selection_method not in ALGORITHM_SELECTION_METHODS:
            raise ConfigurationError(
                "algorithm_selection_method",
                self.algorithm_selection_method,
                f"Must be one of {', '.join(ALGORITHM_SELECTION_METHODS)}!",
            )
        if self.score_normalization_method not in SCORE_NORMALIZATION_METHODS:
            raise ConfigurationError(
                "score_normalization_method",
                self.score_normalization_method,
                f"Must be one of {', '.join(SCORE_NORMALIZATION_METHODS)}!",
            )
        if self.score_aggregation_method not in SCORE_AGGREGATION_METHODS:
            raise ConfigurationError(
                "score_aggregation_method",
                self.score_aggregation_method,
                f"Must be one of {', '.join(SCORE_AGGREGATION_METHODS)}!",
            )

        if self.TRAINING_TIMESERIES_LENGTH < 500:
            raise ConfigurationError(
                "TRAINING_TIMESERIES_LENGTH",
                self.TRAINING_TIMESERIES_LENGTH,
                "Must be at least 500!",
            )
        if self.TRAINING_TIMESERIES_MIN_NO_PERIODS < 3:
            raise ConfigurationError(
                "TRAINING_TIMESERIES_MIN_NO_PERIODS",
                self.TRAINING_TIMESERIES_MIN_NO_PERIODS,
                "Must be at least 3!",
            )

        if 0 < self.training_timeout_s < 10:
            raise ConfigurationError(
                "training_timeout_s", self.training_timeout_s, "Unreasonable duration!"
            )
        if 0 < self.testing_timeout_s < 10:
            raise ConfigurationError(
                "testing_timeout_s", self.testing_timeout_s, "Unreasonable duration!"
            )
        if 0 < self.memory_limit_mb < 512:
            raise ConfigurationError(
                "memory_limit_mb",
                self.memory_limit_mb,
                "Unreasonable memory limit, most algorithms need more memory than 512 MB!",
            )


class DataGenerationSection(ConfigSection):
    """Configuration options for the training data generation process.

    AutoTSAD analyzes the target datasets and extracts base time series from it. It, then, injects known anomalies into
    the base time series to generate labeled training datasets.

    Attributes
    ----------
    autoperiod_max_periods :
        Maximum number of period sizes to consider.
    AUTOPERIOD_MAX_PERIOD_LENGTH :
        Maximum period length considered valid.
    snippets_max_no :
        Maximum number of snippets to consider for each period size.
    SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE :
        (Sub-)window size used to compute the distance profile (`mpdist_vect`) relative to snippet size.
    SNIPPETS_PROFILE_AREA_CHANGE_THRESHOLD :
        If profile area change between snippet profiles is smaller than this threshold, a single snippet is assumed.
    regime_max_sampling_number :
        Maximum number of subsequences to sample from the original timeseries if there is a single snippet only.
    regime_max_sampling_overlap :
        Maximum allowed overlap of subsequences that are sampled from the original timeseries if there is a single
        snippet only.
    REGIME_STRONG_PERCENTAGE :
        Consider only snippets that are present in at least this percentage of the sampled subsequences.
    REGIME_CONSOLIDATION_PERCENTAGE :
        Consolidate regimes that are separated by other regimes shorter than the snippet size percentage.
    REGIME_MIN_PERIODS_PER_SLICE_FOR_VALID_SNIPPET :
        If the mean number of periods per regime slice is smaller than this number, than the snippet is considered
        invalid and removed. Regiming is re-executed without the previous snippet (and smaller number of best snippets).
    enable_regime_overlap_pruning :
        Do not create a multiple base TS for regimes that have a high overlap in their coverage of the original
        timeseries. Just use the regiming result with the smallest coverage and window size as the representative.
    enable_dataset_pruning :
        Remove datasets that are very similar to each other (w.r.t euclidean distance) and just use one representative.
    REGIME_OVERLAP_PRUNING_THRESHOLD :
        If the overlap (percentage of shared points of the overall dataset) between two regime masks is larger than this
        threshold, just the smaller regime mask is selected as a representative and the other regime mask is removed.
    DATASET_PRUNING_SIMILARITY_THRESHOLD :
        If the similarity between two datasets is larger than this threshold, just one dataset is selected as a
        representative (uses the Euclidean distance between Min-Max-normalized time series).
    anom_filter_scoring_threshold_percentile :
        Anomaly score percentile to calculate the scoring threshold for all algorithms.
    ANOM_FILTER_BAD_SCORING_MEAN_LIMIT :
        Bad result filtering heuristic: If the mean of the anomaly scores is smaller than this limit, the algorithm is
        not considered for anomaly filtering because its scoring does not show a small number of significant anomalies.
    ANOM_FILTER_PERIODIC_SCORING_PERCENTAGE :
        Bad result filtering heuristic: If the anomaly scores indicate more anomalies than the percentage of periods in
        the data, the algorithm is not considered for anomaly filtering because it is likely to detect too many false
        positives.
    anom_filter_voting_threshold :
        Percentage of anomaly detection algorithms that must agree to keep a dataset region so that it is not
        removed as a potential anomaly.
    disable_cleaning :
        Avoid expensive time series analysis and cleaning by taking the target time series without modifications as
        the base time series for anomaly injection. Already existing anomalies might still be present in the synthetic
        training data and influence the hyperparameter optimization and ensembling process. Setting this to ``true``
        is not recommended!
    """

    # dataset analysis
    autoperiod_max_periods: int = 8
    AUTOPERIOD_MAX_PERIOD_LENGTH: int = 600

    # snippet discovery
    snippets_max_no: int = 7
    SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE: float = 1 / 2  # stable
    SNIPPETS_PROFILE_AREA_CHANGE_THRESHOLD: float = 0.5  # I wish it would be stable

    # regime extraction
    regime_max_sampling_number: int = 2
    regime_max_sampling_overlap: float = 0.25
    REGIME_STRONG_PERCENTAGE: float = 0.8  # stable
    REGIME_CONSOLIDATION_PERCENTAGE: float = 0.95  # stable
    REGIME_MIN_PERIODS_PER_SLICE_FOR_VALID_SNIPPET: int = 5  # not stable

    # dataset selecting
    enable_regime_overlap_pruning: bool = False
    enable_dataset_pruning: bool = True
    REGIME_OVERLAP_PRUNING_THRESHOLD: float = 0.1  # stable
    DATASET_PRUNING_SIMILARITY_THRESHOLD: float = 0.0005  # stable

    # anomaly filtering
    anom_filter_scoring_threshold_percentile: int = 90
    ANOM_FILTER_BAD_SCORING_MEAN_LIMIT: float = 1 / 4  # stable
    ANOM_FILTER_PERIODIC_SCORING_PERCENTAGE: float = 0.5  # stable
    anom_filter_voting_threshold: float = 0.7

    disable_cleaning: bool = False

    @validate
    def validate_options(self) -> None:
        if self.autoperiod_max_periods < 1:
            raise ConfigurationError(
                "autoperiod_max_periods",
                self.autoperiod_max_periods,
                "At least 1 period is required!",
            )
        if self.snippets_max_no < 1:
            raise ConfigurationError(
                "snippets_max_no",
                self.snippets_max_no,
                "At least 1 snippet is required!",
            )
        if self.snippets_max_no > 10:
            raise ConfigurationError(
                "snippets_max_no",
                self.snippets_max_no,
                "More than 8 snippets is unreasonable and very computationally expensive!",
            )
        for p in (
            "SNIPPETS_DIST_WINDOW_SIZE_PERCENTAGE",
            "SNIPPETS_PROFILE_AREA_CHANGE_THRESHOLD",
            "regime_max_sampling_overlap",
            "REGIME_STRONG_PERCENTAGE",
            "REGIME_CONSOLIDATION_PERCENTAGE",
            "ANOM_FILTER_BAD_SCORING_MEAN_LIMIT",
            "ANOM_FILTER_PERIODIC_SCORING_PERCENTAGE",
            "anom_filter_voting_threshold",
        ):
            if not (0 <= self[p] <= 1):
                raise ConfigurationError(p, self[p], "Must be between 0 and 1!")
        if not (1 <= self.anom_filter_scoring_threshold_percentile <= 99):
            raise ConfigurationError(
                "anom_filter_scoring_threshold_percentile",
                self.anom_filter_scoring_threshold_percentile,
                "Must be between 1 and 99!",
            )


class AnomalyGenerationSection(ConfigSection):
    """Configuration options for the anomalies injected into the training time series.

    Attributes
    ----------
    contamination_threshold :
        Maximum contamination percentage of the training time series.
    possible_anomaly_lengths :
        List of fixed possible anomaly lengths.
    possible_anomaly_length_period_factors :
        List of possible anomaly lengths as a factor of the period size.
    maximum_anomaly_length_fraction :
        Maximum anomaly length as a fraction of the time series length.
    find_position_max_retries :
        Number of retries to find a valid anomaly position.
    anomaly_section_probas :
        Probabilities of an anomaly to be injected at the (beginning, middle, end) of a time series.
    allowed_anomaly_types :
        List of allowed anomaly types (each anomaly must be defined by AutoTSAD).
    same_anomalies_for_all_base_ts :
        Use the same seed to generate anomalies for all base time series.
    generate_multiple_same :
        Enable the generation of datasets with multiple anomalies of the same type.
    generate_multiple_different :
        Enable the generation of datasets with multiple anomalies of different types.
    number_of_anomalies_per_dataset :
        List of the number of anomalies to inject into each training time series.
    number_of_different_anomalies :
        Number of different anomaly types to use for anomaly injection per training time series. The overall number of
        anomalies is limited by `number_of_anomalies_per_dataset`.
    skip_dataset_less_than_desired_anomalies :
        Skip datasets for which the anomaly injection procedure could not inject the desired number of anomalies, e.g.,
        because of too large anomaly lengths, too many cut points, or margin violations.
    skip_dataset_over_contamination_threshold :
        Skip datasets for which the contamination threshold would be exceeded by injecting the desired anomalies.
    """

    contamination_threshold: float = 0.15
    possible_anomaly_lengths: Tuple[int, ...] = (50, 100)
    possible_anomaly_length_period_factors: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    maximum_anomaly_length_fraction: float = 0.1
    find_position_max_retries: int = 100
    anomaly_section_probas: Tuple[float, ...] = (0.3, 0.4, 0.3)
    allowed_anomaly_types: Tuple[str, ...] = tuple(ANOMALY_TYPES)

    same_anomalies_for_all_base_ts: bool = True
    generate_multiple_same: bool = False
    generate_multiple_different: bool = False
    number_of_anomalies_per_dataset: Tuple[int, ...] = (2, 3)
    number_of_different_anomalies: Tuple[int, ...] = (2, 3)
    skip_dataset_less_than_desired_anomalies: bool = True
    skip_dataset_over_contamination_threshold: bool = True

    def anomaly_lengths(self, period_size: int, data_length: int) -> List[int]:
        lengths = np.unique(
            list(self.possible_anomaly_lengths)
            + [
                int(float(l) * period_size)
                for l in self.possible_anomaly_length_period_factors
            ]
        )
        return [
            l
            for l in lengths
            if l <= self.maximum_anomaly_length_fraction * data_length
        ]

    @validate
    def validate_options(self) -> None:
        if self.contamination_threshold > 0.2:
            raise ConfigurationError(
                "contamination_threshold",
                self.contamination_threshold,
                "Contamination above 20% does not indicate an anomaly detection use case!",
            )
        if (
            len(self.possible_anomaly_lengths) == 0
            and len(self.possible_anomaly_length_period_factors) == 0
        ):
            raise ConfigurationError(
                "possible_anomaly_lengths",
                self.possible_anomaly_lengths,
                "Must have at least one possible anomaly length!",
            )
        if any(l <= 0 for l in self.possible_anomaly_lengths) or any(
            l <= 0 for l in self.possible_anomaly_length_period_factors
        ):
            raise ConfigurationError(
                "possible_anomaly_lengths",
                self.possible_anomaly_lengths,
                "Anomaly lengths must be >0!",
            )
        if (
            self.maximum_anomaly_length_fraction <= 0
            or self.maximum_anomaly_length_fraction > self.contamination_threshold
        ):
            raise ConfigurationError(
                "maximum_anomaly_length_fraction",
                self.maximum_anomaly_length_fraction,
                f"Must be between 0 and contamination_threshold ({self.contamination_threshold})!",
            )
        if len(self.anomaly_section_probas) != 3:
            raise ConfigurationError(
                "anomaly_section_probas",
                self.anomaly_section_probas,
                "Must have 3 values for beginning, middle, and end positions!",
            )
        if not np.isclose(sum(self.anomaly_section_probas), 1.0) or any(
            p < 0 for p in self.anomaly_section_probas
        ):
            raise ConfigurationError(
                "anomaly_section_probas",
                self.anomaly_section_probas,
                "Must be >0 and sum up to 1!",
            )
        if any(anom not in ANOMALY_TYPES for anom in self.allowed_anomaly_types):
            raise ConfigurationError(
                "allowed_anomaly_types",
                self.allowed_anomaly_types,
                f"Must be in {ANOMALY_TYPES}!",
            )


class DataGenerationPlottingSection(ConfigSection):
    """Plotting configuration for various parts of the data generation process.

    Attributes
    ----------
    autoperiod :
        Plot the periodogram and ACF plots of Autoperiod.
    profile_area :
        Plot the profile areas, the profile area change, and the automatically selected threshold of the extracted
        snippets.
    subsequence_sampling :
        Plot the extracted subsequence samples if there is only a single regime/snippet.
    snippets :
        Plot the extracted snippets.
    snippet_regimes :
        Plot the extracted snippet regimes.
    snippet_profiles :
        Plot the extracted snippet profiles.
    regime_overlap_pruning :
        Plot the overlap between two regime masks if they are pruned because of their overlap.
    base_ts :
        Plot base time series extracted from the target time series.
    cleaning_algo_scores :
        Plot the scores of the algorithms used for base time series cleaning, and the corresponding cleaning decisions.
    cutouts :
        Plot the regions cut out during the cleaning process.
    truncated_timeseries :
        Plot the subsequence of the time series that is used as training time series after limiting the length.
    injected_anomaly :
        Plot the injected anomaly for every generated training time series.
    """

    autoperiod: bool = False
    profile_area: bool = False
    subsequence_sampling: bool = False
    snippets: bool = False
    snippet_regimes: bool = False
    snippet_profiles: bool = False
    regime_overlap_pruning: bool = False
    base_ts: bool = False

    cleaning_algo_scores: bool = False
    cutouts: bool = False

    truncated_timeseries: bool = False

    injected_anomaly: bool = False

    def __bool__(self) -> bool:
        return any(self[a] for a in self)


class OptimizationSection(ConfigSection):
    """Configuration options for the algorithm hyperparameter optimization and selection process.

    The optimization process consists of roughly three steps:

    1. Sensitivity analysis: Optimize the hyperparameters of the algorithms for datasets covering their sensitive
       dimensions (e.g. anomaly length or anomaly type) for a small number of trials. Use the results of this step as
       starting points for the next step.
    2. Stepwise optimization: Optimize the hyperparameters of the algorithms for all datasets x trials at a time.
       Inbetween, the worst performing algorithms are removed from the optimization process, and for datasets which
       share the same characteristics and similar hyperparameters, a single representative is selected.
    3. Algorithm instance selection: Select the best performing algorithm instance for each dataset.

    Attributes
    ----------
    optuna_storage_type :
        Type of optuna storage to use. Can be 'sqlite', 'postgres', or 'journal'.
    optuna_storage_cleanup :
        Delete the Optuna storage backend after the optimization process has finished.
    optuna_dashboard :
        Start an Optuna dashboard for the optimization process. It can be reached at http://<host>:8080.
    optuna_logging_level :
        Logging level for Optuna.
    n_trials_startup :
        Number of trials for the startup phase that are sampled independently and randomly.
    n_trials_sensitivity :
        Number of trials for the sensitivity analysis.
    n_trials_step :
        Number of trials for each step in the stepwise optimization. The first iteration uses double the number of
        trials.
    max_trails_per_study :
        Maximum number of trials per study. This is the upper limit for any study. As soon as a study reaches this
        limit, it is removed from the optimization process, and considered finished.
    metric_name :
        Name of the metric to optimize. Must be one of the metrics defined in AutoTSAD.
    stop_heuristic_quality_threshold :
        Stop the optimization process for an algorithm study if the quality is above this threshold for at least
        `stop_heuristic_n` trials.
    stop_heuristic_n :
        See `stop_heuristic_quality_threshold`. If `stop_heuristic_n` is 0, the heuristic is disabled.
    stop_heuristic_optuna :
        Use the Optuna Terminator as an additional stop heuristics for studies. A study is terminated when the
        statistical error, e.g. cross-validation error, exceeds the room left for optimization. This feature requires
        botorch, pytorch, and many other dependencies to be installed (install via ``pip install botorch``).
    proxy_allowed_quality_deviation :
        Allowed drop in quality (according to quality metric) for a parameter setting that was optimized on another
        dataset (proxy dataset) to still overwrite the best-so-far result of the current dataset.
    algorithms :
        List of algorithms to optimize. Must be a subset of the algorithms defined in AutoTSAD.
    reintroduce_default_params :
        Reintroduce the default parameters for each algorithm after the hyperparameter optimization for the following
        method selection process.
    disabled :
        Disable the optimization process and instead use the default parameters for each algorithm. This still uses
        Optuna to compute the results on the training data. This is required to determine the proxy metrics for the
        algorithm ranking process.
    init_parameter_set :
        Initial parameter set to start the optimization process. If the optimization is disabled via ``disabled=True``,
        this is the only hyperparameter configuration used for the algorithm ranking process. Must be one of 'all',
        'default', 'bad-default', or 'timeeval'. If ``disabled=False``, must be 'all'.
    """

    optuna_storage_type: str = "postgres"
    optuna_storage_cleanup: bool = False
    optuna_dashboard: bool = False
    optuna_logging_level: int = logging.ERROR

    n_trials_startup: int = 100
    n_trials_sensitivity: int = 200
    n_trials_step: int = 150
    max_trails_per_study: int = 1000

    metric_name: str = "RangePrAUC"
    stop_heuristic_quality_threshold: float = 0.95
    stop_heuristic_n: int = 10
    stop_heuristic_optuna: bool = False

    proxy_allowed_quality_deviation: float = 0.01

    algorithms: FrozenSet[str] = frozenset(
        {
            "subsequence_lof",
            "subsequence_knn",
            "subsequence_if",
            "stomp",
            "kmeans",
            "grammarviz",
            "dwt_mlead",
            "torsk",
        }
    )

    reintroduce_default_params: bool = True

    disabled: bool = False
    init_parameter_set: str = "all"

    def metric(self) -> Metric:
        return METRIC_MAPPING[self.metric_name]

    def use_stop_heuristic(self) -> bool:
        return self.stop_heuristic_n > 0

    def use_optuna_terminator(self) -> bool:
        return self.stop_heuristic_optuna

    def enabled(self) -> bool:
        return not self.disabled

    @validate
    def validate_options(self) -> None:
        if self.optuna_storage_type not in {"sqlite", "postgres", "journal"}:
            raise ConfigurationError(
                "optuna_storage_type",
                self.optuna_storage_type,
                "Must be one of 'sqlite', 'postgres', 'journal'!",
            )
        if self.max_trails_per_study < 1:
            raise ConfigurationError(
                "max_trails_per_study", self.max_trails_per_study, "Must be at least 1!"
            )
        if self.n_trials_sensitivity > self.max_trails_per_study:
            raise ConfigurationError(
                "n_trials_sensitivity",
                self.n_trials_sensitivity,
                "Must be less than or equal to max_trails_per_study!",
            )
        if self.n_trials_step > self.max_trails_per_study:
            raise ConfigurationError(
                "n_trials_step",
                self.n_trials_step,
                "Must be less than or equal to max_trails_per_study!",
            )
        if self.metric_name not in METRIC_MAPPING:
            raise ConfigurationError(
                "metric_name",
                self.metric_name,
                f"Must be one of {METRIC_MAPPING.keys()}!",
            )
        if not (0 < self.stop_heuristic_quality_threshold < 1):
            raise ConfigurationError(
                "stop_heuristic_quality_threshold",
                self.stop_heuristic_quality_threshold,
                "Must be in (0, 1)!",
            )
        if self.stop_heuristic_n > self.max_trails_per_study:
            raise ConfigurationError(
                "stop_heuristic_n",
                self.stop_heuristic_n,
                "Must be less than or equal to max_trails_per_study!",
            )
        if self.proxy_allowed_quality_deviation < 0:
            raise ConfigurationError(
                "proxy_allowed_quality_deviation",
                self.proxy_allowed_quality_deviation,
                "Must be greater than or equal to 0.",
            )
        if not self.algorithms.issubset(ALGORITHMS):
            raise ConfigurationError(
                "algorithms", self.algorithms, f"Must be a subset of {ALGORITHMS}!"
            )
        if self.init_parameter_set not in {"all", "default", "bad-default", "timeeval"}:
            raise ConfigurationError(
                "init_parameter_set",
                self.init_parameter_set,
                "Must be one of 'all', 'default', 'bad-default', 'timeeval'!",
            )
        if not self.disabled and self.init_parameter_set != "all":
            raise ConfigurationError(
                "init_parameter_set",
                self.init_parameter_set,
                "Must be 'all' if optimization is enabled!",
            )


class DatasetConsolidationSection(ConfigSection):
    """Configuration options for the dataset consolidation process (within the optimization process).

    The dataset consolidation process is used to reduce the number of trials for each algorithm. For datasets which
    share the same characteristics and similar hyperparameters, a single representative is selected.

    Attributes
    ----------
    param_selection_strategy :
        Strategy to use for the dataset consolidation process. Must be one of "best", or "threshold".
    param_selection_best_quality_epsilon :
        Epsilon value for the "best" consolidation strategy. Considers all hyperparameter settings of an algorithm for
        a selected dataset that are within this epsilon distance of the best performing setting as equivalent.
    param_selection_quality_threshold :
        Quality threshold for the "threshold" consolidation strategy. Considers all hyperparameter settings of an
        algorithm for a selected dataset that are above this threshold as equivalent.
    dataset_similarity_threshold :
        Threshold for the dataset similarity. Two datasets are considered similar if at least this fraction of their
        characteristics match.
    dataset_selection_strategy :
        Strategy to use for the dataset selection process. Must be one of "best", "fastest", "worst".
    plot :
        Plot the dataset consolidation graph for each algorithm.
    """

    param_selection_strategy: str = "threshold"
    param_selection_best_quality_epsilon: float = 0.001
    param_selection_quality_threshold: float = 0.95
    dataset_similarity_threshold: float = 2 / 3
    dataset_selection_strategy: str = "best"
    plot: bool = False

    @validate
    def validate_options(self) -> None:
        if self.param_selection_strategy not in ("best", "threshold"):
            raise ConfigurationError(
                "param_selection_strategy",
                self.param_selection_strategy,
                "Must be one of 'best', or 'threshold'!",
            )
        if self.param_selection_strategy == "best" and not (
            0 < self.param_selection_best_quality_epsilon < 1
        ):
            raise ConfigurationError(
                "param_selection_best_quality_epsilon",
                self.param_selection_best_quality_epsilon,
                "Must be in (0, 1)!",
            )
        if self.param_selection_strategy == "threshold" and not (
            0 < self.param_selection_quality_threshold < 1
        ):
            raise ConfigurationError(
                "param_selection_quality_threshold",
                self.param_selection_quality_threshold,
                "Must be in (0, 1)!",
            )
        if not (0 < self.dataset_similarity_threshold < 1):
            raise ConfigurationError(
                "dataset_similarity_threshold",
                self.dataset_similarity_threshold,
                "Must be in (0, 1)!",
            )
        if self.dataset_selection_strategy not in ("best", "fastest", "worst"):
            raise ConfigurationError(
                "dataset_selection_strategy",
                self.dataset_selection_strategy,
                "Must be one of 'best', 'fastest', 'worst'!",
            )


class AutoTSADConfig(Config):
    """Configuration of the AutoTSAD system."""

    general: GeneralSection
    data_gen: DataGenerationSection
    data_gen_plotting: DataGenerationPlottingSection
    optimization: OptimizationSection
    consolidation: DatasetConsolidationSection
    anomaly: AnomalyGenerationSection

    def to_json(self) -> str:
        """Convert the current configuration options of AutoTSAD to a JSON string."""

        def _get_annotations(obj: Any) -> List[str]:
            return list(getattr(obj, "__annotations__", {}).keys())

        def _sanitize_entry(section: Any, key: str) -> Any:
            value = getattr(section, key)
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, frozenset):
                return sorted(list(value))
            elif isinstance(value, tuple):
                return list(value)
            return value

        dd = {}
        for section_name in _get_annotations(self):
            section = getattr(self, section_name)
            dd[section_name] = dict(
                [(e, _sanitize_entry(section, e)) for e in _get_annotations(section)]
            )

        return json.dumps(dd)


config: AutoTSADConfig = AutoTSADConfig()
