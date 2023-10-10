from typing import Dict, List, Any

import numpy as np
import streamlit as st

from autotsad.config import (
    ALGORITHM_SELECTION_METHODS as AUTOTSAD_ALGORITHM_SELECTION_METHODS,
    SCORE_NORMALIZATION_METHODS as AUTOTSAD_SCORE_NORMALIZATION_METHODS,
    SCORE_AGGREGATION_METHODS as AUTOTSAD_SCORE_AGGREGATION_METHODS
)
from autotsad.database.autotsad_connection import AutoTSADConnection

ALGORITHM_SELECTION_METHODS = sorted(
    [
        m for m in AUTOTSAD_ALGORITHM_SELECTION_METHODS
        if m not in ["interchange-annotation-overlap", "interchange-euclidean", "training-coverage"]
    ] + ["aggregated-minimum-influence", "aggregated-robust-borda"]
)
SCORE_NORMALIZATION_METHODS = AUTOTSAD_SCORE_NORMALIZATION_METHODS
SCORE_AGGREGATION_METHODS = [m for m in AUTOTSAD_SCORE_AGGREGATION_METHODS if m != "mean"]


def create_autotsad_filters(key: str, conn: AutoTSADConnection, use_columns: bool = True) -> Dict[str, List[Any]]:
    def clear_all() -> None:
        st.session_state[f"{key}-filter_form_experiment_ids"] = []
        st.session_state[f"{key}-filter_form_autotsad_versions"] = []
        st.session_state[f"{key}-filter_form_autotsad_configurations"] = []
        st.session_state[f"{key}-filter_form_ranking_methods"] = []
        st.session_state[f"{key}-filter_form_normalization_methods"] = []
        st.session_state[f"{key}-filter_form_aggregation_methods"] = []

    with st.form(key=f"{key}-filter_form"):
        with st.expander("Filter AutoTSAD versions"):
            experiments = conn.list_experiments()
            experiment_ids = st.multiselect("Experiment",
                                            options=experiments["name"],
                                            key=f"{key}-filter_form_experiment_ids",
                                            help="Select the experiments to include in the results. If empty, all "
                                                 "experiments are included.")
            experiment_ids = [experiments.loc[experiments["name"] == e, "id"].item() for e in experiment_ids]

            autotsad_versions = st.multiselect("Version",
                                               options=conn.list_autotsad_versions(),
                                               key=f"{key}-filter_form_autotsad_versions",
                                               help="Select the AutoTSAD version to include in the results. If empty, "
                                                    "all versions are included.")

            configs = conn.list_autotsad_configs()
            if np.any(configs["description"].isna()):
                st.warning("Some configurations have no description, showing their ID instead.")
                na_mask = configs["description"].isna()
                configs.loc[na_mask, "description"] = configs.loc[na_mask, "id"]
            autotsad_configurations = st.multiselect("Configuration",
                                                     options=configs["description"],
                                                     default=None,
                                                     key=f"{key}-filter_form_autotsad_configurations",
                                                     help="Select the AutoTSAD configurations to include in the "
                                                          "results. If empty, all configurations are included.")
            autotsad_configurations = [
                configs.loc[configs["description"] == c, "id"].item() for c in autotsad_configurations
            ]

            if use_columns:
                columns = st.columns(3)
            else:
                columns = [st, st, st]
            ranking_methods = columns[0].multiselect(
                "Ranking methods",
                options=ALGORITHM_SELECTION_METHODS,
                default=None,
                key=f"{key}-filter_form_ranking_methods",
                help="Select the ranking methods to include in the results. If empty, all ranking methods are "
                     "included.")
            normalization_methods = columns[1].multiselect(
                "Score normalization methods",
                options=SCORE_NORMALIZATION_METHODS,
                default=None,
                key=f"{key}-filter_form_normalization_methods",
                help="Select the score normalization methods to include in the results. If empty, all score "
                     "normalization methods are included.")
            aggregation_methods = columns[2].multiselect(
                "Score aggregation methods",
                options=SCORE_AGGREGATION_METHODS,
                default=None,
                key=f"{key}-filter_form_aggregation_methods",
                help="Select the score aggregation methods to include in the results. If empty, all score aggregation "
                     "methods are included.")

        c0, c1 = st.columns(2)
        c0.form_submit_button("Apply", type="primary", use_container_width=True)
        c1.form_submit_button("Clear", type="secondary", on_click=clear_all, use_container_width=True)

    return {
        "experiment_id": experiment_ids,
        "autotsad_version": autotsad_versions,
        "config_id": autotsad_configurations,
        "ranking_method": ranking_methods,
        "normalization_method": normalization_methods,
        "aggregation_method": aggregation_methods
    }
