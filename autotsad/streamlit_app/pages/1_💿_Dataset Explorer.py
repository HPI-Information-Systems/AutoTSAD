import _lib.preamble  # noqa: F401
import pandas as pd
import streamlit as st
from _lib.plotly_plotting import plot_dataset  # noqa: F401

from autotsad.database.autotsad_connection import AutoTSADConnection
from autotsad.util import mask_to_slices

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="💿",
    layout="wide",
)

conn = st.experimental_connection("autotsad", type=AutoTSADConnection)

st.write("# Dataset Explorer")
datasets = conn.list_datasets().sort_values(["collection", "name"])
datasets = pd.concat([pd.DataFrame([["", ""]], columns=["name", "collection"]), datasets], axis=0).set_index("name")
dataset = st.selectbox("Select Dataset", datasets.index,
                       format_func=lambda x: f"{datasets.loc[x, 'collection']} {x}")

if dataset != "":
    st.write(f"## Dataset {dataset}")
    df = conn.load_dataset(dataset)
    n_anomalies = mask_to_slices(df["is_anomaly"]).shape[0]
    st.table({
        "Property": ["Length", "Dimensions", "Anomalies", "Contamination"],
        "Value": [df.shape[0], df.shape[1] - 2, n_anomalies, f"{df['is_anomaly'].sum() / df.shape[0]:06.2%}"]
    })
    # st.write(f"- Length: {df.shape[0]}")
    # st.write(f"- Dimensions: {df.shape[1] - 2}")
    # st.write(f"- Anomalies: {n_anomalies}")
    # st.write(f"- Contamination: {df['is_anomaly'].sum() / df.shape[0]:06.2%}")

    st.plotly_chart(plot_dataset(dataset, df), use_container_width=True)
