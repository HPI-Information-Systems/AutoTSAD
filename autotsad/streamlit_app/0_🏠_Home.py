import streamlit as st

import _lib.preamble  # noqa: F401

st.set_page_config(
    page_title="AutoTSAD Results",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.write("# AutoTSAD Result Explorer")
st.write("Use the navigation bar on the left to explore the results of the AutoTSAD experiments.")
