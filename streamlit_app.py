"""
ML Playground - interactive demos for the ACT Intro to Machine Learning lecture.

Run locally with:
    streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="ML Playground", page_icon="📉", layout="wide")

pages = {
    "Start": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "How a machine learns": [
        st.Page("app_pages/rolling.py", title="1 · Rolling downhill",
                icon=":material/landscape:"),
        st.Page("app_pages/line_fit.py", title="2 · Fit a line",
                icon=":material/show_chart:"),
        st.Page("app_pages/optimisers.py", title="3 · Optimiser race",
                icon=":material/sports_score:"),
    ],
    "Did it actually learn?": [
        st.Page("app_pages/overfitting.py", title="4 · Under- and overfitting",
                icon=":material/ssid_chart:"),
        st.Page("app_pages/classify.py", title="5 · Decision boundaries",
                icon=":material/scatter_plot:"),
    ],
}

st.navigation(pages, expanded=True).run()
