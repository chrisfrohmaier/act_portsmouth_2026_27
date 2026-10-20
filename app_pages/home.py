"""Landing page."""

import streamlit as st

from mlcore.ui import header

header("ACT · Intro to machine learning", "ML Playground",
       "Five small experiments that show how a machine learns from data, and how to tell "
       "whether it really has. Change a setting, watch what happens, then press ▶ Play.")

st.subheader("How a machine learns")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    st.page_link("app_pages/rolling.py", label="**1 · Rolling downhill**", icon="⛰️")
    st.markdown("Gradient descent on a curve. What the learning rate does, and why "
                "local minima trap you.")
with c2.container(border=True):
    st.page_link("app_pages/line_fit.py", label="**2 · Fit a line**", icon="📈")
    st.markdown("The lecture example: diabetes progression against BMI. The loss "
                "landscape, feature scaling and mini-batches.")
with c3.container(border=True):
    st.page_link("app_pages/optimisers.py", label="**3 · Optimiser race**", icon="🏁")
    st.markdown("Gradient descent, momentum and Adam on surfaces with narrow valleys "
                "and several minima.")

st.subheader("Did it actually learn?")
c4, c5, _ = st.columns(3)
with c4.container(border=True):
    st.page_link("app_pages/overfitting.py", label="**4 · Under- and overfitting**", icon="〰️")
    st.markdown("Polynomials of rising degree on a supernova-like light curve. "
                "Training error against test error.")
with c5.container(border=True):
    st.page_link("app_pages/classify.py", label="**5 · Decision boundaries**", icon="🌗")
    st.markdown("Logistic regression, a neural network, k-nearest neighbours and k-means "
                "on the two-moons data.")

st.divider()
st.markdown(
    "**Using this after the lecture.** All the gradient-descent code lives in "
    "`mlcore/gd.py` as plain NumPy, written to be read and adapted - it is the starting "
    "point for Exercise 1 in *Scikit-learn_basics.ipynb*. Every page also has a "
    "*Show the code* panel. To run the app on your own machine:")
st.code("pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")
