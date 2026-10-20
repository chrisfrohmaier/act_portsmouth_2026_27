"""Demo 5: decision boundaries - linear models, neural networks, k-NN and k-means."""

import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlcore import style as S
from mlcore.ui import header, preset_row, status

DATASETS = ["Two moons", "Circles", "Two blobs"]
MODELS = ["Logistic regression", "Neural network", "k-nearest neighbours",
          "k-means (unsupervised)"]
ACTIVATIONS = {"ReLU": "relu", "tanh": "tanh", "Sigmoid": "logistic",
               "None (identity)": "identity"}
GRID_N = 60

DEFAULTS = dict(c_data=DATASETS[0], c_noise=0.2, c_n=300, c_model=MODELS[1], c_poly=False,
                c_layers=2, c_width=16, c_act="ReLU", c_epochs=200, c_lr=0.01, c_k=15,
                c_clusters=2, c_seed=0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "A straight line": dict(c_data="Two moons", c_model="Logistic regression", c_poly=False),
    "Extra features": dict(c_data="Two moons", c_model="Logistic regression",
                                       c_poly=True),
    "Neural network": dict(c_data="Two moons", c_model="Neural network", c_act="ReLU",
                           c_layers=2, c_width=16, c_epochs=200),
    "No activation": dict(c_data="Two moons", c_model="Neural network",
                                       c_act="None (identity)", c_layers=2, c_width=16,
                                       c_epochs=200),
    "k-NN with k = 1": dict(c_data="Two moons", c_model="k-nearest neighbours", c_k=1,
                            c_noise=0.3),
    "k-means": dict(c_data="Two moons", c_model="k-means (unsupervised)", c_clusters=2),
}


def apply(settings):
    for k, v in settings.items():
        st.session_state[k] = v


@st.cache_data
def make_data(name, n, noise, seed):
    if name == "Two moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    elif name == "Circles":
        X, y = make_circles(n_samples=n, noise=noise * 0.5, factor=0.45, random_state=seed)
        X = X * 1.6
    else:
        X, y = make_blobs(n_samples=n, centers=[(-1, -0.5), (1.5, 1.0)],
                          cluster_std=0.4 + 2 * noise, random_state=seed)
    return X, y


@st.cache_data
def train_network(Xtr, ytr, Xte, yte, layers, width, activation, epochs, lr, seed, grid):
    """Train one epoch at a time so we can watch the boundary form."""
    clf = MLPClassifier(hidden_layer_sizes=(width,) * layers, activation=activation,
                        solver="adam", learning_rate_init=lr, random_state=seed)
    snap_at = set(S.frame_indices(epochs + 1, 40))
    hist = dict(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    snaps = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            clf.partial_fit(Xtr, ytr, classes=[0, 1])
            ptr, pte = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
            hist["train_loss"].append(log_loss(ytr, ptr, labels=[0, 1]))
            hist["test_loss"].append(log_loss(yte, pte, labels=[0, 1]))
            hist["train_acc"].append(accuracy_score(ytr, ptr > 0.5))
            hist["test_acc"].append(accuracy_score(yte, pte > 0.5))
            if epoch in snap_at or epoch == epochs:
                snaps[epoch] = clf.predict_proba(grid)[:, 1].reshape(GRID_N, GRID_N)
    return {k: np.array(v) for k, v in hist.items()}, snaps, clf.predict(Xte)


header("Demo 5 · Did it actually learn?", "Decision boundaries",
       "A classifier splits the plane into regions. Compare a straight line, a neural "
       "network and a nearest-neighbour vote - and a clustering method that never sees "
       "the labels.")

with st.sidebar:
    st.subheader("Data")
    st.selectbox("Data set", DATASETS, key="c_data")
    st.slider("Noise", 0.0, 0.5, key="c_noise", step=0.05)
    st.slider("Number of points", 100, 1000, key="c_n", step=50)
    st.button("Draw new data", width="stretch",
              on_click=lambda: st.session_state.update(c_seed=st.session_state.c_seed + 1))
    st.subheader("Model")
    st.selectbox("Model", MODELS, key="c_model")
    model_name = st.session_state.c_model
    if model_name == "Logistic regression":
        st.toggle("Add polynomial features (degree 3)", key="c_poly")
    elif model_name == "Neural network":
        st.slider("Hidden layers", 1, 3, key="c_layers")
        st.select_slider("Neurons per layer", [2, 4, 8, 16, 32, 64], key="c_width")
        st.selectbox("Activation function", list(ACTIVATIONS), key="c_act")
        st.select_slider("Epochs", [10, 25, 50, 100, 200, 400, 800], key="c_epochs")
        st.select_slider("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1], key="c_lr")
    elif model_name == "k-nearest neighbours":
        st.slider("k (neighbours that vote)", 1, 50, key="c_k")
    else:
        st.slider("Number of clusters", 2, 3, key="c_clusters")

st.caption("Try a preset. For the neural network, press ▶ Play to watch it learn.")
preset_row(PRESETS, apply, "c")

X, y = make_data(st.session_state.c_data, st.session_state.c_n, st.session_state.c_noise,
                 st.session_state.c_seed)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=st.session_state.c_seed,
                                      stratify=y)
pad = 0.5
gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, GRID_N)
gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, GRID_N)
GX, GY = np.meshgrid(gx, gy)
grid = np.column_stack([GX.ravel(), GY.ravel()])


def points(fig, row=1, col=1, colours=None):
    """Training points as filled circles, test points as diamonds."""
    for cls, colour in zip([0, 1], [S.BLUE, S.ORANGE]):
        fig.add_trace(go.Scatter(x=Xtr[ytr == cls, 0], y=Xtr[ytr == cls, 1], mode="markers",
                                 name=f"Class {cls} (train)",
                                 marker=dict(size=9, color=colour,
                                             line=dict(color=S.SURFACE, width=1.5)),
                                 hovertemplate=f"class {cls}, train<extra></extra>"), row, col)
        fig.add_trace(go.Scatter(x=Xte[yte == cls, 0], y=Xte[yte == cls, 1], mode="markers",
                                 name=f"Class {cls} (test)",
                                 marker=dict(size=10, color=colour, symbol="diamond-open",
                                             line=dict(width=2.5)),
                                 hovertemplate=f"class {cls}, test<extra></extra>"), row, col)


def boundary_traces(P):
    return [
        go.Heatmap(x=gx, y=gy, z=P, colorscale=S.DIVERGING, zmin=0, zmax=1,
                   showscale=False, hovertemplate="P(class 1) = %{z:.2f}<extra></extra>"),
        go.Contour(x=gx, y=gy, z=P, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                   line=dict(color=S.INK, width=3), showscale=False, hoverinfo="skip",
                   name="Boundary"),
    ]


def axes(fig, row=1, col=1):
    fig.update_xaxes(range=[gx[0], gx[-1]], title_text="feature 1", row=row, col=col,
                     constrain="domain")
    fig.update_yaxes(range=[gy[0], gy[-1]], title_text="feature 2", row=row, col=col,
                     scaleanchor="x", scaleratio=1, constrain="domain")


def report(y_pred):
    """Accuracy, confusion matrix and precision/recall on the test set."""
    acc = accuracy_score(yte, y_pred)
    cm = confusion_matrix(yte, y_pred, labels=[0, 1])
    c1, c2, c3 = st.columns([0.24, 0.24, 0.52])
    c1.metric("Test accuracy", f"{100 * acc:.1f}%")
    c1.metric("Precision (class 1)", f"{precision_score(yte, y_pred, zero_division=0):.2f}",
              help="Of the points predicted class 1, the fraction that really are")
    c2.metric("Test points", f"{len(yte)}")
    c2.metric("Recall (class 1)", f"{recall_score(yte, y_pred, zero_division=0):.2f}",
              help="Of the points that really are class 1, the fraction we found")
    cmf = go.Figure(go.Heatmap(
        z=cm, x=["Predicted 0", "Predicted 1"], y=["True 0", "True 1"],
        colorscale=S.SEQUENTIAL, showscale=False, text=cm, texttemplate="%{text}",
        textfont=dict(size=20), hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    cmf.update_yaxes(autorange="reversed")
    cmf.update_layout(title=dict(text="Confusion matrix (test set)", x=0, font=dict(size=16)))
    S.style(cmf, height=260, legend=False)
    cmf.update_layout(margin=dict(l=90, r=10, t=50, b=40))
    with c3:
        S.show(st, cmf, key="cm")


# ---------------------------------------------------------------------------
if model_name == "Neural network":
    hist, snaps, y_pred = train_network(
        Xtr, ytr, Xte, yte, st.session_state.c_layers, st.session_state.c_width,
        ACTIVATIONS[st.session_state.c_act], st.session_state.c_epochs, st.session_state.c_lr,
        st.session_state.c_seed, grid)
    epochs = np.arange(1, st.session_state.c_epochs + 1)
    fin_tr, fin_te = hist["train_acc"][-1], hist["test_acc"][-1]
    if st.session_state.c_act == "None (identity)":
        status("No activation function: every layer is linear, so the whole network is "
               "still a straight line", "warn")
    elif fin_te >= 0.95:
        status(f"Learned it: {100 * fin_te:.0f}% of test points correct", "good")
    else:
        status(f"{100 * fin_te:.0f}% of test points correct - try more epochs or neurons", "info")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.1,
                        subplot_titles=("The decision boundary", "Loss after each epoch"))
    last_epoch = max(snaps)
    for tr in boundary_traces(snaps[last_epoch]):
        fig.add_trace(tr, 1, 1)
    points(fig)

    def curves(e):
        return [
            go.Scatter(x=epochs[:e], y=hist["train_loss"][:e], mode="lines",
                       name="Training loss", line=dict(color=S.BLUE, width=2.5),
                       hovertemplate="epoch %{x}<br>loss %{y:.3f}<extra>train</extra>"),
            go.Scatter(x=epochs[:e], y=hist["test_loss"][:e], mode="lines",
                       name="Test loss", line=dict(color=S.ORANGE, width=2.5, dash="dot"),
                       hovertemplate="epoch %{x}<br>loss %{y:.3f}<extra>test</extra>"),
        ]

    for tr in curves(last_epoch):
        fig.add_trace(tr, 1, 2)
    frame_epochs = sorted(snaps)
    fig.frames = [go.Frame(name=str(e), data=boundary_traces(snaps[e]) + curves(e),
                           traces=[0, 1, 6, 7]) for e in frame_epochs]
    axes(fig)
    fig.update_xaxes(title_text="Epoch", range=[0, epochs[-1] + 1], row=1, col=2)
    hi = max(hist["train_loss"].max(), hist["test_loss"].max())
    fig.update_yaxes(title_text="Log loss (cross-entropy)", range=[0, 1.05 * hi], row=1, col=2)
    S.style(fig, height=700)
    S.add_player(fig, [str(e) for e in frame_epochs], duration=150, redraw=True,
                 prefix="Epoch ")
    S.show(st, fig, key="nn_fig")
    report(y_pred)
    st.markdown(
        "**Watch both loss curves.** The network learns by gradient descent (Adam) on the "
        "cross-entropy loss. If the test loss starts rising while the training loss keeps "
        "falling, the network is overfitting - stopping at the lowest test (really, "
        "validation) loss is called *early stopping*.")
    n_weights = (2 * st.session_state.c_width + st.session_state.c_width
                 + (st.session_state.c_layers - 1) * (st.session_state.c_width ** 2
                                                       + st.session_state.c_width)
                 + st.session_state.c_width + 1)
    st.caption(f"This network has {n_weights:,} weights and biases.")
    with st.expander("The scikit-learn code"):
        st.code(
            "from sklearn.neural_network import MLPClassifier\n\n"
            f"clf = MLPClassifier(hidden_layer_sizes=({st.session_state.c_width},)*"
            f"{st.session_state.c_layers},\n"
            f"                    activation='{ACTIVATIONS[st.session_state.c_act]}', "
            f"learning_rate_init={st.session_state.c_lr})\n"
            "clf.fit(X_train, y_train)\n"
            "clf.score(X_test, y_test)    # accuracy on unseen points", language="python")

elif model_name == "k-means (unsupervised)":
    k = st.session_state.c_clusters
    km = KMeans(n_clusters=k, n_init=10, random_state=st.session_state.c_seed).fit(Xtr)
    regions = km.predict(grid).reshape(GRID_N, GRID_N)
    status("k-means never sees the labels: it groups points by distance to a centre", "info")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("What k-means found (no labels used)",
                                        "The true classes"))
    fills = ["rgba(42,111,201,0.16)", "rgba(201,85,32,0.16)", "rgba(23,138,98,0.16)"]
    fig.add_trace(go.Heatmap(x=gx, y=gy, z=regions,
                             colorscale=[[i / max(k - 1, 1), fills[i]] for i in range(k)]
                             if k > 1 else [[0, fills[0]], [1, fills[0]]],
                             showscale=False, hoverinfo="skip"), 1, 1)
    lab = km.labels_
    for c in range(k):
        fig.add_trace(go.Scatter(x=Xtr[lab == c, 0], y=Xtr[lab == c, 1], mode="markers",
                                 name=f"Cluster {c + 1}",
                                 marker=dict(size=9, color=S.SERIES[c], symbol="square",
                                             line=dict(color=S.SURFACE, width=1.5))), 1, 1)
    fig.add_trace(go.Scatter(x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1],
                             mode="markers", name="Cluster centres",
                             marker=dict(symbol="x", size=18, color=S.INK,
                                         line=dict(width=2))), 1, 1)
    points(fig, 1, 2)
    axes(fig, 1, 1)
    axes(fig, 1, 2)
    fig.update_yaxes(scaleanchor="x2", row=1, col=2)
    S.style(fig, height=560)
    S.show(st, fig, key="km_fig")
    if k == 2:
        agree = max(np.mean(lab == ytr), np.mean(lab != ytr))
        st.metric("Points whose cluster matches their class", f"{100 * agree:.0f}%",
                  help="Cluster numbers are arbitrary, so we take the better of the two "
                       "ways of matching clusters to classes")
    st.markdown(
        "**Why it struggles on moons.** k-means assumes each group is a round blob around a "
        "centre. The moons are curved and interlocking, so splitting by distance cuts across "
        "them. On *Two blobs* it does very well. Unsupervised methods find structure, but "
        "only the structure their assumptions can see.")

else:
    if model_name == "Logistic regression":
        if st.session_state.c_poly:
            clf = make_pipeline(PolynomialFeatures(3), StandardScaler(),
                                LogisticRegression(max_iter=2000))
        else:
            clf = LogisticRegression()
    else:
        clf = KNeighborsClassifier(n_neighbors=min(st.session_state.c_k, len(ytr)))
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(grid)[:, 1].reshape(GRID_N, GRID_N)
    y_pred = clf.predict(Xte)
    tr_acc, te_acc = clf.score(Xtr, ytr), accuracy_score(yte, y_pred)
    if model_name == "k-nearest neighbours" and st.session_state.c_k == 1:
        status(f"k = 1 memorises the training set: {100 * tr_acc:.0f}% on training points, "
               f"{100 * te_acc:.0f}% on test points", "warn")
    elif model_name == "Logistic regression" and not st.session_state.c_poly:
        status("Logistic regression draws one straight line: it cannot bend round the moons",
               "info")
    else:
        status(f"{100 * te_acc:.0f}% of test points correct "
               f"({100 * tr_acc:.0f}% of training points)",
               "good" if te_acc > 0.9 else "info")
    fig = go.Figure()
    for tr in boundary_traces(P):
        fig.add_trace(tr)
    points(fig, None, None)
    axes(fig, None, None)
    fig.update_layout(title=dict(text="The decision boundary (black line: 50/50)", x=0,
                                 font=dict(size=17)))
    S.style(fig, height=640)
    S.show(st, fig, key="clf_fig")
    report(y_pred)
    if model_name == "Logistic regression":
        st.markdown(
            "Logistic regression computes a weighted sum of the features and squashes it "
            "with a sigmoid into a probability: a single neuron. Feeding it extra features "
            "(x², xy, y², …) lets one neuron draw a curve - that is *feature engineering*.")
    else:
        st.markdown(
            "k-nearest neighbours has no training step: it stores the data and lets the k "
            "closest training points vote. Small k follows every noisy point (overfitting); "
            "large k smooths everything (underfitting).")
