"""Demo 4: underfitting and overfitting with polynomial regression."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlcore import style as S
from mlcore.problems import CURVES
from mlcore.ui import header, preset_row, status

NAMES = list(CURVES)
MAX_DEGREE = 15
ALPHAS = [0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

DEFAULTS = dict(ov_curve=NAMES[0], ov_n=40, ov_noise=CURVES[NAMES[0]]["noise_default"],
                ov_degree=3, ov_alpha=0.0, ov_test=0.3, ov_seed=2, ov_truth=True)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Underfit (degree 1)": dict(ov_degree=1, ov_alpha=0.0),
    "About right": dict(ov_degree=5, ov_alpha=0.0),
    "Overfit (degree 15)": dict(ov_degree=15, ov_alpha=0.0),
    "Overfit + regularise": dict(ov_degree=15, ov_alpha=1.0),
    "More data": dict(ov_degree=15, ov_alpha=0.0, ov_n=200),
}


def apply(settings):
    for k, v in settings.items():
        st.session_state[k] = v


def new_curve():
    st.session_state.ov_noise = CURVES[st.session_state.ov_curve]["noise_default"]


def new_data():
    st.session_state.ov_seed += 1


@st.cache_data
def make_data(curve, n, noise, seed):
    c = CURVES[curve]
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(*c["t_range"], n))
    return t, c["f"](t) + noise * rng.normal(size=n)


def model_for(degree, alpha):
    """Polynomial features of the (rescaled) input, then a linear model."""
    reg = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), reg)


def rmse(model, t, y, to_unit):
    return float(np.sqrt(np.mean((model.predict(to_unit(t)) - y) ** 2)))


header("Demo 4 · Did it actually learn?", "Underfitting and overfitting",
       "Fit polynomials of increasing degree. The training error always goes down; "
       "the error on points the model never saw tells you the truth.")

with st.sidebar:
    st.subheader("Settings")
    st.selectbox("Data", NAMES, key="ov_curve", on_change=new_curve)
    st.slider("Polynomial degree", 1, MAX_DEGREE, key="ov_degree")
    st.select_slider("Regularisation (ridge α)", ALPHAS, key="ov_alpha",
                     format_func=lambda v: "none" if v == 0 else f"{v:g}")
    st.slider("Number of points", 10, 200, key="ov_n", step=5)
    st.slider("Noise", 0.0, 0.6, key="ov_noise", step=0.01)
    st.slider("Fraction held back as the test set", 0.2, 0.5, key="ov_test", step=0.05)
    st.toggle("Show the true curve", key="ov_truth")
    st.button("Draw new data", on_click=new_data, width="stretch")

st.caption("Try a preset, or drag the degree slider and watch both charts.")
preset_row(PRESETS, apply, "ov")

curve = CURVES[st.session_state.ov_curve]
t, y = make_data(st.session_state.ov_curve, st.session_state.ov_n,
                 st.session_state.ov_noise, st.session_state.ov_seed)
t_tr, t_te, y_tr, y_te = train_test_split(t, y, test_size=st.session_state.ov_test,
                                          random_state=st.session_state.ov_seed)
lo, hi = curve["t_range"]


def to_unit(tt):
    """Map time onto [-1, 1] and make it the [n_samples, n_features] shape sklearn wants."""
    return ((2 * (np.asarray(tt) - lo) / (hi - lo)) - 1).reshape(-1, 1)


degree, alpha = st.session_state.ov_degree, st.session_state.ov_alpha

errors = []
for d in range(1, MAX_DEGREE + 1):
    m = model_for(d, alpha).fit(to_unit(t_tr), y_tr)
    errors.append((rmse(m, t_tr, y_tr, to_unit), rmse(m, t_te, y_te, to_unit)))
train_err, test_err = map(np.array, zip(*errors))
best = int(np.argmin(test_err)) + 1

model = model_for(degree, alpha).fit(to_unit(t_tr), y_tr)
tr_now, te_now = train_err[degree - 1], test_err[degree - 1]
n_params = degree + 1

# ---- status ----------------------------------------------------------------
if n_params >= len(t_tr):
    status(f"{n_params} parameters for {len(t_tr)} training points: the curve can pass "
           "through every training point, noise and all", "warn")
elif te_now > 1.5 * tr_now and degree > best:
    status("Overfitting: great on the training points, poor on the test points", "warn")
elif degree < best and te_now > 1.15 * test_err[best - 1]:
    status("Underfitting: too simple to follow the shape", "info")
else:
    status("About right: training and test errors are close and low", "good")

# ---- figures ---------------------------------------------------------------
left, right = st.columns([0.58, 0.42])

grid = np.linspace(lo, hi, 500)
fit = model.predict(to_unit(grid))
y_all = np.concatenate([y, curve["f"](grid)])
pad = 0.25 * (y_all.max() - y_all.min())
y_range = [y_all.min() - pad, y_all.max() + pad]

f1 = go.Figure()
if st.session_state.ov_truth:
    f1.add_trace(go.Scatter(x=grid, y=curve["f"](grid), mode="lines", name="True curve",
                            line=dict(color=S.MUTED, width=2, dash="dash"), hoverinfo="skip"))
f1.add_trace(go.Scatter(x=t_tr, y=y_tr, mode="markers", name="Training points",
                        marker=dict(size=10, color=S.BLUE, line=dict(color=S.SURFACE, width=1.5)),
                        hovertemplate="%{x:.1f}, %{y:.3f}<extra>train</extra>"))
f1.add_trace(go.Scatter(x=t_te, y=y_te, mode="markers", name="Test points",
                        marker=dict(size=11, color=S.ORANGE, symbol="diamond",
                                    line=dict(color=S.SURFACE, width=1.5)),
                        hovertemplate="%{x:.1f}, %{y:.3f}<extra>test</extra>"))
f1.add_trace(go.Scatter(x=grid, y=np.clip(fit, y_range[0] - 10, y_range[1] + 10),
                        mode="lines", name=f"Degree-{degree} fit",
                        line=dict(color=S.INK, width=3), hoverinfo="skip"))
f1.update_xaxes(title_text=curve["x_label"], range=[lo, hi])
f1.update_yaxes(title_text=curve["y_label"], range=y_range)
f1.update_layout(title=dict(text="The data and the fitted curve", x=0, font=dict(size=17)))
S.style(f1, height=520)
with left:
    S.show(st, f1, key="ov_fit")

degrees = np.arange(1, MAX_DEGREE + 1)
f2 = go.Figure()
f2.add_trace(go.Scatter(x=degrees, y=train_err, mode="lines+markers", name="Training error",
                        line=dict(color=S.BLUE, width=2.5), marker=dict(size=8),
                        hovertemplate="degree %{x}<br>RMSE %{y:.3f}<extra>train</extra>"))
f2.add_trace(go.Scatter(x=degrees, y=test_err, mode="lines+markers", name="Test error",
                        line=dict(color=S.ORANGE, width=2.5),
                        marker=dict(size=9, symbol="diamond"),
                        hovertemplate="degree %{x}<br>RMSE %{y:.3f}<extra>test</extra>"))
f2.add_vline(x=degree, line=dict(color=S.INK, width=2))
f2.add_annotation(x=best, y=np.log10(test_err[best - 1]), text="lowest test error",
                  showarrow=True, arrowhead=2, ax=50, ay=-70, font=dict(size=14),
                  bgcolor="rgba(255,253,248,0.9)")
f2.update_xaxes(title_text="Polynomial degree (model complexity)", dtick=1)
f2.update_yaxes(title_text="Error (RMSE, log scale)", type="log", dtick=1)
f2.update_layout(title=dict(text="Error against complexity", x=0, font=dict(size=17)))
S.style(f2, height=520)
with right:
    S.show(st, f2, key="ov_err")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parameters", f"{n_params}")
c2.metric("Training points", f"{len(t_tr)}")
c3.metric("Training error", f"{tr_now:.3f}")
c4.metric("Test error", f"{te_now:.3f}",
          delta=f"{te_now - tr_now:+.3f} vs training", delta_color="inverse")

st.markdown(
    "**Reading the right-hand chart.** Left of the lowest test error the model is too "
    "simple (*underfitting*: both errors high). Right of it the model starts fitting the "
    "noise (*overfitting*: training error keeps falling, test error rises). Regularisation "
    "and more data both push the overfitting region to the right.")
st.markdown(
    "**In practice** you choose the degree (a *hyperparameter*) using a separate "
    "**validation** set or cross-validation, and touch the **test** set only once, at "
    "the end. Otherwise the test set quietly becomes training data.")
with st.expander("The scikit-learn code"):
    st.code(
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n"
        "from sklearn.linear_model import LinearRegression\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)\n"
        f"model = make_pipeline(PolynomialFeatures({degree}), StandardScaler(), LinearRegression())\n"
        "model.fit(X_train, y_train)          # learn from the training set only\n"
        "model.score(X_test, y_test)          # R² on data the model never saw",
        language="python")
