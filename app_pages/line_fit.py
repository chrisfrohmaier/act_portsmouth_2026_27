"""Demo 2: fit a straight line to real data by gradient descent."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes

from mlcore import style as S
from mlcore.gd import fit_line_gd, mse_gradients, mse_surface
from mlcore.ui import header, preset_row, show_code, status

LR_OPTIONS = [1e-5, 1e-4, 3e-4, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.03, 0.1, 0.3,
              0.5, 0.9, 1.0, 1.05]
STEP_OPTIONS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
BATCH_OPTIONS = {"Batch: all 442 points": None, "Mini-batch: 32 points": 32,
                 "Stochastic: 1 point": 1}


@st.cache_data
def load_bmi():
    """BMI (kg/m²) and disease progression one year later for 442 patients."""
    d = load_diabetes(scaled=False)
    return d.data[:, 2].astype(float), d.target.astype(float)


bmi, y = load_bmi()

DEFAULTS = dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_m0_u=0.0, lf_c0_u=0.0,
                lf_lr=0.001, lf_steps=2000, lf_batch=list(BATCH_OPTIONS)[0])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Last year's notebook": dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_lr=0.001,
                                 lf_steps=2000, lf_batch=list(BATCH_OPTIONS)[0]),
    "Bigger steps": dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_lr=0.1,
                         lf_steps=50, lf_batch=list(BATCH_OPTIONS)[0]),
    "One perfect step": dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_lr=0.5,
                             lf_steps=5, lf_batch=list(BATCH_OPTIONS)[0]),
    "Too big": dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_lr=1.05,
                    lf_steps=20, lf_batch=list(BATCH_OPTIONS)[0]),
    "Forget to scale": dict(lf_scale=False, lf_m0_u=0.0, lf_c0_u=0.0, lf_lr=0.001,
                            lf_steps=2000, lf_batch=list(BATCH_OPTIONS)[0]),
    "Stochastic": dict(lf_scale=True, lf_m0_s=5.0, lf_c0_s=0.0, lf_lr=0.03,
                       lf_steps=500, lf_batch=list(BATCH_OPTIONS)[2]),
}


def apply(settings):
    for k, v in settings.items():
        st.session_state[k] = v


header("Demo 2 · How a machine learns", "Fit a line by gradient descent",
       "Diabetes progression against body-mass index for 442 patients. Each step "
       "nudges the gradient m and intercept c downhill on the loss landscape.")

with st.sidebar:
    st.subheader("Settings")
    scaled = st.toggle("Standardise BMI (mean 0, std 1)", key="lf_scale")
    if scaled:
        st.slider("Starting gradient m", -50.0, 150.0, key="lf_m0_s", step=1.0)
        st.slider("Starting intercept c", -50.0, 300.0, key="lf_c0_s", step=5.0)
    else:
        st.slider("Starting gradient m", -10.0, 30.0, key="lf_m0_u", step=0.5)
        st.slider("Starting intercept c", -300.0, 300.0, key="lf_c0_u", step=5.0)
    st.select_slider("Learning rate η", LR_OPTIONS, key="lf_lr",
                     format_func=lambda v: f"{v:g}")
    st.select_slider("Number of steps", STEP_OPTIONS, key="lf_steps")
    st.radio("Data used for each step", list(BATCH_OPTIONS), key="lf_batch")

st.caption("Try a preset, then press ▶ Play under the chart.")
preset_row(PRESETS, apply, "lf")

# ---- the data and the fit --------------------------------------------------
if scaled:
    x = (bmi - bmi.mean()) / bmi.std()
    m0, c0 = st.session_state.lf_m0_s, st.session_state.lf_c0_s
    x_label = "BMI (standardised)"
else:
    x = bmi
    m0, c0 = st.session_state.lf_m0_u, st.session_state.lf_c0_u
    x_label = "BMI (kg/m²)"

lr, n_steps = st.session_state.lf_lr, st.session_state.lf_steps
batch = BATCH_OPTIONS[st.session_state.lf_batch]
hist = fit_line_gd(x, y, m=m0, c=c0, learning_rate=lr, n_steps=n_steps,
                   batch_size=batch, seed=1)
m_best, c_best = np.polyfit(x, y, 1)  # the exact least-squares answer
loss_best = np.mean((m_best * x + c_best - y) ** 2)

ms, cs, losses = hist["m"], hist["c"], hist["loss"]
diverged = not np.isfinite(losses[-1]) or losses[-1] > 1e3 * losses[0]
# Plotly cannot draw infinities, so blank them out
ms, cs, losses = (np.where(np.isfinite(a), a, np.nan) for a in (ms, cs, losses))
last = len(ms) - 1

# ---- status ----------------------------------------------------------------
gap = (losses[-1] - loss_best) / loss_best if np.isfinite(losses[-1]) else np.inf
if diverged:
    status("Diverged: the learning rate is too big for this landscape", "warn")
elif losses[-1] > losses[0]:
    status("Diverging: every step overshoots further, so the loss is growing", "warn")
elif batch is not None and gap < 0.2:
    status(f"Jittering around the minimum ({100 * gap:.1f}% above the best): noisy "
           "mini-batch gradients never let it settle exactly", "info")
elif gap < 0.001:
    status("Converged: within 0.1% of the best possible line", "good")
elif gap < 0.05:
    status(f"Nearly there: the loss is {100 * gap:.1f}% above the best possible", "info")
else:
    status(f"Still descending: the loss is {100 * gap:.0f}% above the best possible", "info")

# ---- the loss landscape ----------------------------------------------------
pts_m = [m0, m_best] + ([] if diverged else list(ms))
pts_c = [c0, c_best] + ([] if diverged else list(cs))
m_lo, m_hi = min(pts_m), max(pts_m)
c_lo, c_hi = min(pts_c), max(pts_c)
m_pad = 0.25 * max(m_hi - m_lo, 1e-3) + (2 if scaled else 0.5)
c_pad = 0.25 * max(c_hi - c_lo, 1e-3) + (10 if scaled else 10)
m_grid = np.linspace(m_lo - m_pad, m_hi + m_pad, 420)
c_grid = np.linspace(c_lo - c_pad, c_hi + c_pad, 420)
Z = np.log10(mse_surface(x, y, m_grid, c_grid))

# Landscape on the left (first subplot), data and loss curve stacked on the right.
fig = make_subplots(
    rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.62, 0.38],
    specs=[[{"rowspan": 2}, {}], [None, {}]],
    horizontal_spacing=0.1, vertical_spacing=0.22,
    subplot_titles=("The loss landscape (darker = bigger MSE)",
                    "The data and the current line", "Loss after each step"),
)

S.add_landscape(fig, m_grid, c_grid, Z, row=1, col=1)
fig.add_trace(go.Scatter(x=[m_best], y=[c_best], mode="markers", name="Minimum",
                         marker=dict(symbol="star", size=20, color=S.INK,
                                     line=dict(color=S.SURFACE, width=2)),
                         hovertemplate="best: m %{x:.2f}, c %{y:.1f}<extra></extra>"), 1, 1)

x_line = np.array([x.min(), x.max()])
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Patients",
                         marker=dict(size=7, color=S.DATA, opacity=0.55),
                         hovertemplate="BMI %{x:.2f}<br>progression %{y:.0f}<extra></extra>"),
              1, 2)
fig.add_trace(go.Scatter(x=x_line, y=m_best * x_line + c_best, mode="lines",
                         name="Best possible line", line=dict(color=S.INK, width=2, dash="dash"),
                         hoverinfo="skip"), 1, 2)


def moving(k):
    """Traces redrawn in each animation frame, showing step k."""
    yl = ms[k] * x_line + cs[k]
    return [
        go.Scatter(x=ms[:k + 1], y=cs[:k + 1], mode="lines", name="Path",
                   line=dict(color=S.ORANGE, width=2.5), hoverinfo="skip",
                   showlegend=False),
        go.Scatter(x=[ms[k]], y=[cs[k]], mode="markers", name="Now",
                   marker=dict(size=16, color=S.ORANGE, line=dict(color=S.SURFACE, width=3)),
                   hovertemplate="m %{x:.2f}<br>c %{y:.1f}<extra></extra>",
                   showlegend=False),
        go.Scatter(x=x_line, y=yl, mode="lines", name="Current line",
                   line=dict(color=S.ORANGE, width=4), hoverinfo="skip"),
        go.Scatter(x=np.arange(k + 1), y=losses[:k + 1], mode="lines",
                   line=dict(color=S.ORANGE, width=2.5), showlegend=False,
                   hovertemplate="step %{x}<br>MSE %{y:.1f}<extra></extra>"),
    ]


for tr, (r, c) in zip(moving(last), [(1, 1), (1, 1), (1, 2), (2, 2)]):
    fig.add_trace(tr, r, c)
fig.add_hline(y=loss_best, line=dict(color=S.INK, width=1.5, dash="dash"), row=2, col=2)

frames_at = S.frame_indices(len(ms), 60)
fig.frames = [go.Frame(name=str(k), data=moving(k), traces=[3, 4, 5, 6]) for k in frames_at]

y_pad = 0.08 * (y.max() - y.min())
fig.update_xaxes(title_text="gradient m", range=[m_grid[0], m_grid[-1]], row=1, col=1)
fig.update_yaxes(title_text="intercept c", range=[c_grid[0], c_grid[-1]], row=1, col=1)
fig.update_xaxes(title_text=x_label, row=1, col=2)
fig.update_yaxes(title_text="Disease progression", range=[y.min() - y_pad, y.max() + y_pad],
                 row=1, col=2)
fig.update_xaxes(title_text="Step", row=2, col=2)
fig.update_yaxes(title_text="MSE", type="log", row=2, col=2)
S.style(fig, height=820)
S.add_player(fig, [str(k) for k in frames_at], duration=90, redraw=True)
S.show(st, fig, key="line_fig")

# ---- readouts --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gradient m", "∞" if diverged else f"{ms[-1]:.2f}", help="Best possible: "
          f"{m_best:.2f}")
c2.metric("Intercept c", "∞" if diverged else f"{cs[-1]:.1f}", help="Best possible: "
          f"{c_best:.1f}")
c3.metric("MSE now", "∞" if diverged else f"{losses[-1]:,.0f}",
          help=f"Best possible: {loss_best:,.0f}")
if batch is None:
    c4.metric("Steps", f"{last}")
else:
    c4.metric("Steps (epochs)", f"{last} ({last * batch / len(x):.1f})",
              help="One epoch = one pass through all 442 points")

left, right = st.columns(2)
with left:
    st.markdown("**What each step does**")
    st.latex(r"m \leftarrow m - \eta\,\frac{\partial L}{\partial m},\qquad "
             r"c \leftarrow c - \eta\,\frac{\partial L}{\partial c}")
    st.latex(r"\frac{\partial L}{\partial m} = \frac{2}{n}\sum_i (\hat y_i - y_i)\,x_i,\qquad "
             r"\frac{\partial L}{\partial c} = \frac{2}{n}\sum_i (\hat y_i - y_i)")
with right:
    if scaled:
        st.markdown("**Why scale?** With BMI standardised the loss landscape is a round "
                    "bowl, so the downhill direction points at the minimum. Turn the "
                    "toggle off to see what happens with raw BMI.")
    else:
        st.markdown("**Raw BMI (≈ 18–42 kg/m²)** makes the landscape a long, thin, tilted "
                    "valley. A learning rate small enough not to blow up across the valley "
                    "(try 0.002) is far too small to travel along it. Scaling the feature "
                    "fixes the shape.")
    if batch is not None:
        st.markdown("**Mini-batches** estimate the gradient from a few points, so each step "
                    "is cheap but noisy: the path wanders, and the loss wobbles instead of "
                    "falling smoothly.")

with st.expander("Check the answer with scikit-learn"):
    st.code(
        "from sklearn.linear_model import LinearRegression\n\n"
        "model = LinearRegression().fit(x.reshape(-1, 1), y)\n"
        "model.coef_, model.intercept_   # learned parameters end in an underscore",
        language="python")
    st.markdown(f"For these data: m = **{m_best:.2f}**, c = **{c_best:.1f}** "
                "(the dashed line and the star above). Gradient descent should agree once "
                "it has converged.")

show_code(mse_gradients, fit_line_gd)
