"""Demo 1: gradient descent on a function of one variable."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from mlcore import style as S
from mlcore.gd import gradient_descent_1d
from mlcore.problems import FUNCTIONS_1D
from mlcore.ui import header, preset_row, show_code, status

NAMES = list(FUNCTIONS_1D)
BOWL, TWO, BUMPY = NAMES
LR_OPTIONS = [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 1.0, 1.05]
STEP_OPTIONS = [5, 10, 20, 40, 60, 100, 150, 200]

PRESETS = {
    "Too small": dict(fn=BOWL, lr=0.003, mom=0.0, steps=40),
    "About right": dict(fn=BOWL, lr=0.1, mom=0.0, steps=40),
    "One step": dict(fn=BOWL, lr=0.5, mom=0.0, steps=10),
    "Bounces": dict(fn=BOWL, lr=1.0, mom=0.0, steps=20),
    "Blows up": dict(fn=BOWL, lr=1.05, mom=0.0, steps=40),
    "Stuck in a dip": dict(fn=BUMPY, lr=0.05, mom=0.0, steps=100),
    "With momentum": dict(fn=BUMPY, lr=0.05, mom=0.9, steps=100),
}

DEFAULTS = dict(r_fn=BOWL, r_lr=0.1, r_mom=0.0, r_steps=40)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("r_x0", FUNCTIONS_1D[st.session_state.r_fn]["start"])


def apply(settings):
    st.session_state.r_fn = settings["fn"]
    st.session_state.r_lr = settings["lr"]
    st.session_state.r_mom = settings["mom"]
    st.session_state.r_steps = settings["steps"]
    st.session_state.r_x0 = FUNCTIONS_1D[settings["fn"]]["start"]


def reset_start():
    st.session_state.r_x0 = FUNCTIONS_1D[st.session_state.r_fn]["start"]


header("Demo 1 · How a machine learns", "Rolling downhill",
       "Gradient descent on a function of one variable. The learning rate η decides "
       "whether you crawl, converge, bounce or blow up.")

# ---- controls --------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")
    st.selectbox("Function", NAMES, key="r_fn", on_change=reset_start)
    fn = FUNCTIONS_1D[st.session_state.r_fn]
    lo, hi = fn["x_range"]
    st.slider("Starting point x₀", float(lo), float(hi), key="r_x0", step=0.05)
    st.select_slider("Learning rate η", LR_OPTIONS, key="r_lr")
    st.select_slider("Number of steps", STEP_OPTIONS, key="r_steps")
    st.slider("Momentum β (0 = plain gradient descent)", 0.0, 0.95, key="r_mom",
              step=0.05)

st.caption("Try a preset, then press ▶ Play under the chart.")
preset_row(PRESETS, apply, "r")

lr, mom = st.session_state.r_lr, st.session_state.r_mom
xs = gradient_descent_1d(fn["grad"], st.session_state.r_x0, lr,
                         st.session_state.r_steps, momentum=mom)
with np.errstate(over="ignore", invalid="ignore"):
    fs = fn["f"](xs)

# ---- diagnosis -------------------------------------------------------------
diverged = (not np.all(np.isfinite(xs))) or np.max(np.abs(xs)) > 10 * max(abs(lo), abs(hi))
final_grad = abs(fn["grad"](xs[-1])) if not diverged else np.inf
moves = np.diff(xs[-12:])
bouncing = len(moves) > 3 and np.all(moves[1:] * moves[:-1] < 0) and abs(moves[-1]) > 1e-3

if diverged:
    status("Diverged: each step overshoots by more than the last, so x runs away", "warn")
elif bouncing:
    status("Bouncing: the steps keep overshooting the bottom", "warn")
elif final_grad < 1e-2 and abs(xs[-1] - xs[-2]) < 1e-2:
    status(f"Converged: the slope is flat at x = {xs[-1]:.3f}", "good")
else:
    status("Still rolling: not at the bottom yet - more steps or a bigger η", "info")

# ---- figure ----------------------------------------------------------------
grid = np.linspace(lo, hi, 400)
in_view = np.where((xs >= lo) & (xs <= hi), xs, np.nan)
f_view = fn["f"](in_view)
f_lo, f_hi = fn["f"](grid).min(), fn["f"](grid).max()
pad = 0.08 * (f_hi - f_lo)

fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.09,
                    subplot_titles=("The function and the path taken",
                                    "f(x) after each step"))
fig.add_trace(go.Scatter(x=grid, y=fn["f"](grid), mode="lines", name="f(x)",
                         line=dict(color=S.INK, width=2), hoverinfo="skip"), 1, 1)


def path_traces(k):
    """Traces that change during the animation, drawn up to step k."""
    steps = np.arange(k + 1)
    return [
        go.Scatter(x=in_view[:k + 1], y=f_view[:k + 1], mode="lines+markers",
                   name="Path", line=dict(color=S.ORANGE, width=2),
                   marker=dict(size=7, color=S.ORANGE),
                   hovertemplate="x = %{x:.3f}<br>f = %{y:.3f}<extra></extra>"),
        go.Scatter(x=[in_view[k]], y=[f_view[k]], mode="markers", name="Ball",
                   marker=dict(size=22, color=S.ORANGE, line=dict(color=S.SURFACE, width=3)),
                   hovertemplate="step %{text}<extra></extra>", text=[str(k)],
                   showlegend=False),
        go.Scatter(x=steps, y=fs[:k + 1], mode="lines+markers", name="f(x)",
                   line=dict(color=S.ORANGE, width=2), marker=dict(size=6),
                   showlegend=False,
                   hovertemplate="step %{x}<br>f = %{y:.4f}<extra></extra>"),
    ]


last = len(xs) - 1
for i, tr in enumerate(path_traces(last)):
    fig.add_trace(tr, 1, 1 if i < 2 else 2)

frames_at = S.frame_indices(len(xs), 60)
fig.frames = [go.Frame(name=str(k), data=path_traces(k), traces=[1, 2, 3])
              for k in frames_at]

fig.update_xaxes(title_text="x", range=[lo, hi], row=1, col=1)
fig.update_yaxes(title_text="f(x)", range=[f_lo - pad, f_hi + pad], row=1, col=1)
fig.update_xaxes(title_text="Step", range=[-0.5, max(last, 1) + 0.5], row=1, col=2)
finite = fs[np.isfinite(fs)]
if not diverged and len(finite):
    span = finite.max() - finite.min()
    fig.update_yaxes(range=[finite.min() - 0.1 * span - 0.05, finite.max() + 0.1 * span + 0.05],
                     row=1, col=2)
fig.update_yaxes(title_text="f(x)", row=1, col=2)
S.style(fig, height=600, legend=False)
S.add_player(fig, [str(k) for k in frames_at], duration=120)
S.show(st, fig, key="rolling_fig")

# ---- readouts --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Start", f"x₀ = {xs[0]:.2f}")
c2.metric("Finish", "∞" if diverged else f"x = {xs[-1]:.3f}")
c3.metric("f at finish", "∞" if diverged else f"{fs[-1]:.3f}")
c4.metric("Steps taken", f"{last}")

left, right = st.columns(2)
with left:
    st.markdown("**The function**")
    st.latex(fn["latex"])
    st.markdown(fn["note"])
with right:
    st.markdown("**The update rule**")
    if mom == 0:
        st.latex(r"x_{\text{new}} = x - \eta\,\frac{df}{dx}")
        st.markdown("Step downhill, by an amount proportional to the slope. "
                    "Near the bottom the slope shrinks, so the steps shrink too.")
    else:
        st.latex(r"v_{\text{new}} = \beta\,v - \eta\,\frac{df}{dx},\qquad x_{\text{new}} = x + v_{\text{new}}")
        st.markdown("The ball keeps a fraction β of its previous velocity, so it "
                    "can coast through small dips and over small bumps.")

show_code(gradient_descent_1d)
