"""Demo 3: plain gradient descent vs momentum vs Adam on 2D surfaces."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from mlcore import style as S
from mlcore.gd import OPTIMISERS, adam_step, momentum_step, run_optimiser
from mlcore.problems import SURFACES_2D
from mlcore.ui import header, show_code

NAMES = list(SURFACES_2D)
OPT_NAMES = list(OPTIMISERS)
COLOURS = dict(zip(OPT_NAMES, S.SERIES))


def lr_key(surface, opt):
    return f"o_lr_{NAMES.index(surface)}_{OPT_NAMES.index(opt)}"


def reset(surface=None):
    surface = surface or st.session_state.o_surface
    s = SURFACES_2D[surface]
    st.session_state.o_x0, st.session_state.o_y0 = s["start"]
    st.session_state.o_steps = s["steps"]
    for opt in OPT_NAMES:
        st.session_state[lr_key(surface, opt)] = s["learning_rates"][opt]


st.session_state.setdefault("o_surface", NAMES[0])
if "o_x0" not in st.session_state:
    reset(NAMES[0])
for o in OPT_NAMES:
    st.session_state.setdefault(f"o_use_{o}", True)
    st.session_state.setdefault(lr_key(st.session_state.o_surface, o),
                                SURFACES_2D[st.session_state.o_surface]["learning_rates"][o])

header("Demo 3 · How a machine learns", "Optimiser race",
       "Three ways to go downhill on the same surface. Real loss landscapes have "
       "millions of dimensions, but the same problems show up in two.")

with st.sidebar:
    st.subheader("Settings")
    st.selectbox("Surface", NAMES, key="o_surface", on_change=reset)
    surf = SURFACES_2D[st.session_state.o_surface]
    (xlo, xhi), (ylo, yhi) = surf["x_range"], surf["y_range"]
    st.slider("Start x", float(xlo), float(xhi), key="o_x0", step=0.1)
    st.slider("Start y", float(ylo), float(yhi), key="o_y0", step=0.1)
    st.slider("Number of steps", 10, 1000, key="o_steps", step=10)
    st.markdown("**Optimisers** (each with its own learning rate)")
    for opt in OPT_NAMES:
        on = st.checkbox(opt, key=f"o_use_{opt}")
        st.number_input(f"η for {opt}", min_value=1e-5, max_value=2.0,
                        key=lr_key(st.session_state.o_surface, opt), format="%.4f",
                        step=0.0005, disabled=not on, label_visibility="collapsed")
    st.button("Reset this surface", on_click=reset, width="stretch")

chosen = [o for o in OPT_NAMES if st.session_state[f"o_use_{o}"]]
start = (st.session_state.o_x0, st.session_state.o_y0)
paths = {}
for opt in chosen:
    p = run_optimiser(OPTIMISERS[opt], surf["grad"], start,
                      st.session_state[lr_key(st.session_state.o_surface, opt)],
                      st.session_state.o_steps)
    paths[opt] = np.where(np.isfinite(p), p, np.nan)

# ---- figure ----------------------------------------------------------------
gx = np.linspace(xlo, xhi, 480)
gy = np.linspace(ylo, yhi, 420)
GX, GY = np.meshgrid(gx, gy)
Z = surf["f"](np.stack([GX, GY], axis=-1))
Zplot = np.log10(Z + 1e-3) if surf["log_colour"] else np.sqrt(Z)

fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.09,
                    subplot_titles=("Paths on the surface", "Height f after each step (log scale)"))
S.add_landscape(fig, gx, gy, Zplot, n_levels=24, row=1, col=1)
mx, my = zip(*surf["minima"])
fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", name="Minimum",
                         marker=dict(symbol="star", size=18, color=S.INK,
                                     line=dict(color=S.SURFACE, width=2)),
                         hoverinfo="skip"), 1, 1)
fig.add_trace(go.Scatter(x=[start[0]], y=[start[1]], mode="markers", name="Start",
                         marker=dict(symbol="circle-open", size=16, color=S.INK,
                                     line=dict(width=3)), hoverinfo="skip"), 1, 1)

n_total = max((len(p) for p in paths.values()), default=1)


def moving(k):
    traces = []
    for opt in chosen:
        p = paths[opt]
        j = min(k, len(p) - 1)
        with np.errstate(over="ignore", invalid="ignore"):
            f = surf["f"](p[: j + 1])
        traces += [
            go.Scatter(x=p[: j + 1, 0], y=p[: j + 1, 1], mode="lines", name=opt,
                       line=dict(color=COLOURS[opt], width=2.5), legendgroup=opt,
                       hoverinfo="skip"),
            go.Scatter(x=[p[j, 0]], y=[p[j, 1]], mode="markers", showlegend=False,
                       marker=dict(size=14, color=COLOURS[opt],
                                   line=dict(color=S.SURFACE, width=2.5)),
                       legendgroup=opt, hovertemplate=f"{opt}<br>step {j}<extra></extra>"),
            go.Scatter(x=np.arange(j + 1), y=np.maximum(f, 1e-12), mode="lines",
                       showlegend=False, line=dict(color=COLOURS[opt], width=2.5),
                       legendgroup=opt,
                       hovertemplate=f"{opt}<br>step %{{x}}<br>f %{{y:.3g}}<extra></extra>"),
        ]
    return traces


cells = []
for _ in chosen:
    cells += [(1, 1), (1, 1), (1, 2)]
for tr, (r, c) in zip(moving(n_total - 1), cells):
    fig.add_trace(tr, r, c)

frames_at = S.frame_indices(n_total, 60)
moving_ids = list(range(2, 2 + 3 * len(chosen)))
fig.frames = [go.Frame(name=str(k), data=moving(k), traces=moving_ids) for k in frames_at]

fig.update_xaxes(title_text="x", range=[xlo, xhi], row=1, col=1)
fig.update_yaxes(title_text="y", range=[ylo, yhi], row=1, col=1)
fig.update_xaxes(title_text="Step", row=1, col=2)
fig.update_yaxes(title_text="f(x, y)", type="log", dtick=1, row=1, col=2)
S.style(fig, height=700)
S.add_player(fig, [str(k) for k in frames_at], duration=90, redraw=True)
S.show(st, fig, key="opt_fig")

# ---- readouts --------------------------------------------------------------
if chosen:
    cols = st.columns(len(chosen))
    for col, opt in zip(cols, chosen):
        p = paths[opt]
        end = p[-1]
        if not np.all(np.isfinite(end)):
            col.metric(opt, "Diverged")
        else:
            col.metric(opt, f"f = {surf['f'](end):.3g}",
                       help=f"Finished at ({end[0]:.2f}, {end[1]:.2f}) after {len(p) - 1} steps")

left, right = st.columns(2)
with left:
    st.markdown("**The surface**")
    st.latex(surf["latex"])
    st.markdown(surf["note"])
with right:
    st.markdown("**The three update rules**")
    st.latex(r"\text{GD:}\quad p \leftarrow p - \eta\,\nabla f")
    st.latex(r"\text{Momentum:}\quad v \leftarrow \beta v - \eta\,\nabla f,\quad p \leftarrow p + v")
    st.latex(r"\text{Adam:}\quad p \leftarrow p - \eta\,\frac{\hat m}{\sqrt{\hat s}+\epsilon}")
    st.markdown("Adam keeps a running average of the gradient (m̂) and of its square (ŝ). "
                "It is the usual default for training neural networks in PyTorch: "
                "`torch.optim.Adam(model.parameters(), lr=1e-3)`.")

show_code(momentum_step, adam_step, run_optimiser)
