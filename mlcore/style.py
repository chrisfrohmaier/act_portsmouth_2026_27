"""Shared colours and Plotly helpers so every page looks like one app."""

import numpy as np
import plotly.graph_objects as go

# Colours (checked for colour-blind separation on the chart surface)
INK = "#1c2130"
MUTED = "#6a7080"
GRID = "#e6e2d8"
SURFACE = "#fffdf8"
BLUE = "#2a6fc9"
ORANGE = "#c95520"
GREEN = "#178a62"
DATA = "#7d8494"  # neutral grey for plain data points

SERIES = [BLUE, ORANGE, GREEN]

# One hue, light -> dark: for "how big is the loss here?"
SEQUENTIAL = [
    [0.0, "#fffdf8"], [0.15, "#e3ebf5"], [0.35, "#b8cbe5"],
    [0.55, "#8aa8d3"], [0.75, "#5a82bd"], [1.0, "#2b4a80"],
]

# Two hues with a neutral middle: for "which class is more likely?"
DIVERGING = [
    [0.0, "#9bbbe6"], [0.25, "#c9daf1"], [0.5, "#f4f1ea"],
    [0.75, "#f4cdb6"], [1.0, "#e9a37e"],
]

FONT = dict(family="IBM Plex Sans, Inter, Helvetica, Arial, sans-serif",
            size=15, color=INK)


def style(fig, height=520, legend=True):
    """Apply the house style to a Plotly figure (call before add_player)."""
    fig.update_layout(
        height=height,
        font=FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        margin=dict(l=64, r=24, t=104 if legend else 60, b=64),
        showlegend=legend,
        # The legend sits in its own band at the very top, clear of subplot titles
        legend=dict(orientation="h", yref="container", y=0.995, yanchor="top",
                    x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT["family"], size=14)),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, showline=True,
                ticks="outside", tickcolor=MUTED)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _ramp(t, stops):
    """Interpolate a [[position, "#rrggbb"], ...] colour scale at positions t in [0, 1]."""
    pos = np.array([p for p, _ in stops])
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops], float)
    return np.stack([np.interp(t, pos, rgb[:, k]) for k in range(3)], axis=-1)


def add_landscape(fig, x, y, z, n_levels=22, row=1, col=1):
    # x, y: 1D grid coordinates; z[i, j] is the loss at (x[j], y[i]). Use a fine
    # grid (a few hundred points each way) so the band edges look smooth.
    """
    Draw a loss landscape behind a subplot as banded colour (like filled contours).

    It is a picture placed under the axes rather than a Plotly trace, because
    Streamlit's Plotly component blanks heatmap and filled-contour traces while an
    animation plays. Darker bands mean a bigger loss.
    """
    from PIL import Image  # Pillow is installed with Streamlit

    z = np.asarray(z, dtype=float)
    lo, hi = np.nanmin(z), np.nanmax(z)
    t = np.clip((z - lo) / (hi - lo + 1e-12), 0, 1)
    bands = np.floor(t * n_levels) / n_levels          # quantise into contour bands
    rgb = _ramp(bands, SEQUENTIAL)
    edges = np.zeros_like(t, dtype=bool)               # thin darker line between bands
    edges[:-1, :] |= bands[:-1, :] != bands[1:, :]
    edges[:, :-1] |= bands[:, :-1] != bands[:, 1:]
    rgb[edges] = rgb[edges] * 0.9
    img = Image.fromarray(np.flipud(rgb).astype(np.uint8))  # image row 0 is the top

    xref = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")
    yref = fig.get_subplot(row, col).yaxis.plotly_name.replace("axis", "")
    fig.add_layout_image(dict(source=img, xref=xref, yref=yref, x=x[0], y=y[-1],
                              sizex=x[-1] - x[0], sizey=y[-1] - y[0], xanchor="left",
                              yanchor="top", sizing="stretch", layer="below"))
    fig.update_xaxes(showgrid=False, row=row, col=col)  # grid lines would sit on top
    fig.update_yaxes(showgrid=False, row=row, col=col)
    return fig


def frame_indices(n_points, max_frames=60):
    """Pick at most max_frames evenly spaced step numbers, always including both ends."""
    if n_points <= max_frames:
        return list(range(n_points))
    idx = np.unique(np.linspace(0, n_points - 1, max_frames).round().astype(int))
    return idx.tolist()


def add_player(fig, frame_names, duration=80, redraw=False, prefix="Step "):
    """
    Add Play / Pause buttons and a scrubber below an animated figure.

    The figure is drawn at its final state; Play always starts from step 0,
    so you see the answer straight away and can then replay how it got there.
    Call this after style(), because the controls are placed in pixels below
    the plotting area.
    """
    height = fig.layout.height or 520
    top = fig.layout.margin.t or 60
    bottom = 200
    plot_h = max(height - top - bottom, 100)
    below = lambda px: -px / plot_h  # paper coordinates of a point px under the plot

    play = dict(
        label="▶  Play",
        method="animate",
        args=[frame_names, dict(frame=dict(duration=duration, redraw=redraw),
                                transition=dict(duration=0), mode="immediate",
                                fromcurrent=False)],
    )
    pause = dict(
        label="❚❚  Pause",
        method="animate",
        args=[[None], dict(frame=dict(duration=0, redraw=False),
                           transition=dict(duration=0), mode="immediate")],
    )
    steps = [
        dict(method="animate", label=str(name),
             args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=redraw),
                                transition=dict(duration=0))])
        for name in frame_names
    ]
    fig.update_layout(
        margin=dict(b=bottom),
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0, xanchor="left", y=below(112), yanchor="top",
                          pad=dict(r=8, t=0), buttons=[play, pause],
                          bgcolor=SURFACE, bordercolor=MUTED, font=dict(size=15))],
        sliders=[dict(active=len(frame_names) - 1, x=0.22, len=0.78, y=below(96),
                      yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(prefix=prefix, font=dict(size=15, color=INK),
                                        xanchor="left"),
                      steps=steps, bgcolor=GRID, bordercolor=MUTED, tickcolor=GRID,
                      ticklen=4, font=dict(size=10, color=MUTED))],
    )
    return fig


def show(st, fig, key=None):
    """Draw a figure with our styling (not Streamlit's) and a tidy toolbar."""
    st.plotly_chart(fig, theme=None, key=key,
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
