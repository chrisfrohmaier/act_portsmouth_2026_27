"""Small page-layout helpers shared by every page."""

import inspect

import streamlit as st

_CSS = """
<style>
  /* Slightly larger text so the app reads well on a lecture-theatre projector */
  html { font-size: 17px; }
  h1 { font-weight: 650 !important; letter-spacing: -0.01em; font-size: 2.6rem !important; }
  h1 span[data-heading-text] { font-size: inherit !important; }
  .mlp-kicker { font-family: "IBM Plex Mono", ui-monospace, monospace;
                font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase;
                color: #b04618; margin-bottom: -0.6rem; }
  .mlp-lede { font-size: 1.12rem; color: #4a5060; max-width: 60rem; }
  .mlp-status { padding: 0.55rem 0.9rem; border-radius: 0.5rem; font-weight: 600;
                display: inline-block; margin: 0.2rem 0 0.6rem; }
  .mlp-good { background: #e3f1ea; color: #0f5c41; }
  .mlp-warn { background: #fbeadf; color: #8a3a12; }
  .mlp-info { background: #e6edf7; color: #22508f; }
</style>
"""


def header(kicker, title, lede):
    """Page title block: a small kicker line, the title and one sentence."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f'<p class="mlp-kicker">{kicker}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="mlp-lede">{lede}</p>', unsafe_allow_html=True)


def status(text, kind="info"):
    """A coloured status pill: kind is 'good', 'warn' or 'info'."""
    st.markdown(f'<span class="mlp-status mlp-{kind}">{text}</span>',
                unsafe_allow_html=True)


def show_code(*functions, label="Show the code behind this page"):
    """Print the real source of the functions this page uses."""
    with st.expander(label):
        st.code("\n\n".join(inspect.getsource(f) for f in functions),
                language="python")


def preset_row(presets, apply_fn, key_prefix):
    """A row of buttons; each one applies a named group of settings."""
    cols = st.columns(len(presets))
    for col, (name, settings) in zip(cols, presets.items()):
        col.button(name, key=f"{key_prefix}_{name}", on_click=apply_fn,
                   args=(settings,), width="stretch")
