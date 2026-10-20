"""
Smoke tests: every page and every preset button runs without an exception.

    pip install pytest
    pytest -q
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

PAGES = ["app_pages/home.py", "app_pages/rolling.py", "app_pages/line_fit.py",
         "app_pages/optimisers.py", "app_pages/overfitting.py", "app_pages/classify.py"]


def open_page(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = open_page(page)
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES[1:])
def test_every_preset(page):
    at = open_page(page)
    n = len(at.main.button)
    for i in range(n):
        at.main.button[i].click()
        at.run()
        assert not at.exception, (page, i, at.exception)


@pytest.mark.parametrize("surface", [0, 1, 2])
def test_each_surface(surface):
    at = open_page("app_pages/optimisers.py")
    box = at.sidebar.selectbox[0]
    box.set_value(box.options[surface])
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("model", [0, 1, 2, 3])
@pytest.mark.parametrize("data", [0, 1, 2])
def test_each_model_and_dataset(model, data):
    at = open_page("app_pages/classify.py")
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[data]).run()
    at.sidebar.selectbox[1].set_value(at.sidebar.selectbox[1].options[model]).run()
    assert not at.exception, at.exception
