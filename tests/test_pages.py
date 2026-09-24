"""Smoke tests for the multi-page app's additional pages, using Streamlit's
AppTest framework -- same pattern and rationale as tests/test_app.py:
these catch import errors and wiring mistakes between modules and
Streamlit-API misuse, not modeling correctness (covered elsewhere).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_available(), reason="no network access in this environment")
def test_portfolio_page_runs_without_exception_on_default_inputs():
    at = AppTest.from_file(str(PAGES_DIR / "1_Portfolio_X-Ray.py"), default_timeout=180)
    at.run()
    assert not at.exception


@pytest.mark.skipif(not _network_available(), reason="no network access in this environment")
def test_fund_vs_fund_page_runs_without_exception_on_default_inputs():
    at = AppTest.from_file(str(PAGES_DIR / "2_Fund_vs_Fund.py"), default_timeout=180)
    at.run()
    assert not at.exception
