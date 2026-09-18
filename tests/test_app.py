"""Smoke test for app.py using Streamlit's official AppTest framework.

This is deliberately a single, network-dependent smoke test (skipped
without connectivity, like the data_loader/crowding_score smoke tests):
it exists to catch import errors, wiring mistakes between modules, and
Streamlit-API misuse (e.g. a bad widget default) that unit tests on the
individual src/ modules can't see, not to re-verify the modeling logic
itself -- that's what the rest of the test suite is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_available(), reason="no network access in this environment")
def test_app_runs_without_exception_on_default_inputs():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 5
