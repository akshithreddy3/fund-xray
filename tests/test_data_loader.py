"""Unit tests for src/data_loader.py.

The pure transformation functions (log returns, monthly resampling) are
tested against synthetic data so they run offline and fast. A single
network-backed smoke test exercises the real yfinance / Fama-French
integration and is skipped automatically if there's no network access,
so the suite stays green in sandboxed CI environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_to_log_returns_matches_manual_calculation():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 108.9]},
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    returns = DataLoader.to_log_returns(prices)

    expected = np.log(prices["A"] / prices["A"].shift(1)).iloc[1:]
    pd.testing.assert_series_equal(returns["A"], expected, check_names=False)
    assert len(returns) == len(prices) - 1


def test_to_log_returns_flags_missing_data(caplog):
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    with caplog.at_level("WARNING"):
        returns = DataLoader.to_log_returns(prices)

    assert returns["A"].isna().sum() == 2
    assert "NaN return observations" in caplog.text


def test_resample_to_monthly_sums_log_returns():
    daily = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-02-01", "2024-02-02"]),
    )
    monthly = DataLoader.resample_to_monthly(daily)

    assert len(monthly) == 2
    assert monthly.iloc[0] == pytest.approx(0.03)
    assert monthly.iloc[1] == pytest.approx(0.02)


def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_available(), reason="no network access in this environment")
def test_build_fund_dataset_smoke(tmp_path):
    loader = DataLoader(cache_dir=tmp_path)
    dataset = loader.build_fund_dataset("SPY", "2022-01-01", "2022-06-01")

    assert not dataset.fund_returns.empty
    assert list(dataset.factor_returns.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    assert dataset.fund_returns.index.equals(dataset.factor_returns.index)
    assert not dataset.factor_returns.isna().any().any()
