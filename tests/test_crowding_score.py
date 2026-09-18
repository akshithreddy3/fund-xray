"""Unit tests for the crowding-score functions in src/drift_score.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.drift_score import (
    average_pairwise_cosine_similarity,
    build_peer_exposure_vectors,
    compute_crowding_score,
)

FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _make_betas(vector: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = np.tile(vector, (len(dates), 1))
    betas = pd.DataFrame(data, index=dates, columns=FACTOR_COLS)
    betas.insert(0, "const", 0.0)
    return betas


def test_average_pairwise_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, -1.0, 0.5, 0.0, 0.3])
    vectors = {"A": v, "B": v.copy(), "C": v * 2}  # same direction, different magnitude
    assert average_pairwise_cosine_similarity(vectors) == pytest.approx(1.0, abs=1e-10)


def test_average_pairwise_cosine_similarity_orthogonal_vectors_is_zero():
    vectors = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    assert average_pairwise_cosine_similarity(vectors) == pytest.approx(0.0, abs=1e-10)


def test_average_pairwise_cosine_similarity_matches_manual_three_way_average():
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])
    vectors = {"A": a, "B": b, "C": c}

    def cos_sim(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    expected = np.mean([cos_sim(a, b), cos_sim(a, c), cos_sim(b, c)])
    assert average_pairwise_cosine_similarity(vectors) == pytest.approx(expected)


def test_crowding_score_is_one_when_all_peers_identical():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    v = np.array([1.0, -0.2, 0.1, 0.0, 0.3, 0.05])
    peer_betas = {name: _make_betas(v, dates) for name in ["A", "B", "C"]}

    result = compute_crowding_score(peer_betas)

    assert np.allclose(result.crowding.to_numpy(), 1.0, atol=1e-8)
    assert (result.n_peers_used == 3).all()
    assert result.peer_tickers == ["A", "B", "C"]


def test_crowding_score_handles_ragged_peer_histories():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    full_history = _make_betas(v, dates)
    short_history = _make_betas(v, dates[50:])  # peer "B" only has data for the second half

    result = compute_crowding_score({"A": full_history, "B": short_history})

    assert len(result.crowding) == 50  # only dates where >=2 peers have data
    assert (result.n_peers_used == 2).all()


def test_crowding_score_requires_at_least_min_peers():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        compute_crowding_score({"A": _make_betas(v, dates)})


def test_crowding_score_raises_on_factor_column_mismatch():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a = _make_betas(v, dates)
    b = _make_betas(v, dates).rename(columns={"Mom": "Momentum"})
    with pytest.raises(ValueError):
        compute_crowding_score({"A": a, "B": b})


def test_crowding_score_carries_through_skipped_tickers():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    peer_betas = {"A": _make_betas(v, dates), "B": _make_betas(v, dates)}
    result = compute_crowding_score(peer_betas, skipped_tickers=["DELISTED_TICKER"])
    assert result.skipped_tickers == ["DELISTED_TICKER"]


def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=3)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_available(), reason="no network access in this environment")
def test_build_peer_exposure_vectors_smoke(tmp_path):
    loader = DataLoader(cache_dir=tmp_path)
    peer_betas, skipped = build_peer_exposure_vectors(
        loader, ["SPY", "NOT_A_REAL_TICKER_XYZ"], "2022-01-01", "2022-12-31"
    )

    assert "SPY" in peer_betas
    assert "NOT_A_REAL_TICKER_XYZ" in skipped
    assert not peer_betas["SPY"].empty
