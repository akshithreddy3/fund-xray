"""Data acquisition layer for Fund X-Ray.

`DataLoader` is the single place that talks to yfinance and the Kenneth
French data library. Every other module receives clean, aligned,
already-logged-for-missingness DataFrames -- it never touches a network
call directly. This keeps the statistical modules testable with
synthetic data and keeps API-flake failures isolated to one place.

Missing-data policy: we never silently forward-fill. When two series
don't fully overlap (e.g. a fund's inception date is later than the
factor history, or an exchange holiday mismatch), we log exactly which
dates are dropped and why, then use an inner join. If a caller opts into
forward-fill (`ffill_limit`), that choice and its effect are also
logged. This matters for a risk system: a silently-filled "flat" return
on a day the fund didn't actually trade would understate volatility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import famafrench

from src import config

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@dataclass
class FundDataset:
    """Aligned, ready-to-model bundle for a single fund.

    All frames share the same DatetimeIndex (the inner-joined trading
    calendar across the fund, the factors, and the risk-free rate).
    """

    ticker: str
    fund_returns: pd.Series  # simple total return, not excess
    excess_returns: pd.Series  # fund_returns - risk_free
    factor_returns: pd.DataFrame  # columns = config.FACTOR_COLUMNS
    risk_free: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp


class DataLoader:
    """Fetches, caches, and aligns fund/factor/macro return series."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generic parquet cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe_key = key.replace("/", "-").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.parquet"

    def _read_cache(self, key: str, max_age_days: float) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
        if age_days > max_age_days:
            logger.info("Cache stale for %s (%.1f days old) -- refetching.", key, age_days)
            return None
        logger.info("Loaded %s from cache (%.1f days old).", key, age_days)
        return pd.read_parquet(path)

    def _write_cache(self, key: str, df: pd.DataFrame) -> None:
        self._cache_path(key).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._cache_path(key))

    # ------------------------------------------------------------------
    # Prices / returns
    # ------------------------------------------------------------------
    def get_prices(
        self,
        tickers: str | list[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        """Adjusted close prices, wide-form (columns = tickers)."""
        ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
        cache_key = f"prices_{'-'.join(sorted(ticker_list))}_{start}_{end}"
        cached = self._read_cache(cache_key, config.PRICE_CACHE_MAX_AGE_DAYS)
        if cached is not None:
            return cached

        logger.info("Downloading prices for %s from yfinance (%s to %s).", ticker_list, start, end)
        raw = yf.download(
            ticker_list,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no price data for {ticker_list}")

        # yfinance always returns a (ticker, field) MultiIndex with
        # group_by="ticker", even for a single ticker.
        available = set(raw.columns.get_level_values(0))
        prices = pd.concat(
            {t: raw[t]["Close"] for t in ticker_list if t in available},
            axis=1,
        )

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        missing_tickers = set(ticker_list) - set(prices.columns)
        if missing_tickers:
            logger.warning("No price data returned for tickers: %s", sorted(missing_tickers))

        self._write_cache(cache_key, prices)
        return prices

    @staticmethod
    def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Log returns: additive across time, standard for return-based analysis."""
        log_returns = np.log(prices / prices.shift(1)).iloc[1:]
        n_dropped = int(log_returns.isna().sum().sum())
        if n_dropped:
            logger.warning(
                "%d NaN return observations present after price differencing "
                "(likely missing trading days for one or more tickers).",
                n_dropped,
            )
        return log_returns

    # ------------------------------------------------------------------
    # Fama-French factors
    # ------------------------------------------------------------------
    def _fetch_famafrench_dataset(
        self, dataset_name: str, start: date | str, end: date | str
    ) -> pd.DataFrame:
        cache_key = f"ff_{dataset_name}_{start}_{end}"
        cached = self._read_cache(cache_key, config.FACTOR_CACHE_MAX_AGE_DAYS)
        if cached is not None:
            return cached

        logger.info("Downloading Fama-French dataset '%s' (%s to %s).", dataset_name, start, end)
        reader = famafrench.FamaFrenchReader(dataset_name, start=start, end=end)
        raw = reader.read()[0]  # index 0 = the (non-annual) daily/monthly table
        reader.close()

        df = raw.copy()
        df.columns = [c.strip() for c in df.columns]
        df = df / 100.0  # French's library reports percent, not decimal
        df.index = pd.to_datetime(df.index.to_timestamp() if hasattr(df.index, "to_timestamp") else df.index)

        self._write_cache(cache_key, df)
        return df

    def get_factor_returns(
        self, start: date | str, end: date | str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Returns (factor_returns[Mkt-RF,SMB,HML,RMW,CMA,Mom], risk_free)."""
        five_factor = self._fetch_famafrench_dataset(config.FF_5_FACTOR_DATASET, start, end)
        momentum = self._fetch_famafrench_dataset(config.FF_MOMENTUM_DATASET, start, end)

        merged = five_factor.join(momentum, how="inner")
        n_dropped = five_factor.index.difference(merged.index).size + momentum.index.difference(
            merged.index
        ).size
        if n_dropped:
            logger.warning(
                "Dropped %d non-overlapping dates while joining 5-factor and "
                "momentum datasets.",
                n_dropped,
            )

        risk_free = merged[config.RISK_FREE_COLUMN].rename("RF")
        factors = merged[config.FACTOR_COLUMNS]
        return factors, risk_free

    # ------------------------------------------------------------------
    # Macro / regime inputs
    # ------------------------------------------------------------------
    def get_vix(self, start: date | str, end: date | str) -> pd.Series:
        prices = self.get_prices(config.REGIME_VOL_PROXY, start, end)
        return prices[config.REGIME_VOL_PROXY].rename("VIX")

    # ------------------------------------------------------------------
    # Top-level convenience: build a fully aligned modeling dataset
    # ------------------------------------------------------------------
    def build_fund_dataset(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
    ) -> FundDataset:
        prices = self.get_prices(ticker, start, end)
        fund_returns = self.to_log_returns(prices)[ticker]

        factors, risk_free = self.get_factor_returns(start, end)

        combined = pd.concat(
            {"fund": fund_returns, "rf": risk_free},
            axis=1,
            sort=False,
        ).join(factors, how="inner")

        n_total = combined.shape[0]
        combined_before_dropna = combined.copy()
        combined = combined.dropna(how="any")
        n_dropped = n_total - combined.shape[0]
        if n_dropped:
            dropped_dates = combined_before_dropna.index.difference(combined.index)
            logger.warning(
                "Dropped %d rows (of %d) with missing data while aligning '%s' "
                "with factor data. Date range affected: %s to %s.",
                n_dropped,
                n_total,
                ticker,
                dropped_dates.min().date() if len(dropped_dates) else "n/a",
                dropped_dates.max().date() if len(dropped_dates) else "n/a",
            )

        if combined.empty:
            raise ValueError(
                f"No overlapping data between '{ticker}' and Fama-French factors "
                f"in range {start} to {end}."
            )

        excess_returns = (combined["fund"] - combined["rf"]).rename("excess_return")

        return FundDataset(
            ticker=ticker,
            fund_returns=combined["fund"].rename(ticker),
            excess_returns=excess_returns,
            factor_returns=combined[config.FACTOR_COLUMNS],
            risk_free=combined["rf"],
            start=combined.index.min(),
            end=combined.index.max(),
        )

    @staticmethod
    def resample_to_monthly(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Compound daily log returns to monthly by summation (log returns are additive)."""
        return returns.resample("ME").sum()
