#!/usr/bin/env python

"""
Download the S&P 500 constituents and two years of daily prices, then cache a
random sample of 100 tickers with complete price history.

The script runs once. The notebooks read the cached files in `data/` and never
call it.

# Usage Example

- Download the data into `data/`:
> download_data.py

- Show what would be downloaded and written, without doing it:
> download_data.py --dry_run

Import as:

import class_project.data605.Fall2026.projects.UmdTask959_DATA605_Fall2026_NetworkX_Link_Prediction_in_a_Stock_Correlation_Network.download_data as dowdata
"""

import argparse
import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hparser as hparser
import helpers.hselect_input_output as hseinout

_LOG = logging.getLogger(__name__)

# Seed of the only random step, the draw of the sample in `_sample_tickers()`.
_SEED = 42
_NUM_TICKERS = 100
# Year one is 2024 and year two is 2025; yfinance treats the end date as
# exclusive.
_START_DATE = "2024-01-01"
_END_DATE = "2026-01-01"
_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)


# #############################################################################
# Tickers.
# #############################################################################


def _to_yahoo_ticker(symbol: str) -> str:
    """
    Convert a symbol from the constituents list to the Yahoo Finance form.

    The constituents list writes share classes with a dot and Yahoo Finance
    with a dash, e.g., `BRK.B` -> `BRK-B`.

    :param symbol: ticker as written in the constituents list
    :return: ticker as written by Yahoo Finance
    """
    yahoo_ticker = symbol.replace(".", "-")
    return yahoo_ticker


def _load_constituents(url: str) -> pd.DataFrame:
    """
    Load the S&P 500 constituents and add the Yahoo Finance form of each ticker.

    The original `Symbol` is kept next to `yahoo_ticker`, so prices downloaded
    under the Yahoo Finance form can be matched back to their GICS sector.

    :param url: CSV with `Symbol`, `Security`, `GICS Sector`, and
        `GICS Sub-Industry` columns
    :return: one row per constituent, with an extra `yahoo_ticker` column
    """
    _LOG.info("Loading constituents from '%s'", url)
    df = pd.read_csv(url)
    columns = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    hdbg.dassert_is_subset(columns, df.columns)
    df = df[columns].copy()
    hdbg.dassert_no_duplicates(df["Symbol"].tolist())
    hdbg.dassert(
        df["GICS Sector"].notna().all(), "Some constituents have no GICS sector"
    )
    df["yahoo_ticker"] = df["Symbol"].map(_to_yahoo_ticker)
    # Two symbols mapping to the same Yahoo Finance ticker would merge columns.
    hdbg.dassert_no_duplicates(df["yahoo_ticker"].tolist())
    _LOG.info("Loaded '%s' constituents", len(df))
    return df


# #############################################################################
# Prices.
# #############################################################################


def _download_prices(yahoo_tickers: List[str]) -> pd.DataFrame:
    """
    Download daily adjusted close prices for 2024 and 2025.

    Tickers are fetched in batches of 100 with a pause between batches to stay
    under the Yahoo Finance rate limit. A ticker with no price at all, e.g.,
    one listed after 2025, is kept as an all-NaN column and later flagged as
    ineligible. A batch that comes back empty or without any price, or more
    than 10 tickers without prices in total, stops the script instead of
    being retried.

    :param yahoo_tickers: tickers in the Yahoo Finance form
    :return: adjusted close prices indexed by date, one column per ticker
    """
    batches = [
        yahoo_tickers[start : start + 100]
        for start in range(0, len(yahoo_tickers), 100)
    ]
    closes = []
    for num, batch in enumerate(batches, start=1):
        _LOG.info(
            "Downloading batch '%s' of '%s' with '%s' tickers",
            num,
            len(batches),
            len(batch),
        )
        # With `auto_adjust=True`, `Close` is adjusted for splits and
        # dividends and yfinance returns no separate `Adj Close` column.
        data = yf.download(
            batch,
            start=_START_DATE,
            end=_END_DATE,
            auto_adjust=True,
            actions=False,
            group_by="column",
            multi_level_index=True,
            progress=False,
        )
        hdbg.dassert(not data.empty, "Batch '%s' returned no data", num)
        # yfinance gives a failed ticker a placeholder frame that includes an
        # `Adj Close` column of NaN; tickers with prices have no such column
        # with `auto_adjust=True`.
        adj_close = data.loc[:, data.columns.get_level_values(0) == "Adj Close"]
        hdbg.dassert(
            adj_close.isna().all().all(),
            "Batch '%s' has unadjusted 'Adj Close' prices",
            num,
        )
        close = data["Close"]
        hdbg.dassert_set_eq(close.columns, batch)
        # A batch where no ticker has any price is a download failure, e.g., a
        # rate limit, not a set of tickers without history.
        hdbg.dassert(
            close.notna().any().any(), "Batch '%s' has no prices at all", num
        )
        closes.append(close)
        if num < len(batches):
            time.sleep(2)
    prices = pd.concat(closes, axis=1)
    prices.index.name = "Date"
    # A few tickers without prices are expected, e.g., recent listings; many
    # point to a download problem.
    no_prices = prices.columns[prices.isna().all()].tolist()
    _LOG.info("'%s' tickers have no prices: '%s'", len(no_prices), no_prices)
    hdbg.dassert_lte(
        len(no_prices), 10, "Too many tickers without prices: '%s'", no_prices
    )
    return prices


def _get_reference_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Get the trading days on which at least 95% of the tickers have a price.

    Market holidays have no rows at all, so they never count as missing, and a
    date present for only a few tickers does not enter the calendar.

    :param prices: adjusted close prices, one column per ticker
    :return: dates of the reference calendar
    """
    coverage = prices.notna().mean(axis=1)
    calendar = prices.index[coverage >= 0.95]
    _LOG.info(
        "Reference calendar: '%s' of '%s' dates", len(calendar), len(prices)
    )
    return calendar


def _find_complete_tickers(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Count the calendar days each ticker has a price on, per year.

    A ticker is eligible when it has a price on every calendar day of both
    years. Tickers listed after the start of 2024, delisted, or with gaps are
    not eligible. The reason is `no prices`, `incomplete 2024`, or
    `incomplete 2025`, and empty for an eligible ticker.

    :param prices: adjusted close prices, one column per ticker
    :param calendar: dates of the reference calendar
    :return: one row per ticker with `days_2024`, `days_2025`, `eligible`,
        and `reason`
    """
    df = pd.DataFrame(index=prices.columns)
    reason = pd.Series("", index=prices.columns)
    for year in range(int(_START_DATE[:4]), int(_END_DATE[:4])):
        days = calendar[calendar.year == year]
        hdbg.dassert_lt(0, len(days), "No calendar days in '%s'", year)
        df[f"days_{year}"] = prices.loc[days].notna().sum()
        # Keep the first reason found, so 2024 wins over 2025.
        incomplete = (df[f"days_{year}"] < len(days)) & (reason == "")
        reason[incomplete] = f"incomplete {year}"
        _LOG.info("Year '%s' has '%s' calendar days", year, len(days))
    # A ticker without any price is reported as such, not as incomplete.
    reason[prices.isna().all()] = "no prices"
    df["eligible"] = reason == ""
    df["reason"] = reason
    _LOG.info(
        "Excluded tickers per reason: '%s'",
        reason[reason != ""].value_counts().to_dict(),
    )
    return df


# #############################################################################
# Sampling.
# #############################################################################


def _sample_tickers(eligible: List[str], num: int, seed: int) -> List[str]:
    """
    Draw a reproducible random sample of tickers without replacement.

    The pool is sorted first, so the sample depends only on the set of eligible
    tickers and the seed, not on their order in the constituents list.

    :param eligible: tickers with complete history in both years
    :param num: number of tickers to draw
    :param seed: seed of the random generator
    :return: the sampled tickers, sorted
    """
    hdbg.dassert_lte(
        num,
        len(eligible),
        "Only '%s' eligible tickers, '%s' needed",
        len(eligible),
        num,
    )
    rng = np.random.default_rng(seed)
    pool = sorted(eligible)
    sample = sorted(rng.choice(pool, size=num, replace=False).tolist())
    return sample


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    hseinout.add_dst_dir_arg(
        parser, dst_dir_required=False, dst_dir_default="data"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    if args.dry_run:
        _LOG.warning(
            "[DRY_RUN] Would download constituents from '%s', prices from "
            "'%s' to '%s', and write '%s' tickers drawn with seed '%s' to '%s'",
            _CONSTITUENTS_URL,
            _START_DATE,
            _END_DATE,
            _NUM_TICKERS,
            _SEED,
            args.dst_dir,
        )
    else:
        # Refuse to run when the destination exists, unless `--overwrite`.
        dst_dir = hseinout.parse_dst_dir_arg(args)
        # Download.
        constituents = _load_constituents(_CONSTITUENTS_URL)
        prices_all = _download_prices(constituents["yahoo_ticker"].tolist())
        # Rename the columns back to the constituents-list symbols, so each
        # price column matches the GICS sector of its ticker.
        to_symbol = dict(
            zip(constituents["yahoo_ticker"], constituents["Symbol"])
        )
        prices_all = prices_all.rename(columns=to_symbol)
        # Keep only the reference calendar and flag complete tickers.
        calendar = _get_reference_calendar(prices_all)
        prices_all = prices_all.loc[calendar]
        coverage = _find_complete_tickers(prices_all, calendar)
        constituents = constituents.join(coverage, on="Symbol")
        excluded = constituents.loc[~constituents["eligible"], "Symbol"].tolist()
        _LOG.info(
            "Excluded '%s' tickers without complete history: '%s'",
            len(excluded),
            excluded,
        )
        # Sample only after the incomplete tickers are excluded, so the sample
        # cannot come up short.
        eligible = constituents.loc[constituents["eligible"], "Symbol"].tolist()
        sample = _sample_tickers(eligible, _NUM_TICKERS, _SEED)
        prices = prices_all[sample]
        hdbg.dassert(prices.notna().all().all(), "Sampled prices have gaps")
        # Write.
        raw_dir = os.path.join(dst_dir, "raw")
        hio.create_dir(raw_dir, incremental=True)
        constituents.to_csv(
            os.path.join(dst_dir, "constituents.csv"), index=False
        )
        prices.to_csv(os.path.join(dst_dir, "prices.csv"))
        prices_all.to_csv(os.path.join(raw_dir, "prices_all.csv"))
        _LOG.info(
            "Wrote '%s' constituents and prices for '%s' tickers on '%s' days "
            "to '%s'",
            len(constituents),
            prices.shape[1],
            prices.shape[0],
            dst_dir,
        )


if __name__ == "__main__":
    _main(_parse())
