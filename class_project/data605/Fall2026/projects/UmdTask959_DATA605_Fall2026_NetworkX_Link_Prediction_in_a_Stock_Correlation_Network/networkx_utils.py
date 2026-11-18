"""
Utility functions for the NetworkX notebooks: loading the cached stock data,
building the correlation network, and describing it.

Import as:

import class_project.data605.Fall2026.projects.UmdTask959_DATA605_Fall2026_NetworkX_Link_Prediction_in_a_Stock_Correlation_Network.networkx_utils as utils
"""

import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Data.
# #############################################################################


def load_constituents(data_dir: str) -> pd.DataFrame:
    """
    Load the cached S&P 500 constituents with their GICS sectors.

    :param data_dir: directory written by `download_data.py`
    :return: one row per constituent, with `Symbol`, `Security`, `GICS Sector`,
        and the eligibility columns
    """
    file_name = os.path.join(data_dir, "constituents.csv")
    hdbg.dassert_file_exists(file_name)
    # Keep empty `reason` cells as empty strings instead of NaN.
    constituents = pd.read_csv(file_name, keep_default_na=False)
    hdbg.dassert_is_subset(
        ["Symbol", "Security", "GICS Sector", "eligible"], constituents.columns
    )
    hdbg.dassert_no_duplicates(constituents["Symbol"].tolist())
    return constituents


def load_prices(data_dir: str) -> pd.DataFrame:
    """
    Load the cached adjusted close prices of the sampled tickers.

    :param data_dir: directory written by `download_data.py`
    :return: prices indexed by trading day, one column per ticker
    """
    file_name = os.path.join(data_dir, "prices.csv")
    hdbg.dassert_file_exists(file_name)
    prices = pd.read_csv(file_name, index_col="Date", parse_dates=True)
    hdbg.dassert(prices.notna().all().all(), "Cached prices have gaps")
    hdbg.dassert_lt(0, prices.min().min(), "Prices must be positive")
    return prices


# #############################################################################
# Returns and correlation.
# #############################################################################


def compute_log_returns(prices: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Compute the daily log returns of one calendar year.

    Returns are computed within the year only: the first trading day of the
    year has no return, so no information crosses from one year into the next.

    :param prices: prices indexed by trading day, one column per ticker
    :param year: calendar year to use
    :return: log returns $\\log(P_t / P_{t-1})$, one row per trading day after
        the first one of the year
    """
    year_prices = prices.loc[prices.index.year == year]
    hdbg.dassert_lt(1, len(year_prices), "No prices for year '%s'", year)
    returns = np.log(year_prices).diff().iloc[1:]
    hdbg.dassert(returns.notna().all().all(), "Returns have gaps")
    return returns


def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix of the returns.

    :param returns: log returns, one column per ticker
    :return: square correlation matrix indexed and labeled by ticker
    """
    corr = returns.corr(method="pearson")
    hdbg.dassert_eq(corr.shape, (returns.shape[1], returns.shape[1]))
    return corr


# #############################################################################
# Graph.
# #############################################################################


def build_correlation_graph(
    corr: pd.DataFrame, sectors: pd.Series, top_fraction: float
) -> nx.Graph:
    """
    Link the most correlated pairs of tickers in an undirected graph.

    Every ticker is a node with its GICS sector as the `sector` attribute,
    including tickers that end up with no edge. All unordered pairs are ranked
    by correlation, highest first; ties are broken by the first ticker name,
    then the second, both in alphabetical order, so the ranking is
    deterministic. The top `top_fraction` of the pairs become edges, which
    fixes the number of edges exactly. Each edge stores its correlation as
    `weight`, and the correlation of the last linked pair is stored as
    `graph.graph["threshold"]`.

    :param corr: square correlation matrix indexed and labeled by ticker
    :param sectors: GICS sector of each ticker, indexed by ticker
    :param top_fraction: share of all pairs that become edges, in (0, 1]
    :return: the correlation graph
    """
    hdbg.dassert_lt(0, top_fraction)
    hdbg.dassert_lte(top_fraction, 1)
    tickers = sorted(corr.columns)
    hdbg.dassert_is_subset(tickers, sectors.index)
    # List each unordered pair once, with the tickers in alphabetical order.
    rows, cols = np.triu_indices(len(tickers), k=1)
    pairs = pd.DataFrame(
        {
            "ticker_1": [tickers[i] for i in rows],
            "ticker_2": [tickers[j] for j in cols],
        }
    )
    values = corr.loc[tickers, tickers].to_numpy()
    pairs["correlation"] = values[rows, cols]
    # Rank by correlation, then by ticker names to break ties.
    pairs = pairs.sort_values(
        ["correlation", "ticker_1", "ticker_2"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    num_edges = round(top_fraction * len(pairs))
    edges = pairs.iloc[:num_edges]
    # Build the graph.
    graph = nx.Graph()
    for ticker in tickers:
        graph.add_node(ticker, sector=sectors[ticker])
    for ticker_1, ticker_2, correlation in edges.itertuples(index=False):
        graph.add_edge(ticker_1, ticker_2, weight=correlation)
    graph.graph["threshold"] = float(edges["correlation"].iloc[-1])
    _LOG.debug(
        "num_pairs=%s num_edges=%s threshold=%s",
        len(pairs),
        num_edges,
        graph.graph["threshold"],
    )
    return graph


def compute_graph_stats(graph: nx.Graph) -> pd.DataFrame:
    """
    Summarize the size, density, and degree distribution of a graph.

    :param graph: correlation graph from `build_correlation_graph()`
    :return: one row per statistic, with a single `value` column
    """
    degrees = pd.Series(dict(graph.degree()))
    components = list(nx.connected_components(graph))
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4),
        "correlation threshold": round(graph.graph["threshold"], 4),
        "min degree": int(degrees.min()),
        "median degree": float(degrees.median()),
        "mean degree": round(float(degrees.mean()), 2),
        "max degree": int(degrees.max()),
        "isolated nodes": nx.number_of_isolates(graph),
        "connected components": len(components),
        "largest component": max(len(component) for component in components),
    }
    stats_df = pd.DataFrame({"value": pd.Series(stats, dtype=object)})
    return stats_df


# #############################################################################
# Plots.
# #############################################################################


def plot_degree_histogram(
    graph: nx.Graph,
    *,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot the number of nodes for each degree value.

    The mean and median degree are drawn as vertical lines and the highest
    degree nodes are named in the title.

    :param graph: correlation graph from `build_correlation_graph()`
    :param figsize: figure size, by default the notebook's
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    degrees = pd.Series(dict(graph.degree()), name="degree")
    _, ax = plt.subplots(figsize=figsize)
    sns.histplot(x=degrees, discrete=True, ax=ax)
    ax.axvline(
        degrees.mean(),
        color="black",
        linestyle="--",
        label=f"mean {degrees.mean():.1f}",
    )
    ax.axvline(
        degrees.median(),
        color="gray",
        linestyle=":",
        label=f"median {degrees.median():.0f}",
    )
    # Name the hubs, ordered by degree and then by ticker.
    hubs = degrees.sort_index().sort_values(ascending=False, kind="mergesort")
    hub_text = ", ".join(f"{t} ({d})" for t, d in hubs.iloc[:3].items())
    ax.set_title(
        f"Degree distribution: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges\nHighest degree: {hub_text}"
    )
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.legend()
    plt.show()
