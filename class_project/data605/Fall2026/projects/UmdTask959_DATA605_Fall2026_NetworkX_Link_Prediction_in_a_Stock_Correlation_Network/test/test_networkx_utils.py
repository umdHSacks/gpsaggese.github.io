"""
Import as:

import class_project.data605.Fall2026.projects.UmdTask959_DATA605_Fall2026_NetworkX_Link_Prediction_in_a_Stock_Correlation_Network.test.test_networkx_utils as ttneut
"""

import logging
from typing import Tuple

import networkx as nx
import pandas as pd

import class_project.data605.Fall2026.projects.UmdTask959_DATA605_Fall2026_NetworkX_Link_Prediction_in_a_Stock_Correlation_Network.networkx_utils as nxutils
import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


def _get_corr_and_sectors() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a 5-ticker correlation matrix and the sector of each ticker.

    The 10 pairs, highest correlation first:
    - `A-B` 0.9 and `C-D` 0.8
    - `A-C` and `B-D` tie at 0.5; the tie is broken by ticker name, so `A-C`
      ranks first
    - Every other pair is below 0.5, and `E` is weakly correlated with all

    :return: correlation matrix and sectors, both indexed by ticker
    """
    tickers = ["A", "B", "C", "D", "E"]
    corr = pd.DataFrame(0.1, index=tickers, columns=tickers)
    for ticker_1, ticker_2, value in [
        ("A", "B", 0.9),
        ("C", "D", 0.8),
        ("A", "C", 0.5),
        ("B", "D", 0.5),
        ("A", "D", 0.3),
        ("B", "C", 0.2),
    ]:
        corr.loc[ticker_1, ticker_2] = value
        corr.loc[ticker_2, ticker_1] = value
    for ticker in tickers:
        corr.loc[ticker, ticker] = 1.0
    sectors = pd.Series(
        {
            "A": "Energy",
            "B": "Energy",
            "C": "Utilities",
            "D": "Utilities",
            "E": "Financials",
        }
    )
    return corr, sectors


# #############################################################################
# Test_build_correlation_graph
# #############################################################################


class Test_build_correlation_graph(hunitest.TestCase):
    """
    Test `networkx_utils.build_correlation_graph()`.
    """

    def test1(self) -> None:
        """
        Test that the top 30% of 10 pairs give 3 edges, with ties broken by
        ticker name.
        """
        # Prepare inputs.
        corr, sectors = _get_corr_and_sectors()
        top_fraction = 0.3
        # Prepare outputs.
        expected = "[('A', 'B', 0.9), ('A', 'C', 0.5), ('C', 'D', 0.8)]"
        # Run test.
        graph = nxutils.build_correlation_graph(corr, sectors, top_fraction)
        # Check outputs.
        # Write each edge with its tickers in alphabetical order.
        actual = sorted(
            (*sorted((u, v)), weight)
            for u, v, weight in graph.edges(data="weight")
        )
        self.assert_equal(str(actual), expected)
        self.assertEqual(graph.graph["threshold"], 0.5)

    def test2(self) -> None:
        """
        Test that every node carries its GICS sector as the `sector` attribute.
        """
        # Prepare inputs.
        corr, sectors = _get_corr_and_sectors()
        top_fraction = 0.3
        # Prepare outputs.
        expected = (
            "{'A': 'Energy', 'B': 'Energy', 'C': 'Utilities', "
            "'D': 'Utilities', 'E': 'Financials'}"
        )
        # Run test.
        graph = nxutils.build_correlation_graph(corr, sectors, top_fraction)
        # Check outputs.
        actual = dict(sorted(graph.nodes(data="sector")))
        self.assert_equal(str(actual), expected)

    def test3(self) -> None:
        """
        Test that a ticker with no edge stays in the graph as an isolated node.
        """
        # Prepare inputs.
        corr, sectors = _get_corr_and_sectors()
        top_fraction = 0.3
        # Prepare outputs.
        expected = "nodes=5 isolates=['E']"
        # Run test.
        graph = nxutils.build_correlation_graph(corr, sectors, top_fraction)
        # Check outputs.
        actual = (
            f"nodes={graph.number_of_nodes()} "
            f"isolates={sorted(nx.isolates(graph))}"
        )
        self.assert_equal(actual, expected)
