# Explainer

This document explains the project in full: what each component does, why it is
built the way it is, and what the results mean. The README covers how to build
and run it. The code carries only the comments a reader cannot do without; the
reasoning is here.

Each section is written in the same session as the code it describes.

## 1. Overview

What the project does and what it finds, in a few paragraphs, for a reader who has
not seen the code or the notebooks.

## 2. Architecture

How the files fit together: `download_data.py` writes the cache in `data/`;
`networkx_utils.py` holds every function; `networkx.API.ipynb` teaches the library
on small synthetic graphs; `networkx.example.ipynb` runs Project 1 on the cached
data; `test/` checks the utils and the Docker build. Why logic lives in the utils
and the notebooks only orchestrate.

## 3. API reference

The public functions of `networkx_utils.py`, grouped by section, one line each on
what they take and return.

## 4. Data

The tickers and their GICS sectors come from the S&P 500 constituents list
published by `datasets/s-and-p-500-companies`. Daily prices for 2024 and 2025 come
from Yahoo Finance through `yfinance` 1.8.0. The prices are the `Close` column
with `auto_adjust=True`, so they are adjusted for splits and dividends, and a log
return computed from them includes the dividend and has no jump on a split date.

`download_data.py` downloads everything once, on 2026-11-18, and the results are
committed in `data/`. The notebooks read only these files. There are three reasons
for caching instead of downloading at run time: the project has to run locally in
the class container, `yfinance` is an unofficial interface that is rate limited
and whose data can be revised, and the notebooks have to run within the time of
the video. `data/README.md` records the details needed to regenerate the files.

The sample of 100 tickers is drawn only from tickers with complete history. The
script downloads prices for all 503 constituents, builds a trading calendar from
the days on which at least 95% of them have a price (252 days in 2024 and 250 in
2025), and keeps a ticker only if it has a price on every one of those days. Eight
tickers fail: three have no prices at all because they were listed after 2025
(FDXF, HONA, VYLR), and five were listed during 2024 or 2025 (GEV, Q, RDDT, SNDK,
SOLV). The 100 are then drawn without replacement from the remaining 495, with the
seed fixed at 42. Excluding before drawing means the sample can never come up
short, and sorting the pool before the draw makes the sample depend only on the
set of eligible tickers and the seed. Drawing twice from the cached file gives the
same 100 tickers.

The constituents list writes share classes with a dot (`BRK.B`) and Yahoo Finance
with a dash (`BRK-B`). The script converts the symbol only for the download and
renames the price columns back afterward, so every price column matches a row of
the constituents file and its sector. The list has two share classes, `BRK.B` and
`BF.B`; both have complete history, and neither was drawn.

The constituents list is the index as of the download date, not as of 2024. A
company that left the S&P 500 during 2024 or 2025 cannot be drawn, so the sample
has survivorship bias: it contains only companies that were still in the index in
late 2026. The study compares the correlation structure of the same companies in
two consecutive years, so the bias narrows the population the results describe
rather than invalidating the comparison. A point-in-time constituents list would
remove it and is outside the scope of this project.

## 5. Graph construction

The graph for a year starts from the daily log returns
$r_t = \log(P_t / P_{t-1})$ of the 100 tickers. Returns are computed within the
year only: the first trading day has no return, so 2024 gives 251 returns and no
information from one year enters the graph of the other. Log returns are used
because they add up over time and are closer to symmetric than simple returns,
which suits a correlation measure.

The Pearson correlation of every pair of tickers gives a 100 by 100 matrix and
4,950 distinct pairs. In 2024 the average pairwise correlation is 0.19 and the
largest is 0.97. The 10% of the pairs with the highest correlation become edges,
which is exactly 495 edges. Pairs are ranked by signed correlation, as the project
description asks: a strong negative correlation does not create an edge. Negative
correlations are weak in this sample, the most negative being -0.29, so no pair
that a ranking by absolute value would add is lost. Ties in the ranking are broken
by ticker name, so the graph is identical on every run.

Ranking pairs and keeping a fixed share, instead of cutting at a fixed correlation
value, gives both years a graph with the same number of edges even if the overall
level of correlation changes between them. That matters for the prediction task:
with a fixed cutoff, a year with higher market-wide correlation would gain edges
everywhere, and the labels would reflect the market level rather than the
structure. The cost is that the correlation cutoff itself moves; in 2024 it is
0.37. The share of 10% follows the project description, and how the results
depend on it is a stretch item.

Every ticker is a node, including those with no edge, and each node carries its
GICS sector as the `sector` attribute. Keeping isolated tickers matters later:
they are part of the candidate pairs, and the density and degree statistics would
be misleading without them.

The statistics of the 2024 graph:

| Statistic | Value |
| :-------- | :---- |
| Nodes | 100 |
| Edges | 495 |
| Density | 0.10 |
| Correlation threshold | 0.3701 |
| Degree: min, median, mean, max | 0, 6, 9.9, 35 |
| Isolated nodes | 11 |
| Connected components | 13 |
| Largest component | 87 nodes |

The density is 0.10 by construction: 495 of 4,950 possible pairs are linked
because the method links exactly that share. It says nothing about the data, and
neither do the edge count or the mean degree of 9.9, which follow from the same
choice. The degree distribution is the informative statistic. It is strongly right
skewed: the median degree of 6 is well below the mean, 25 tickers have at most one
edge and 11 have none, while 22 tickers have 20 or more and the highest, IEX, has
35. A few highly connected tickers, mostly industrials, materials, and financials,
account for a large share of the edges. Apart from the 11 isolated tickers and one
separate pair (COR and MCK, two drug distributors), every ticker belongs to a
single component of 87. Sector explains part of the structure but not most of it:
30% of the edges join two tickers of the same sector.

This shape has a direct consequence for the scores in the next sections. The
Jaccard coefficient and the Adamic-Adar index count shared neighbors, so they are
zero for every pair that involves an isolated or nearly isolated ticker, while
preferential attachment, the product of the two degrees, favors pairs of hubs.

## 6. The prediction problem

How the year two graph defines labels, which pairs are candidates, the class
balance that results, and why scores are computed on year one only.

## 7. Link prediction scores and communities

What each score measures and what structural assumption it encodes: Jaccard
coefficient, Adamic-Adar index, preferential attachment. Then community
detection, the NMI comparison against sectors, and the same-community flag as an
additional score.

## 8. Evaluation

The three baselines and why each is there. Why both ROC-AUC and average precision
are reported. What the numbers say, including the case where the correlation
baseline wins.

## 9. Visualization

What the figure shows, the encoding choices for color and size, and what it shows
that the numbers do not.

## 10. Library comparison

The igraph and SNAP comparison: the algorithm chosen, graph sizes, how time and
memory were measured, the results, and any installation problem and the
substitute used.

## 11. Use of the course skills

Required by the project description. One entry per use, logged when it happens:
the skill (`notebook.*`, `coding.*`, `docker.*`, `readme_dir.*`, or an
instruction-file run), what it produced, and what I reviewed, changed, or
rejected by hand.

| Date | Skill | Applied to | What it produced | What I changed |
| --- | --- | --- | --- | --- |
| 2026-11-18 | `docker.use_standard_style` | Docker files of the project directory | A comparison with `class_project/project_template`: all 19 template files present with the same content and modes, except `IMAGE_NAME=umd_project_networkx` in `docker_name.sh` and the build log `docker_build.version.log`, which is regenerated by each build and ignored. No change needed. | Nothing; I confirmed both differences are intended. |

## 12. Design decisions and trade-offs

The choices I would be asked to defend in review, each with the alternatives I
considered and why I rejected them.

## 13. Limitations

What this project does not establish, and what would be needed to establish it.
