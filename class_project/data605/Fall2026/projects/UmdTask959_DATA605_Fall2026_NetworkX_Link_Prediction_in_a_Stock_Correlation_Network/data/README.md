# Data

This directory holds the cached input data of the project: the S&P 500
constituents with their GICS sectors, and daily prices for a random sample of 100
of them over 2024 and 2025. `download_data.py` wrote these files once. The notebooks
read them and never download anything.

## Files

- `constituents.csv`: One row per S&P 500 constituent (503 rows)
  - **Source columns**: `Symbol`, `Security`, `GICS Sector`, `GICS Sub-Industry`
  - **Added columns**:
    - `yahoo_ticker`: The Yahoo Finance form of `Symbol`
    - `days_2024`, `days_2025`: Trading days with a price in each year
    - `eligible`: Whether the ticker has a price on every trading day of both years
    - `reason`: Why a ticker is not eligible, empty when it is
- `prices.csv`: Daily adjusted close prices of the 100 sampled tickers
  - One row per trading day (502 rows), one column per ticker, named by `Symbol`
  - No missing values
- `raw/prices_all.csv`: The same prices for all 503 constituents
  - Kept so that eligibility can be checked again without a new download
  - Not committed (`data/raw/` is in `.gitignore`)

## Sources

- **Constituents**: The `datasets/s-and-p-500-companies` list,
  [constituents.csv](https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv)
- **Prices**: Yahoo Finance through `yfinance` 1.8.0, from 2024-01-01 to 2025-12-31
- **Downloaded**: 2026-11-18

## Prices

- Prices are the `Close` column returned by `yfinance.download()` with
  `auto_adjust=True`, which is also its default
  - They are adjusted for splits and dividends, so log returns computed from them
    include dividends and have no jumps at split dates
  - With this setting `yfinance` returns no separate `Adj Close` column
- Share classes are written with a dot in the constituents list and with a dash on
  Yahoo Finance, e.g., `BRK.B` and `BRK-B`
  - The download uses the dash form; the columns of the price files use the dot
    form, so every price column matches a row of `constituents.csv`
  - The list has two share classes, `BRK.B` and `BF.B`. Both have complete
    history; neither was drawn into the sample

## Trading Calendar

- The reference calendar is the set of days on which at least 95% of the 503
  tickers have a price
  - 252 days in 2024 and 250 days in 2025
- Weekdays with no row are market closures shared by all tickers, e.g., national
  holidays and the market closure of 2025-01-09

## Replacement Rule

- A ticker is eligible when it has a price on every day of the reference calendar
  in 2024 and in 2025
- Tickers that fail this rule are excluded before the sample is drawn, so the
  sample cannot come up short
  - The next draw replaces each excluded ticker implicitly, because the sample is
    drawn only from the eligible tickers
- 8 of the 503 tickers were excluded, leaving 495 eligible:

  | Reason            | Count | Tickers                       |
  | :---------------- | :---- | :---------------------------- |
  | `no prices`       | 3     | FDXF, HONA, VYLR              |
  | `incomplete 2024` | 5     | GEV, Q, RDDT, SNDK, SOLV      |
  | `incomplete 2025` | 0     |                               |

- `no prices` marks tickers listed after 2025, e.g., spin-offs, and
  `incomplete 2024` marks tickers listed during 2024 or 2025
  - A ticker incomplete in both years is reported as `incomplete 2024`

## Sample

- 100 tickers drawn without replacement from the 495 eligible tickers, with
  `numpy.random.default_rng(42)`
- **Seed**: 42, set as `_SEED` in `download_data.py` and used only by the sampling
  step
- The eligible tickers are sorted before the draw, so the sample depends only on
  the set of eligible tickers and the seed
- Drawing again from `constituents.csv` with the same seed returns the same 100
  tickers

## Limitations

- The constituents list is the index as of the download date, not as of 2024
  - Companies that left the S&P 500 during 2024 or 2025 cannot be drawn, so the
    sample has survivorship bias: it holds only companies that were still in the
    index in 2026
  - The study asks how the correlation structure among these companies changes
    from one year to the next, so the bias narrows the population it speaks for
    rather than invalidating the comparison
- `yfinance` is an unofficial interface to Yahoo Finance; prices can be revised and
  tickers can change. The cached files fix the inputs of the analysis

## How to Regenerate

- Run from the project directory, inside the container:
  ```bash
  > ./docker_cmd.sh "pip install -q yfinance==1.8.0 && \
      cd /git_root/class_project/data605/Fall2026/projects/<project_dir> && \
      python download_data.py --overwrite"
  ```
  - Without `--overwrite`, the script refuses to run when `data/` exists
  - A new download uses the current constituents list and current Yahoo Finance
    data, so its files can differ from the committed ones
- The container runs as `root`, so give the outputs back to the host user:
  ```bash
  > ./docker_cmd.sh "chown -R 1000:1000 \
      /git_root/class_project/data605/Fall2026/projects/<project_dir>/data"
  ```
