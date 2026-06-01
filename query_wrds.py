"""
Top 5 US ETF fund managers by AUM:
1. BlackRock (iShares)
2. Vanguard
3. State Street Global Advisors (SPDR)
4. Invesco
5. Charles Schwab

For each manager, we select their top 5 most liquid / well-known ETFs covering different asset classes (equity, fixed income, sector, international).
"""

import wrds
import pandas as pd
import  numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. ETF UNIVERSE DEFINITION
# ---------------------------------------------------------------------------
ETF_UNIVERSE = {
    "BlackRock (iShares)": [
        "IVV",   # iShares Core S&P 500 ETF
        "AGG",   # iShares Core U.S. Aggregate Bond ETF
        "EFA",   # iShares MSCI EAFE ETF
        "IWM",   # iShares Russell 2015 ETF
        "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond ETF
    ],
    "Vanguard": [
        "VOO",   # Vanguard S&P 500 ETF
        "VTI",   # Vanguard Total Stock Market ETF
        "BND",   # Vanguard Total Bond Market ETF
        "VEA",   # Vanguard FTSE Developed Markets ETF
        "VWO",   # Vanguard FTSE Emerging Markets ETF
    ],
    "State Street (SPDR)": [
        "SPY",   # SPDR S&P 500 ETF Trust
        "GLD",   # SPDR Gold Shares
        "XLF",   # Financial Select Sector SPDR Fund
        "XLE",   # Energy Select Sector SPDR Fund
        "XLK",   # Technology Select Sector SPDR Fund
    ],
    "Invesco": [
        "QQQ",   # Invesco QQQ Trust (Nasdaq-100)
        "RSP",   # Invesco S&P 500 Equal Weight ETF
        "BKLN",  # Invesco Senior Loan ETF
        "PGX",   # Invesco Preferred ETF
        "EMLC",  # VanEck EM Local Currency Bond (proxy for Invesco EM exposure)
    ],
    "Charles Schwab": [
        "SCHB",  # Schwab U.S. Broad Market ETF
        "SCHX",  # Schwab U.S. Large-Cap ETF
        "SCHF",  # Schwab International Equity ETF
        "SCHD",  # Schwab U.S. Dividend Equity ETF
        "SCHI",  # Schwab 5-10 Year Corporate Bond ETF
    ],
}
 
# Flatten to list of all tickers
ALL_TICKERS = [ticker for tickers in ETF_UNIVERSE.values() for ticker in tickers]
 
# ---------------------------------------------------------------------------
# 2. WRDS CONNECTION & DATA PULL
# ---------------------------------------------------------------------------
def connect_wrds(username: str = None) -> wrds.Connection:
    """
    Connect to WRDS. Pass your WRDS username or leave None for interactive prompt.
    WRDS will use ~/.pgpass if configured, otherwise prompts for password.
    """
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username=username)
    print("Connected successfully.\n")
    return db

def fetch_crsp_etf_data(db: wrds.Connection, tickers: list, start_date: str = '2015-01-01', end_date: str = None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"[DEBUG] Fetching CRSP daily data for {len(tickers)} ETFs: {start_date} -> {end_date}")

    query = """
        SELECT
            d.date,
            n.ticker,
            d.prc        AS price,
            d.ret        AS daily_return,
            d.shrout     AS shares_out,
            d.cfacpr     AS cum_factor_price,
            d.cfacshr    AS cum_factor_shr,
            ABS(d.prc) * d.vol AS dollar_volume
        FROM crsp.dsf AS d
        JOIN crsp.dsenames AS n
            ON d.permno = n.permno
            AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, CURRENT_DATE)
        WHERE n.ticker = ANY(%(tickers)s)
          AND n.shrcd IN (73)
          AND d.date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY n.ticker, d.date
    """

    params = {
        "tickers":    tickers,
        "start_date": start_date,
        "end_date":   end_date,
    }

    df = db.raw_sql(query, params=params, date_cols=["date"])

    print(f"    -> Retrieved {len(df):,} rows across {df['ticker'].nunique()} tickers.\n")
    return df


def fetch_etf_characteristics(db: wrds.Connection, tickers: list) -> pd.DataFrame:
    print("Fetching ETF characteristics from CRSP fund summary...")

    # Convert list to PostgreSQL array literal: {'IVV','AGG','EFA',...}
    ticker_array = "{" + ",".join(tickers) + "}"

    query = f"""
        SELECT DISTINCT ON (m.ticker)
            m.ticker,
            m.fund_name,
            m.mgmt_name,
            m.et_flag,
            m.index_fund_flag,
            m.first_offer_dt,
            m.end_dt,
            m.dead_flag,
            f.tna_latest,
            f.tna_latest_dt,
            f.nav_latest,
            f.nav_latest_dt,
            f.per_com,
            f.per_bond,
            f.per_cash,
            f.caldt
        FROM crsp.fund_names AS m
        LEFT JOIN crsp.fund_summary AS f ON m.crsp_fundno = f.crsp_fundno
        WHERE m.ticker = ANY('{ticker_array}'::text[])
          AND m.et_flag = 'F'
          AND (m.dead_flag IS NULL OR m.dead_flag != 'D')
        ORDER BY m.ticker, f.caldt DESC
    """

    try:
        df = db.raw_sql(query, date_cols=["first_offer_dt", "end_dt",
                                           "tna_latest_dt", "nav_latest_dt", "caldt"])
        print(f"  → Retrieved characteristics for {df['ticker'].nunique()} ETFs.\n")
        return df
    except Exception as e:
        print(f"  Warning: Could not fetch fund characteristics: {e}")
        print("  Falling back to yfinance for metadata.\n")
        return pd.DataFrame()
    

def fetch_avg_volume(db: wrds.Connection, tickers: list,
                     lookback_days: int = 252) -> pd.DataFrame:
    """
    Compute average daily share and dollar volume from crsp.dsf directly.
    Used for liquidity filter in validate_universe().
    """
    ticker_array = "{" + ",".join(tickers) + "}"
    cutoff_date  = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    query = f"""
        SELECT
            n.ticker,
            AVG(d.vol)               AS avg_daily_volume,
            AVG(ABS(d.prc) * d.vol)  AS avg_dollar_volume
        FROM crsp.dsf AS d
        JOIN crsp.dsenames AS n
            ON d.permno = n.permno
            AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, CURRENT_DATE)
        WHERE n.ticker = ANY('{ticker_array}'::text[])
          AND n.shrcd IN (73)
          AND d.date >= '{cutoff_date}'
        GROUP BY n.ticker
        ORDER BY n.ticker
    """

    df = db.raw_sql(query)
    print(f"  → Avg volume computed for {len(df)} ETFs over last {lookback_days} trading days.\n")
    return df

# ---------------------------------------------------------------------------
# 3. UNVERSE VALIDATION & LIQUDITY FILTERS
# ---------------------------------------------------------------------------
def validate_universe(prices: pd.DataFrame,
                       metadata: pd.DataFrame,
                       min_history_years: float = 5.0,
                       min_aum_millions: float = 500.0,
                       min_avg_volume: int = 500_000,
                       min_data_completeness: float = 0.95) -> pd.DataFrame:

    print("=" * 60)
    print("UNIVERSE VALIDATION")
    print("=" * 60)

    results = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()

        # History check
        if len(series) < 2:
            history_years = 0
        else:
            history_years = (series.index[-1] - series.index[0]).days / 365.25

        # Data completeness
        total_days     = len(prices)
        available_days = series.count()
        completeness   = available_days / total_days if total_days > 0 else 0

        # Metadata checks — use .get() with fallbacks for missing columns
        meta_row = metadata[metadata["ticker"] == ticker]
        if not meta_row.empty:
            aum     = meta_row["aum_millions"].values[0] or 0
            name    = meta_row["long_name"].values[0] if "long_name" in meta_row.columns else ticker

            # avg_volume: present in yfinance metadata, absent in WRDS
            avg_vol = meta_row["avg_volume"].values[0] if "avg_volume" in meta_row.columns else None

            # is_etn: present in yfinance metadata, absent in WRDS
            is_etn  = meta_row["is_etn"].values[0] if "is_etn" in meta_row.columns else False
        else:
            aum, avg_vol, name, is_etn = 0, None, ticker, False

        # Leverage/inverse keyword filter
        leverage_keywords = [
            "leveraged", "2x", "3x", "ultra", "inverse", "short",
            "bear", "bull 2", "bull 3", "daily"
        ]
        is_leveraged = any(kw in name.lower() for kw in leverage_keywords)

        # Pass/fail — skip volume filter if no volume data available
        pass_history      = history_years >= min_history_years
        pass_aum          = aum >= min_aum_millions
        pass_volume       = avg_vol >= min_avg_volume if avg_vol is not None else True
        pass_leverage     = not is_leveraged and not is_etn
        pass_completeness = completeness >= min_data_completeness
        passes_all        = all([pass_history, pass_aum, pass_volume,
                                 pass_leverage, pass_completeness])

        results.append({
            "ticker":            ticker,
            "long_name":         name,
            "history_years":     round(history_years, 1),
            "aum_millions":      aum,
            "avg_daily_volume":  avg_vol,
            "completeness_pct":  round(completeness * 100, 1),
            "is_leveraged":      is_leveraged,
            "is_etn":            is_etn,
            "pass_history":      pass_history,
            "pass_aum":          pass_aum,
            "pass_volume":       pass_volume,
            "pass_leverage":     pass_leverage,
            "pass_completeness": pass_completeness,
            "PASSES_ALL":        passes_all,
        })

    validation_df = pd.DataFrame(results).sort_values("ticker")

    passed = validation_df[validation_df["PASSES_ALL"]]
    failed = validation_df[~validation_df["PASSES_ALL"]]

    print(f"\nFilter criteria:")
    print(f"  Min history  : {min_history_years} years")
    print(f"  Min AUM      : ${min_aum_millions:,.0f}M")
    print(f"  Min volume   : {min_avg_volume:,} shares/day (skipped if no data)")
    print(f"  Completeness : {min_data_completeness*100:.0f}%")
    print(f"\nResult: {len(passed)}/{len(validation_df)} ETFs passed all filters.")

    if not failed.empty:
        print(f"Failed tickers: {list(failed['ticker'])}")

    return validation_df

# ---------------------------------------------------------------------------
# 4. RETURN COMPUTATION
# ---------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame, 
                    valid_tickers: list) -> pd.DataFrame:
    """
    Compute daily log returns and simple returns for validated ETFs.
    """
    prices_clean = prices[valid_tickers].copy()
 
    # Forward-fill gaps (weekends already excluded), then drop leading NaNs
    prices_clean = prices_clean.ffill().dropna(how="all")
 
    simple_returns = prices_clean.pct_change().dropna(how="all")
    log_returns    = np.log(prices_clean / prices_clean.shift(1)).dropna(how="all")
 
    print(f"Returns computed: {simple_returns.shape[0]} trading days × "
          f"{simple_returns.shape[1]} ETFs")
    print(f"Date range: {simple_returns.index[0].date()} → "
          f"{simple_returns.index[-1].date()}\n")
 
    return simple_returns, log_returns

# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main(wrds_username: str = None,
         use_wrds: bool = True,
         start_date: str = "2015-01-01"):
    """
    Main pipeline: connect → fetch prices → fetch metadata → validate → compute returns.

    Parameters
    ----------
    wrds_username : str
        Your WRDS username. Leave None for interactive prompt.
    use_wrds : bool
        If True, attempt WRDS connection. No yfinance fallback.
    start_date : str
        Start date for historical data pull (YYYY-MM-DD).

    Returns
    -------
    prices          : pd.DataFrame  — adjusted close prices (wide format)
    simple_returns  : pd.DataFrame  — daily simple returns (validated ETFs)
    log_returns     : pd.DataFrame  — daily log returns (validated ETFs)
    validation      : pd.DataFrame  — universe validation results
    metadata        : pd.DataFrame  — ETF metadata from WRDS
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("QF623 Portfolio Management — ETF Universe Construction")
    print("=" * 60 + "\n")

    print("ETF Universe (top 5 managers × top 5 ETFs each):")
    for manager, tickers in ETF_UNIVERSE.items():
        print(f"  {manager}: {', '.join(tickers)}")
    print()

    # ── Steps 1 & 2: Fetch all data from WRDS ────────────────
    prices   = None
    metadata = None
    db       = None

    try:
        db = connect_wrds(wrds_username)

        # ── Step 1: Prices ────────────────────────────────────
        crsp_df = fetch_crsp_etf_data(db, ALL_TICKERS, start_date, end_date)

        if crsp_df is None or crsp_df.empty:
            raise RuntimeError("CRSP returned empty price data.")

        prices = (
            crsp_df
            .assign(adj_price=lambda x: x["price"].abs() / x["cum_factor_price"])
            .pivot_table(index="date", columns="ticker", values="adj_price")
        )
        print(f"  → Price matrix: {prices.shape[0]} dates × {prices.shape[1]} tickers\n")

        # ── Step 2: Metadata ──────────────────────────────────
        characteristics = fetch_etf_characteristics(db, ALL_TICKERS)
        avg_volume      = fetch_avg_volume(db, ALL_TICKERS)

        if characteristics.empty:
            raise RuntimeError("CRSP returned empty ETF characteristics.")

        metadata = (
            characteristics
            .rename(columns={"tna_latest": "aum_millions", "fund_name": "long_name"})
            .merge(avg_volume, on="ticker", how="left")
        )

    except Exception as e:
        raise RuntimeError(f"WRDS fetch failed: {e}")

    finally:
        # ── Step 3: Always close WRDS connection ──────────────
        if db is not None:
            db.close()
            print("  WRDS connection closed.\n")

    # ── Step 4: Validate universe ─────────────────────────────
    validation    = validate_universe(prices, metadata)
    valid_tickers = validation[validation["PASSES_ALL"]]["ticker"].tolist()

    if not valid_tickers:
        raise ValueError("No ETFs passed validation filters. Check your filter thresholds.")

    print("\nValidated ETF Universe:")
    display_cols = ["ticker", "long_name", "history_years", "aum_millions",
                    "avg_daily_volume", "PASSES_ALL"]
    display_cols = [c for c in display_cols if c in validation.columns]
    print(validation[display_cols].to_string(index=False))

    # ── Step 5: Compute returns ───────────────────────────────
    print()
    simple_returns, log_returns = compute_returns(prices, valid_tickers)

    # ── Step 6: Save outputs ──────────────────────────────────
    import os
    os.makedirs("data", exist_ok=True)

    prices[valid_tickers].to_csv("data/etf_prices.csv")
    simple_returns.to_csv("data/etf_simple_returns.csv")
    log_returns.to_csv("data/etf_log_returns.csv")
    validation.to_csv("data/etf_validation.csv", index=False)
    metadata.to_csv("data/etf_metadata.csv", index=False)

    print("\nOutputs saved:")
    print("  data/etf_prices.csv")
    print("  data/etf_simple_returns.csv")
    print("  data/etf_log_returns.csv")
    print("  data/etf_validation.csv")
    print("  data/etf_metadata.csv")

    print(f"\n✓ Universe construction complete.")
    print(f"  Final universe: {len(valid_tickers)} ETFs → {valid_tickers}")

    return prices, simple_returns, log_returns, validation, metadata


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    WRDS_USERNAME = "ngchunyue"        # e.g. "jsmith"
    USE_WRDS      = True
    START_DATE    = "2015-01-01"

    prices, simple_returns, log_returns, validation, metadata = main(
        wrds_username=WRDS_USERNAME,
        use_wrds=USE_WRDS,
        start_date=START_DATE,
    )