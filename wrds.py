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
# ETF UNIVERSE DEFINITION
# ---------------------------------------------------------------------------
ETF_UNIVERSE = {
    "BlackRock (iShares)": [
        "IVV",   # iShares Core S&P 500 ETF
        "AGG",   # iShares Core U.S. Aggregate Bond ETF
        "EFA",   # iShares MSCI EAFE ETF
        "IWM",   # iShares Russell 2000 ETF
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
# 3. WRDS CONNECTION & DATA PULL
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

def fetch_crsp_etf_data(db: wrds.Connection, tickers: list, start_date: str = '2000-01-01', end_date: str = None) -> pd.DataFrame:
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
    
# ---------------------------------------------------------------------------
# 3. YFINANCE FALLBACK /  SUPPLEMENTAL DATA
# ---------------------------------------------------------------------------
def fetch_yfinance_data(tickers: list, 
                        start_date: str = "2000-01-01",
                        end_date: str = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance as fallback or supplement.
    Returns a wide DataFrame: index=date, columns=tickers.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
 
    print(f"Fetching Yahoo Finance data for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start_date, end=end_date, 
                      auto_adjust=True, progress=False)
 
    # Extract Close prices (adjusted via auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
 
    print(f"  → Shape: {prices.shape} (dates × tickers)\n")
    return prices
 
 
def fetch_yfinance_metadata(tickers: list) -> pd.DataFrame:
    """
    Pull key ETF metadata from yfinance: AUM, expense ratio, category, 
    avg volume, inception date. Used for universe validation.
    """
    records = []
    print("Fetching ETF metadata from Yahoo Finance...")
 
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker":          ticker,
                "long_name":       info.get("longName", ""),
                "category":        info.get("category", ""),
                "fund_family":     info.get("fundFamily", ""),
                "expense_ratio":   info.get("annualReportExpenseRatio", None),
                "aum_millions":    round(info.get("totalAssets", 0) / 1e6, 1),
                "avg_volume":      info.get("averageVolume", None),
                "avg_volume_10d":  info.get("averageVolume10days", None),
                "currency":        info.get("currency", ""),
                "exchange":        info.get("exchange", ""),
                "is_etn":          info.get("isEtn", False),
            })
        except Exception as e:
            print(f"  Warning: Could not fetch info for {ticker}: {e}")
 
    df = pd.DataFrame(records)
    print(f"  → Metadata retrieved for {len(df)} tickers.\n")
    return df

# ---------------------------------------------------------------------------
# 4. UNVERSE VALIDATION & LIQUDITY FILTERS
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
# 5. RETURN COMPUTATION
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
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main(wrds_username: str = None,
         use_wrds: bool = True,
         start_date: str = "2000-01-01"):
    """
    Main pipeline: connect → fetch prices → fetch metadata → validate → compute returns.

    Parameters
    ----------
    wrds_username : str
        Your WRDS username. Leave None for interactive prompt.
    use_wrds : bool
        If True, attempt WRDS connection first; falls back to yfinance.
        If False, use yfinance only (useful for testing without WRDS access).
    start_date : str
        Start date for historical data pull (YYYY-MM-DD).

    Returns
    -------
    prices          : pd.DataFrame  — adjusted close prices (wide format)
    simple_returns  : pd.DataFrame  — daily simple returns (validated ETFs)
    log_returns     : pd.DataFrame  — daily log returns (validated ETFs)
    validation      : pd.DataFrame  — universe validation results
    metadata        : pd.DataFrame  — ETF metadata (WRDS or yfinance)
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("QF623 Portfolio Management — ETF Universe Construction")
    print("=" * 60 + "\n")

    print("ETF Universe (top 5 managers × top 5 ETFs each):")
    for manager, tickers in ETF_UNIVERSE.items():
        print(f"  {manager}: {', '.join(tickers)}")
    print()

    # ── Step 1: Fetch price data ──────────────────────────────
    prices      = None
    db          = None
    crsp_df     = None

    if use_wrds:
        try:
            db = connect_wrds(wrds_username)
            crsp_df = fetch_crsp_etf_data(db, ALL_TICKERS, start_date, end_date)

            if crsp_df is not None and not crsp_df.empty:
                # Reconstruct adjusted price: abs(prc) / cum_factor_price
                prices = (
                    crsp_df
                    .assign(adj_price=lambda x: x["price"].abs() / x["cum_factor_price"])
                    .pivot_table(index="date", columns="ticker", values="adj_price")
                )
                print(f"  → Price matrix: {prices.shape[0]} dates × {prices.shape[1]} tickers\n")
            else:
                print("  Warning: CRSP returned empty data, falling back to yfinance for prices.\n")

        except Exception as e:
            print(f"  WRDS price fetch failed: {e}")
            print("  Falling back to yfinance for prices.\n")

    if prices is None:
        prices = fetch_yfinance_data(ALL_TICKERS, start_date, end_date)

    # ── Step 2: Fetch metadata ────────────────────────────────
    metadata = None

    if use_wrds and db is not None:
        try:
            characteristics = fetch_etf_characteristics(db, ALL_TICKERS)

            if not characteristics.empty:
                # Rename WRDS columns to match validate_universe() expectations
                metadata = characteristics.rename(columns={
                    "tna_latest": "aum_millions",
                    "fund_name":  "long_name",
                }).copy()
                # fund_summary has no volume data — set to None (filter will be skipped)
                metadata["avg_volume"] = None
                print("  Using WRDS metadata, skipping yfinance.\n")

        except Exception as e:
            print(f"  WRDS metadata fetch failed: {e}\n")

    if metadata is None:
        print("  WRDS metadata unavailable, falling back to yfinance...")
        metadata = fetch_yfinance_metadata(ALL_TICKERS)

    # ── Step 3: Close WRDS connection ────────────────────────
    if db is not None:
        db.close()
        print("  WRDS connection closed.\n")

    # ── Step 4: Validate universe ─────────────────────────────
    validation = validate_universe(prices, metadata)

    valid_tickers = validation[validation["PASSES_ALL"]]["ticker"].tolist()

    print("\nValidated ETF Universe:")
    display_cols = ["ticker", "long_name", "history_years", "aum_millions",
                    "avg_daily_volume", "PASSES_ALL"]
    # Only show columns that exist (avg_volume may be None if from WRDS)
    display_cols = [c for c in display_cols if c in validation.columns]
    print(validation[display_cols].to_string(index=False))

    if not valid_tickers:
        raise ValueError("No ETFs passed validation filters. Check your filter thresholds.")

    # ── Step 5: Compute returns ───────────────────────────────
    print()
    simple_returns, log_returns = compute_returns(prices, valid_tickers)

    # ── Step 6: Save outputs ──────────────────────────────────
    prices[valid_tickers].to_csv("data/etf_prices.csv")
    simple_returns.to_csv("data/etf_simple_returns.csv")
    log_returns.to_csv("data/etf_log_returns.csv")
    validation.to_csv("data/etf_validation.csv", index=False)
    metadata.to_csv("data/etf_metadata.csv", index=False)

    print("\nOutputs saved:")
    print("  etf_prices.csv")
    print("  etf_simple_returns.csv")
    print("  etf_log_returns.csv")
    print("  etf_validation.csv")
    print("  etf_metadata.csv")

    print(f"\n✓ Universe construction complete.")
    print(f"  Final universe: {len(valid_tickers)} ETFs → {valid_tickers}")

    return prices, simple_returns, log_returns, validation, metadata


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    WRDS_USERNAME = "ngchunyue"        # e.g. "jsmith"
    USE_WRDS      = True
    START_DATE    = "2000-01-01"

    prices, simple_returns, log_returns, validation, metadata = main(
        wrds_username=WRDS_USERNAME,
        use_wrds=USE_WRDS,
        start_date=START_DATE,
    )