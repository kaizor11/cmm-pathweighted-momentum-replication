import os
import sys
import pandas as pd
import polars as pl
from sqlalchemy import create_engine
from dotenv import load_dotenv
import argparse
import tempfile
import time
from constants import RAW_DATA_PATH, DATA_PATH, RENAME, FEATURES

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

def fetch_data():
    """
    Fetches data from WIND SQL database saved to OUT_PATH_RAW
    Runtime ~12min
    """

    uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    query = """
        SELECT 
            S_INFO_WINDCODE, TRADE_DT, S_DQ_ADJOPEN, S_DQ_ADJHIGH, S_DQ_ADJLOW, S_DQ_ADJCLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_PCTCHANGE
        FROM ASHAREEODPRICES
        WHERE
            TRADE_DT >= '20150101' AND TRADE_DT < '20260731'
        ;
    """

    # data fetch
    engine = create_engine(uri)
    print(f"[fetch] fetching data")

    df = pl.read_database(query, connection=engine)
    df.write_parquet(RAW_DATA_PATH, compression="snappy")
    engine.dispose()

    # wind data is not automatically sorted
    print("[fetch] sorting raw parquet by S_INFO_WINDCODE, TRADE_DT (streaming)...")
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=os.path.dirname(RAW_DATA_PATH))
    os.close(fd) # close fd because not needed

    try:
        (
        pl.scan_parquet(RAW_DATA_PATH)
        .sort(["S_INFO_WINDCODE", "TRADE_DT"])
        .sink_parquet(tmp)
        )
        os.replace(tmp, RAW_DATA_PATH)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[fetch] raw parquet sorted -> {RAW_DATA_PATH}")

def compute_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    # DB prices arrive as decimal[38,4]; rolling/window math needs floats.
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        lf = lf.with_columns(pl.col(c).cast(pl.Float64))

    ret = pl.col("ret")
    mkt_ret = pl.col("mkt_ret")

    # daily returns
    lf = lf.with_columns(
        (pl.col("close").pct_change()
         .over("stock")).alias("ret")
    )

    # ---- gap-bridge nulling (Bug #1 fix, part 1 of 2) ----
    # A daily return is a SUSPENSION GAP-BRIDGE when the PRIOR trading day had
    # volume == 0 (the stock was halted). close.pct_change() then compresses the
    # whole suspension (weeks/months/years) into one fake "daily" return -- e.g.
    # 000670.SZ +488% bridging a 2.4y halt, or 000520.SZ +738%. We null that
    # return to 0.0 (we do NOT delete the row, and we never touch `close`):
    #   - the panel date grid stays intact (no join / look-ahead shift),
    #   - the NEXT day's return (= close[t+1]/close[t]-1) is computed from real
    #     prices, so it is completely untouched by this nulling,
    #   - ret_z no longer spikes to the +/-5 clip bound and collapses the
    #     softmax in model.py,
    #   - every other ret-derived feature for that day is de-corrupted too.
    # Already-suspended days (volume==0 with carried-forward close) have ret==0
    # naturally, so nulling them is a harmless no-op; only the resumption-day
    # bridge is actually changed. fill_null(1) keeps the first row per stock
    # (prev_volume is null there) from being nulled.
    prev_volume = pl.col("volume").shift(1).over("stock").fill_null(1)
    lf = lf.with_columns(
        pl.when(prev_volume == 0).then(0.0).otherwise(pl.col("ret")).alias("ret")
    )

    # equal-weight market return per date (feeds beta + idio-vol).
    mkt = lf.group_by("date").agg(pl.col("ret").mean().alias("mkt_ret"))
    lf = lf.join(mkt, on="date", how="left")
    # join does not guarantee row order -> re-sort so every rolling window is time-ordered
    lf = lf.sort(["stock", "date"])

    # ---- volatility (rolling std of daily return) ----
    f_vol_60 = ret.rolling_std(60, min_samples=60, ddof=1).over("stock")
    f_vol_120 = ret.rolling_std(120, min_samples=120, ddof=1).over("stock")

    # ---- amihud illiquidity: |ret| / (amount in thousand), 60d mean ----
    ret0 = pl.col("ret").fill_null(0.0)
    amihud = pl.when(pl.col("amount") > 0).then(
        ret0.abs() / (pl.col("amount") / 1e3)
    ).otherwise(0.0)
    f_amihud_60 = amihud.rolling_mean(60, min_samples=60).over("stock")

    # ---- beta vs market (252d): cov(ret, mkt) / var(mkt) ----
    cov252 = ((ret * mkt_ret).rolling_mean(252, min_samples=252).over("stock")
              - ret.rolling_mean(252, min_samples=252).over("stock")
              * mkt_ret.rolling_mean(252, min_samples=252).over("stock"))
    var_m252 = mkt_ret.rolling_var(252, min_samples=252, ddof=1).over("stock")
    f_beta_252 = pl.when(var_m252 > 0).then(cov252 / var_m252).otherwise(None)

    # ---- log amount + amount trend ----
    la60 = (pl.col("amount").rolling_mean(60, min_samples=60).over("stock") + 1.0).log()
    la252 = (pl.col("amount").rolling_mean(252, min_samples=252).over("stock") + 1.0).log()
    f_log_amount_60 = la60
    f_log_amount_252 = la252
    f_amount_trend = la60 - la252

    # ---- mean absolute return (60d) ----
    f_mean_abs_ret_60 = ret.abs().rolling_mean(60, min_samples=60).over("stock")

    # ---- return shape ----
    f_ret_21d = (pl.col("close") / pl.col("close").shift(21).over("stock") - 1.0)
    f_ret_126d = (pl.col("close") / pl.col("close").shift(126).over("stock") - 1.0)
    f_max_ret_21d = ret.rolling_max(21, min_samples=21).over("stock")
    f_price_to_high_252 = pl.col("close") / pl.col("high").rolling_max(252, min_samples=252).over("stock")

    # ---- distribution: skew & kurtosis (manual Fisher-Pearson, matching pandas) ----
    """
    Because we are using a rolling window, we don't use the direct formulas for Skew and Kurtosis as it is computationally intensive (we have to calculate the mean at each step). 
    Instead, we use raw moments, which are the running averages for r, r2, r3, r4. We then calculate Skew and Kurt using a algebraic-equivalent formulas using raw moments.
    Note: because the sample is only 60, we use a correction factor
    """
    n = 60
    m1 = ret.rolling_mean(n, min_samples=n).over("stock")
    m2r = (ret * ret).rolling_mean(n, min_samples=n).over("stock")
    m3r = (ret * ret * ret).rolling_mean(n, min_samples=n).over("stock")
    m4r = (ret * ret * ret * ret).rolling_mean(n, min_samples=n).over("stock")
    m2 = m2r - m1 * m1
    m3 = m3r - 3 * m1 * m2r + 2 * m1 ** 3
    m4 = m4r - 4 * m1 * m3r + 6 * m1 * m1 * m2r - 3 * m1 ** 4
    f_skew_60 = pl.when(m2 > 1e-12).then(
        (((n * (n - 1)) ** 0.5) / (n - 2)) * (m3 / (m2 ** 1.5))
    ).otherwise(None)
    f_kurt_60 = pl.when(m2 > 1e-12).then(
        ((n - 1) * (n + 1) / ((n - 2) * (n - 3))) * (m4 / (m2 * m2))
        - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    ).otherwise(-3.0)

    # ---- idiosyncratic vol (60d std of residual vs market) ----
    cov60 = ((ret * mkt_ret).rolling_mean(60, min_samples=60).over("stock")
             - ret.rolling_mean(60, min_samples=60).over("stock")
             * mkt_ret.rolling_mean(60, min_samples=60).over("stock"))
    var_m60 = mkt_ret.rolling_var(60, min_samples=60, ddof=1).over("stock")
    beta60 = pl.when(var_m60 > 0).then(cov60 / var_m60).otherwise(0.0)
    resid = ret - beta60 * mkt_ret
    f_idio_vol_60 = resid.rolling_std(60, min_samples=60, ddof=1).over("stock")

    # ---- zero-return ratio (60d) ----
    zr = (
        pl.when(pl.col("ret").is_null()).then(0.0)
        .when(pl.col("ret").abs() < 1e-6).then(1.0)
        .otherwise(0.0)
    )
    f_zero_ret_ratio_60 = zr.rolling_mean(60, min_samples=60).over("stock")

    feat_exprs = {
        "f_vol_60": f_vol_60, "f_vol_120": f_vol_120, "f_amihud_60": f_amihud_60,
        "f_beta_252": f_beta_252, "f_log_amount_60": f_log_amount_60,
        "f_log_amount_252": f_log_amount_252, "f_amount_trend": f_amount_trend,
        "f_mean_abs_ret_60": f_mean_abs_ret_60, "f_ret_21d": f_ret_21d,
        "f_ret_126d": f_ret_126d, "f_max_ret_21d": f_max_ret_21d,
        "f_price_to_high_252": f_price_to_high_252, "f_skew_60": f_skew_60,
        "f_kurt_60": f_kurt_60, "f_idio_vol_60": f_idio_vol_60,
        "f_zero_ret_ratio_60": f_zero_ret_ratio_60,
    }
    lf = lf.with_columns([e.alias(name) for name, e in feat_exprs.items()])

    # fill NaN and null
    lf = lf.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0).alias(c) for c in FEATURES + ["ret"]])

    # z-score standardization all cols except ohlcv, then winsorize to [-5, 5].
    z_cols = FEATURES + ["ret"]
    z_exprs = []
    for c in z_cols:
        mu = pl.col(c).mean().over("date")
        sd = pl.col(c).std(ddof=0).over("date")
        z_exprs.append(
            pl.when(sd > 0)
            .then(((pl.col(c) - mu) / sd).clip(-5.0, 5.0))
            .otherwise(0.0)
            .alias(c + "_z")
        )
    lf = lf.with_columns(z_exprs)

    # ---- tradable universe flag (mirrors backtest filter, requires full feature history) ----
    lf = lf.with_columns(
        (
            (pl.col("f_beta_252") != 0)                                   # >=252d listed -> all 16 features populated
            & (pl.col("amount").rolling_mean(20, min_samples=20).over("stock") >= 20_000) # 20d avg amount >= 20M RMB (WIND amount is 千元)
            & (pl.col("volume") > 0)                                      # not suspended that day
        ).alias("tradable")
    )


    z_features = [c + "_z" for c in z_cols]
    keep = ["date","stock","open","high","low","close","volume","amount","ret"] + z_features + ["tradable"]
    return lf.select(keep)

def main():
    start_time = time.time()

    # check missing env var
    missing = [k for k in ("DB_USER","DB_PASS","DB_HOST","DB_PORT","DB_NAME") if not os.getenv(k)]
    if missing:
        sys.exit(f"[data] missing env vars: {', '.join(missing)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="fetch + sort raw data")
    parser.add_argument("--limit-stocks", type=int, default=0,
                        help="if N>0, process only the first N stocks (smoke test)")
    args = parser.parse_args()

    # fetching new data from WIND
    if args.fetch:
        fetch_data()
    else:
        if not os.path.exists(RAW_DATA_PATH):
            sys.exit("[data] raw file missing — run: python data.py --fetch")

    # rename col, sort, remove BSE
    lf = pl.scan_parquet(RAW_DATA_PATH).rename(RENAME).sort(["stock", "date"])
    lf = lf.filter(~pl.col("stock").str.ends_with(".BJ"))

    # smoke test: limit to first N stocks
    if args.limit_stocks:
        keep_stocks = (lf.select("stock").unique().sort("stock")
                        .limit(args.limit_stocks).collect()["stock"].to_list())
        lf = lf.filter(pl.col("stock").is_in(keep_stocks))

    print(f"[data] computing features")
    out = compute_features(lf)

    if args.limit_stocks:
        df = out.collect()
        print(f"[features] smoke test -> {df.height:,} rows x {df.width} cols")
        print(df.head(3))
    else:
        out.sink_parquet(DATA_PATH, compression="snappy")
        n = pl.scan_parquet(DATA_PATH).select(pl.len()).collect().item()
        print(f"[features] saved -> {DATA_PATH}  ({n:,} rows)")

    print(f"Runtime: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()

# Runtime: ~150s without --fetch