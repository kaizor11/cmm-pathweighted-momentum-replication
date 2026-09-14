"""
backtest.py  --  A-share backtest engine for the CMM baseline factor.

Consumes:
    output/ashare_2015-2016.parquet   (OHLCV + 16 features)
    output/signals.parquet            (month-end cmm_signal, sm_signal per stock)

Backtest rules (mirror A-share):
* Rebalance monthly: signal formed at month-end t, trade at the CLOSE of the first
  trading day of the next month, hold to next rebalance.
* Universe filter (month-end): listed > 120 trading days, 20-day average turnover
  amount >= 20,000,000 RMB, not suspended (volume > 0).
* Costs: commission 0.025% per side, stamp duty 0.05% on sells, slippage 0.10% per
  side; lot size = 100 shares; T+1 (we hold >=21 days so no intraday sell of fresh buys);
  +/-10% daily price limit (trades skipped if a name is at limit on the trade date).
* Portfolio: equal-weight Top-K by signal (baseline = no market-cap / industry
  neutralization). Long-only.

Outputs (in output/):
    backtest_metrics.csv     per-strategy performance
    backtest_ic.csv          monthly IC / RankIC for CMM vs SM
    equity_curve.png         CMM/SM top-K vs equal-weight universes (CMM & SM)
    binned_performance.png   CMM signal-sorted quantile bins (G1..G{BINS}),
                             bar + line of each bin's performance (growth of 1.0
                             or cumulative return, per --axis)
    backtest_summary.txt     printed + saved narrative
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from constants import OUT_DIR, IN_DIR, DATA_PATH, SIG_PATH
import time


# A-share cost model
COMMISSION = 0.00025
STAMP = 0.0005          # sells only
SLIPPAGE = 0.001
LOT = 100
PRICE_LIMIT = 0.10
MIN_LIST_DAYS = 120
MIN_AMOUNT_20D = 20_000  # 20-day avg turnover >= 20M RMB. NOTE: amount is WIND
                         # S_DQ_AMOUNT in 千元 (thousand RMB), so 20M RMB = 20_000 千元.
INIT_CAPITAL = 1_000_000_000.0

TOP_K = 200               # top k stocks ranked by signal (to buy)
BINS = 10                 # signal-sorted quantile buckets for binned_performance.png.
                          # G1 = lowest-signal bucket, G{BINS} = highest-signal bucket.
TEST_LO = "2020-01-01"
TEST_HI = "2026-07-01"   # month-ends up to 2026-06


# --------------------------------------------------------------------------- #
def load_data(sig_path):
    """
    Purpose: Load the signal file and the matching OHLCV/feature parquet, restrict
        the price data to the stocks actually present in the signal file (so a
        subsampled signals_s<N>.parquet only materializes its N stocks via
        predicate pushdown), and pivot everything into wide date x stock matrices
        aligned to the union of trading dates.
    Used by: main()
    Returns: tuple (dates, stocks, close_p, ret_p, amount_p, vol_p, sig)
        - dates: pd.DatetimeIndex of sorted unique trading dates
        - stocks: list of stock codes present in the data
        - close_p / ret_p / amount_p / vol_p: pd.DataFrame pivoted to
          (index=date, columns=stock), forward-filled close for delisted names
        - sig: the raw signal DataFrame (columns include stock/date/cmm_signal/sm_signal)
    """
    sig = pd.read_parquet(sig_path)
    # Restrict the backtest universe to the stocks actually present in the signal
    # file: a subsampled run (signals_s<N>.parquet) must only see its N stocks, and
    # the predicate pushdown avoids materializing the full 11.7M rows for those runs.
    sig_stocks = sorted(sig["stock"].unique())
    df = pd.read_parquet(DATA_PATH, filters=[("stock", "in", sig_stocks)])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")   # string "20150105" -> Timestamp
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    stocks = list(df["stock"].unique())
    close_p = df.pivot(index="date", columns="stock", values="close").reindex(dates).ffill() # fill NaN for delisted stocks
    ret_p = df.pivot(index="date", columns="stock", values="ret").reindex(dates)
    amount_p = df.pivot(index="date", columns="stock", values="amount").reindex(dates)
    vol_p = df.pivot(index="date", columns="stock", values="volume").reindex(dates)
    
    return dates, stocks, close_p, ret_p, amount_p, vol_p, sig


def tradable_at(date, dates, idx, stocks, amount_p, vol_p):
    """
    Purpose: Apply the A-share universe filter at a single month-end `date`:
        require at least MIN_LIST_DAYS of listing history, a 20-day average
        turnover amount >= MIN_AMOUNT_20D, and positive volume (not suspended).
    Used by: main() (via the tradable_fn closure), ic_stats(), build_bin_schedules()
    Returns: list[str] of stock codes passing the filter; [] if too early in
        the sample or none pass.
    """
    p = idx[date]
    # listing age proxy: require >= MIN_LIST_DAYS of history (all synthetic names
    # are listed from 2015, so this only guards the very start of the sample)
    if p < MIN_LIST_DAYS:
        return []
    amt20 = amount_p.iloc[p - 20:p].mean(axis=0)          # (stocks,)
    vol = vol_p.iloc[p]                                    # (stocks,)
    mask = (amt20 >= MIN_AMOUNT_20D) & (vol > 0)
    return [s for s in stocks if mask.get(s, False)]


def run_portfolio(dates, idx, price_p, ret_p, schedule, tag):
    """
    Purpose: Simulate a long-only equal-weight portfolio through a given rebalance
        schedule under the full A-share cost model. On each scheduled trade date it
        rebalances to the target stock list (skipping names at daily price limits)
        and otherwise marks the portfolio to market using `price_p`.
    Used by: main() (for the top-K / universe strategies and each signal bin)
    Returns: tuple (nav_s, avg_turnover)
        - nav_s: pd.Series of total portfolio value indexed by trading date
        - avg_turnover: float, mean per-rebalance turnover ratio (0.0 if no rebalance)
    """
    if not schedule:
        return pd.Series(dtype=float), 0.0
    nav = {}
    shares = {}
    cash = INIT_CAPITAL
    last_total = INIT_CAPITAL
    turnovers = []
    # iterate all trading days from first trade date to end
    first_td = schedule[0][0]
    sd = idx[first_td]
    sched_by_date = {d: t for d, t in schedule}
    # precompute price-limit up/down at each date for tradable check
    for i in range(sd, len(dates)):
        d = dates[i]
        prices = price_p.iloc[i]
        if d in sched_by_date:
            targets = sched_by_date[d]
            # skip names at limit-up (cannot buy) / limit-down (cannot sell)
            r = ret_p.iloc[i]
            clean_targets = []
            for s in targets:
                rr = float(r[s]) if s in ret_p.columns else 0.0
                # skip limit-up (>= +10%, cannot buy) and limit-down (<= -10%, cannot sell)
                if rr >= PRICE_LIMIT - 1e-9 or rr <= -PRICE_LIMIT + 1e-9:
                    continue
                clean_targets.append(s)
            shares, cash, to = _rebalance(shares, cash, clean_targets, prices)
            turnovers.append(to)
        total = cash + sum(shares[s] * float(prices[s]) for s in shares)
        nav[d] = total
        last_total = total
    nav_s = pd.Series(nav).sort_index() # portfolio value at each date sorted by date
    return nav_s, float(np.mean(turnovers)) if turnovers else 0.0


def _rebalance(shares, cash, targets, prices):
    """
    Purpose: Execute a single rebalance: sell names rotating out, buy/adjust to an
        equal weight across `targets`, in 100-share lots, charging commission +
        slippage on every trade and stamp duty on sells.
    Used by: run_portfolio()
    Returns: tuple (shares, cash, turnover_ratio)
        - shares: dict {stock: share count} after rebalancing
        - cash: float cash balance after buys/sells and costs
        - turnover_ratio: float, total traded value / pre-rebalance total value
    """
    # prices: full close-price Series for ALL stocks on the trade date, so outgoing
    # (rotated-out) positions can be marked-to-market and sold correctly.
    total = cash + sum(v * float(prices[s]) for s, v in shares.items())
    w = 1.0 / len(targets) if targets else 0.0
    desired = {}
    for s in targets:
        p = float(prices[s])
        if not np.isfinite(p) or p <= 0:
            continue
        val = total * w
        sh = int(val / (p * LOT)) * LOT
        desired[s] = max(sh, 0)
    traded = 0.0
    # sells (names leaving or removed)
    for s in list(shares.keys()):
        if s not in desired or desired[s] == 0:
            proceeds = shares[s] * float(prices[s])
            cost = proceeds * (COMMISSION + SLIPPAGE + STAMP)
            cash += proceeds - cost
            traded += proceeds
            del shares[s]
    # buys / adjusts
    for s in targets:
        p = float(prices[s])
        if not np.isfinite(p) or p <= 0:
            continue
        cur = shares.get(s, 0)
        tgt = desired.get(s, 0)
        diff = tgt - cur
        if diff > 0:
            buy_val = diff * p
            cost = buy_val * (COMMISSION + SLIPPAGE)
            if cash >= buy_val + cost:
                shares[s] = tgt
                cash -= buy_val + cost
                traded += buy_val
        elif diff < 0:
            sell_val = (-diff) * p
            cost = sell_val * (COMMISSION + SLIPPAGE + STAMP)
            cash += sell_val - cost
            traded += sell_val
            shares[s] = tgt
    return shares, cash, traded / total if total > 0 else 0.0


# --------------------------------------------------------------------------- #
def metrics(nav):
    """
    Purpose: Compute standard performance statistics from a NAV series:
        annualized return and volatility, Sharpe ratio, and max drawdown.
    Used by: main()
    Returns: dict with keys annual_return, annual_vol, sharpe, max_drawdown,
        final_nav; returns {} if the NAV series has fewer than 2 returns.
    """
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return {}
    n = len(rets)
    cum = nav.iloc[-1] / nav.iloc[0]
    ann_ret = cum ** (252.0 / n) - 1.0
    ann_vol = rets.std() * np.sqrt(252.0)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    running_max = nav.cummax()
    mdd = ((nav - running_max) / running_max).min()
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "final_nav": nav.iloc[-1],
    }


def axis_series(nav, axis):
    """
    Purpose: Convert a NAV series into the y-axis quantity selected by --axis.
        "growth"  -> cumulative growth of 1.0 (nav / nav[0])
        "returns" -> cumulative return (nav / nav[0] - 1), starting at 0.0
    Used by: main() plotting code (equity curve + binned performance).
    Returns: pd.Series indexed like `nav` (same length).
    """
    g = nav / nav.iloc[0]
    if axis == "growth":
        return g
    return (g - 1.0)


def ic_stats(dates, idx, close_p, sig, test_month_ends, tradable_fn):
    """
    Purpose: Compute monthly cross-sectional IC and RankIC for the CMM and SM
        signals against the forward return (close[t+22] / close[t+1] - 1), using
        only tradable names that also have a signal row.
    Used by: main()
    Returns: dict {signal_name: {"mean_IC": float, "ICIR": float,
        "positive_ratio": float}} for CMM, SM, CMM_rank, SM_rank.
    """
    rows = []
    for t in test_month_ends:
        pt = idx[t]
        if pt + 21 >= len(dates):
            continue
        trad = tradable_fn(t) # check whether stock is tradable
        if len(trad) < 10:
            continue
        sm = sig[sig["date"] == pd.Timestamp(t)]
        if len(sm) == 0:
            continue                                              # signals missing for this month-end
        sm = sm[sm["stock"].isin(trad)].set_index("stock")
        if len(sm) == 0:
            continue                                              # no tradable stock has a signal row
        common = sm.index                                          # signal ∩ tradable
        y = (close_p.iloc[pt + 22].loc[common].values / close_p.iloc[pt+1].loc[common].values - 1.0) # t+1 because signal is created using data on t, so we can only trade on the signal starting t+1
        cmm = sm.loc[common, "cmm_signal"].values
        smm = sm.loc[common, "sm_signal"].values
        if len(y) < 5:
            continue
        ic_c = np.corrcoef(cmm, y)[0, 1]
        ic_s = np.corrcoef(smm, y)[0, 1]
        ric_c = np.corrcoef(pd.Series(cmm).rank().values, pd.Series(y).rank().values)[0, 1]
        ric_s = np.corrcoef(pd.Series(smm).rank().values, pd.Series(y).rank().values)[0, 1]
        rows.append((ic_c, ic_s, ric_c, ric_s))
    arr = np.array(rows)
    out = {}
    for name, col in [("CMM", 0), ("SM", 1), ("CMM_rank", 2), ("SM_rank", 3)]:
        x = arr[:, col]
        out[name] = {
            "mean_IC": np.nanmean(x),
            "ICIR": np.nanmean(x) / np.nanstd(x) if np.nanstd(x) > 0 else np.nan,
            "positive_ratio": np.nanmean(x > 0),
        }
    return out


def build_bin_schedules(sig, dates, idx, test_month_ends, tradable_fn, col, bins):
    """
    Purpose: Split each month-end's tradable cross-section into `bins`
        signal-sorted quantile buckets (descending on `col`) and build one
        equal-weight rebalance schedule per bucket. Buckets are formed by sorting
        the month-end tradable cross-section on `col` (descending) and slicing it
        into `bins` equal-sized groups, so bin 1 is the lowest-signal bucket (G1)
        and bin `bins` is the highest (G{bins}).
    Used by: main()
    Returns: tuple (schedules, avg_signal)
        - schedules: dict {bin: [(trade_date, [stocks]), ...]} for bin in 1..bins;
          each trade_date is the first trading day of the month following the
          month-end.
        - avg_signal: dict {bin: float}, the bucket's mean cross-sectional z-score
          of its members vs that month's full tradable cross-section, averaged
          over rebalances (nan if the bucket is never populated). Mirrors
          build_schedule()'s signal-strength stat.
    """
    schedules = {g: [] for g in range(1, bins + 1)}
    strengths = {g: [] for g in range(1, bins + 1)}
    for t in test_month_ends:
        td = dates[idx[t] + 1]                 # first trading day of next month
        trad = tradable_fn(t)
        if not trad:
            continue
        sm = sig[sig["date"] == pd.Timestamp(t)]
        cross = sm[sm["stock"].isin(trad)].dropna(subset=[col]).sort_values(
            col, ascending=False)
        if len(cross) < bins:
            continue                          # too few names to form `bins` buckets
        vals = cross[col].values
        mu = vals.mean()
        sd = vals.std(ddof=0)
        zvals = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)
        # groups[0] = top-signal bucket, groups[bins-1] = bottom-signal bucket.
        # NB: `cross` keeps the original integer RangeIndex, so the stock codes
        # live in the "stock" column (not the index). Split row indices so the
        # same slice drives both the stock list and its z-score average.
        stocks_arr = cross["stock"].values
        groups = np.array_split(np.arange(len(vals)), bins)
        for g in range(1, bins + 1):
            gi = groups[bins - g]
            if len(gi) == 0:
                continue
            schedules[g].append((td, list(stocks_arr[gi])))
            strengths[g].append(float(zvals[gi].mean()))
    avg_signal = {g: float(np.mean(strengths[g])) if strengths[g] else float("nan")
                  for g in range(1, bins + 1)}
    return schedules, avg_signal


# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()

    ap = argparse.ArgumentParser(description="A-share backtest for the CMM baseline factor.")
    ap.add_argument("--stocks", type=int, default=None,
                    help="backtest the subsampled signals_s<N>.parquet (mirrors "
                         "model.py --stocks); default = full-universe signals.parquet")
    ap.add_argument("--bins", type=int, default=BINS,
                    help="number of signal-sorted quantile bins for "
                         "binned_performance.png (default 10 = deciles)")
    ap.add_argument("--axis", type=str, choices=["growth", "returns"],
                    default="growth",
                    help="y-axis for the three plots: 'growth' = growth of 1.0 "
                         "(default), 'returns' = cumulative return")
    args = ap.parse_args()
    bins = args.bins
    axis = args.axis
    if axis == "returns":
        YLABEL = "Cumulative Returns"
        BIN_BAR_TITLE = "Final cumulative return per bin"
        BIN_LINE_TITLE = "Cumulative return over time per bin"
        BIN_HLINE = 0.0
    else:
        YLABEL = "Growth of 1.0"
        BIN_BAR_TITLE = "Final growth of 1.0 per bin"
        BIN_LINE_TITLE = "Growth over time per bin"
        BIN_HLINE = 1.0

    sig_path = (os.path.join(OUT_DIR, f"signals_s{args.stocks}.parquet")
                if args.stocks is not None else SIG_PATH)

    B_OUT_DIR = (os.path.join(OUT_DIR, f"backtest_s{args.stocks}")
                   if args.stocks is not None else os.path.join(OUT_DIR, "backtest"))
    os.makedirs(B_OUT_DIR, exist_ok=True)

    dates, stocks, close_p, ret_p, amount_p, vol_p, sig = load_data(sig_path)
    idx = {d: i for i, d in enumerate(dates)}

    # test month-ends (last trading day of each month within range)
    me_all = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).max().sort_values()
    test_month_ends = [pd.Timestamp(d) for d in me_all.values
                       if pd.Timestamp(TEST_LO) <= pd.Timestamp(d) < pd.Timestamp(TEST_HI)]
    print(f"[backtest] test month-ends: {len(test_month_ends)} "
          f"({test_month_ends[0].date()} .. {test_month_ends[-1].date()})")

    def tradable_fn(t):
        """
        Purpose: Thin closure that binds main()'s local data (dates, idx, stocks,
            amount_p, vol_p) so callers can request the tradable universe for a
            given month-end using just the date.
        Used by: build_schedule(), ic_stats(), build_bin_schedules()
        Returns: list[str] of stock codes passing the universe filter at `t`.
        """
        return tradable_at(t, dates, idx, stocks, amount_p, vol_p)

    # build rebalance schedules for each strategy, and the per-strategy average
    # signal strength of the held names: each stock's signal is z-scored against
    # that month's FULL tradable cross-section, then averaged over the selected
    # top-K names, then averaged across rebalances. This measures "how many std-dev
    # above the cross-sectional mean does the held portfolio sit, on average".
    def build_schedule(col, topk):
        """
        Purpose: Build a rebalance schedule for one strategy by, at each month-end,
            taking the tradable cross-section sorted descending on signal column
            `col` and selecting the top `topk` names (all names if topk is None).
            It also computes the strategy's average signal strength, measured as
            the mean z-score of the selected names against that month's full
            tradable cross-section, averaged across rebalances.
        Used by: main() (for the four strategies CMM_topK / SM_topK / Universe_CMM
            / Universe_SM)
        Returns: tuple (sched, avg_signal)
            - sched: list of (trade_date, [stocks]) rebalance entries
            - avg_signal: float, mean cross-sectional z-score of held names (nan if
              no rebalance)
        """
        sched = []
        strengths = []
        for t in test_month_ends:
            td = dates[idx[t] + 1]                 # first trading day of next month
            trad = tradable_fn(t)
            if not trad:
                continue
            sm = sig[sig["date"] == pd.Timestamp(t)]
            cross = sm[sm["stock"].isin(trad)].dropna(subset=[col]).sort_values(
                col, ascending=False)
            if len(cross) == 0:
                continue
            sel = cross if topk is None else cross.head(topk)
            targets = list(sel["stock"])
            if not targets:
                continue
            mu = cross[col].mean()
            sd = cross[col].std(ddof=0)
            zsel = (sel[col] - mu) / sd if sd > 0 else pd.Series(0.0, index=sel.index)
            strengths.append(float(zsel.mean()))
            sched.append((td, targets))
        avg_signal = float(np.mean(strengths)) if strengths else float("nan")
        return sched, avg_signal

    strategies = {}
    sig_strength = {}
    for name, (col, topk) in {
        f"CMM_top{TOP_K}": ("cmm_signal", TOP_K),
        f"SM_top{TOP_K}": ("sm_signal", TOP_K),
        "Universe_EW": ("cmm_signal", None),   # all tradable, equal weight
    }.items():
        sched, avg_sig = build_schedule(col, topk)
        strategies[name] = sched
        sig_strength[name] = avg_sig

    navs = {}
    turns = {}
    for name, sched in strategies.items():
        nav, to = run_portfolio(dates, idx, close_p, ret_p, sched, name)
        navs[name] = nav
        turns[name] = to
        print(f"[backtest] {name}: {len(sched)} rebalances, avg monthly turnover={to:.2%}")

    # metrics
    mrows = []
    for name in strategies:
        m = metrics(navs[name])
        mrows.append({
            "strategy": name,
            "annual_return": m.get("annual_return"),
            "annual_vol": m.get("annual_vol"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "avg_monthly_turnover": turns[name],
            "final_nav": m.get("final_nav"),
            "avg_signal": sig_strength[name],
        })
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(os.path.join(B_OUT_DIR, "backtest_metrics.csv"), index=False)

    # IC
    ic = ic_stats(dates, idx, close_p, sig, test_month_ends, tradable_fn)
    icdf = pd.DataFrame([{"signal": k, **v} for k, v in ic.items()])
    icdf.to_csv(os.path.join(B_OUT_DIR, "backtest_ic.csv"), index=False)

    # plots growth rate of portfolio value for each strategy.
    # NOTE: Universe_CMM / Universe_SM hold every tradable stock equal-weight, so
    # the signal value is never used to rank or size positions. Since cmm_signal
    # and sm_signal are both non-null for the same stock set, the two universe
    # curves are numerically identical and trace the same line -- give each a
    # distinct color/linestyle so all four show up in the legend and any future
    # divergence (e.g. signal-weighted sizing) is visible.
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    plot_style = {
        f"CMM_top{TOP_K}":  {"color": "tab:red",    "linestyle": "-"},
        f"SM_top{TOP_K}":   {"color": "tab:blue",   "linestyle": "-"},
        "Universe_EW":     {"color": "tab:orange",  "linestyle": ":"},
    }
    for name in [f"CMM_top{TOP_K}", f"SM_top{TOP_K}", "Universe_EW"]:
        nav = navs[name]
        if len(nav) == 0:
            continue
        y = axis_series(nav, axis)
        plt.plot(y.index, y.values, label=name, linewidth=1.6, **plot_style[name])
        ax.annotate(f"{y.iloc[-1]:.2f}", (y.index[-1], y.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=plot_style[name]["color"], fontsize=9,
                    va="center", ha="left")
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.06)   # room for the end-of-line labels
    plt.title("Strategies Performance Comparison")
    plt.ylabel(YLABEL)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(B_OUT_DIR, "equity_curve.png"), dpi=130)
    plt.close()

    # Binned CMM performance: split the tradable cross-section into `bins`
    # G1 = lowest-signal bucket, G{bins} = highest-signal bucket.
    bin_schedules, bin_sig_strength = build_bin_schedules(
        sig, dates, idx, test_month_ends, tradable_fn, "cmm_signal", bins)
    bin_navs = {}
    bin_final = {}
    bin_turns = {}
    for g in range(1, bins + 1):
        nav, to = run_portfolio(dates, idx, close_p, ret_p, bin_schedules[g], f"CMM_G{g}")
        bin_navs[g] = nav
        bin_turns[g] = to
        bin_final[g] = axis_series(nav, axis).iloc[-1] if len(nav) > 1 else np.nan

    cmap = plt.cm.coolwarm                                 # G1 (cold) -> G{bins} (hot)
    bin_color = {g: cmap((g - 1) / max(bins - 1, 1)) for g in range(1, bins + 1)}
    labels = [f"G{g}" for g in range(1, bins + 1)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"CMM: signal-sorted bins "
                 f"(G1 = lowest signal stocks, G{bins} = highest signal stocks)", fontsize=13)

    # Top panel: bar chart of final cumulative growth per bin.
    ax = axes[0]
    heights = [bin_final[g] for g in range(1, bins + 1)]
    bars = ax.bar(labels, heights, color=[bin_color[g] for g in range(1, bins + 1)])
    ax.axhline(BIN_HLINE, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(BIN_BAR_TITLE)
    ax.set_ylabel(YLABEL)
    ax.grid(axis="y", alpha=0.3)
    finite_h = [h for h in heights if np.isfinite(h)]
    max_h = max(finite_h) if finite_h else 1.0
    for b, h in zip(bars, heights):
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02 * max_h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Bottom panel: line chart of cumulative growth per bin (normalized to 1.0).
    ax = axes[1]
    for g in range(1, bins + 1):
        nav = bin_navs[g]
        if len(nav) < 2:
            continue
        y = axis_series(nav, axis)
        ax.plot(y.index, y.values, label=f"G{g}", linewidth=1.4,
                color=bin_color[g])
        ax.annotate(f"{y.iloc[-1]:.2f}", (y.index[-1], y.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=bin_color[g], fontsize=8, va="center", ha="left")
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo, xhi + (xhi - xlo) * 0.06)   # room for the end-of-line labels
    ax.set_title(BIN_LINE_TITLE)
    ax.set_ylabel(YLABEL)
    ax.set_xlabel("Date")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(B_OUT_DIR, "binned_performance.png"), dpi=130)
    plt.close(fig)

    # summary text
    lines = []
    lines.append("=== CMM baseline backtest (A-share rules) ===")
    lines.append(f"Test window: {test_month_ends[0].date()} -> {test_month_ends[-1].date()}")
    lines.append(f"Universe: {len(stocks)} stocks, monthly filter (20d amount>=20M).")
    lines.append("")
    lines.append("Performance:")
    for _, r in mdf.iterrows():
        lines.append(f"  {r['strategy']:<12} ann_ret={r['annual_return']:.2%}  "
                     f"vol={r['annual_vol']:.2%}  sharpe={r['sharpe']:.2f}  "
                     f"maxDD={r['max_drawdown']:.2%}  turnover={r['avg_monthly_turnover']:.2%}  "
                     f"avg_signal={r['avg_signal']:.4f}")
    lines.append("")
    lines.append("CMM Binned Performance (ordered highest signal -> lowest):")
    for g in range(bins, 0, -1):
        label = f"G{g}"
        m = metrics(bin_navs[g])
        if not m:
            lines.append(f"  {label:<12} (no data)")
            continue
        lines.append(f"  {label:<12} ann_ret={m['annual_return']:.2%}  "
                     f"vol={m['annual_vol']:.2%}  sharpe={m['sharpe']:.2f}  "
                     f"maxDD={m['max_drawdown']:.2%}  turnover={bin_turns[g]:.2%}  "
                     f"avg_signal={bin_sig_strength[g]:.4f}")
    lines.append("")
    lines.append("Monthly IC (signal vs next-21d return):")
    for _, r in icdf.iterrows():
        lines.append(f"  {r['signal']:<10} meanIC={r['mean_IC']:.4f}  "
                     f"ICIR={r['ICIR']:.2f}  pos_ratio={r['positive_ratio']:.2%}")
    summary = "\n".join(lines)
    with open(os.path.join(B_OUT_DIR, "backtest_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    print("[backtest] outputs written to", B_OUT_DIR)

    print(f"Runtime: {time.time() - start_time:.2f}s")
if __name__ == "__main__":
    main()

# all stocks: ~100s