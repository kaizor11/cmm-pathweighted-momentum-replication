import argparse
import os
import sys
import math
import random
import time

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SEED, DATA_PATH, SIG_PATH, DIAG_PATH, FEATURES

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NF = len(FEATURES)
WIN = 231          # formation window length (trading days)
SKIP = 21          # skip most-recent 21 days (avoid short-term reversal)
HOLD = 21          # forward return horizon (trading days)

RET_COL = "ret"
CLOSE_COL = "close"

def log(*a):
    """
    Purpose: Print a log line with flush=True so progress is visible immediately
        (not buffered) during long training / signal-generation runs.
    Used by: main() throughout (loading, training, signal-writing diagnostics).
    Returns: None
    """
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class CMMNet(nn.Module):
    """Path-weighted momentum (CMM), faithful to the study.

    Pipeline (report p5): X (NF x WIN feature path) -> temporal conv (kernel=5)
    -> GELU -> Flatten (NF*WIN) -> FFN funnel NF*WIN -> 16 -> 8 -> 1 (GELU +
    Dropout on hidden layers) -> state scalar theta (B,) -> w = softmax(theta*R)
    -> CMM = sum_t w_t * r_t.

    The temporal conv uses einsum (not nn.Conv1d) because PyTorch's CPU Conv1d
    is ~4x slower for this (large-batch, few-channel) shape (measured ~1.6s vs
    ~0.35s per forward on 5454 x 16 x 231); the math is identical to Conv1d.
    """

    def __init__(self, hidden=(16, 8), dropout=0.2):
        """
        Purpose: Build the CMM network: a learnable temporal-convolution weight
            (kernel=5, no bias) plus a global FFN head that funnels the flattened
            feature path down to a single state scalar `theta` per stock.
        Used by: train_model() (via CMMNet()).
        Returns: None (registers parameters/Modules on the instance).
        """
        super().__init__()
        # Temporal convolution (kernel=5, padding=2, no bias) as a learnable
        # weight, applied via 5 einsums over shifted slices in _temporal_conv.
        self.conv_weight = nn.Parameter(torch.empty(NF, NF, 5))
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        self.act = nn.GELU()
        # Global FFN funnel over the FLATTENED path -> a single state scalar
        # theta per stock (NOT a per-day head). `hidden` = the study's chosen
        # [16, 8] widths; Dropout sits on each hidden activation (report p5).
        layers = []
        in_dim = NF * WIN
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # -> theta, no dropout on scalar
        self.head = nn.Sequential(*layers)

    def _temporal_conv(self, x):
        """
        Purpose: Apply the temporal convolution (kernel=5, pad=2, no bias) via five
            einsums over shifted slices — mathematically identical to
            Conv1d(NF, NF, k=5, pad=2, bias=False) but faster on CPU for this shape.
        Used by: forward().
        Returns: torch.Tensor of shape (B, NF, L) — the convolved feature path.
        """
        # x: (B, NF, L) -> (B, NF, L), equivalent to Conv1d(NF, NF, k=5, pad=2, bias=False)
        xp = F.pad(x, (2, 2))
        L = x.shape[-1]
        out = torch.einsum("oi,bil->bol", self.conv_weight[:, :, 0], xp[:, :, 0:L])
        for k in range(1, 5):
            out = out + torch.einsum("oi,bil->bol", self.conv_weight[:, :, k], xp[:, :, k:k + L])
        return out

    def forward(self, X, R):
        """
        Purpose: Run the full CMM pipeline on one batch: temporal conv -> GELU ->
            flatten -> FFN funnel -> state scalar theta -> softmax(theta * R) day
            weights -> path-weighted momentum signal.
        Used by: train_model() and predict().
        Returns: tuple (signal, w)
            - signal: torch.Tensor (B,) — one path-weighted momentum score per stock
            - w: torch.Tensor (B, WIN) — softmax day weights over the return path
        """
        # X: (B, NF, WIN)  R: (B, WIN)
        h = self.act(self._temporal_conv(X))     # (B, NF, WIN)
        h = h.flatten(start_dim=1)               # (B, NF*WIN)
        theta = self.head(h).squeeze(-1)         # (B,) one state scalar per stock
        w = torch.softmax(theta.unsqueeze(-1) * R, dim=1)         # (B, WIN) day weights
        signal = (w * R).sum(dim=1)              # (B,) path-weighted momentum
        return signal, w


def pearson(a, b):
    """
    Purpose: Compute the Pearson correlation between two tensors, centered and
        normalized, with a grad-safe guard that maps any residual NaN/inf (constant
        or poisoned slice) to 0.
    Used by: train_model() (loss = -pearson(signal, y)).
    Returns: torch.Tensor scalar — the correlation coefficient.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    corr = (a * b).sum() / denom
    # grad-safe guard: any residual NaN/inf (constant or poisoned slice) -> 0
    return torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_pivots(max_stocks=None):
    """
    Purpose: Read the long parquet with Polars (optionally subsampling to a seeded
        random draw of `max_stocks`), then fill dense (date x stock) numpy arrays
        via a vectorized (row, col) index mapping — far faster than 18 Polars
        pivots. Returns the wide close / return-path / feature / tradable arrays
        needed for training and scoring.
    Used by: main().
    Returns: tuple (dates, idx, stocks, close_p, ret_z_p, z_p, tradable_p)
        - dates: pd.DatetimeIndex of sorted unique trading dates
        - idx: dict {Timestamp: int} fast date -> row lookup
        - stocks: sorted list of stock codes
        - close_p: (D, N) float64 raw close (for the forward-return label y)
        - ret_z_p: (D, N) float32 z-scored return path (NaN -> 0)
        - z_p: dict {feature: (D, N) float32} z-scored features (NaN -> 0)
        - tradable_p: (D, N) bool tradable-universe flag (NaN -> False)
    """
    lf = pl.scan_parquet(DATA_PATH)
    if max_stocks is not None:
        all_stk = sorted(lf.select("stock").unique().collect()["stock"].to_list())
        k = min(int(max_stocks), len(all_stk))
        keep = np.random.default_rng(SEED).choice(all_stk, size=k, replace=False)
        lf = lf.filter(pl.col("stock").is_in(sorted(keep.tolist())))
    df = (
        lf.with_columns(pl.col("date").str.to_date(format="%Y%m%d"))
        .sort(["date", "stock"])
        .collect()
    )
    dates = pd.DatetimeIndex(df["date"].unique().sort().to_list())
    stocks = sorted(df["stock"].unique().to_list())
    idx = {d: i for i, d in enumerate(dates)}  # {Timestamp: int} for fast lookup

    # vectorized (row, col) index for every row -> one dense scatter per column
    rows = dates.get_indexer(pd.to_datetime(df["date"].to_numpy()))
    cols = pd.Index(stocks).get_indexer(df["stock"].to_numpy())
    n_dates, n_stocks = len(dates), len(stocks)

    def to_wide_numpy(col):
        """
        Purpose: Scatter one long DataFrame column into a dense (n_dates, n_stocks)
            numpy array using the precomputed (rows, cols) index mapping.
        Used by: load_pivots() (for close, each *_z feature, ret_z, tradable).
        Returns: np.ndarray of shape (n_dates, n_stocks), NaN where no row exists.
        """
        out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        out[rows, cols] = df[col].to_numpy()
        return out

    close_p = to_wide_numpy(CLOSE_COL)   # (D, N) raw close -> used only for the fwd-return label y

    # Features (and the return path) are ALREADY z-scored + winsorized in data.py
    # (see compute_features: per-date cross-section, clipped to [-5, 5]). We just read
    # the precomputed *_z columns here -- single source of truth, no recomputation.
    # Pre-listing (never-listed) cells are NaN in the wide pivot -> map to 0.0 to keep
    # the tensors finite (matches the old behavior of feeding 0 for unlisted names).
    z_p = {}
    for f in FEATURES:
        z_p[f] = np.nan_to_num(to_wide_numpy(f + "_z"), nan=0.0).astype(np.float32)
    ret_z_p = np.nan_to_num(to_wide_numpy(RET_COL + "_z"), nan=0.0).astype(np.float32)
    # tradable universe flag (Boolean). Pre-listing cells are NaN in the wide pivot
    # -> map to False so unlisted names are excluded from training at every month-end.
    tradable_p = np.nan_to_num(to_wide_numpy("tradable"), nan=0.0).astype(bool)
    return dates, idx, stocks, close_p, ret_z_p, z_p, tradable_p


def month_ends(dates):
    """
    Purpose: Return the last trading date of each calendar month present in `dates`.
    Used by: main() (to derive the month-end training/scoring keys).
    Returns: list of Timestamps — one month-end date per month, sorted ascending.
    """
    s = pd.Series(dates)
    me = s.groupby(s.dt.to_period("M")).max().sort_values()
    return list(me.values)


def _bounds(idx, t, n):
    """
    Purpose: Convert a month-end `t` into its (pt, start, end) window indices,
        where the formation window is [pt - (SKIP+WIN-1), pt - SKIP] (skip the most
        recent SKIP days, then WIN days of history). Returns None if there isn't
        enough history before, or enough forward days for the HOLD label after.
    Used by: build_one(), build_label().
    Returns: tuple (pt, start, end) of ints, or None if the window is invalid.
    """
    pt = idx[pd.Timestamp(t)]
    start = pt - (SKIP + WIN - 1)   # = pt - 251
    end = pt - SKIP                 # = pt - 21  (inclusive)
    if start < 0 or pt + HOLD >= n:
        return None
    return pt, start, end


def build_one(dates, idx, close_p, ret_z_p, z_p, t, tradable_p=None):
    """
    Purpose: Build (X, R, y, mask) for a single month-end `t` using numpy slices
        (no pandas iloc). X is the (N, NF, WIN) feature-path tensor, R the (N, WIN)
        z-scored return path, y the (N,) forward HOLD-day return label, and mask the
        boolean tradable-universe flag. When tradable_p is given, the rows are
        masked down to tradable names (Bug #3 fix); otherwise mask is all-True.
    Used by: main() (signal generation) and _month_tensors() (training).
    Returns: tuple (X, R, y, mask) or None if the window is invalid
        - X: np.ndarray (N, NF, WIN) float32
        - R: np.ndarray (N, WIN) float32
        - y: np.ndarray (N,) float32
        - mask: np.ndarray (N,) bool
    """
    n = len(dates)
    b = _bounds(idx, t, n)
    if b is None:
        return None
    pt, start, end = b
    N = ret_z_p.shape[1]
    # zero-fill pre-listing NaN in the return path (matches how `y` treats unlisted
    # stocks below), so the weighted signal stays finite for recently-listed names.
    R = np.nan_to_num(ret_z_p[start:end + 1, :].T, nan=0.0).astype(np.float32)   # (N, 231) z-scored return path
    X = np.empty((N, NF, WIN), dtype=np.float32)
    for k, f in enumerate(FEATURES):
        X[:, k, :] = z_p[f][start:end + 1, :].T
    y = np.nan_to_num(close_p[pt + HOLD, :] / close_p[pt, :] - 1.0, nan=0.0).astype(np.float32)
    if tradable_p is not None:
        mask = np.asarray(tradable_p[pt], dtype=bool)
        R, X, y = R[mask], X[mask], y[mask]   # drop non-tradable rows from the Pearson loss
    else:
        mask = np.ones(N, dtype=bool)
    return X, R, y, mask


def build_label(dates, idx, close_p, t):
    """
    Purpose: Build only the forward-return label y for a month-end `t` (cheap
        slice, no feature tensors), used by diagnostics.
    Used by: diagnostics / external callers.
    Returns: np.ndarray (N,) float32 forward return (NaN -> 0), or None if the
        window is invalid.
    """
    n = len(dates)
    b = _bounds(idx, t, n)
    if b is None:
        return None
    pt, _, _ = b
    return np.nan_to_num(close_p[pt + HOLD, :] / close_p[pt, :] - 1.0, nan=0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _to_tensors(sample):
    """
    Purpose: Convert a (X, R, y) numpy sample tuple into PyTorch tensors.
    Used by: predict().
    Returns: tuple (torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(y)).
    """
    X, R, y = sample
    return (torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(y))


def _month_tensors(dates, idx, close_p, ret_z_p, z_p, t, tradable_p=None):
    """
    Purpose: Build one month's (X, R, y) as PyTorch tensors on demand from the
        in-memory pivot arrays (avoids pre-building an ~7 GB samples dict that
        swap-thrashes an 8 GB machine). When tradable_p is given, tensors are
        masked to the tradable universe (Bug #3 fix).
    Used by: train_model() (per training and validation month).
    Returns: tuple (torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(y)).
    """
    X, R, y, _ = build_one(dates, idx, close_p, ret_z_p, z_p, t, tradable_p)
    return torch.from_numpy(X), torch.from_numpy(R), torch.from_numpy(y)


def train_model(dates, idx, close_p, ret_z_p, z_p, train_keys, val_keys,
                tradable_p=None, epochs=200, patience=12, lr=1e-3, min_delta=5e-4, weight_decay=1e-4):
    """
    Purpose: Train a CMMNet with Adam to maximize the Pearson IC between the
        model's momentum signal and the forward return. Iterates over shuffled
        training months, tracks validation IC, and early-stops (restoring the best
        weights) when validation IC stops improving by min_delta for `patience`
        epochs.
    Used by: main() (once per retrain date in the rolling schedule).
    Returns: tuple (model, best_val)
        - model: CMMNet with the best-validation state loaded
        - best_val: float, best mean validation IC (Pearson)
    """
    model = CMMNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1e9
    best_state = None
    wait = 0
    for ep in range(epochs):
        model.train()
        random.shuffle(train_keys)
        for k in train_keys:
            X, R, y = _month_tensors(dates, idx, close_p, ret_z_p, z_p, k, tradable_p)
            opt.zero_grad()
            """Internally: temporal conv → GELU → flatten → FFN funnel (3696→16→8→1)
            → state scalar theta (B,) → softmax(theta * R) → 231 day weights w
            → signal = (w * R).sum(dim=1). Output is one score per stock, shape (B,).
            The _ throws away the 231 day-weights — interesting for analysis, useless for the loss."""
            sig, _ = model(X, R)
            loss = -pearson(sig, y)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clips weights to max 1; stops one pathological month from firing an update so large it wrecks the weights
                opt.step()
        # validation
        model.eval()
        vs = [] # collects one IC for each validation month
        with torch.no_grad():
            for k in val_keys:
                X, R, y = _month_tensors(dates, idx, close_p, ret_z_p, z_p, k, tradable_p)
                sig, _ = model(X, R)
                vs.append(pearson(sig, y).item()) # .item() converts the PyTorch tensor into a python float (avoids memory leak)
        vmean = float(np.mean(vs)) if vs else -1e9 # avg IC store across validation months for each stock
        """store best val and early stopping"""
        if vmean > best_val + min_delta:
            best_val = vmean
            best_state = {p: v.detach().clone() for p, v in model.state_dict().items()}
            wait = 0 # no-imporvement counter
        else: # if no-improvement for 'patience' epochs, break
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(model, sample):
    """
    Purpose: Run the model in eval mode on one sample and return both the CMM
        signal (path-weighted momentum) and the simple-mean (SM) signal (the
        equal-weighted mean of the return path).
    Used by: main() (to generate month-end cmm_signal / sm_signal rows).
    Returns: tuple (cmm, sm)
        - cmm: np.ndarray (N,) float64 model signal
        - sm: np.ndarray (N,) float64 mean of the return path
    """
    model.eval()
    X, R, y = _to_tensors(sample)
    with torch.no_grad():
        sig, _ = model(X, R)
        sm = R.mean(dim=1)
    return sig.numpy().astype(np.float64), sm.numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Retrain schedule (rolling, no look-ahead)
# --------------------------------------------------------------------------- #
def select_keys(me_keys, lo=None, hi=None):
    """
    Purpose: Filter a list of month-end keys to the half-open interval
        [lo, hi) (inclusive lower, exclusive upper), skipping unset bounds.
    Used by: build_schedule() (to carve out train/val windows).
    Returns: list of month-end keys (Timestamps) within [lo, hi).
    """
    out = []
    for k in me_keys:
        d = pd.Timestamp(k)
        if lo is not None and d < pd.Timestamp(lo):
            continue
        if hi is not None and d >= pd.Timestamp(hi):
            continue
        out.append(k)
    return out


def build_schedule(me_keys):
    """
    Purpose: Build the rolling (no-look-ahead) retrain schedule.
        - 36-month rolling window
        - Train dates: 2017-2020 initial train, 2020-2026 every 6 months) (11 total trains)
        - Validation dates: 12 months following train end
        - Test set: 6 months following validation end
        - Shift entire window forward 6 months after every train
    Used by: main() (iterates the schedule to train one model per retrain date).
    Returns: list of tuples (retrain_date, train_keys, val_keys, cover_hi)
        - retrain_date: str "YYYY-MM-DD" when the model is (re)trained
        - train_keys / val_keys: lists of month-end keys for train / validation
        - cover_hi: str "YYYY-MM-DD" exclusive upper bound the model covers
    """
    me_keys = sorted(me_keys)
    sched = []
    # init_train = select_keys(me_keys, lo="2017-01-01", hi="2023-01-01")   # 2017-2022
    # init_val = select_keys(me_keys, lo="2023-01-01", hi="2024-01-01")     # 2023
    # sched.append(("2024-01-01", init_train, init_val, "2024-07-01"))
    init_train = select_keys(me_keys, lo="2017-01-01", hi="2020-01-01")
    init_val = select_keys(me_keys, lo="2020-01-01", hi="2021-01-01")
    sched.append(("2021-01-01", init_train, init_val, "2021-07-01"))

    # retrain_dates = ["2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"]
    retrain_dates = ["2021-07-01", 
                     "2022-01-01", "2022-07-01", 
                     "2023-01-01", "2023-07-01", 
                     "2024-01-01", "2024-07-01", 
                     "2025-01-01", "2025-07-01", 
                     "2026-01-01"]
    for i, d in enumerate(retrain_dates):
        dt = pd.Timestamp(d)
        val_lo = (dt - pd.DateOffset(months=12)).strftime("%Y-%m-%d") # validate on latest 12 months (up to retrain_date)
        val_hi = d
        train_lo = (dt - pd.DateOffset(months=48)).strftime("%Y-%m-%d") # train on 36 months (up to val_lo)
        train_hi = val_lo
        tr = select_keys(me_keys, lo=train_lo, hi=train_hi)
        va = select_keys(me_keys, lo=val_lo, hi=val_hi)
        cover_hi = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else "2026-07-01"
        sched.append((d, tr, va, cover_hi))
    return sched


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    """
    Purpose: Entry point that orchestrates the full pipeline: parse CLI args, load
        and pivot data, derive usable month-ends, build the rolling retrain
        schedule, train one CMMNet per retrain date, and generate per-month-end
        cmm_signal / sm_signal rows saved to the signals parquet (plus a
        diagnostics text file).
    Used by: the `__main__` guard at the bottom of this module.
    Returns: None (writes signals and diagnostics to output/).
    """
    ap = argparse.ArgumentParser(description="Train CMMNet and dump month-end signals.")
    ap.add_argument("--stocks", type=int, default=None,
                    help="subsample to N stocks (seeded random draw). Writes to "
                         "output/signals_s<N>.parquet and output/model_diagnostics_s<N>.txt "
                         "so a quick run can never overwrite the full-universe results.")
    args = ap.parse_args()

    start_time = time.time()

    log("[model] loading data ...")
    dates, idx, stocks, close_p, ret_z_p, z_p, tradable_p = load_pivots(args.stocks)
    me_list = month_ends(dates)
    n = len(dates)
    log(f"[model] {n} days, {len(stocks)} stocks, {len(me_list)} month-ends")

    # a subsampled run is a debugging aid, not a result -> keep the canonical
    # outputs (which backtest.py reads) untouched
    if args.stocks is not None:
        sig_path = SIG_PATH.replace(".parquet", f"_s{len(stocks)}.parquet")
        diag_path = DIAG_PATH.replace(".txt", f"_s{len(stocks)}.txt")
        log(f"[model] SUBSAMPLED to {len(stocks)} stocks -> {os.path.basename(sig_path)}")
    else:
        sig_path, diag_path = SIG_PATH, DIAG_PATH

    # valid month-ends (enough history and a forward 21d label exist)
    valid_mes = []
    for t in me_list:
        pt = idx[t]
        if pt - (SKIP + WIN - 1) >= 0 and pt + HOLD < n:
            valid_mes.append(t)
    me_keys = sorted(valid_mes) # last day of month (that are valid for train)
    log(f"[model] {len(me_keys)} usable month-ends (have full history + 21d fwd label)")

    sched = build_schedule(me_keys)
    for d, tr, va, hi in sched: # retrain_date, train_keys, val_keys, cover_hi
        log(f"  retrain {d}: train={len(tr)} months, val={len(va)} months, covers <{hi}")

    # train all models (samples are built on-demand inside train_model -> low memory)
    models = {}
    diag = []
    for d, tr, va, hi in sched:
        log(f"[model] training model for retrain {d} (train {len(tr)}, val {len(va)}) ...")
        train_start_time = time.time()
        m, v = train_model(dates, idx, close_p, ret_z_p, z_p, tr, va, tradable_p=tradable_p)
        if v <= -1e8:
            log(f"[model] WARNING: retrain {d} never produced a finite validation IC "
                f"(loss may be NaN; check data / date parsing)")
        models[d] = m
        diag.append(f"retrain {d}: best_val_IC={v:.4f}  train_months={len(tr)} val_months={len(va)}")
        log(f"[model] Runtime (for above train): {time.time() - train_start_time:2f}s")
    log("[model] all models trained.")

    # map each month-end to the applicable model (largest retrain date <= month)
    retrain_sorted = sorted(models.keys())

    def model_for(k):
        """
        Purpose: Pick the model applicable to a month-end `k` — the one with the
            largest retrain date <= k (i.e. the most recently trained model whose
            coverage window includes k).
        Used by: main() (during signal generation, per month-end).
        Returns: CMMNet — the model to score month-end `k`.
        """
        chosen = retrain_sorted[0]
        for rd in retrain_sorted:
            if pd.Timestamp(rd) <= pd.Timestamp(k):
                chosen = rd
            else:
                break
        return models[chosen]

    # generate signals for every usable month-end (build one month at a time)
    log("[model] generating signals for all month-ends ...")
    recs = []
    for k in me_keys:
        m = model_for(k)
        s = build_one(dates, idx, close_p, ret_z_p, z_p, k, tradable_p)
        if s is None:
            continue
        X, R, y, mask = s
        cmm, sm = predict(m, (X, R, y))
        trad_stocks = [stk for stk, mm in zip(stocks, mask) if mm]
        for j, stk in enumerate(trad_stocks):
            recs.append((k, stk, cmm[j], sm[j]))
    sig_df = pd.DataFrame(recs, columns=["date", "stock", "cmm_signal", "sm_signal"])
    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df.to_parquet(sig_path, index=False)
    log(f"[model] saved signals -> {sig_path} ({len(sig_df):,} rows)")

    with open(diag_path, "w") as f:
        f.write("\n".join(diag) + "\n")
    log("[model] diagnostics:\n" + "\n".join(diag))

    print(f"Runtime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()

# s1000: ~7min
# all stocks: ~40min