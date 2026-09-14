# CMM Path-Weighted Momentum (A-share Replication)

Code replication of the *path-weighted momentum based on CMM* research report by 华福金工-李杨团队 (reference PDF),
(`input/ashare_2015-2026.parquet`, sourced from WIND `ASHAREEODPRICES`,
2015–2026, ~11.48M rows / 5,454 stocks / 2,812 trading days).

## Structure

| Path | Role |
|---|---|
| `constants.py` | Central config: paths, `SEED`, WIND→friendly column `RENAME`, and the 16 `FEATURES` names. |
| `data.py` | Data pipeline. `--fetch` pulls full daily bars from WIND via SQLAlchemy+pymysql; otherwise computes the 16-feature Polars pipeline → `input/ashare_2015-2026.parquet` (`DATA_PATH`). |
| `model.py` | CMMNet training + signal generation (cross-sectional z-score + winsorize in `load_pivots`; rolling retrain; era-IC diagnostics). |
| `backtest.py` | A-share backtest engine: monthly rebalance, universe filter, costs, lot size 100, T+1, ±10% limit. |
| `input/` | Raw + model-ready data (tracked). |
| `output/` | Signals and backtest results (tracked). |

## Setup

```bash
# quant_env conda environment (Python 3.10)
pip install -r requirements.txt
```

Database credentials are read from `.env` (`DB_USER` / `DB_PASS` / `DB_HOST` /
`DB_PORT` / `DB_NAME`). 

## Workflow

```bash
# 1. Fetch raw data from WIND (optional, writes input/ashare_2015-2026_raw.parquet)
python data.py --fetch

# 2. Compute features (raw -> DATA_PATH)
python data.py

# 3. Train + generate signals (full / sampled)
python model.py
python model.py --stocks 1000

# 4. Backtest (full / sampled; needs the matching signals file)
python backtest.py
python backtest.py --stocks 1000
```