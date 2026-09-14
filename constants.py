import os

SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
IN_DIR = os.path.join(HERE, "input")
# DATA_PATH = os.path.join(IN_DIR, "synthetic_ashare.parquet")
RAW_DATA_PATH = os.path.join(IN_DIR, "ashare_2015-2026_raw.parquet")
DATA_PATH = os.path.join(IN_DIR, "ashare_2015-2026.parquet")
SIG_PATH = os.path.join(OUT_DIR, "signals.parquet")
DIAG_PATH = os.path.join(OUT_DIR, "model_diagnostics.txt")

# WIND columns rename
RENAME = {
    "S_INFO_WINDCODE": "stock",
    "TRADE_DT": "date",
    "S_DQ_ADJOPEN": "open",
    "S_DQ_ADJHIGH": "high",
    "S_DQ_ADJLOW": "low",
    "S_DQ_ADJCLOSE": "close",
    "S_DQ_VOLUME": "volume",
    "S_DQ_AMOUNT": "amount",
}

FEATURES = [
    "f_vol_60", "f_vol_120", "f_amihud_60", "f_beta_252", "f_log_amount_60",
    "f_log_amount_252", "f_amount_trend", "f_mean_abs_ret_60", "f_ret_21d",
    "f_ret_126d", "f_max_ret_21d", "f_price_to_high_252", "f_skew_60",
    "f_kurt_60", "f_idio_vol_60", "f_zero_ret_ratio_60",
]