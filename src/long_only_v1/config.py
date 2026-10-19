import pandas as pd

TICKERS = ["LLY", "PANW", "TSM", "KO", "JPM"]
START_DATE = "2013-01-01"
END_DATE = "2026-10-01"
VOL_WINDOW = 60

TRAIN_END = pd.Period("2021-06", freq="M")

MOMENTUM_MONTHS = 6
TREND_WINDOW = 200
MIN_WEIGHT_GRID = [p / 100 for p in range(20, 51)]

RF_ANNUAL = 0.04
RF_MONTHLY = (1 + RF_ANNUAL) ** (1 / 12) - 1