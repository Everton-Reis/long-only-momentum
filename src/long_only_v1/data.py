import pandas as pd
import yfinance as yf

def dowload_prices(tickers, start, end):
    raw = yf.dowload(tickers, start = start, end = end, auto_adjust=True, progress=False)
    return raw["Close"].sort_index().dropna()

def validate_prices(prices, tickers, end_date):
    assert prices.columns.isin(tickers).all()
    assert prices.index.is_unique
    assert (prices > 0).all().all()
    assert prices.index.max() < pd.Timestamp(end_date)
    last_month = (pd.Timestamp(end_date) - pd.Timedelta(days=1)).to_period("M")
    assert prices.index.max().to_period("M") == last_month
