import pandas as pd
import pytest

from long_only_v1 import returns as R


def test_monthly_prices_use_last_trading_day(prices):
    monthly = R.to_monthly_prices(prices)
    assert monthly.index[0] == pd.Period("2013-01")
    assert monthly.index[-1] == pd.Period("2026-09")
    last_day_jan = prices.loc["2013-01"].iloc[-1]
    pd.testing.assert_series_equal(monthly.loc[pd.Period("2013-01")], last_day_jan, check_names=False)


def test_first_monthly_return_is_nan(prices):
    monthly_returns = R.to_monthly_returns(R.to_monthly_prices(prices))
    assert monthly_returns.iloc[0].isna().all()
    assert monthly_returns.iloc[1:].notna().all().all()


def test_benchmark_is_equal_weight_rebalanced_monthly(prices):
    monthly_prices = R.to_monthly_prices(prices)
    monthly_returns = R.to_monthly_returns(monthly_prices)
    benchmark = R.equal_weight_benchmark(monthly_returns)

    # Cálculo independente: $1 dividido em 5 partes iguais no início do mês 10 e vendido no fim dele
    start, end = monthly_prices.iloc[9], monthly_prices.iloc[10]
    shares = (1 / len(start)) / start
    value_at_end = (shares * end).sum()

    assert benchmark.iloc[10] == pytest.approx(value_at_end - 1)


def test_growth_of_one_compounds_returns():
    r = pd.Series([0.10, -0.10, 0.05])
    assert R.growth_of_one(r).iloc[-1] == pytest.approx(1.10 * 0.90 * 1.05)