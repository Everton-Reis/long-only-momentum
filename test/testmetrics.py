import numpy as np
import pandas as pd
import pytest

from long_only_v1 import metrics as M


def test_total_and_annualized_return_over_one_year():
    r = pd.Series([0.10] * 12)
    m = M.performance_metrics(r, rf_monthly=0.0)
    assert m["Meses"] == 12
    assert m["Retorno acumulado"] == pytest.approx(1.10**12 - 1)
    assert m["Retorno anualizado"] == pytest.approx(1.10**12 - 1)  # 12 meses = 1 ano
    assert m["Máximo drawdown"] == pytest.approx(0.0)
    assert m["Volatilidade anualizada"] == pytest.approx(0.0, abs=1e-12)


def test_single_observation_has_no_volatility_or_sharpe():
    m = M.performance_metrics(pd.Series([0.01]), rf_monthly=0.0)
    assert np.isnan(m["Volatilidade anualizada"])
    assert np.isnan(m["Sharpe"])


def test_max_drawdown():
    r = pd.Series([0.10, -0.10, 0.05])      # riqueza: 1.10 -> 0.99 -> 1.0395
    m = M.performance_metrics(r, rf_monthly=0.0)
    assert m["Máximo drawdown"] == pytest.approx(0.99 / 1.10 - 1)


def test_drawdown_counts_loss_in_first_month():
    r = pd.Series([-0.20, 0.10])            # parte do capital inicial de 1
    m = M.performance_metrics(r, rf_monthly=0.0)
    assert m["Máximo drawdown"] == pytest.approx(-0.20)


def test_sharpe_matches_formula():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 36))
    rf = 0.003
    excess = r - rf
    expected = np.sqrt(12) * excess.mean() / excess.std(ddof=1)
    assert M.performance_metrics(r, rf)["Sharpe"] == pytest.approx(expected)


def test_volatility_is_annualized_with_sqrt_12():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.04, 36))
    assert M.performance_metrics(r, 0.0)["Volatilidade anualizada"] == pytest.approx(r.std(ddof=1) * np.sqrt(12))


def test_empty_series_returns_empty():
    assert M.performance_metrics(pd.Series([], dtype=float), 0.0).empty


def test_split_train_test_includes_train_end_in_train():
    idx = pd.period_range("2021-05", "2021-08", freq="M")
    r = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    train, test = M.split_train_test(r, pd.Period("2021-06", freq="M"))
    assert list(train.index) == [pd.Period("2021-05"), pd.Period("2021-06")]
    assert list(test.index) == [pd.Period("2021-07"), pd.Period("2021-08")]