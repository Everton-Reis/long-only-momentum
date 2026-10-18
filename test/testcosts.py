import numpy as np
import pandas as pd
import pytest

from long_only_v1 import costs as C
from long_only_v1 import metrics as M


def make_weights():
    idx = pd.period_range("2020-01", periods=5, freq="M")
    w = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])
    w.iloc[1] = [0.6, 0.4, 0.0]     # entra na carteira
    w.iloc[2] = [0.4, 0.6, 0.0]     # só troca a ordem
    w.iloc[3] = [0.0, 0.6, 0.4]     # sai A, entra C
    w.iloc[4] = [0.0, 0.6, 0.4]     # sem mudança
    return w


def test_turnover_values_are_charged_in_the_following_month():
    w = make_weights()
    t = C.monthly_turnover(w, w.index)
    # mudança decidida em t cobra em t+1; turnover = metade da soma de |Δpeso|
    assert t.iloc[0] == 0 and t.iloc[1] == 0
    assert t.iloc[2] == pytest.approx(0.5)    # entrada: |0.6|+|0.4| = 1.0 -> 0.5
    assert t.iloc[3] == pytest.approx(0.2)    # troca de ordem: 0.2+0.2 = 0.4 -> 0.2
    assert t.iloc[4] == pytest.approx(0.4)    # sai A (0.4) entra C (0.4): 0.8 -> 0.4


def test_cost_formula():
    r = pd.Series([0.02, 0.02], index=pd.period_range("2020-01", periods=2, freq="M"))
    turnover = pd.Series([0.5, 0.0], index=r.index)
    net = C.apply_transaction_costs(r, turnover, cost_bps=10)
    # custo = 2 * turnover * 10 bps = 2 * 0.5 * 0.001 = 0.001
    assert net.iloc[0] == pytest.approx(0.02 - 0.001)
    assert net.iloc[1] == pytest.approx(0.02)


def test_zero_cost_equals_gross_metrics_and_more_cost_lowers_sharpe():
    idx = pd.period_range("2018-01", periods=60, freq="M")
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.01, 0.03, 60), index=idx)
    turnover = pd.Series(0.25, index=idx)
    train_end = pd.Period("2020-12", freq="M")

    table = C.evaluate_transaction_costs(r, turnover, train_end, "test", rf_monthly=0.003)
    gross = M.performance_metrics(r.loc[r.index > train_end], 0.003)
    pd.testing.assert_series_equal(table["0 bps"], gross, check_names=False)

    sharpes = table.loc["Sharpe"].to_numpy()
    assert (np.diff(sharpes) < 0).all()      # mais custo -> menor Sharpe


def test_invalid_period_raises():
    r = pd.Series([0.01], index=pd.period_range("2020-01", periods=1, freq="M"))
    with pytest.raises(ValueError):
        C.evaluate_transaction_costs(r, r * 0, pd.Period("2020-01", freq="M"), "x", 0.0)