import numpy as np
import pandas as pd
import pytest

from long_only_v1 import config
from long_only_v1 import strategy as S
from long_only_v1.config import RF_MONTHLY, TRAIN_END


def run_vol(prices):
    return S.run_strategy_inverse_vol(prices)


def test_weights_follow_inverse_volatility_formula():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    cols = ["A", "B", "C"]
    volatility = pd.DataFrame([[0.02, 0.01, 0.05], [0.01, 0.01, 0.05]], index=idx, columns=cols)
    pairs = {idx[0]: ["A", "B"], idx[1]: ["B", "A"]}

    w = S.build_weights_inverse_vol(pairs, idx, cols, volatility)

    # σ_A = 2·σ_B  ->  w_A = (1/0.02) / (1/0.02 + 1/0.01) = 1/3  e  w_B = 2/3
    assert w.loc[idx[0], "A"] == pytest.approx(1 / 3)
    assert w.loc[idx[0], "B"] == pytest.approx(2 / 3)
    assert w.loc[idx[1], "A"] == pytest.approx(0.5) and w.loc[idx[1], "B"] == pytest.approx(0.5)
    assert (w["C"] == 0).all()


def test_missing_volatility_falls_back_to_equal_weights():
    idx = pd.period_range("2020-01", periods=1, freq="M")
    volatility = pd.DataFrame([[np.nan, 0.01]], index=idx, columns=["A", "B"])
    w = S.build_weights_inverse_vol({idx[0]: ["A", "B"]}, idx, ["A", "B"], volatility)
    assert w.iloc[0].tolist() == [0.5, 0.5]


def test_trailing_volatility_matches_manual_calculation(prices):
    vol = S.trailing_volatility(prices, window=60)
    month = pd.Period("2018-05", freq="M")
    daily = prices["KO"].pct_change()
    last_day = prices.loc["2018-05"].index[-1]
    expected = daily.loc[:last_day].iloc[-60:].std()
    assert vol.loc[month, "KO"] == pytest.approx(expected)


def test_always_two_positions_and_fully_invested(prices):
    weights = run_vol(prices)["weights"]
    active = weights.loc[weights.sum(axis=1) > 0]
    assert len(active) > 100
    assert np.allclose(active.sum(axis=1), 1.0)
    assert ((active > 0).sum(axis=1) == 2).all()
    assert (weights >= 0).all().all()


def test_same_stock_selection_as_the_first_strategy(prices):
    first = S.run_strategy(prices, RF_MONTHLY, TRAIN_END)
    second = run_vol(prices)
    assert second["ranked_pairs"] == first["ranked_pairs"]
    assert ((second["weights"] > 0) == (first["weights"] > 0)).all().all()


def test_lower_volatility_stock_gets_the_larger_weight(prices):
    out = run_vol(prices)
    for date, (first, second) in list(out["ranked_pairs"].items())[:60]:
        vol_first, vol_second = out["volatility"].loc[date, first], out["volatility"].loc[date, second]
        w_first, w_second = out["weights"].loc[date, first], out["weights"].loc[date, second]
        assert (w_first > w_second) == (vol_first < vol_second) or w_first == pytest.approx(w_second)


def test_no_lookahead_in_last_month(prices):
    base = run_vol(prices)
    changed = prices.copy()
    changed.loc["2026-09-01":, "LLY"] = changed.loc["2026-09-01":, "LLY"] * 3
    other = run_vol(changed)

    last = base["strategy_returns"].index[-1]
    pd.testing.assert_series_equal(
        other["strategy_returns"].loc[:last - 1], base["strategy_returns"].loc[:last - 1]
    )


def test_vol_window_is_read_from_config_at_call_time(prices, monkeypatch):
    base = run_vol(prices)["weights"]
    monkeypatch.setattr(config, "VOL_WINDOW", 30)
    other = run_vol(prices)["weights"]
    assert not np.allclose(base.to_numpy(), other.to_numpy())