import numpy as np
import pandas as pd
import pytest

from long_only_v1 import config
from long_only_v1 import strategy as S
from long_only_v1.config import RF_MONTHLY, TRAIN_END


def run(prices):
    return S.run_strategy(prices, RF_MONTHLY, TRAIN_END)


def test_always_two_positions_and_fully_invested(prices):
    weights = run(prices)["weights"]
    active = weights.loc[weights.sum(axis=1) > 0]
    assert len(active) > 100
    assert np.allclose(active.sum(axis=1), 1.0)
    assert ((active > 0).sum(axis=1) == 2).all()
    assert (weights >= 0).all().all()          # long-only, sem venda a descoberto


def test_second_ranked_gets_min_weight_and_first_gets_complement(prices):
    out = run(prices)
    for date, (first, second) in out["ranked_pairs"].items():
        assert out["weights"].loc[date, first] == pytest.approx(out["top_weight"])
        assert out["weights"].loc[date, second] == pytest.approx(out["min_weight"])


def test_chosen_weight_is_inside_grid(prices):
    out = run(prices)
    assert round(out["min_weight"], 2) in [round(w, 2) for w in config.MIN_WEIGHT_GRID]
    assert out["top_weight"] == pytest.approx(1 - out["min_weight"])
    assert out["min_weight"] >= 0.20 and out["min_weight"] <= 0.50


def test_grid_is_read_from_config_at_call_time(prices, monkeypatch):
    # Regressão: mudar o config.py precisa mudar a busca, sem reiniciar o módulo strategy
    monkeypatch.setattr(config, "MIN_WEIGHT_GRID", [0.30, 0.40])
    out = run(prices)
    assert list(out["weight_table"].index) == [0.30, 0.40]
    assert out["min_weight"] in (0.30, 0.40)


def test_weight_choice_does_not_use_test_period(prices):
    base = run(prices)

    changed = prices.copy()
    rng = np.random.default_rng(99)
    after_train = changed.index > pd.Timestamp("2021-06-30")
    noise = np.exp(np.cumsum(rng.normal(0, 0.05, (after_train.sum(), changed.shape[1])), axis=0))
    changed.loc[after_train, :] = changed.loc[after_train, :].to_numpy() * noise

    other = run(changed)
    assert other["min_weight"] == base["min_weight"]
    pd.testing.assert_series_equal(
        other["weight_table"]["Sharpe treino"], base["weight_table"]["Sharpe treino"]
    )


def test_no_lookahead_in_last_month(prices):
    base = run(prices)

    changed = prices.copy()
    changed.loc["2026-09-01":, "LLY"] = changed.loc["2026-09-01":, "LLY"] * 3   # choque só no último mês
    other = run(changed)

    last = base["strategy_returns"].index[-1]
    # Nada antes do último mês pode mudar: retornos e escolhas
    pd.testing.assert_series_equal(
        other["strategy_returns"].loc[:last - 1], base["strategy_returns"].loc[:last - 1]
    )
    for date, pair in base["ranked_pairs"].items():
        if date < last:
            assert other["ranked_pairs"][date] == pair


def test_weights_apply_to_next_month_only(prices):
    out = run(prices)
    weights, monthly_returns = out["weights"], out["monthly_returns"]
    t = weights.index[50]
    expected = (weights.loc[t] * monthly_returns.loc[t + 1]).sum()   # pesos de t rendem o mês t+1
    assert out["strategy_returns"].loc[t + 1] == pytest.approx(expected)


def test_build_weights_rejects_min_weight_above_half(prices):
    out = run(prices)
    with pytest.raises(AssertionError):
        S.build_weights(out["ranked_pairs"], out["monthly_prices"].index, out["monthly_prices"].columns, 0.6)


def test_strategy_label():
    assert S.strategy_label(0.40) == "Estrategia 60/40"
    assert S.strategy_label(0.20) == "Estrategia 80/20"