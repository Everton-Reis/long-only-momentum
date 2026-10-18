import numpy as np
import pandas as pd
import pytest

from long_only_v1 import robustness as RB


def test_halves_are_different_and_cover_the_period():
    idx = pd.period_range("2020-01", periods=10, freq="M")
    df = pd.DataFrame({"A": np.linspace(0.00, 0.09, 10)}, index=idx)   # retornos crescentes

    out = RB.compare_halves(df, "X", rf_monthly=0.0)

    first = out[("X - primeira metade", "A")]
    second = out[("X - segunda metade", "A")]
    assert first["Meses"] == 5 and second["Meses"] == 5
    assert second["Retorno acumulado"] > first["Retorno acumulado"]      # não pode ser a mesma metade
    assert first["Retorno acumulado"] == pytest.approx((1 + df["A"].iloc[:5]).prod() - 1)
    assert second["Retorno acumulado"] == pytest.approx((1 + df["A"].iloc[5:]).prod() - 1)


def test_odd_length_puts_extra_month_in_second_half():
    idx = pd.period_range("2020-01", periods=7, freq="M")
    df = pd.DataFrame({"A": np.full(7, 0.01)}, index=idx)
    out = RB.compare_halves(df, "X", 0.0)
    assert out[("X - primeira metade", "A")]["Meses"] == 3
    assert out[("X - segunda metade", "A")]["Meses"] == 4