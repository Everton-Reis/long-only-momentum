"""Robustez: estabilidade do desempenho ao longo do tempo."""

import pandas as pd

from .metrics import performance_metrics


def compare_halves(comparison_period: pd.DataFrame, label: str, rf_monthly: float) -> pd.DataFrame:
    """Métricas de cada coluna na primeira e na segunda metade do período.

    `comparison_period`: DataFrame de retornos mensais (uma coluna por carteira).
    """
    midpoint = len(comparison_period) // 2

    first = comparison_period.iloc[:midpoint]
    second = comparison_period.iloc[midpoint:]

    # As metades não podem se sobrepor e, juntas, cobrem o período inteiro
    assert first.index.max() < second.index.min()
    assert len(first) + len(second) == len(comparison_period)

    halves = {
        f"{label} - primeira metade": first,
        f"{label} - segunda metade": second,
    }

    results = {}
    for half_name, half_returns in halves.items():
        for column in half_returns.columns:
            results[(half_name, column)] = performance_metrics(half_returns[column], rf_monthly)

    return pd.concat(results, axis=1)