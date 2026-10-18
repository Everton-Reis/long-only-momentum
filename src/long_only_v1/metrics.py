"""Métricas de desempenho e risco calculadas sobre retornos mensais."""

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def performance_metrics(returns: pd.Series, rf_monthly: float) -> pd.Series:
    """Métricas de uma série de retornos MENSAIS.

    - Retorno acumulado e anualizado (geométrico).
    - Volatilidade anualizada: desvio amostral * sqrt(12).
    - Máximo drawdown, partindo do capital inicial de 1.
    - Sharpe anualizado: sqrt(12) * média(r - rf) / desvio(r - rf).

    `rf_monthly` é a taxa livre de risco MENSAL (por exemplo, (1 + 0.04) ** (1/12) - 1).
    """
    r = returns.dropna()

    if len(r) == 0:
        return pd.Series(dtype=float)

    # Crescimento de 1$
    wealth = (1 + r).cumprod()
    total_return = wealth.iloc[-1] - 1

    annual_volatility = r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) if len(r) > 1 else np.nan

    # Capital inicial de 1$, seguido pelo crescimento mensal
    wealth_path = np.r_[1.0, wealth.to_numpy()]
    running_peak = np.maximum.accumulate(wealth_path)
    max_drawdown = (wealth_path / running_peak - 1).min()

    # Taxa anualizada equivalente
    annualized_return = (
        (1 + total_return) ** (MONTHS_PER_YEAR / len(r)) - 1
        if 1 + total_return > 0
        else np.nan
    )

    excess_returns = r - rf_monthly
    excess_vol = excess_returns.std(ddof=1)
    sharpe = (
        np.sqrt(MONTHS_PER_YEAR) * excess_returns.mean() / excess_vol
        if pd.notna(excess_vol) and excess_vol > 0
        else np.nan
    )

    return pd.Series({
        "Meses": len(r),
        "Retorno acumulado": total_return,
        "Retorno anualizado": annualized_return,
        "Volatilidade anualizada": annual_volatility,
        "Máximo drawdown": max_drawdown,
        "Sharpe": sharpe,
    })


def metrics_table(returns: pd.DataFrame, rf_monthly: float) -> pd.DataFrame:
    """Tabela de métricas: uma coluna por série de retornos."""
    return pd.concat(
        {col: performance_metrics(returns[col], rf_monthly) for col in returns.columns},
        axis=1,
    )


def split_train_test(returns, train_end: pd.Period):
    """Divide em treino (índice <= train_end) e teste (índice > train_end)."""
    return returns.loc[returns.index <= train_end], returns.loc[returns.index > train_end]