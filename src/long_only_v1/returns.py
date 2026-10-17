"""Retornos diários e mensais, crescimento acumulado  e benchmark igualmente dividido"""
import pandas as pd

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorno simples diário"""
    returns = prices.pct_change(fill_method=None)
    return returns.dropna(how="all")

def growth_of_one(returns):
    return (1 + returns).cumprod()

def to_monthly_prices(prices):
    """Preço ajustado no último pregão de cada mês"""
    return prices.groupby(prices.index.to_period("M")).last()

def to_monthly_returns(monthly_prices):
    """Retorno mensal: definido como a variação entre o fechamento de dois meses consecutivos
    A primeira linha é Nan (não há mês anterior)
    """
    return monthly_prices.pct_change(fill_method=None)

def equal_weight_benchmark(monthly_returns: pd.DataFrame) -> pd.Series:
    """Benchmark: 20% em cada ação que é rebalanceado todo mês
    A média simples dos retornos mensais equivale a manter pesos iguais no início de cada mês
    """
    return monthly_returns.mean(axis=1)

