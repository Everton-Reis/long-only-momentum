"""Turnover e custos de transação (modelo simples: bps sobre o valor negociado)."""

import pandas as pd

from .metrics import performance_metrics

# Custos testados, em pontos-base (bps) POR LADO (compra ou venda)
COST_BPS_GRID = (0, 5, 10, 25)


def monthly_turnover(weights: pd.DataFrame, index) -> pd.Series:
    """Turnover mensal: fração do capital vendida (e comprada) em cada rebalanceamento.

    - A mudança de pesos decidida no fechamento do mês t é cobrada no mês t+1,
      que é quando os novos pesos passam a render (por isso o `shift(1)`).
    - Dividimos por 2 porque cada real vendido vira um real comprado.
    - ATENÇÃO: subestima o turnover real, pois ignora o "drift" dos pesos durante o mês
      (mesmo sem trocar de ação, voltar ao peso-alvo exige negociar um pouco).

    `index` é o índice dos retornos da estratégia (para alinhar os meses).
    """
    weight_changes = weights.diff().abs().sum(axis=1)
    turnover = (weight_changes / 2).shift(1)
    return turnover.reindex(index).fillna(0)


def turnover_summary(turnover: pd.Series) -> pd.Series:
    """Média, mediana e máximo do turnover mensal."""
    return pd.Series({
        "Turnover médio mensal": turnover.mean(),
        "Turnover mediano mensal": turnover.median(),
        "Turnover máximo mensal": turnover.max(),
    })


def apply_transaction_costs(returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Retorno líquido: desconta `cost_bps` por lado sobre o total negociado (compras + vendas)."""
    cost_rate = cost_bps / 10_000
    # Total negociado = 2 * turnover (vende X e compra X)
    costs = 2 * turnover * cost_rate
    return returns - costs


def evaluate_transaction_costs(
    returns: pd.Series,
    turnover: pd.Series,
    train_end: pd.Period,
    period: str,
    rf_monthly: float,
    cost_bps_grid=COST_BPS_GRID,
) -> pd.DataFrame:
    """Métricas da estratégia líquida de custos, uma coluna por custo, no treino ou no teste."""
    if period == "train":
        mask = returns.index <= train_end
    elif period == "test":
        mask = returns.index > train_end
    else:
        raise ValueError("period deve ser 'train' ou 'test'.")

    results = {}
    for cost_bps in cost_bps_grid:
        net_returns = apply_transaction_costs(returns, turnover, cost_bps).loc[mask]
        results[f"{cost_bps} bps"] = performance_metrics(net_returns, rf_monthly)

    return pd.DataFrame(results)