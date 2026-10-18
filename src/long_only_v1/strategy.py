"""Estratégia long-only: duas ações por mês, escolhidas por momentum + filtros.

Regras (todas decididas no fechamento do mês t e aplicadas ao mês t+1):
1. Momentum relativo: ranking das ações pelo retorno de `lookback` meses, pulando o
   último mês (`skip`).
2. Filtros: momentum absoluto positivo e preço acima da média móvel de `window` dias.
3. Escolhem-se as 2 melhores que passam nos filtros; se menos de 2 passam, completa-se
   pelo ranking de momentum entre as restantes (a carteira sempre tem 2 ações).
4. Pesos: a 2ª do ranking recebe `min_weight` e a 1ª recebe `1 - min_weight`.
   `min_weight` é escolhido entre inteiros de 20% a 50% pelo maior Sharpe no TREINO.
"""

import pandas as pd

from . import config
from .metrics import performance_metrics
from .returns import equal_weight_benchmark, to_monthly_prices, to_monthly_returns


def momentum_scores(monthly_prices, lookback=None, skip=1):
    """M_t = P_{t-skip} / P_{t-skip-lookback} - 1.

    Com lookback=6 e skip=1: P_{t-1} / P_{t-7} - 1 (ignora o mês mais recente).
    Se `lookback` for None, usa config.MOMENTUM_MONTHS (lido na hora da chamada).
    """
    lookback = config.MOMENTUM_MONTHS if lookback is None else lookback
    return monthly_prices.shift(skip) / monthly_prices.shift(skip + lookback) - 1


def trend_filter(prices, monthly_prices, window=None):
    """True quando o preço do fim do mês está acima da média móvel diária de `window` dias.

    Se `window` for None, usa config.TREND_WINDOW (lido na hora da chamada).
    """
    window = config.TREND_WINDOW if window is None else window
    daily_ma = prices.rolling(window=window).mean()
    monthly_ma = daily_ma.groupby(daily_ma.index.to_period("M")).last()
    return monthly_prices > monthly_ma


def rank_pairs(momentum, trend_ok):
    """Em cada mês, devolve [1ª do ranking, 2ª do ranking]. Não depende dos pesos."""
    ranked_pairs = {}

    for date in momentum.index:
        scores = momentum.loc[date].dropna()
        if len(scores) < 2:
            continue

        trend_flags = trend_ok.loc[date].reindex(scores.index).fillna(False)

        # Elegível: momentum > 0 e preço > média móvel
        eligible = scores.index[(scores > 0) & trend_flags]

        selected = list(
            scores.loc[eligible].sort_values(ascending=False).index[:2]
        )

        # Menos de duas elegíveis: completa pelo ranking de momentum entre as restantes
        if len(selected) < 2:
            remaining = scores.drop(index=selected)
            fallback = list(
                remaining.sort_values(ascending=False).index[: 2 - len(selected)]
            )
            selected.extend(fallback)

        ranked_pairs[date] = selected

    return ranked_pairs


def top_weight(min_weight):
    """Peso da maior posição (complemento)."""
    return round(1 - min_weight, 2)


def strategy_label(min_weight):
    """Rótulo para tabelas e gráficos, por exemplo 'Estrategia 60/40'."""
    return f"Estrategia {round(top_weight(min_weight) * 100)}/{round(min_weight * 100)}"


def build_weights(ranked_pairs, index, columns, min_weight):
    """Pesos-alvo decididos no fim de cada mês (linhas sem par definido ficam em 0)."""
    assert 0 < min_weight <= 0.5, "min_weight deve estar em (0, 0.5]"

    weights = pd.DataFrame(0.0, index=index, columns=columns)
    for date, selected in ranked_pairs.items():
        weights.loc[date, selected[0]] = top_weight(min_weight)
        weights.loc[date, selected[1]] = min_weight
    return weights


def strategy_returns(weights, monthly_returns):
    """Retorno mensal da carteira. Pesos decididos no mês t valem para o retorno do mês t+1."""
    shifted = weights.shift(1)
    returns = (shifted * monthly_returns).sum(axis=1)
    # Evita registrar retorno 0 antes de existir carteira com duas ações
    return returns.where(shifted.sum(axis=1) > 0)


def search_weight(
    ranked_pairs, monthly_prices, monthly_returns, rf_monthly, train_end,
    grid=None,
):
    """Escolhe o peso da menor posição pelo maior Sharpe NO TREINO.

    Devolve (peso_escolhido, tabela). A coluna de teste é só diagnóstico e não entra na escolha.
    Se `grid` for None, usa config.MIN_WEIGHT_GRID (lido na hora da chamada, não no import).
    """
    grid = config.MIN_WEIGHT_GRID if grid is None else grid
    rows = {}
    for min_weight in grid:
        weights = build_weights(
            ranked_pairs, monthly_prices.index, monthly_prices.columns, min_weight
        )
        r = strategy_returns(weights, monthly_returns).dropna()
        rows[round(min_weight, 2)] = {
            "Sharpe treino": performance_metrics(r.loc[r.index <= train_end], rf_monthly)["Sharpe"],
            "Sharpe teste (diagnóstico)": performance_metrics(r.loc[r.index > train_end], rf_monthly)["Sharpe"],
        }

    table = pd.DataFrame(rows).T
    table.index.name = "Peso da menor posição"
    best = float(table["Sharpe treino"].idxmax())
    return best, table


def run_strategy(prices, rf_monthly, train_end, grid=None, lookback=None, window=None):
    """Pipeline completo: preços diários -> pesos escolhidos -> retornos da estratégia.

    Devolve um dicionário com os objetos intermediários, para o notebook e o script usarem.
    """
    monthly_prices = to_monthly_prices(prices)
    monthly_returns = to_monthly_returns(monthly_prices)

    momentum = momentum_scores(monthly_prices, lookback=lookback)
    trend_ok = trend_filter(prices, monthly_prices, window=window)
    pairs = rank_pairs(momentum, trend_ok)

    min_weight, weight_table = search_weight(
        pairs, monthly_prices, monthly_returns, rf_monthly, train_end, grid
    )
    weights = build_weights(pairs, monthly_prices.index, monthly_prices.columns, min_weight)

    return {
        "monthly_prices": monthly_prices,
        "monthly_returns": monthly_returns,
        "benchmark_returns": equal_weight_benchmark(monthly_returns),
        "momentum": momentum,
        "trend_ok": trend_ok,
        "ranked_pairs": pairs,
        "min_weight": min_weight,
        "top_weight": top_weight(min_weight),
        "weight_table": weight_table,
        "weights": weights,
        "strategy_returns": strategy_returns(weights, monthly_returns),
    }