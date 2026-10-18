"""Gráficos. Nenhuma função altera o estilo global; chame `set_style()` se quiser o 'ggplot'.

Todas aceitam:
- `save_path`: se informado, salva a figura (cria a pasta se preciso).
- `show`: True (padrão) exibe no notebook; use False em scripts (a figura é fechada).
As funções não retornam nada, para o notebook não mostrar a figura duas vezes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_style(style: str = "ggplot") -> None:
    plt.style.use(style)


def _finish(fig, save_path, show) -> None:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------- Preparação dos dados dos gráficos (testável sem desenhar nada) ----------

def _with_initial_row(df: pd.DataFrame, value: float = 1.0) -> pd.DataFrame:
    """Acrescenta o capital inicial (valor 1) no mês anterior ao primeiro retorno."""
    initial = pd.DataFrame(
        [np.full(len(df.columns), value)],
        columns=df.columns,
        index=pd.PeriodIndex([df.index[0] - 1], freq="M"),
    )
    return pd.concat([initial, df])


def growth_path(comparison_period: pd.DataFrame) -> pd.DataFrame:
    """Crescimento de $1 de cada coluna, começando em 1 no mês anterior ao primeiro retorno."""
    growth = (1 + comparison_period).cumprod()
    path = _with_initial_row(growth)
    path.index = path.index.to_timestamp()
    return path


def drawdown_path(comparison_period: pd.DataFrame) -> pd.DataFrame:
    """Drawdown (queda desde o pico) de cada coluna, partindo do capital inicial de $1."""
    wealth = _with_initial_row((1 + comparison_period).cumprod())
    drawdown = wealth / wealth.cummax() - 1
    drawdown.index = drawdown.index.to_timestamp()
    return drawdown


# ---------- Gráficos ----------

def plot_prices(prices, title="Preços ajustados das ações selecionadas", save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    prices.plot(ax=ax, title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(title="Ticker")
    _finish(fig, save_path, show)


def plot_daily_growth(daily_growth, title="Crescimento a partir de $1 investido em cada ação",
                      save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_growth.plot(ax=ax, title=title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor $")
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)


def plot_monthly_growth(monthly_stock_growth, benchmark_growth, strategy_growth=None,
                        strategy_label="Estratégia", save_path=None, show=True):
    """Ações + benchmark (+ estratégia, se informada), em crescimento mensal de $1."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker in monthly_stock_growth.columns:
        ax.plot(monthly_stock_growth.index.to_timestamp(), monthly_stock_growth[ticker],
                alpha=0.65, label=ticker)

    ax.plot(benchmark_growth.index.to_timestamp(), benchmark_growth,
            linewidth=3, label="Benchmark: 20% cada ação")

    if strategy_growth is not None:
        ax.plot(strategy_growth.index.to_timestamp(), strategy_growth,
                linewidth=3, label=strategy_label)

    title = ("Crescimento acumulado: estratégia x benchmark x ações" if strategy_growth is not None
             else "Crescimento acumulado: ações e benchmark")
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Crescimento de US$ 1 investido")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path, show)


def plot_growth(comparison_period, title, save_path=None, show=True):
    """Crescimento de $1 de cada coluna (estratégia, benchmark, SPY...)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    growth_path(comparison_period).plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor de $1 investido")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, save_path, show)


def plot_drawdown(comparison_period, title, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    drawdown_path(comparison_period).plot(ax=ax, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, save_path, show)


def plot_weight_search(weight_table, min_weight, save_path=None, show=True):
    """Sharpe por peso da menor posição, com o peso escolhido marcado."""
    top_weight = round(1 - min_weight, 2)
    fig, ax = plt.subplots(figsize=(10, 4))

    x = weight_table.index.to_numpy() * 100
    ax.plot(x, weight_table["Sharpe treino"], marker="o", label="Treino (usado na escolha)")
    ax.plot(x, weight_table["Sharpe teste (diagnóstico)"], marker="o", alpha=0.6,
            label="Teste (só diagnóstico)")
    ax.axvline(min_weight * 100, color="black", linestyle="--",
               label=f"Escolhido: {top_weight:.0%}/{min_weight:.0%}")

    ax.set_title("Sharpe da estratégia por peso da menor posição")
    ax.set_xlabel("Peso da menor posição (%)")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, save_path, show)