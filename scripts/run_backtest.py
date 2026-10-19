"""Roda o backtest completo e salva tabelas e figuras em results/<versão>/.

Uso (na raiz do projeto, com o ambiente ativado):
    python scripts/run_backtest.py                           # estratégia 1 (busca de pesos) -> results/v1
    python scripts/run_backtest.py --strategy inverse_vol    # estratégia 2 (vol. inversa)   -> results/v2
    python scripts/run_backtest.py --use-cache               # lê/gera o cache em data/raw/ (não vai para o Git)
    python scripts/run_backtest.py --version teste           # força outro nome de pasta de resultados

O notebook faz a mesma análise com texto e gráficos na tela; este script serve para
reproduzir tudo com um comando e sem abrir o Jupyter.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janelas: as figuras só são salvas (precisa vir antes dos imports abaixo)

import pandas as pd  # noqa: E402

from long_only_v1 import config  # noqa: E402
from long_only_v1 import costs as C  # noqa: E402
from long_only_v1 import metrics as M  # noqa: E402
from long_only_v1 import plots as P  # noqa: E402
from long_only_v1 import returns as R  # noqa: E402
from long_only_v1 import robustness as RB  # noqa: E402
from long_only_v1 import strategy as S  # noqa: E402
from long_only_v1.config import END_DATE, RF_MONTHLY, START_DATE, TICKERS, TRAIN_END  # noqa: E402
from long_only_v1.data import download_prices, validate_prices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def get_prices(use_cache: bool):
    """Preços das 5 ações e do SPY. Com --use-cache, lê (ou cria) CSVs em data/raw/."""
    cache = ROOT / "data" / "raw"
    stocks_file, spy_file = cache / "prices.csv", cache / "spy.csv"

    if use_cache and stocks_file.exists() and spy_file.exists():
        prices = pd.read_csv(stocks_file, index_col=0, parse_dates=True)
        spy = pd.read_csv(spy_file, index_col=0, parse_dates=True)["SPY"]
        print(f"Preços lidos do cache: {cache}")
    else:
        prices = download_prices(TICKERS, START_DATE, END_DATE)
        spy = download_prices(["SPY"], START_DATE, END_DATE)["SPY"]
        if use_cache:
            cache.mkdir(parents=True, exist_ok=True)
            prices.to_csv(stocks_file)
            spy.to_frame("SPY").to_csv(spy_file)
            print(f"Preços salvos em cache: {cache}")

    validate_prices(prices, TICKERS, END_DATE)
    return prices, spy


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Backtest long-only de 2 ações entre 5.")
    parser.add_argument("--strategy", choices=["weight_search", "inverse_vol"], default="weight_search",
                        help="weight_search: pesos 20%%-50%% escolhidos no treino; inverse_vol: pesos por volatilidade inversa")
    parser.add_argument("--version", default=None,
                        help="pasta em results/ (padrão: v1 para weight_search, v2 para inverse_vol)")
    parser.add_argument("--use-cache", action="store_true", help="usar/gerar CSVs em data/raw/")
    args = parser.parse_args(argv)
    version = args.version or {"weight_search": "v1", "inverse_vol": "v2"}[args.strategy]

    figures = ROOT / "results" / version / "figures"
    tables = ROOT / "results" / version / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    def save_table(df, name):
        (df.to_frame() if isinstance(df, pd.Series) else df).to_csv(tables / f"{name}.csv")

    def fig(name):
        return dict(save_path=figures / name, show=False)

    # 1) Dados -------------------------------------------------------------------------
    prices, spy_prices = get_prices(args.use_cache)
    print(f"Dados: {prices.index.min().date()} a {prices.index.max().date()} | ações: {list(prices.columns)}")
    P.plot_prices(prices, **fig("01_precos.png"))
    P.plot_daily_growth(R.growth_of_one(R.daily_returns(prices)), **fig("02_crescimento_diario.png"))

    # 2) Estratégia (parâmetros vêm do config.py) --------------------------------------
    if args.strategy == "weight_search":
        out = S.run_strategy(prices, RF_MONTHLY, TRAIN_END)
        min_weight = out["min_weight"]
        label = S.strategy_label(min_weight)
        print(f"Pesos escolhidos: {out['top_weight']:.0%} / {min_weight:.0%} (maior Sharpe no treino)")
        save_table(out["weight_table"], "busca_de_pesos")
        P.plot_weight_search(out["weight_table"], min_weight, **fig("04_busca_de_pesos.png"))
    else:
        out = S.run_strategy_inverse_vol(prices)
        label = S.INVERSE_VOL_LABEL
        print(f"Pesos por volatilidade inversa (janela de {config.VOL_WINDOW} pregões)")
        save_table(out["volatility"], "volatilidade_mensal")

    weights, strategy_returns = out["weights"], out["strategy_returns"]
    benchmark_returns, monthly_returns = out["benchmark_returns"], out["monthly_returns"]
    save_table(weights, "pesos_finais")
    save_table(strategy_returns.rename("retorno_estrategia"), "retornos_estrategia")

    stock_growth = R.growth_of_one(monthly_returns)
    benchmark_growth = R.growth_of_one(benchmark_returns)
    strategy_growth = R.growth_of_one(strategy_returns.dropna())
    P.plot_monthly_growth(stock_growth, benchmark_growth, **fig("03_acoes_benchmark.png"))

    # 3) Estratégia vs benchmark (treino e teste) --------------------------------------
    comparison = pd.concat(
        [strategy_returns.rename("Estrategia"), benchmark_returns.rename("Benchmark")], axis=1
    ).dropna()
    train, test = M.split_train_test(comparison, TRAIN_END)
    metrics_train = M.metrics_table(train, RF_MONTHLY)
    metrics_test = M.metrics_table(test, RF_MONTHLY)
    save_table(metrics_train, "metricas_treino")
    save_table(metrics_test, "metricas_teste")

    P.plot_growth(train, "Treino: estratégia vs. benchmark", **fig("05_crescimento_treino.png"))
    P.plot_growth(test, "Out-of-sample: estratégia vs. benchmark", **fig("06_crescimento_teste.png"))
    P.plot_drawdown(train, "Drawdown no treino", **fig("07_drawdown_treino.png"))
    P.plot_drawdown(test, "Drawdown out-of-sample", **fig("08_drawdown_teste.png"))

    # 4) Referência: S&P 500 (SPY) -----------------------------------------------------
    spy_returns = R.to_monthly_returns(R.to_monthly_prices(spy_prices))
    comparison_all = pd.concat(
        [strategy_returns.rename(label), benchmark_returns.rename("Benchmark 5 ativos"),
         spy_returns.rename("S&P 500 (proxy SPY)")],
        axis=1, join="inner",
    ).dropna()
    all_train, all_test = M.split_train_test(comparison_all, TRAIN_END)
    save_table(M.metrics_table(all_train, RF_MONTHLY), "metricas_com_spy_treino")
    save_table(M.metrics_table(all_test, RF_MONTHLY), "metricas_com_spy_teste")
    P.plot_growth(all_test, "Out-of-sample: estratégia vs. benchmark vs. S&P 500", **fig("09_teste_com_spy.png"))

    # 5) Robustez, turnover e custos ---------------------------------------------------
    save_table(RB.compare_halves(all_train, "Treino", RF_MONTHLY), "robustez_treino")
    save_table(RB.compare_halves(all_test, "Teste", RF_MONTHLY), "robustez_teste")

    turnover = C.monthly_turnover(weights, strategy_returns.index)
    save_table(C.turnover_summary(turnover), "turnover_resumo")
    for period, name in [("train", "custos_treino"), ("test", "custos_teste")]:
        save_table(
            C.evaluate_transaction_costs(strategy_returns, turnover, TRAIN_END, period, RF_MONTHLY), name
        )

    # 6) Gráfico final -----------------------------------------------------------------
    P.plot_monthly_growth(stock_growth, benchmark_growth, strategy_growth,
                          strategy_label=label, **fig("10_final.png"))

    print("\nSharpe (treino):\n" + metrics_train.loc["Sharpe"].round(3).to_string())
    print("\nSharpe (teste):\n" + metrics_test.loc["Sharpe"].round(3).to_string())
    print(f"\nTurnover médio mensal: {turnover.mean():.1%}")
    print(f"Resultados salvos em: {ROOT / 'results' / version}")


if __name__ == "__main__":
    main()