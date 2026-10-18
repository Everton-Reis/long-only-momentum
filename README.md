# Long-Only: 2 de 5 ações com momentum

Backtest de uma estratégia **long-only** que mantém sempre **2 de 5 ações** em carteira (100% comprada, sem alavancagem), comparada a um benchmark igualmente ponderado e rebalanceado todo mês.
Projeto de um desafio técnico de estratégia quantitativa. O foco é o **método** (premissas claras, ausência de look-ahead, custos e análise crítica dos resultados), e não a lucratividade.

> **Aviso:** material educacional. Não é recomendação de investimento. Resultados passados, sobretudo de um universo escolhido com visão posterior, não garantem resultados futuros.

## Resumo

- **Universo:** LLY, PANW, TSM, KO e JPM (preços diários ajustados, em USD).
- **Estratégia:** momentum de 6 meses (pulando o último mês) + filtros de tendência; a cada mês, escolhe as 2 melhores ações.
- **Pesos:** a 2ª ação recebe um peso inteiro entre 20% e 50% e a 1ª recebe o complemento. O peso é escolhido pelo **maior Sharpe no período de treino**.
- **Benchmark:** 20% em cada ação, rebalanceado mensalmente (o S&P 500, via SPY, entra como referência extra).
- **Conclusão:** <!-- PREENCHER com 2 ou 3 linhas: a estratégia superou o benchmark fora da amostra? Quão confiante você está? -->

## Estrutura do repositório

```
.
├── src/long_only_v1/        # código da estratégia (importável)
│   ├── config.py            # tickers, datas, período de treino, grade de pesos, taxa livre de risco
│   ├── data.py              # download e validação dos preços
│   ├── returns.py           # retornos diários/mensais e benchmark
│   ├── strategy.py          # seleção, pesos e retornos da estratégia
│   ├── metrics.py           # Sharpe, drawdown e demais métricas
│   ├── costs.py             # turnover e custos de transação
│   ├── robustness.py        # primeira e segunda metade de cada período
│   └── plots.py             # gráficos
├── notebooks/long_only.ipynb  # análise completa, com texto e gráficos
├── scripts/run_backtest.py    # reproduz tudo por linha de comando
├── tests/                     # testes automatizados (sem internet)
└── results/v1/                # tabelas (.csv) e figuras (.png) geradas
```

## Como reproduzir

Na raiz do projeto:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Pelo script** (salva tudo em `results/v1/`):

```bash
python scripts/run_backtest.py              # baixa os preços do Yahoo Finance
python scripts/run_backtest.py --use-cache  # guarda/usa uma cópia local em data/raw/
```

**Pelo notebook:** instale `requirements-dev.txt`, abra `notebooks/long_only.ipynb` com o kernel do ambiente e use *Restart & Run All*.

A janela de dados é **fixa** (de 2013-01-01 até 2026-10-01, com `end` exclusivo, então o último pregão é 30/09/2026). Os preços ajustados do Yahoo podem variar um pouco com novos dividendos; use `--use-cache` para congelar os dados.

## Metodologia e premissas

**Dados**
- Preços diários ajustados (`auto_adjust=True`), em dólares, via `yfinance`. As ações foram escolhidas pelo autor (setores diferentes), com visão posterior.
- Período: janeiro/2013 a setembro/2026 (o último mês é completo).

**Regras (decididas no fechamento do mês *t* e aplicadas ao mês *t+1*)**
1. **Momentum:** `P(t-1) / P(t-7) - 1` (6 meses, ignorando o mês mais recente).
2. **Filtros:** momentum positivo e preço acima da média móvel de 200 dias.
3. **Seleção:** as 2 melhores que passam nos filtros. Se menos de 2 passam, completa-se pelo ranking de momentum (a carteira sempre tem 2 ações).
4. **Pesos:** 2ª ação = `w`, 1ª ação = `1 - w`, com `w` inteiro entre 20% e 50%, escolhido pelo Sharpe do treino.
5. **Rebalanceamento:** mensal, sem alavancagem e sem venda a descoberto.

**Treino e teste**
- Treino: set/2013 a jun/2021. Teste (fora da amostra): jul/2021 a set/2026.
- Só o peso `w` é escolhido com dados de treino; o teste não participa dessa escolha.

**Métricas**
- Sharpe anualizado = `√12 × média(r - rf) / desvio(r - rf)`, com retornos mensais e taxa livre de risco fixa de 4% ao ano.
- Também: retorno acumulado e anualizado, volatilidade, máximo drawdown (mensal).

**Custos**
- 0, 5, 10 e 25 bps **por lado**, sobre o total negociado, usando o turnover mensal.
- O turnover ignora o *drift* dos pesos durante o mês (subestima o real).

**Execução**
- Supõe-se que as ordens são executadas ao preço de fechamento do último pregão do mês, sem *slippage*, impostos nem câmbio.

## Resultados

<!-- PREENCHER depois de rodar: copie as tabelas de results/v1/tables/ -->

| | Estratégia | Benchmark | SPY |
|---|---|---|---|
| Sharpe (treino) | 1.31 | 1.25 | 0.83 |
| Sharpe (teste) | 1.23 | 1.49 | 0.62 |
| Retorno anualizado (teste) | 34.7 % | 31.8 % | 13.1 % |
| Máx. drawdown (teste) | -17.8 % | -15.0 % | -23.9 % |

![Crescimento acumulado](results/v1/figures/10_final.png)

Todas as tabelas e figuras ficam em [`results/v1/`](results/v1/).

## Limitações

- Apenas 5 ações, escolhidas com visão posterior: os números absolutos não se generalizam.
- Um único corte treino/teste, sem walk-forward.
- Parâmetros de momentum (6 meses) e média móvel (200 dias) fixos; só o peso é escolhido no treino.
- Custos estimados e aplicados só à estratégia; o turnover ignora o *drift* dos pesos.
- Métricas calculadas em retornos mensais: drawdown e volatilidade intramês ficam subestimados.
- Taxa livre de risco constante (4% ao ano).
- Com ~13 anos e 5 ações, a incerteza estatística é grande: diferenças pequenas de Sharpe entre estratégia e benchmark não são conclusivas.

## Possíveis melhorias

- Curva diária com rebalanceamento mensal e drawdown diário.
- Teste placebo (sortear 2 das 5 ações) e bootstrap em blocos para a diferença de Sharpe.
- Análise de sensibilidade (janelas de momentum, média móvel, pesos).
- Universo maior, definido por regra objetiva, para reduzir o viés de seleção.
- Taxa livre de risco real, custos também no benchmark e turnover com *drift*.
- Métricas de cauda (VaR e CVaR).git 

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes usam preços sintéticos (não precisam de internet) e verificam, entre outras coisas: sempre 2 posições e 100% investido, ausência de look-ahead, que a escolha do peso não usa o período de teste, o cálculo do Sharpe e do drawdown, e o turnover e os custos.

## Uso de IA

<!-- AJUSTE ao que de fato aconteceu -->
Ferramentas de IA foram usadas como apoio (revisão, organização do código e testes). Entretanto, mesmo com a precisão da IA, todos os resultados foram revisados manualmente.
