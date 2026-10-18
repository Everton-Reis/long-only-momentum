"""Fixtures compartilhadas: preços sintéticos (sem internet) para testar a lógica."""

import numpy as np
import pandas as pd
import pytest

TICKERS_TESTE = ["LLY", "PANW", "TSM", "KO", "JPM"]


@pytest.fixture(scope="session")
def prices():
    """Cinco séries de preços diários sintéticas, de 02/01/2013 a 30/09/2026 (não alterar nos testes)."""
    index = pd.bdate_range("2013-01-02", "2026-09-30")
    cols = {}
    for k, ticker in enumerate(TICKERS_TESTE):
        rng = np.random.default_rng(10 * k + 3)
        daily = rng.normal(0.0006 + 0.0001 * k, 0.017 + 0.002 * k, len(index))
        cols[ticker] = 50 * np.exp(np.cumsum(daily))
    return pd.DataFrame(cols, index=index)