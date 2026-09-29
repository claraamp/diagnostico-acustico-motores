"""
Testes da subamostragem do treino na curva de aprendizado do run_protocol.

Garante que (1) só saem ids de treino, então o teste e a faixa de descarte
continuam fora, (2) cada classe recebe o número pedido, ou tudo o que tem se
tiver menos, e (3) a mesma semente sorteia os mesmos segmentos.
"""

from __future__ import annotations

import numpy as np

from validation.run_protocol import subamostrar_treino

Y = np.array(["normal"] * 10 + ["falha"] * 30)
TREINO = list(range(2, 8)) + list(range(12, 40))     # 6 normais e 28 falhas


def test_so_ids_de_treino():
    sub = subamostrar_treino(TREINO, Y, 5, np.random.default_rng(0))
    assert set(sub) <= set(TREINO)
    assert sub == sorted(sub)


def test_n_por_classe_e_classe_menor_entra_inteira():
    sub = subamostrar_treino(TREINO, Y, 5, np.random.default_rng(0))
    assert sum(Y[sub] == "normal") == 5
    assert sum(Y[sub] == "falha") == 5
    tudo = subamostrar_treino(TREINO, Y, 30, np.random.default_rng(0))
    assert sum(Y[tudo] == "normal") == 6             # só há 6 normais no treino
    assert sum(Y[tudo] == "falha") == 28


def test_mesma_semente_mesmo_sorteio():
    a = subamostrar_treino(TREINO, Y, 3, np.random.default_rng(42))
    b = subamostrar_treino(TREINO, Y, 3, np.random.default_rng(42))
    c = subamostrar_treino(TREINO, Y, 3, np.random.default_rng(43))
    assert a == b
    assert a != c
