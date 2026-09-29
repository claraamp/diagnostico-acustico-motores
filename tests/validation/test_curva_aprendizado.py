"""
Testes da subamostragem do treino na curva de aprendizado do run_protocol.

Garante que (1) só saem ids de treino, então o teste e a faixa de descarte
continuam fora, (2) cada classe recebe o número pedido, ou tudo o que tem se
tiver menos, (3) a classe que junta várias gravações é repartida entre elas, e
(4) a mesma semente sorteia os mesmos segmentos.
"""

from __future__ import annotations

import numpy as np

from validation.run_protocol import cotas_por_gravacao, subamostrar_treino

# 10 segmentos normais e 30 de falha, vindos de três gravações de 10
GRAV = np.array(["normal"] * 10 + ["f1"] * 10 + ["f2"] * 10 + ["f3"] * 10)
Y = np.where(GRAV == "normal", "normal", "falha")
TREINO = list(range(2, 8)) + list(range(12, 40))     # 6 normais; 8 + 10 + 10 de falha


def _sub(n, semente=0):
    return subamostrar_treino(TREINO, Y, GRAV, n, np.random.default_rng(semente))


def test_so_ids_de_treino():
    sub = _sub(5)
    assert set(sub) <= set(TREINO)
    assert sub == sorted(sub)


def test_n_por_classe_e_classe_menor_entra_inteira():
    sub = _sub(5)
    assert sum(Y[sub] == "normal") == 5
    assert sum(Y[sub] == "falha") == 5
    tudo = _sub(30)
    assert sum(Y[tudo] == "normal") == 6             # só há 6 normais no treino
    assert sum(Y[tudo] == "falha") == 28


def test_falha_repartida_entre_as_gravacoes():
    for semente in range(20):
        sub = _sub(6, semente)
        _, n = np.unique(GRAV[sub][Y[sub] == "falha"], return_counts=True)
        assert list(n) == [2, 2, 2]
        sub = _sub(2, semente)                        # 2 segmentos, 3 gravações
        assert len(set(GRAV[sub][Y[sub] == "falha"])) == 2


def test_cota_passa_para_as_outras_quando_a_gravacao_acaba():
    cotas = cotas_por_gravacao({"f1": 1, "f2": 10, "f3": 10}, 9, np.random.default_rng(0))
    assert cotas["f1"] == 1
    assert cotas["f2"] + cotas["f3"] == 8
    assert abs(cotas["f2"] - cotas["f3"]) <= 1


def test_mesma_semente_mesmo_sorteio():
    assert _sub(3, 42) == _sub(3, 42)
    assert _sub(3, 42) != _sub(3, 43)
