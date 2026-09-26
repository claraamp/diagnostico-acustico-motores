"""
Testes da montagem do treino com aumento no run_protocol.

Garante que (1) só entram as variantes aceitas no fold, (2) cada variante herda
o rótulo de treino do seu segmento de origem — inclusive o embaralhado do
controle de permutação — e (3) sem aumento nada muda.
"""

from __future__ import annotations

import numpy as np

from validation import particao
from validation.run_protocol import montar_treino


def _fold(treino, teste):
    return particao.Fold(protocolo="B", nome="f", treino=list(treino),
                         teste=list(teste), descartados=[])


def test_sem_aumento_e_o_treino_de_sempre():
    X = np.arange(12.0).reshape(6, 2)
    f = _fold([0, 1, 2], [4, 5])
    y_tr = np.array(["a", "b", "a"])
    X_fit, y_fit, n = montar_treino(f, X, y_tr, None)
    np.testing.assert_array_equal(X_fit, X[[0, 1, 2]])
    np.testing.assert_array_equal(y_fit, y_tr)
    assert n == 0


def test_so_variantes_aceitas_e_rotulo_da_origem():
    X = np.zeros((6, 2))
    f = _fold([0, 1, 2], [4, 5])
    # rótulos de treino JÁ embaralhados (como no --permutar): a variante segue a origem
    y_tr = np.array(["falha", "normal", "falha"])
    aumento = {
        "X": np.array([[10.0, 10], [11, 11], [12, 12], [13, 13]]),
        "origem": np.array([1, 2, 4, 2]),
        # var0 lê (1,) ok; var1 lê (2, 3) — 3 é descarte/teste → recusada;
        # var2 é de um segmento de teste → recusada; var3 lê (1, 2) ok
        "lidos": [(1,), (2, 3), (4,), (1, 2)],
    }
    X_fit, y_fit, n = montar_treino(f, X, y_tr, aumento)
    assert n == 2
    np.testing.assert_array_equal(X_fit[3:], [[10, 10], [13, 13]])
    np.testing.assert_array_equal(y_fit, ["falha", "normal", "falha", "normal", "falha"])
