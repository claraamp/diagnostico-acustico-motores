"""
Testes da montagem do treino com aumento no run_protocol.

Garante que (1) só entram as variantes aceitas no fold, (2) cada variante herda
o rótulo de treino do seu segmento de origem — inclusive o embaralhado do
controle de permutação — e (3) sem aumento nada muda. Confere também o nome da
pasta da rodada e os parâmetros do aumento que vão para o registro.
"""

from __future__ import annotations

import numpy as np

from validation import particao
from validation.run_protocol import montar_treino, parametros_aumento, rotulo_aumento


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


# --------------------------------------------------------------- nome e parâmetros da rodada
_INFO = {"copias": 4, "tecnicas": ["deslocamento", "estiramento", "ruido"],
         "modo_estiramento": "tempo", "desloc_max_s": 0.4, "estir_taxas": [0.95, 1.05],
         "snr_db": [20.0, 35.0], "pv_nfft": 512, "pv_hop": 128}


def test_rotulo_distingue_o_modo_velocidade():
    assert rotulo_aumento(_INFO) == "aum4-desl-estir-ruido"
    assert rotulo_aumento({**_INFO, "modo_estiramento": "velocidade"}) == "aum4-desl-estir-ruido-vel"
    # sem estiramento, o modo não se aplica e não entra no nome
    so_ruido = {**_INFO, "tecnicas": ["ruido"], "modo_estiramento": "velocidade"}
    assert rotulo_aumento(so_ruido) == "aum4-ruido"


def test_parametros_registram_o_phase_vocoder():
    p = parametros_aumento(_INFO)
    assert p["aumento_pv_nfft"] == 512 and p["aumento_pv_hop"] == 128
    assert parametros_aumento(None) == {"aumento": "nenhum"}
