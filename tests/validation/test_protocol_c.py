"""
Testes do run_protocol_c.py, sem data/.

Garantias: o limite superior do alarme falso é o do card (zero alarmes em 8 dá 31 %, em
20 dá 14 %), com casos intermediários conferidos, e a decisão contra X segue os dois
limites; o pente só roda onde a máscara do eixo deixa dentes (20 e 30 Hz); o D2 do C2
tira o limiar fora do bloco, só da calibração, igual a uma conta independente com o
Ledoit–Wolf do sklearn, e marca um ponto longe; e os segmentos
em que o pente não foi avaliado ficam fora da conta.
"""

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from validation import run_protocol_c as C


def test_limite_superior_do_alarme_falso():
    assert C.limite_superior(0, 8) == pytest.approx(1 - 0.05 ** (1 / 8))        # 31 %
    assert C.limite_superior(0, 8) == pytest.approx(0.312, abs=1e-3)
    assert C.limite_superior(0, 20) == pytest.approx(0.139, abs=1e-3)
    assert C.limite_superior(8, 8) == 1.0
    lo, hi = C.intervalo(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 10))
    # casos intermediários (0 < k < n), onde a beta pode errar
    assert C.limite_superior(1, 4) == pytest.approx(0.751, abs=1e-3)
    assert C.limite_superior(3, 4) == pytest.approx(0.987, abs=1e-3)
    assert C.intervalo(3, 4) == pytest.approx((0.194, 0.994), abs=1e-3)


def test_decisao_contra_x():
    assert C.decidir(0, 8) == "inconclusivo"        # limite superior de 31 %
    assert C.decidir(0, 59) == "aprova"             # 1 - 0,05^(1/59) < 5 %
    assert C.decidir(0, 58) == "inconclusivo"
    assert C.decidir(3, 4) == "reprova"             # limite inferior de 25 %
    assert C.decidir(8, 8) == "reprova"
    assert C.decidir(1, 100) == "aprova"            # limite superior de 4,7 %
    assert C.decidir(2, 40) == "inconclusivo"       # 5 % medido: limites de 0,9 % e 15 %


def test_alarme_esperado_pelo_maximo():
    assert C.alarme_esperado_pelo_maximo(8) == pytest.approx(1 / 9)
    assert C.alarme_esperado_pelo_maximo(4, 2) == pytest.approx(1 - 0.8 ** 2)


def test_pente_so_onde_a_mascara_deixa_dentes():
    assert [C.pente_viavel(v) for v in (5, 10, 20, 30)] == [False, False, True, True]


def test_d2_do_c2_calibra_so_na_calibracao_e_marca_o_ponto_longe():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 26))
    calib = np.zeros(12, dtype=bool)
    calib[:8] = True
    X[11] += 20                                     # um segmento de medida muito longe da normal
    dispara, lim, encolhimento = C.mahalanobis_c2(X, calib)
    assert dispara[11] and lim > 0
    # conta independente: padroniza e ajusta o Ledoit–Wolf do sklearn direto, deixando
    # cada segmento de calibração de fora, e a distância ao quadrado é a forma quadrática
    loo = []
    for i in range(8):
        A = np.delete(X[:8], i, axis=0)
        mu, sd = A.mean(axis=0), A.std(axis=0)
        lw = LedoitWolf().fit((A - mu) / sd)
        z = (X[i] - mu) / sd
        loo.append(z @ np.linalg.inv(lw.covariance_) @ z)
    assert lim == pytest.approx(max(loo), rel=1e-9)
    assert 0 < encolhimento["final"] < 1 and encolhimento["n_ajuste_fora_do_bloco"] == 7
    # mexer num segmento que não é de calibração não muda o limiar
    X2 = X.copy()
    X2[9] += 50
    assert C.mahalanobis_c2(X2, calib)[1] == pytest.approx(lim)


def test_pente_nao_avaliado_fica_fora_da_conta():
    segs = [C.SegHust("x", "H" if i < 4 else "BF", "normal" if i < 4 else "falha", v, 1, np.zeros(1))
            for i, v in enumerate((5, 20, 5, 30, 10, 20, 30, 5))]
    pont = np.array([[np.nan, np.nan], [1, 1], [np.nan, np.nan], [3, 0],
                     [np.nan, np.nan], [9, 0], [0, 9], [np.nan, np.nan]])
    d = C.dispara_pente(pont, np.array([2.0, 2.0]))
    normal = np.array([True] * 4 + [False] * 4)
    r = C.leitura(segs, d, np.zeros(8, dtype=bool), normal, ~normal)["sozinho"]
    assert (r["alarme"]["k"], r["alarme"]["n"]) == (1, 2)          # só 20 e 30 Hz contam
    assert (r["deteccao"]["k"], r["deteccao"]["n"]) == (2, 2)
    assert set(r["por_rotacao"]) == {20, 30}
