"""
Testes do run_protocol_c.py, sem data/.

Garantias: o limite superior do alarme falso é o do card (zero alarmes em 8 dá 31 %, em
20 dá 14 %), com casos intermediários conferidos, e a decisão contra X segue os dois
limites; o pente só roda onde a máscara do eixo deixa dentes (20 e 30 Hz); o D2 do C2
tira o limiar fora do bloco, só da calibração, igual a uma conta independente com o
Ledoit–Wolf do sklearn, e marca um ponto longe; e os segmentos
em que o pente não foi avaliado ficam fora da conta.
"""

import json

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


# ---------------------------------------------------------------------------
# Recalibração da LDA pela normal da máquina nova (Registro, 29/10)
# ---------------------------------------------------------------------------
def _maquina(seed=0, n_rot=(5, 10, 20, 30), por_rot=5):
    """Segmentos de uma 'máquina' com rotação, classe e bloco, no formato do C2."""
    rng = np.random.default_rng(seed)
    rot, cls, bl = [], [], []
    for v in n_rot:
        for c in ("H", "BF"):
            for b in range(1, por_rot + 1):
                rot.append(v), cls.append(c), bl.append(b)
    rot, cls, bl = np.array(rot), np.array(cls), np.array(bl)
    X = rng.standard_normal((len(rot), 26))
    calib = (cls == "H") & np.isin(bl, [1, 2])
    return X, rot, cls, bl, calib


def _ref(seed=1):
    rng = np.random.default_rng(seed)
    return {"media": rng.standard_normal(26), "desvio": rng.uniform(0.5, 2.0, 26), "n": 59}


def test_principal_desloca_so_as_13_medias_e_centra_a_calibracao():
    X, rot, _, _, calib = _maquina()
    ref = _ref()
    Xr, info = C.recalibrar(X, rot, calib, ref, "medias13")
    np.testing.assert_array_equal(Xr[:, 13:], X[:, 13:])          # os desvios ficam como medidos
    for v in (5, 10, 20, 30):
        c = calib & (rot == v)
        np.testing.assert_allclose(Xr[c, :13].mean(axis=0), ref["media"][:13], atol=1e-12)
        m = rot == v                                               # o mesmo deslocamento em toda a rotação
        delta = (ref["media"] - X[c].mean(axis=0))[:13]
        np.testing.assert_allclose(Xr[m, :13] - X[m, :13], np.broadcast_to(delta, (m.sum(), 13)), atol=1e-12)
    assert info["n_calibracao"] == 8 and set(info["media_por_rotacao"]) == {5, 10, 20, 30}


def test_filtro_fixo_por_rotacao_e_desfeito_pela_principal():
    # a máquina de referência tem a normal de calibração centrada na referência (média zero);
    # a máquina nova é a mesma com um "filtro fixo" diferente em cada rotação, que soma uma
    # constante às 13 médias. A principal devolve as características originais.
    X0, rot, _, _, calib = _maquina(seed=3)
    for v in (5, 10, 20, 30):
        X0[rot == v, :13] -= X0[calib & (rot == v), :13].mean(axis=0)
    ref = {"media": np.zeros(26), "desvio": np.ones(26), "n": 59}
    rng = np.random.default_rng(4)
    offset = {v: np.r_[rng.normal(0, 5, 13), np.zeros(13)] for v in (5, 10, 20, 30)}
    X = X0 + np.array([offset[v] for v in rot])
    Xr, _ = C.recalibrar(X, rot, calib, ref, "medias13")
    np.testing.assert_allclose(Xr, X0, atol=1e-10)


def test_controle_sem_c0_neutraliza_a_media_e_o_desvio_do_c0():
    X, rot, _, _, calib = _maquina()
    ref = _ref()
    Xr, _ = C.recalibrar(X, rot, calib, ref, "sem_c0")
    Xp, _ = C.recalibrar(X, rot, calib, ref, "medias13")
    assert np.all(Xr[:, 0] == ref["media"][0]) and np.all(Xr[:, 13] == ref["media"][13])
    outras = [j for j in range(26) if j not in (0, 13)]
    np.testing.assert_array_equal(Xr[:, outras], Xp[:, outras])


def test_variante_de_26_e_variante_com_desvio():
    X, rot, _, _, calib = _maquina()
    ref = _ref()
    X26, _ = C.recalibrar(X, rot, calib, ref, "medias26")
    for v in (5, 10, 20, 30):
        np.testing.assert_allclose(X26[calib & (rot == v)].mean(axis=0), ref["media"], atol=1e-12)
    # desvio: n − rotações graus de liberdade e o piso de 0,5 · σ_Jung
    X[calib, 7] = X[calib, 7] * 1e-6                                 # desvio quase zero numa característica
    Xd, info = C.recalibrar(X, rot, calib, ref, "desvio")
    assert info["graus_de_liberdade"] == 8 - 4 and info["caracteristicas_no_piso"] >= 1
    c = calib & (rot == 10)
    mu = X[c].mean(axis=0)
    j = 7                                                            # no piso: divide por 0,5 · σ_Jung
    np.testing.assert_allclose(Xd[rot == 10, j], ref["media"][j] + (X[rot == 10, j] - mu[j]) / 0.5)


def test_recalibracao_recusa_rotacao_sem_calibracao_e_variante_sem_graus_de_liberdade():
    X, rot, cls, bl, calib = _maquina()
    sem_5hz = calib & (rot != 5)
    with pytest.raises(ValueError, match="sem segmento de calibração"):
        C.recalibrar(X, rot, sem_5hz, _ref(), "medias13")
    um_por_rot = (cls == "H") & (bl == 1)
    with pytest.raises(ValueError, match="graus de liberdade"):
        C.recalibrar(X, rot, um_por_rot, _ref(), "desvio")
    with pytest.raises(ValueError, match="variante"):
        C.recalibrar(X, rot, calib, _ref(), "outra")


def test_principal_entra_so_no_bias_no_firmware():
    X, rot, _, _, calib = _maquina()
    ref = _ref()
    rng = np.random.default_rng(5)
    p = {"dobrada": {"pesos": rng.standard_normal(26).tolist(), "bias": 0.3}}
    Xr, info = C.recalibrar(X, rot, calib, ref, "medias13")
    w = np.asarray(p["dobrada"]["pesos"])
    for v, mu in info["media_por_rotacao"].items():
        m = rot == v
        np.testing.assert_allclose(Xr[m] @ w + 0.3, X[m] @ w + C.bias_medias13(p, ref, np.asarray(mu)), atol=1e-10)


def _segs_c2():
    segs = []
    for v in (5, 10, 20, 30):
        for c in ("H", "BF"):
            for b in range(1, 6):
                segs.append(C.SegHust("x", c, "normal" if c == "H" else "falha", v, b, np.zeros(1)))
    cls = np.array([s.classe for s in segs])
    bl = np.array([s.bloco for s in segs])
    rot = np.array([s.rotacao for s in segs])
    return segs, cls, bl, rot, (cls == "H") & np.isin(bl, [4, 5]), cls == "BF"


def test_criterio_12_de_15_a_10_20_e_30_hz_com_5_hz_fora_da_conta():
    segs, cls, bl, rot, mn, mf = _segs_c2()
    d = mf & (rot != 5)                                              # tudo a 10–30 Hz, nada a 5 Hz
    r = C.avaliar_recalibracao(segs, d, mn, mf)
    assert (r["deteccao_criterio"]["k"], r["deteccao_criterio"]["n"], r["deteccao_criterio"]["minimo"]) == (15, 15, 12)
    assert r["deteccao_criterio"]["passa"] and r["deteccao_fora_do_criterio"]["k"] == 0
    d2 = d.copy()
    d2[np.flatnonzero(mf & (rot == 30))[:4]] = False                 # 11 de 15: não passa
    assert not C.avaliar_recalibracao(segs, d2, mn, mf)["deteccao_criterio"]["passa"]
    d3 = d.copy()
    d3[np.flatnonzero(mf & (rot == 30))[:3]] = False                 # 12 de 15: passa
    assert C.avaliar_recalibracao(segs, d3, mn, mf)["deteccao_criterio"]["passa"]


def test_classificacao_funciona_nivel_e_nao_funciona():
    segs, cls, bl, rot, mn, mf = _segs_c2()
    passa = C.avaliar_recalibracao(segs, mf.copy(), mn, mf)
    nao_passa = C.avaliar_recalibracao(segs, np.zeros(len(segs), bool), mn, mf)
    assert C.classificar_recalibracao(passa, passa) == "funciona"
    assert C.classificar_recalibracao(passa, nao_passa) == "detecção pelo nível"
    assert C.classificar_recalibracao(nao_passa, passa) == "não funciona"
    # 2 alarmes em 8 ainda não reprovam (limite inferior de 4,6 %); 3 reprovam (11 %)
    for k, esperado in ((2, "funciona"), (3, "não funciona")):
        d = mf.copy()
        d[np.flatnonzero(mn)[:k]] = True
        r = C.avaliar_recalibracao(segs, d, mn, mf)
        assert C.classificar_recalibracao(r, passa) == esperado


def test_conferencia_no_protocolo_b_e_a_identidade():
    from validation.particao import Fold
    rng = np.random.default_rng(6)
    y = np.array(["normal"] * 40 + ["falha"] * 40)
    X = rng.standard_normal((80, 26)) + np.where(y == "falha", 1.5, 0.0)[:, None]
    folds = [Fold("B", f"B_x_bloco{k}", treino=[i for i in range(80) if i % 4 != k],
                  teste=[i for i in range(80) if i % 4 == k], descartados=[], info={"falha_de_fora": "x"})
             for k in range(4)]
    r = C.conferencia_protocolo_b(X, y, folds)
    assert r["decisoes_diferentes"] == 0 and r["acc_bal_media_sem"] == r["acc_bal_media_com"]
    assert r["maior_mudanca_nas_caracteristicas"] < 1e-12 and r["n_folds"] == 4


def test_divisao_conferida_e_decisoes_por_segmento(tmp_path):
    segs, cls, bl, rot, mn, mf = _segs_c2()
    segs = [C.SegHust(f"{'h' if s.classe == 'H' else 'bf'}_{s.rotacao}hz", s.classe, s.binario, s.rotacao,
                      s.bloco, s.x) for s in segs]
    calib = (cls == "H") & np.isin(bl, [1, 2])
    papel = C.papel_c2(calib, mn, mf, cls == "H")
    assert set(papel[(cls == "H") & (bl == 3)]) == {"descarte"}
    div = {"c2": {"por_rotacao": {str(v): {
        "calibracao": [f"h_{v}hz_a{a}" for a in (1, 2)], "descarte": [f"h_{v}hz_a3"],
        "medida_normal": [f"h_{v}hz_a{a}" for a in (4, 5)],
        "medida_falha": [f"bf_{v}hz_a{a}" for a in range(1, 6)]} for v in (5, 10, 20, 30)}}}
    caminho = tmp_path / "divisao.json"
    caminho.write_text(json.dumps(div), encoding="utf-8")
    assert len(C.conferir_divisao(segs, papel, caminho)) == 10
    errado = papel.copy()
    errado[0] = "medida_normal"                                     # h_5hz_a1 com outro papel
    with pytest.raises(SystemExit, match="difere"):
        C.conferir_divisao(segs, errado, caminho)
    n = len(segs)
    escores = {"sem_recal": np.arange(n, dtype=float), "medias13": -np.arange(n, dtype=float)}
    dispara = {"sem_recal": np.ones(n, bool), "medias13": np.zeros(n, bool), "d2_c2": mf, "combinado": mf}
    C.gravar_decisoes(tmp_path / "d.csv", segs, papel, escores, dispara)
    linhas = (tmp_path / "d.csv").read_text(encoding="utf-8").splitlines()
    assert linhas[0].startswith("segmento,classe,rotacao,aquisicao,papel_c2,escore_sem_recal,falha_sem_recal")
    assert len(linhas) == n + 1 and linhas[1].startswith("h_5hz_a1,H,5,1,calibracao,0.000000,1,")
