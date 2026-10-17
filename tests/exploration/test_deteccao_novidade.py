"""
Testes da detecção de novidade (inspect_deteccao_novidade.py), sem data/.

Garantias: o D1b é o espectro do critério; a série sintética do D4 sai na
amplitude pedida e só entra no treino de cada fold a partir de segmentos normais
de treino; o Mahalanobis separa a normal de um ponto distante; os pentes do
controle ficam longe de BPFO, BPFI e das frações simples delas; o pente fixo
pontua alto só onde há série; e a regra de escolha descarta o D3 que não supera
o D2 e aplica o controle relativo.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import config
from validation import particao

_CAMINHO = Path(__file__).resolve().parents[2] / "scripts" / "exploration" / "inspect_deteccao_novidade.py"
_spec = importlib.util.spec_from_file_location("inspect_deteccao_novidade", _CAMINHO)
N = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = N
_spec.loader.exec_module(N)
P = N.P

FS = config.FS_TRABALHO


def test_d1b_e_o_espectro_do_criterio():
    c = N.CAND_D1B
    assert (c.fs, c.n_fft, c.n_blocos, c.fmax) == (6400.0, 2048, 3, 2800.0)
    assert c.resolucao == pytest.approx(3.125) and P.inviavel(c) is None
    f, p = P.espectro(np.random.default_rng(0).standard_normal(FS), c)
    assert len(f) == len(p) == 1025 and f[-1] == pytest.approx(3200.0)
    assert P.custo_stm32(c)["ram_kb"] == pytest.approx(4 * (2 * 2048 + 1025) / 1024)


def test_serie_sintetica_sai_na_amplitude_pedida():
    x = 0.01 * np.random.default_rng(1).standard_normal(FS)
    rng = np.random.default_rng(2)
    estado = rng.bit_generator.state
    y, info = N.injetar_serie(x, rng)
    assert info["familia"] in ("bpfo", "bpfi")
    assert N.D4_HARMONICOS[0] <= info["n_harmonicos"] <= N.D4_HARMONICOS[1]
    # refaz o sorteio para saber os dB pedidos de cada dente
    rng.bit_generator.state = estado
    rng.integers(2); rng.uniform(-1, 1)
    rng.integers(N.D4_HARMONICOS[0], N.D4_HARMONICOS[1] + 1)
    db = rng.uniform(*N.D4_DB, size=info["n_harmonicos"])
    pot = np.abs(np.fft.rfft(y * np.hanning(FS))) ** 2
    pot0 = np.abs(np.fft.rfft(x * np.hanning(FS))) ** 2
    erro = []
    for k in range(1, info["n_harmonicos"] + 1):
        b = int(round(k * info["f0"]))
        fundo = np.median(pot0[np.r_[b - 10:b - 2, b + 3:b + 11]])
        erro.append(10 * np.log10(pot[b - 1:b + 2].max() / fundo) - db[k - 1])
    # o ruído no bin do dente e o f0 fora do centro do bin (até 1,4 dB na Hann)
    # espalham o erro de cada dente; a mediana tem de ficar perto de zero
    erro = np.array(erro)
    assert abs(np.median(erro)) < 1.0
    assert np.mean(np.abs(erro) < 3.0) >= 0.8


@pytest.fixture(scope="module")
def sintetico():
    lista = P.sintetico()
    part = particao.gerar_particoes(lista)
    return lista, particao.segmentos_de(part), particao.folds_de(part, "B")


def test_d4_usa_so_normais_de_treino(sintetico):
    lista, segmentos, folds = sintetico
    clipes = {c.rotulo: c for c in lista}
    y = np.array([s.binario for s in segmentos])
    X = np.random.default_rng(3).standard_normal((len(segmentos), 26))
    f = folds[0]
    normais_treino = [i for i in f.treino if y[i] == "normal"]
    # só as cópias permitidas existem: qualquer outra leitura levantaria KeyError
    sint = {i: X[i] + 5 for i in normais_treino}
    N.rodar_b([f], segmentos, y, X, "D4", sinteticas=sint)
    todas = N.features_sinteticas({c.rotulo: c for c in lista}, segmentos[:120], config.SEMENTE)
    assert set(todas) == {s.id for s in segmentos[:120] if s.binario == "normal"}
    del clipes


def test_mahalanobis_separa_normal_de_ponto_distante():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((40, 26))
    m = N.Mahalanobis().fit(X)
    perto, longe = m.pontuar(X.mean(axis=0, keepdims=True)), m.pontuar(X.mean(axis=0, keepdims=True) + 10)
    assert perto[0] < 1 and longe[0] > 50


def test_razoes_do_controle_ficam_longe_das_de_rolamento():
    for r in (r for par in N.razoes_controle(50) for r in par):
        assert N.CTRL_RAZOES[0] <= r <= N.CTRL_RAZOES[1]
        assert abs(r - round(r)) > N.CTRL_LONGE_INTEIRO
        for b in (P.RAZAO_BPFO, P.RAZAO_BPFI):
            for fr in N.FRACOES_BPF:
                assert abs(r / (b * fr) - 1) > N.CTRL_LONGE_BPF


def test_pente_fixo_pontua_alto_so_onde_ha_serie():
    rng = np.random.default_rng(5)
    t = np.arange(FS) / FS
    f0 = P.RAZAO_BPFO * N.EIXO_NOMINAL_HZ
    x = 0.01 * rng.standard_normal(FS) + sum(0.01 * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 25))
    c = N.CAND_B
    f, p = P.espectro(x, c)
    pico, fundo = P.pico_e_fundo(f, p, c.fundo_hz)
    na_serie, _ = N.pente_fixo(f, pico, fundo, N.EIXO_NOMINAL_HZ, P.RAZAO_BPFO, c)
    fora, _ = N.pente_fixo(f, pico, fundo, N.EIXO_NOMINAL_HZ, N.razoes_controle(1)[0][0], c)
    assert na_serie > 15 and fora < 6


def _resultado(alvo, espec=1.0, outras=1.0):
    pf = {f: {"acuracia_balanceada": outras} for f in ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_1.0mm")}
    pf["bpfo_0.3mm"] = {"acuracia_balanceada": alvo}
    return {"resumo": {"especificidade_media": espec, "por_falha": pf}}


def test_regra_de_escolha():
    ok = {"passa": True}
    assert N.avaliar(_resultado(0.9), ok)["situacao"].startswith("candidato")
    assert N.avaliar(_resultado(0.9), {"passa": False})["situacao"] == "não elegível"
    assert N.avaliar(_resultado(0.9, espec=0.9), ok)["situacao"] == "não elegível"
    assert N.avaliar(_resultado(0.6), ok)["situacao"].startswith("elegível, mas")
    assert N.avaliar(_resultado(0.9), ok, d2_alvo=0.89)["situacao"].startswith("descartado")
    assert N.avaliar(_resultado(0.95), ok, d2_alvo=0.90)["situacao"].startswith("candidato")
    assert N.avaliar(_resultado(0.9), ok, decide=False)["situacao"].startswith("só reporte")
