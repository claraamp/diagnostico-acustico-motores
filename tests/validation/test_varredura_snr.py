"""
Testes da varredura de SNR (run_varredura_snr.py).

Não dependem de data/. Conferem que (1) o ruído tem a SNR pedida no segmento e o
rosa cai com 1/f, (2) o recorte a 51,2 kHz decimado reproduz o `resample_clip`
do `02`, (3) a saturação no int16 é contada, (4) o limite E do escore segue a
fórmula da página de 05/11, (5) o escore da emulação é julgado por E e não pela
tolerância relativa, e (6) as duas regras decidem como estão escritas.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import config
import dsp
import pcm_io
from validation import particao
from validation import run_varredura_snr as V


@pytest.mark.parametrize("tipo", V.RUIDOS)
@pytest.mark.parametrize("snr", (30, 10, 0))
def test_ruido_tem_a_snr_pedida_no_segmento(tipo, snr):
    rng = np.random.default_rng(0)
    entrada = 3000 * np.sin(np.arange(60000) * 0.01) + rng.normal(0, 100, 60000)
    parte = slice(5000, 55000)
    base = V.ruido_base(tipo, len(entrada), np.random.default_rng(1))
    ruido = V.somar_ruido(entrada, base, parte, snr) - entrada
    medida = 10 * math.log10(np.mean(entrada[parte] ** 2) / np.mean(ruido[parte] ** 2))
    assert medida == pytest.approx(snr, abs=1e-9)


def test_ruido_rosa_cai_com_1_sobre_f():
    r = V.ruido_base("rosa", 1 << 18, np.random.default_rng(2))
    f, p = dsp.psd(r, 51200)
    banda = (f > 20) & (f < 20000)
    inclinacao = np.polyfit(np.log10(f[banda]), np.log10(p[banda]), 1)[0]
    assert inclinacao == pytest.approx(-1.0, abs=0.1)
    assert r.std() == pytest.approx(1.0)


def test_saturacao_no_int16_e_contada():
    pcm, n = V.saturar_int16(np.array([0.4, 32767.4, 32767.6, -32768.4, -32769.0, 40000.0]))
    assert pcm.tolist() == [0, 32767, 32767, -32768, -32768, 32767]
    assert n == 3


def test_recorte_decimado_reproduz_o_02():
    n_seg = 3
    fs12 = config.FS_TRABALHO
    # com sobra no fim, como nas gravações: o resample_clip perde o atraso de grupo lá
    pcm = np.random.default_rng(3).integers(-20000, 20000, (n_seg + 1) * config.FATOR_DECIMACAO * fs12,
                                            dtype=np.int16)
    clipe = pcm_io.Clip(rotulo="normal", binario="normal", fs=config.FS_ORIGINAL, pcm=pcm)
    spec = dsp.design_decimation(config.FS_ORIGINAL, fs12)
    decimado = pcm_io.para_pcm(dsp.resample_clip(clipe.x, config.FS_ORIGINAL, fs12, spec))
    segs = [particao.Segmento(i, "normal", "normal", i, 0, i * fs12, (i + 1) * fs12) for i in range(n_seg)]
    dec = V.Decimador({"normal": clipe}, segs)
    for s in segs:            # o primeiro lê antes do começo da gravação (zeros)
        e, parte = dec.entrada(s)
        assert np.array_equal(e[parte], pcm[4 * s.inicio:4 * s.fim])
        y, _ = V.saturar_int16(dec.decimar(s, e) * config.INT16_FULL)
        assert np.array_equal(y, decimado[s.inicio:s.fim])


def test_limite_do_escore_segue_a_formula_da_pagina():
    nc = config.MFCC_N_COEFS
    pesos = np.arange(1, 2 * nc + 1, dtype=float) * np.where(np.arange(2 * nc) % 2, -1, 1)
    x = np.linspace(-3, 5, 2 * nc)
    bias = -7.0
    delta = np.r_[np.full(nc, 1e-4), 5e-5 * np.abs(x[nc:])]
    u = 2.0 ** -24
    gama = 27 * u / (1 - 27 * u)
    esperado = np.sum(np.abs(pesos) * delta) + gama * (abs(bias) + np.sum(np.abs(pesos * x)))
    assert V.limite_escore(x, pesos, bias)[0] == pytest.approx(esperado, rel=1e-12)


def test_escore_perto_do_limiar_e_julgado_pelo_limite_absoluto():
    # escore de -0,68 com erro de 4e-4: 59× a tolerância relativa, 0,1 % de E
    pesos = np.zeros(2 * config.MFCC_N_COEFS)
    pesos[0] = 1.0

    class Cadeia:
        def escala(self, x): return x
        def janela(self, x): return x
        def espectro(self, x): return x
        def mel(self, x): return x
        def logmel(self, x): return x
        def mfcc(self, x): return x
        def resumo(self, x): return x[0], x[1]
        def escore(self, c): return np.array([-0.6765 + 4e-4])

    um = np.ones((2, 2 * config.MFCC_N_COEFS))
    ref = {e: um for e in ("entrada", "escala", "janela", "espectro", "mel", "logmel", "mfcc")}
    ref |= {"media": um[0], "desvio": um[1], "escore": np.array([-0.6765])}
    r = V.emular(Cadeia(), ref, E=0.41)
    assert r["escore_rel"] > 50
    assert r["escore"] == pytest.approx(4e-4 / 0.41)


def _curva(acc_com, esp_com, dp=0.01):
    linhas = []
    for ruido in V.RUIDOS:
        for snr in V.SNRS_DB:
            for k, sem in enumerate(V.SEMENTES):
                d = dp * (k - 1)
                linhas.append({"modelo": "sem_aumento", "ruido": ruido, "snr_db": snr,
                               "acuracia_balanceada": 0.6 + d, "especificidade": 0.5 + d})
                linhas.append({"modelo": "com_aumento", "ruido": ruido, "snr_db": snr,
                               "acuracia_balanceada": acc_com(ruido, snr) + d,
                               "especificidade": esp_com(ruido, snr) + d})
    return linhas


def test_aumento_so_troca_o_modelo_se_ganhar_em_todos_os_pontos_ate_10_db():
    # o desvio entre sementes é 0,01: ganhar por 0,05 passa
    d = V.decidir_aumento(_curva(lambda r, s: 0.65, lambda r, s: 0.55))
    assert d["modelo_final"] == "com_aumento"
    assert len(d["pontos"]) == 2 * 3 * 2                 # 2 ruídos × (10, 5, 0 dB) × 2 métricas
    # perder um ponto (especificidade a 5 dB, rosa) mantém o modelo; 30 dB não conta
    d = V.decidir_aumento(_curva(lambda r, s: 0.4 if s == 30 else 0.65,
                                 lambda r, s: 0.5 if (r, s) == ("rosa", 5) else 0.55))
    assert d["modelo_final"] == "sem_aumento (exp239)"
    assert d["nao_passa"] == ["especificidade a 5 dB (rosa)"]
    # ganhar por menos que um desvio não basta
    d = V.decidir_aumento(_curva(lambda r, s: 0.605, lambda r, s: 0.505))
    assert d["modelo_final"] == "sem_aumento (exp239)"


def test_placa_necessaria_onde_ha_segmento_perto_do_limiar_ou_emulacao_fora():
    base = {"perto_limiar": 0, "emul_frac_tol_max": 0.1, "decisoes_diferentes": 0}
    margem = [{"ruido": None, "snr_db": None, **base, "perto_limiar": 3},   # o limpo não decide
              {"ruido": "branco", "snr_db": 10, **base},
              {"ruido": "rosa", "snr_db": 0, **base}]
    assert V.decidir_placa(margem) == {"placa": "só confirmação (opcional)", "pontos": []}
    margem[1]["perto_limiar"] = 1
    margem[2]["emul_frac_tol_max"] = 1.2
    assert V.decidir_placa(margem) == {"placa": "necessária",
                                       "pontos": [("branco", 10), ("rosa", 0)]}
