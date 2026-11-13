"""
Testes da varredura de SNR (run_varredura_snr.py).

Não dependem de data/. Conferem que (1) o ruído tem a SNR pedida no segmento e o
rosa cai com 1/f, (2) o recorte a 51,2 kHz decimado reproduz o `resample_clip`
do `02`, (3) a saturação no int16 é contada, (4) o limite E do escore segue a
fórmula da página de 05/11, (5) o escore da emulação sai pela tolerância relativa
do A4 e por |Δs|/E, (6) as duas regras decidem como estão escritas (a do aumento,
pareada, com o empate, o sinal limpo e o ponto ausente), e (7) o
recorte exportado para a placa dá, no script de envio, o escore e a decisão gravados.
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


def test_escore_perto_do_limiar_sai_pelas_duas_medidas():
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
    assert r["escore"] > 50                                # a tolerância relativa do A4
    assert r["escore_E"] == pytest.approx(4e-4 / 0.41)


def _curva(valor, sem_pontos=()):
    """Curva com todos os pontos e sementes; valor(modelo, ruido, snr, k) → (acc. bal., especificidade)."""
    linhas = []
    pontos = [(None, None)] + [(r, s) for r in V.RUIDOS for s in V.SNRS_DB]
    for ruido, snr in pontos:
        if (ruido, snr) in sem_pontos:
            continue
        for k, sem in enumerate(V.SEMENTES):
            for m in V.MODELOS:
                acc, esp = valor(m, ruido, snr, k)
                linhas.append({"modelo": m, "ruido": ruido, "snr_db": snr, "semente": sem,
                               "acuracia_balanceada": acc, "especificidade": esp})
    return linhas


def _ganho(g, ruido=0.01):
    """O com aumento ganha `g` em todo ponto; o ruído de semente entra igual nos dois modelos."""
    def valor(m, r, s, k):
        d = ruido * (k % 3 - 1) + (g if m == "com_aumento" else 0.0)
        return 0.6 + d, 0.5 + d
    return valor


def test_regra_pareada_passa_com_ganho_constante_por_semente():
    # dₖ = 0,005 em toda semente: desvio pareado 0, passa; a regra antiga (0,005 < desvio 0,01) não passaria
    d = V.decidir_aumento(_curva(_ganho(0.005)))
    assert d["modelo_final"] == "com_aumento"
    assert len(d["pontos"]) == 2 * 3 * 2                 # 2 ruídos × (10, 5, 0 dB) × 2 métricas
    assert d["limpo"]["passa"]


def test_regra_pareada_falha_quando_o_desvio_pareado_passa_da_media():
    # sem e com variam em sentidos opostos: a diferença das médias (0,05) passa do maior desvio (0,03),
    # mas dₖ alterna entre −0,01 e 0,11, e o desvio de dₖ (0,06) passa da média (0,05)
    def valor(m, r, s, k):
        e = 0.03 if k % 2 else -0.03
        return (0.65 - e, 0.55 - e) if m == "com_aumento" else (0.6 + e, 0.5 + e)
    d = V.decidir_aumento(_curva(valor))
    assert d["modelo_final"] == "sem_aumento (exp239)"
    assert all(not p["passa"] for p in d["pontos"])


def test_empate_exato_falha():
    d = V.decidir_aumento(_curva(_ganho(0.0)))
    assert d["modelo_final"] == "sem_aumento (exp239)"
    assert all(p["media_d"] == 0 and p["desvio_d"] == 0 and not p["passa"] for p in d["pontos"])
    assert "empatam" in V.frase_aumento(d)


def test_com_aumento_pior_sem_ruido_falha():
    def valor(m, r, s, k):
        acc, esp = _ganho(0.05)(m, r, s, k)
        return (acc - 0.2, esp) if (r is None and m == "com_aumento") else (acc, esp)
    d = V.decidir_aumento(_curva(valor))
    assert d["modelo_final"] == "sem_aumento (exp239)"
    assert d["nao_passa"] == ["acuracia_balanceada sem ruído menor com aumento"]


def test_ponto_ausente_conta_como_falha():
    d = V.decidir_aumento(_curva(_ganho(0.05), sem_pontos={("rosa", 5)}))
    assert d["modelo_final"] == "sem_aumento (exp239)"
    assert d["nao_passa"] == ["acuracia_balanceada a 5 dB (rosa): sem dados",
                              "especificidade a 5 dB (rosa): sem dados"]
    # 30 dB não entra na regra: faltar não muda nada
    assert V.decidir_aumento(_curva(_ganho(0.05), sem_pontos={("rosa", 30)}))["modelo_final"] == "com_aumento"


def test_placa_pela_tolerancia_do_a4_e_proposta_por_e():
    base = {"perto_limiar": 0, "emul_frac_tol_max": 0.1, "emul_frac_tol_max_E": 0.1, "decisoes_diferentes": 0}
    margem = [{"ruido": None, "snr_db": None, **base, "perto_limiar": 3},   # o limpo não decide
              {"ruido": "branco", "snr_db": 10, **base},
              {"ruido": "rosa", "snr_db": 0, **base}]
    d = V.decidir_placa(margem)
    assert (d["placa"], d["pontos"]) == ("só confirmação (opcional)", [])
    # escore fora da tolerância relativa e dentro de E: a regra manda, a proposta não
    margem[2] |= {"emul_frac_tol_max": 65.0, "emul_frac_tol_max_E": 0.1}
    d = V.decidir_placa(margem)
    assert (d["placa"], d["pontos"]) == ("necessária", [("rosa", 0)])
    assert d["proposta_E"] == {"placa": "só confirmação (opcional)", "pontos": []}
    margem[1]["perto_limiar"] = 1                      # |s| < E vale nos dois critérios
    assert V.decidir_placa(margem)["proposta_E"]["pontos"] == [("branco", 10)]


def test_exportacao_para_a_placa_grava_o_int16_e_as_referencias(tmp_path):
    import csv
    import json

    from validation import matriz_serial_placa as envio
    n_seg, fs12 = 3, config.FS_TRABALHO
    pcm = (8000 * np.sin(np.arange((n_seg + 1) * 4 * fs12) * 0.003)).astype(np.int16)
    clipe = pcm_io.Clip(rotulo="normal", binario="normal", fs=config.FS_ORIGINAL, pcm=pcm)
    segs = [particao.Segmento(i, "normal", "normal", i, 0, i * fs12, (i + 1) * fs12) for i in range(n_seg)]
    pesos = np.random.default_rng(4).normal(0, 1, 2 * config.MFCC_N_COEFS)
    modelo = tmp_path / "lda.json"
    modelo.write_text(json.dumps({"id": "expX", "dobrada": {"pesos": pesos.tolist(), "bias": 0.5}}))
    m = V.exportar_placa(V.Decimador({"normal": clipe}, segs), segs, pesos, 0.5, modelo,
                         bin_path=tmp_path / "seg.bin", out_dir=tmp_path / "placa")
    linhas = list(csv.DictReader((tmp_path / "placa" / "referencias.csv").open(encoding="utf-8")))
    outras = [sem for sem in V.SEMENTES if sem != V.RECORTE_SEMENTE]
    assert m["n_segmentos"] == len(linhas) == (len(V.RUIDOS) * len(V.RECORTE_SNRS_DB) * n_seg
                                               + len(V.PERTO_DO_LIMIAR) * len(outras))
    corpo = np.fromfile(tmp_path / "seg.bin", dtype="<i2").reshape(len(linhas), fs12)
    for l, x in zip(linhas, corpo):     # o .bin, na ordem do CSV, dá o escore e a decisão gravados
        e = envio.escore_python(x, pesos, 0.5)
        assert e == pytest.approx(float(l["escore"]), abs=1e-6)
        assert l["decisao"] == envio.decisao_de(e)
        assert int(l["grupo"]) == V.GRUPO_PLACA[l["ruido"]]
        extra = 0 if int(l["semente"]) == V.RECORTE_SEMENTE else int(l["semente"]) * 100000
        assert int(l["id"]) == extra + int(l["snr_db"]) * 1000 + int(l["segmento"])
    assert len({(l["grupo"], l["id"]) for l in linhas}) == len(linhas)
    # nos pontos perto do limiar, uma linha por semente a mais: a de menor |s|/E daquela semente
    for ruido, snr in V.PERTO_DO_LIMIAR:
        extras = [l for l in linhas if l["ruido"] == ruido and int(l["snr_db"]) == snr
                  and int(l["semente"]) != V.RECORTE_SEMENTE]
        assert sorted(int(l["semente"]) for l in extras) == outras
