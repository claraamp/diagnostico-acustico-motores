"""
Confere, contra o código, cada valor do docs/contrato_numerico.md.

O contrato diz ao porte em C o que reproduzir. Se a cadeia do Python mudar, este
teste falha, e o documento tem que mudar junto. Cada teste recalcula o estágio
por uma fórmula escrita à mão, independente do `dsp.py`, e compara.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
import dsp

FS = config.FS_TRABALHO
N_FFT = 512
BORDAS_MEL = [0, 4, 7, 11, 16, 21, 27, 33, 40, 48, 57, 67, 78, 90, 104, 119, 136,
              155, 177, 200, 227, 256]


def _segmento(semente=0):
    return np.random.default_rng(semente).normal(0, 0.1, config.AMOSTRAS_POR_SEGMENTO)


def test_valores_do_config_coincidem_com_o_dsp():
    # do config, o dsp.py só lê o MFCC_FMIN; os outros descrevem a cadeia sem ser lidos
    assert config.MFCC_FMIN == 20.0
    assert config.MFCC_FMAX == FS / 2
    assert config.MFCC_JANELA == "hann"
    assert config.N_FFT == N_FFT == 1 << (config.WINDOW_LENGTH - 1).bit_length()
    assert config.INT16_FULL == 32767.0


def test_quadros():
    """98 quadros de 320 amostras com passo de 128; as 64 últimas amostras ficam de fora."""
    x = _segmento()
    assert dsp.log_mel(x, FS).shape == (98, 20)
    y = x.copy()
    y[12736:] = 1e3          # mexer só no que nenhum quadro lê não muda nada
    np.testing.assert_array_equal(dsp.log_mel(x, FS), dsp.log_mel(y, FS))


def test_janela_simetrica():
    w = np.hanning(320)
    n = np.arange(320)
    np.testing.assert_allclose(w, 0.5 - 0.5 * np.cos(2 * np.pi * n / 319), atol=1e-15)
    assert w[0] == w[319] == 0.0 and w[159] == w[160] == pytest.approx(0.9999757531750084)


def test_bordas_e_forma_do_banco_de_mel():
    mels = np.linspace(2595 * np.log10(1 + 20 / 700), 2595 * np.log10(1 + (FS / 2) / 700), 22)
    hz = 700 * (10 ** (mels / 2595) - 1)
    assert np.floor(513 * hz / FS).astype(int).tolist() == BORDAS_MEL
    fb = dsp.mel_filterbank(FS, N_FFT, 20)
    for m in range(20):
        l, c, r = BORDAS_MEL[m:m + 3]
        esperado = np.zeros(257)
        esperado[l:c] = (np.arange(l, c) - l) / (c - l)
        esperado[c:r] = (r - np.arange(c, r)) / (r - c)
        np.testing.assert_allclose(fb[m], esperado, atol=1e-15)
        assert fb[m].max() == 1.0                  # pico 1, sem normalização por área
    assert np.all(fb[:, 0] == 0) and np.all(fb[:, 256] == 0)   # DC e Nyquist fora


def test_cadeia_completa_escrita_a_mao():
    """Hann simétrica → FFT de 512 com zeros no fim → potência → Mel → ln(+1e-10) → DCT-II ortonormal."""
    x = _segmento(1)
    w = np.hanning(320)
    fb = dsp.mel_filterbank(FS, N_FFT, 20)
    k, n = np.arange(13)[:, None], np.arange(20)[None, :]
    D = np.sqrt(2 / 20) * np.cos(np.pi * k * (2 * n + 1) / 40)
    D[0] *= np.sqrt(0.5)                            # s(0) = √(1/20)
    linhas = []
    for q in range(98):
        quadro = np.concatenate([x[128 * q:128 * q + 320] * w, np.zeros(192)])
        potencia = np.abs(np.fft.rfft(quadro)) ** 2  # sem dividir por N
        linhas.append(D @ np.log(fb @ potencia + 1e-10))
    np.testing.assert_allclose(dsp.mfcc(x, FS), np.array(linhas), rtol=1e-12, atol=1e-10)


def test_log_natural_com_epsilon_somado():
    assert dsp.log_mel(np.zeros(12800), FS)[0, 0] == pytest.approx(np.log(1e-10))


def test_resumo_media_e_desvio_populacional():
    """26 valores: médias de c0…c12 e depois desvios, com ddof=0."""
    import importlib.util
    from pathlib import Path
    caminho = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "04_extract_features.py"
    spec = importlib.util.spec_from_file_location("extract_features", caminho)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    x = _segmento(2)
    m = dsp.mfcc(x, FS)
    linha = E.resumo_mfcc(x, norm_clipe=False)
    np.testing.assert_array_equal(linha[:13], m.mean(axis=0))
    np.testing.assert_allclose(linha[13:], np.sqrt(((m - m.mean(axis=0)) ** 2).sum(axis=0) / 98))
    assert not np.allclose(linha[13:], m.std(axis=0, ddof=1))


def _entrada(fonte: str) -> dict:
    import json
    from pathlib import Path
    from models import lda
    if fonte == "cadeia_entrada":
        return lda.cadeia_entrada()
    caminho = Path(__file__).resolve().parents[1] / "reports" / "modelo_final" / "lda_final.json"
    return json.loads(caminho.read_text(encoding="utf-8"))["entrada"]


@pytest.mark.parametrize("fonte", ["cadeia_entrada", "lda_final.json"])
def test_contrato_coincide_com_o_bloco_entrada(fonte):
    """O documento, o `lda.cadeia_entrada()` e o bloco `entrada` do JSON entregue ao firmware dizem o mesmo."""
    c = _entrada(fonte)
    assert c["fs_hz"] == FS and c["amostras_por_segmento"] == 12800
    assert c["escala_entrada"]["divisor"] == 32767.0
    q = c["quadros"]
    assert (q["amostras_por_quadro"], q["passo"], q["n_quadros"]) == (320, 128, 98)
    assert (c["janela"]["tipo"], c["janela"]["simetrica"]) == ("hann", True)
    assert c["fft"]["n_fft"] == N_FFT
    mel = c["mel"]
    assert (mel["n_filtros"], mel["fmin_hz"], mel["fmax_hz"]) == (20, 20.0, FS / 2)
    assert mel["bins"] == BORDAS_MEL and not mel["bins_coincidentes"]
    assert (c["log"]["funcao"], c["log"]["epsilon"]) == ("ln", 1e-10)
    assert (c["dct"]["tipo"], c["dct"]["norma"], c["dct"]["coeficientes"]) == ("II", "ortonormal", 13)
    assert "ddof=0" in c["resumo_por_segmento"]
    o = c["origem_do_sinal"]
    assert (o["fs_original_hz"], o["fator_decimacao"]) == (51200, 4)
    assert (o["fir"]["numtaps"], o["fir"]["corte_hz"], o["fir"]["transicao_hz"]) == (147, 5760.0, 640.0)
    assert "descarte das primeiras 73 amostras" in o["descricao"]


# --------------------------------------------------------------------------- #
# Decimação (seção "Decimação" do contrato)
# --------------------------------------------------------------------------- #
FS_ORIG = config.FS_ORIGINAL
SPEC = dsp.design_decimation(FS_ORIG, FS)


def _decimador_causal(x: np.ndarray, taps: np.ndarray, M: int, fase: int = 0) -> np.ndarray:
    """y[m] = Σ b[k]·x[M·m + p − k], estado inicial zerado; p = 0 é o arm_fir_decimate_f32."""
    from scipy.signal import lfilter
    return lfilter(taps, 1.0, x)[fase::M]


def _gravacao(n_seg=3, semente=3):
    # ruído com banda larga: qualquer desalinhamento de uma amostra aparece
    return np.random.default_rng(semente).normal(0, 0.2, n_seg * FS_ORIG + 2000)


def test_filtro_da_decimacao():
    taps = dsp.taps_decimacao(SPEC, FS_ORIG)
    assert SPEC.down == 4 and SPEC.numtaps == len(taps) == 147
    assert SPEC.cutoff_hz == 5760.0
    np.testing.assert_array_equal(taps, taps[::-1])     # simétrico: a ordem invertida da CMSIS não muda nada
    assert taps.sum() == pytest.approx(1.0, abs=1e-3)    # ganho unitário em DC


def test_amostra_decimada_j_usa_x_de_4j_menos_73_a_4j_mais_73():
    x = _gravacao()
    y = dsp.resample_clip(x, FS_ORIG, FS, SPEC)
    j = 5000
    for i, muda in ((4 * j - 74, False), (4 * j - 73, True), (4 * j + 73, True), (4 * j + 74, False)):
        x2 = x.copy()
        x2[i] += 1.0
        assert (dsp.resample_clip(x2, FS_ORIG, FS, SPEC)[j] != y[j]) == muda, i


@pytest.mark.parametrize("fase, margens", [(0, (75, 73, 37)), (3, (74, 70, 36))])
@pytest.mark.parametrize("segmento", [1, 2])
def test_recorte_reproduz_o_segmento_decimado(segmento, fase, margens):
    """Decimador causal sobre x[a:b], descartando as primeiras saídas = o resample_clip.
    Fase 0 é a do arm_fir_decimate_f32; fase 3 (M − 1), a de um decimador que espera o bloco."""
    x = _gravacao()
    y = dsp.resample_clip(x, FS_ORIG, FS, SPEC)
    n, inicio = config.AMOSTRAS_POR_SEGMENTO, segmento * config.AMOSTRAS_POR_SEGMENTO
    a, b, descartar = dsp.recorte_para_decimar(inicio, n, SPEC, fase)
    assert (inicio * 4 - a, b - (inicio + n) * 4, descartar) == margens
    assert (b - a) % SPEC.down == 0
    saida = _decimador_causal(x[a:b], dsp.taps_decimacao(SPEC, FS_ORIG), SPEC.down, fase)
    np.testing.assert_allclose(saida[descartar:descartar + n], y[inicio:inicio + n], rtol=0, atol=1e-12)
    assert len(saida) == descartar + n


def test_impulso_identifica_a_fase():
    """O teste de impulso do contrato: fase 0 dá b[0], b[4], …; fase 3 dá b[3], b[7], …"""
    taps = dsp.taps_decimacao(SPEC, FS_ORIG)
    impulso = np.zeros(64)
    impulso[0] = 1.0
    np.testing.assert_array_equal(_decimador_causal(impulso, taps, 4, 0)[:4], taps[[0, 4, 8, 12]])
    np.testing.assert_array_equal(_decimador_causal(impulso, taps, 4, 3)[:4], taps[[3, 7, 11, 15]])
    with pytest.raises(ValueError):
        dsp.recorte_para_decimar(12800, 10, SPEC, fase=4)


def test_primeiro_segmento_precisa_de_zeros_antes():
    """No início da gravação o lfilter preenche com zeros: o recorte começa antes de 0."""
    x = _gravacao()
    y = dsp.resample_clip(x, FS_ORIG, FS, SPEC)
    n = config.AMOSTRAS_POR_SEGMENTO
    a, b, descartar = dsp.recorte_para_decimar(0, n, SPEC)
    assert a == -75
    entrada = np.concatenate([np.zeros(-a), x[:b]])
    saida = _decimador_causal(entrada, dsp.taps_decimacao(SPEC, FS_ORIG), SPEC.down)
    np.testing.assert_allclose(saida[descartar:descartar + n], y[:n], rtol=0, atol=1e-12)


def test_pcm_decimado_e_quantizado_para_int16():
    """O 02 grava o decimado em int16 (round meio-para-par, saturado) e o 04 lê esse int16."""
    import pcm_io
    v = np.array([0.5, 1.5, 2.5, -0.5, 1e6, -1e6]) / config.INT16_FULL
    np.testing.assert_array_equal(pcm_io.para_pcm(v), [0, 2, 2, 0, 32767, -32768])


# --------------------------------------------------------------------------- #
# Média e desvio em float32 (seção "Média e desvio em float32" do contrato)
# --------------------------------------------------------------------------- #
def _mfcc_de_referencia() -> np.ndarray:
    """A matriz 98 × 13 de MFCC do reference_data.h (segmento 0 da normal, dado real)."""
    from pathlib import Path
    import re
    texto = (Path(__file__).resolve().parents[1] / "reports" / "c_reference"
             / "reference_data.h").read_text(encoding="ascii")
    corpo = re.search(r"ref_mfcc_matrix\[[^\]]*\] = \{(.*?)\};", texto, re.S).group(1)
    valores = [float(v.strip().rstrip("f")) for v in corpo.split(",") if v.strip()]
    return np.array(valores, dtype=np.float32).reshape(98, 13)


@pytest.mark.parametrize("metodo", dsp.METODOS_RESUMO)
def test_metodos_reproduzem_media_e_desvio_populacional(metodo):
    q32 = _mfcc_de_referencia()
    q = q32.astype(np.float64)
    media, desvio = dsp.media_desvio_float32(q32, metodo)
    assert media.dtype == desvio.dtype == np.float32
    np.testing.assert_allclose(media, q.mean(axis=0), rtol=1e-5, atol=1e-5)
    rtol = 1e-3 if metodo == "soma_quadrados" else 1e-5
    np.testing.assert_allclose(desvio, q.std(axis=0), rtol=rtol)       # ddof=0


def test_soma_dos_quadrados_perde_precisao_com_media_grande():
    """Média grande perto do desvio: Welford e dois passos se mantêm; a soma dos quadrados não."""
    q32 = np.random.default_rng(4).normal(1000.0, 0.1, (98, 13)).astype(np.float32)
    ref = q32.astype(np.float64).std(axis=0)
    erro = {m: np.max(np.abs(dsp.media_desvio_float32(q32, m)[1] - ref) / ref)
            for m in dsp.METODOS_RESUMO}
    assert erro["welford"] < 1e-3 and erro["dois_passos"] < 1e-3
    assert erro["soma_quadrados"] > 1e-2


def test_metodo_desconhecido():
    with pytest.raises(ValueError):
        dsp.media_desvio_float32(np.zeros((98, 13)), "kahan")
