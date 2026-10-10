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
    # o dsp.py não lê estes valores do config; o contrato diz que coincidem
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
