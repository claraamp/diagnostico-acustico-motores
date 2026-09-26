"""
Testes das transformações de sinal do aumento de dados.

Usam sinais sintéticos (senoides, ruído); não dependem de data/.

    pytest tests/augmentation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import config
from augmentation import transformacoes as T

FS = config.FS_TRABALHO
N = config.AMOSTRAS_POR_SEGMENTO


def _senoide(f_hz: float, n: int) -> np.ndarray:
    return np.sin(2 * np.pi * f_hz * np.arange(n) / FS)


def _pico_hz(y: np.ndarray) -> float:
    espectro = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.fft.rfftfreq(len(y), 1 / FS)[np.argmax(espectro)])


# --------------------------------------------------------------- estiramento
def test_taxa_1_nao_altera():
    x = np.random.default_rng(0).normal(size=N)
    np.testing.assert_array_equal(T.estirar_tempo(x, 1.0), x)
    np.testing.assert_array_equal(T.mudar_velocidade(x, 1.0), x)


@pytest.mark.parametrize("taxa", [0.95, 1.05])
def test_tempo_preserva_frequencia_e_muda_duracao(taxa):
    x = _senoide(1000.0, 2 * N)
    y = T.estirar_tempo(x, taxa)
    assert abs(len(y) - len(x) / taxa) <= config.AUMENTO_PV_NFFT
    assert abs(_pico_hz(y) - 1000.0) < 2.0


@pytest.mark.parametrize("taxa", [0.95, 1.05])
def test_velocidade_escala_frequencia(taxa):
    x = _senoide(1000.0, 2 * N)
    y = T.mudar_velocidade(x, taxa)
    assert len(y) == round(len(x) / taxa)
    assert abs(_pico_hz(y) - 1000.0 * taxa) < 2.0


def test_estirar_despacha_modo_e_recusa_desconhecido():
    x = _senoide(500.0, N)
    np.testing.assert_array_equal(T.estirar(x, 1.02, "tempo"), T.estirar_tempo(x, 1.02))
    np.testing.assert_array_equal(T.estirar(x, 1.02, "velocidade"), T.mudar_velocidade(x, 1.02))
    with pytest.raises(ValueError):
        T.estirar(x, 1.02, "outro")


def test_janela_de_leitura_rende_um_segmento_inteiro():
    """O comprimento lido pelo variantes.py rende ≥ N amostras nos dois extremos."""
    for taxa in config.AUMENTO_ESTIR_TAXAS:
        n_lido = math.ceil(N * taxa) + 2 * config.AUMENTO_PV_NFFT
        x = np.random.default_rng(1).normal(size=n_lido)
        assert len(T.estirar_tempo(x, taxa)) >= N
        assert len(T.mudar_velocidade(x, taxa)) >= N


# --------------------------------------------------------------- ruído
@pytest.mark.parametrize("snr_db", [20.0, 35.0])
def test_ruido_respeita_snr(snr_db):
    x = 0.3 * _senoide(800.0, 20 * N)
    y = T.adicionar_ruido(x, snr_db, np.random.default_rng(2))
    snr_medida = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert abs(snr_medida - snr_db) < 0.2


def test_ruido_independe_do_ganho():
    """Mesma SNR e mesma semente: o ruído escala com o sinal."""
    x = _senoide(800.0, N)
    y1 = T.adicionar_ruido(x, 25.0, np.random.default_rng(3))
    y10 = T.adicionar_ruido(10 * x, 25.0, np.random.default_rng(3))
    np.testing.assert_allclose(y10, 10 * y1)


def test_ruido_em_silencio_nao_cria_sinal():
    x = np.zeros(N)
    np.testing.assert_array_equal(T.adicionar_ruido(x, 20.0, np.random.default_rng(4)), x)


# --------------------------------------------------------------- recorte
def test_recorte_central():
    x = np.arange(10)
    np.testing.assert_array_equal(T.recortar_centro(x, 4), [3, 4, 5, 6])
    with pytest.raises(ValueError):
        T.recortar_centro(x, 11)
