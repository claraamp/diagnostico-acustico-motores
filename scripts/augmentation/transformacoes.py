"""
transformacoes.py — transformações de sinal usadas no aumento de dados.

Blocos puros, como os do `dsp.py`: recebem um vetor e parâmetros e devolvem um
vetor novo, sem estado e sem sorteio próprio. Quem decide os parâmetros (e a
semente) é o `variantes.py`; aqui só se aplica.

Conteúdo
--------
estirar_tempo      estiramento/compressão temporal por phase vocoder (modo "tempo")
mudar_velocidade   reamostragem, como tocar mais rápido ou devagar (modo "velocidade")
adicionar_ruido    ruído branco gaussiano com SNR fixada
recortar_centro    recorte central de comprimento fixo

O deslocamento de janela não aparece aqui porque não é uma transformação do
sinal: é a escolha de ONDE ler a gravação, e fica no `variantes.py`.

Os dois modos de estiramento — o que cada um preserva
----------------------------------------------------
Medido num sinal sintético de defeito (impactos a 107 Hz excitando uma
ressonância de 3 kHz), com taxa 0,9:

  modo          frequência dos impactos     ressonância
  "tempo"       107 Hz → 107 Hz (mantém)    3000 Hz → 3000 Hz (mantém)
  "velocidade"  107 Hz → 96 Hz  (× 0,9)     3000 Hz → 2700 Hz (× 0,9)

- "tempo" (phase vocoder, o que `librosa.effects.time_stretch` faz e a técnica
  citada por Carrera et al., 2022): muda a escala de tempo e preserva o
  espectro. Como os quadros de 40 ms resolvem as raias em múltiplos da BPFO,
  a periodicidade dos impactos também é preservada — do mesmo jeito que o
  phase vocoder preserva o tom da voz. O que muda é a estrutura temporal
  lenta, acima de um quadro, e surgem pequenos artefatos de fase
  ("phasiness") nos transientes. É uma perturbação suave, que não altera a
  assinatura da falha.
- "velocidade" (reamostragem, a "speed perturbation" de reconhecimento de
  fala): multiplica todas as frequências pela taxa. Simula a variação de
  rotação na frequência de falha, mas desloca junto as ressonâncias, que no
  motor real não mudam com a rotação.

Nenhum dos dois reproduz exatamente uma mudança de rotação (impactos mudam,
ressonância fica). Por isso a faixa de taxas é estreita (±5 %, ver
`config.AUMENTO_ESTIR_TAXAS`) e o modo é um parâmetro registrado da rodada
(`config.AUMENTO_ESTIR_MODO`).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sg

import config


def estirar_tempo(x: np.ndarray, taxa: float,
                  n_fft: int = config.AUMENTO_PV_NFFT,
                  hop: int = config.AUMENTO_PV_HOP) -> np.ndarray:
    """
    Estira (taxa < 1) ou comprime (taxa > 1) `x` no tempo, sem mudar o espectro.

    A saída tem aproximadamente `len(x) / taxa` amostras. Algoritmo: STFT →
    leitura dos quadros em passos fracionários de `taxa` → magnitude
    interpolada entre quadros vizinhos → fase acumulada com o avanço de fase
    medido em cada bin (o que mantém cada senoide na sua frequência) → ISTFT.
    """
    x = np.asarray(x, dtype=np.float64)
    if taxa <= 0:
        raise ValueError(f"taxa de estiramento tem que ser positiva (recebi {taxa})")
    if taxa == 1.0:
        return x.copy()

    noverlap = n_fft - hop
    _, _, Z = sg.stft(x, nperseg=n_fft, noverlap=noverlap, window="hann")
    n_bins, n_quadros = Z.shape

    passos = np.arange(0.0, n_quadros - 1, taxa)
    # avanço de fase esperado, por hop, para o centro de cada bin
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft
    fase = np.angle(Z[:, 0])
    saida = np.empty((n_bins, len(passos)), dtype=np.complex128)
    for j, s in enumerate(passos):
        i = int(s)
        frac = s - i
        mag = (1.0 - frac) * np.abs(Z[:, i]) + frac * np.abs(Z[:, i + 1])
        saida[:, j] = mag * np.exp(1j * fase)
        # desvio de fase em relação ao esperado, trazido para (-π, π]
        dphi = np.angle(Z[:, i + 1]) - np.angle(Z[:, i]) - omega
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        fase = fase + omega + dphi

    _, y = sg.istft(saida, nperseg=n_fft, noverlap=noverlap, window="hann")
    return y


def mudar_velocidade(x: np.ndarray, taxa: float) -> np.ndarray:
    """
    Reamostra `x` para `round(len(x) / taxa)` amostras, mantendo a taxa de
    amostragem nominal: com taxa > 1 o sinal fica mais curto e todas as
    frequências sobem pelo fator `taxa`; com taxa < 1, o contrário.
    """
    x = np.asarray(x, dtype=np.float64)
    if taxa <= 0:
        raise ValueError(f"taxa tem que ser positiva (recebi {taxa})")
    if taxa == 1.0:
        return x.copy()
    return sg.resample(x, int(round(len(x) / taxa)))


def estirar(x: np.ndarray, taxa: float, modo: str = config.AUMENTO_ESTIR_MODO) -> np.ndarray:
    """Despacha para o modo de estiramento configurado ("tempo" ou "velocidade")."""
    if modo == "tempo":
        return estirar_tempo(x, taxa)
    if modo == "velocidade":
        return mudar_velocidade(x, taxa)
    raise ValueError(f"modo de estiramento desconhecido: {modo!r}")


def adicionar_ruido(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Soma ruído branco gaussiano com potência `P_sinal / 10^(snr_db/10)`.

    A potência de referência é a do próprio trecho, então a SNR é a mesma para
    uma gravação alta e uma baixa — o ruído não vira pista de classe. Trecho
    silencioso (potência ~0) volta sem ruído, como no `dsp.normalizar_rms_clipe`.
    """
    x = np.asarray(x, dtype=np.float64)
    p_sinal = float(np.mean(x ** 2))
    if p_sinal < 1e-12:
        return x.copy()
    p_ruido = p_sinal / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(p_ruido), size=x.shape)


def recortar_centro(x: np.ndarray, n: int) -> np.ndarray:
    """Os `n` valores centrais de `x`. Falha se `x` for mais curto que `n`."""
    if len(x) < n:
        raise ValueError(f"recorte de {n} amostras pedido num vetor de {len(x)}")
    ini = (len(x) - n) // 2
    return x[ini:ini + n]
