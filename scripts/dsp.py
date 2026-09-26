"""
dsp.py — funções de processamento de sinais compartilhadas pelo pipeline.

Blocos puros, sem estado e sem efeito colateral: recebem sinal e parâmetros,
devolvem resultado. Ficam aqui para que a etapa de decimação, a extração de
features e a validação do porte em C da Fase 2 usem exatamente a mesma
implementação — se cada etapa tivesse a sua, elas poderiam divergir em silêncio
e os resultados deixariam de ser comparáveis, que é o que o CONVENTIONS.md
(seção 5) existe para evitar.

Os parâmetros têm valor padrão vindo do `config.py`, mas são todos explícitos na
assinatura: um estudo que varre taxas precisa calcular alguns deles por taxa sob
teste, e não pode herdar os de produção.

Conteúdo:
    Espectro          psd, envelope_spectrum, peak_snr
    Banco de Mel      hz_to_mel, mel_to_hz, mel_filterbank, mfcc
    Decimação         FirSpec, design_decimation, resample_clip,
                      aliasing_energy_db, usable_band
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sg
from scipy.fft import dct, rfft, rfftfreq

import config


@dataclass
class FirSpec:
    numtaps: int
    cutoff_hz: float
    transition_hz: float
    stopband_atten_db: float
    passband_ripple_db: float
    integer_factor: bool
    up: int
    down: int

def normalizar_rms_clipe(x: np.ndarray) -> np.ndarray:
    """
    Normaliza o segmento temporal pelo seu valor RMS (sem estado).
    Garante que a normalização ocorra isoladamente por clipe.
    """
    rms = np.sqrt(np.mean(x**2))
    # segmento quase silencioso: dividir pelo RMS só amplificaria ruído digital
    if rms > 1e-6:
        return x / rms
    return x

def psd(x: np.ndarray, fs: float, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    f, p = sg.welch(x, fs=fs, nperseg=min(nperseg, len(x)), noverlap=None, window="hann")
    return f, p


def envelope_spectrum(
    x: np.ndarray, fs: float, band: tuple[float, float], seg_s: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Banda-passante → envelope de Hilbert → espectro de amplitude médio sobre
    segmentos. Retorna (frequências, amplitude).
    """
    nyq = fs / 2
    lo = max(band[0], 20.0)
    hi = min(band[1], nyq * 0.98)
    if hi - lo < 50.0:
        return np.array([0.0]), np.array([0.0])

    sos = sg.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    y = sg.sosfiltfilt(sos, x)

    nseg = max(int(seg_s * fs), 1024)
    n_full = (len(y) // nseg) * nseg
    if n_full == 0:
        nseg, n_full = len(y), len(y)
    segs = y[:n_full].reshape(-1, nseg)

    acc = None
    for s in segs:
        env = np.abs(sg.hilbert(s))
        env = env - env.mean()
        env *= np.hanning(len(env))
        mag = np.abs(rfft(env)) * 2.0 / len(env)
        acc = mag if acc is None else acc + mag
    acc /= len(segs)
    freqs = rfftfreq(nseg, 1.0 / fs)
    return freqs, acc


def usable_band(lo: float, hi: float, fs: float) -> tuple[tuple[float, float], bool, bool]:
    """
    Banda para demodular numa taxa reduzida. Se a banda ressonante de referência
    couber (mesmo parcialmente) abaixo de Nyquist, usa o trecho que sobrou. Se
    ficar inteiramente acima, a portadora original não existe mais: cai para a
    banda alta ainda disponível e sinaliza `substituta`, porque aí a pergunta
    passa a ser se *alguma* banda remanescente ainda carrega a modulação.
    """
    nyq_max = fs / 2 * 0.98
    if lo + 100.0 <= nyq_max:
        return (lo, min(hi, nyq_max)), bool(hi > nyq_max), False
    return (max(0.45 * nyq_max, 200.0), nyq_max), True, True


def peak_snr(freqs: np.ndarray, mag: np.ndarray, f_target: float,
             limiar_db: float = 6.0) -> dict:
    """Pico dentro de ±1,5 Hz vs. piso mediano em ±25 Hz (excluindo ±3 Hz)."""
    if len(freqs) < 8 or f_target >= freqs[-1]:
        return {"presente": False, "motivo": "fora da banda analisável", "snr_db": None, "f_pico_hz": None}

    win = np.abs(freqs - f_target) <= 1.5
    if not win.any():
        return {"presente": False, "motivo": "resolução insuficiente", "snr_db": None, "f_pico_hz": None}
    i_peak = int(np.argmax(np.where(win, mag, -np.inf)))
    peak = float(mag[i_peak])

    near = (np.abs(freqs - f_target) <= 25.0) & (np.abs(freqs - f_target) > 3.0)
    floor = float(np.median(mag[near])) if near.any() else float(np.median(mag))
    snr = 20 * math.log10((peak + 1e-30) / (floor + 1e-30))
    return {
        "presente": bool(snr >= limiar_db),
        "snr_db": round(snr, 2),
        "f_pico_hz": round(float(freqs[i_peak]), 3),
        "amplitude": peak,
    }


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(fs: float, n_fft: int, n_mels: int, fmin: float = 20.0) -> np.ndarray:
    fmax = fs / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = min(l + 1, n_fft // 2)
        if r == c:
            r = min(c + 1, n_fft // 2)
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb

def log_mel(x: np.ndarray, fs: float,
            frame_ms: float = float(config.MFCC_WINDOW_MS),
            hop_ms: float = float(config.MFCC_HOP_MS),
            n_mels: int = config.MFCC_N_MELS) -> np.ndarray:
    """
    Calcula as energias log-Mel por quadro (etapa anterior à DCT).
    Exposta separadamente para permitir análise direta das bandas de frequência
    na investigação de falhas sutis (ex: bpfo_0.3mm).
    """
    n_frame = int(round(frame_ms / 1000 * fs))
    n_hop = int(round(hop_ms / 1000 * fs))
    n_fft = 1 << (n_frame - 1).bit_length()

    if len(x) < n_frame:
        return np.zeros((0, n_mels))

    # A janela de Hanning é aplicada internamente em quadros pequenos (ex: 25ms),
    # nunca sobre o segmento de 1s inteiro.
    win = np.hanning(n_frame)
    n_frames = 1 + (len(x) - n_frame) // n_hop

    # Fatiamento estrito dentro do segmento (sem vazar para clipes vizinhos)
    idx = np.arange(n_frame)[None, :] + n_hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win

    spec = np.abs(rfft(frames, n=n_fft, axis=1)) ** 2
    fb = mel_filterbank(fs, n_fft, n_mels)
    mel = spec @ fb.T

    # Logaritmo com proteção contra zeros
    return np.log(mel + 1e-10)


def mfcc(x: np.ndarray, fs: float,
         frame_ms: float = float(config.MFCC_WINDOW_MS),
         hop_ms: float = float(config.MFCC_HOP_MS),
         n_mels: int = config.MFCC_N_MELS,
         n_mfcc: int = config.MFCC_N_COEFS) -> np.ndarray:
    """MFCC no formato que será portado: Hanning → FFT → mel → log → DCT-II."""
    lmel = log_mel(x, fs, frame_ms, hop_ms, n_mels)
    if lmel.shape[0] == 0:
        return np.zeros((0, n_mfcc))
    return dct(lmel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def design_decimation(fs_in: float, fs_out: float,
                      atten_db: float = float(config.FIR_ATTENUATION)) -> FirSpec:
    """
    Projeta o anti-aliasing. Fator inteiro → FIR único (o que o CMSIS-DSP faz).
    Fator racional → L/M, reportado como tal.
    """
    frac = Fraction(int(fs_out), int(fs_in)).limit_denominator(10_000)
    up, down = frac.numerator, frac.denominator
    integer_factor = up == 1

    nyq_out = fs_out / 2
    cutoff = 0.45 * fs_out                      # margem de 10 % até Nyquist
    transition = nyq_out - cutoff               # largura da transição
    fs_work = fs_in * up                        # taxa em que o filtro opera
    numtaps, beta = sg.kaiserord(atten_db, 2 * transition / (fs_work / 2))
    numtaps = int(numtaps) | 1                  # ímpar → fase linear, atraso inteiro

    taps = sg.firwin(numtaps, cutoff / (fs_work / 2), window=("kaiser", beta))
    w, h = sg.freqz(taps, worN=8192, fs=fs_work)
    hdb = 20 * np.log10(np.abs(h) + 1e-12)
    stop = hdb[w >= nyq_out]
    # O `firwin` define `cutoff` no ponto de -6 dB, então medir a ondulação até
    # lá dentro inclui a descida do próprio roll-off e reporta ~6 dB de ripple
    # num filtro perfeitamente plano. A banda passante útil vai até 0,9·corte.
    passb = hdb[w <= 0.9 * cutoff]
    return FirSpec(
        numtaps=numtaps,
        cutoff_hz=cutoff,
        transition_hz=transition,
        stopband_atten_db=float(-np.max(stop)) if stop.size else float("nan"),
        passband_ripple_db=float(np.max(passb) - np.min(passb)) if passb.size else float("nan"),
        integer_factor=integer_factor,
        up=up,
        down=down,
    )


def resample_clip(x: np.ndarray, fs_in: float, fs_out: float, spec: FirSpec,
                  atten_db: float = float(config.FIR_ATTENUATION)) -> np.ndarray:
    if spec.integer_factor:
        fs_work = fs_in
        nyq = fs_work / 2
        numtaps, beta = sg.kaiserord(atten_db, 2 * spec.transition_hz / nyq)
        taps = sg.firwin(int(numtaps) | 1, spec.cutoff_hz / nyq, window=("kaiser", beta))
        y = sg.lfilter(taps, 1.0, x)
        delay = (len(taps) - 1) // 2
        y = y[delay:]                            # compensa o atraso de grupo
        return y[:: spec.down]
    return sg.resample_poly(x, spec.up, spec.down)


def aliasing_energy_db(x_ref: np.ndarray, fs_ref: float, x_dec: np.ndarray, fs_dec: float) -> float:
    """
    Energia espúria: compara a PSD do decimado com a PSD do original restrita à
    nova banda. Diferença grande em alta frequência indica aliasing.
    """
    f_r, p_r = psd(x_ref, fs_ref)
    f_d, p_d = psd(x_dec, fs_dec)
    band = (f_d >= 0.30 * fs_dec / 2) & (f_d <= 0.98 * fs_dec / 2)
    if not band.any():
        return float("nan")
    p_r_i = np.interp(f_d[band], f_r, p_r)
    e_ref = float(np.trapezoid(p_r_i, f_d[band])) + 1e-30
    e_dec = float(np.trapezoid(p_d[band], f_d[band])) + 1e-30
    return 10 * math.log10(e_dec / e_ref)