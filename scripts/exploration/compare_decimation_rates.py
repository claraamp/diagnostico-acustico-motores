#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_decimation_rates.py — qual taxa de decimação usar, e por quê
=====================================================================

Estudo pontual que decidiu a taxa de trabalho do projeto. Compara cinco taxas
candidatas contra o sinal original de 51,2 kHz e produz as métricas, as figuras
e o relatório que sustentam a escolha.

Não faz parte do pipeline: ele respondeu uma pergunta uma vez. A aplicação da
decisão — decimar os PCM na taxa escolhida — é
`scripts/pipeline/02_decimate_pcm.py`. Fica aqui para que a decisão continue
reproduzível e auditável, não para ser reexecutado a cada rodada.

O critério
----------
Para cada classe de falha, `D(f) = max(0, PSD_falha(f) - PSD_normal(f))` é a
energia que ela tem a mais que a condição normal em cada frequência. A fração
dessa diferença abaixo do novo Nyquist é o que sobrevive à decimação. Exige-se
um piso (padrão 95 %) em **todas** as classes, não na média: as classes severas
têm 6-7x mais energia e esconderiam as incipientes, que são as que decidem.

Duas medidas ficam fora do critério, de propósito:

- **A acurácia do classificador.** Com uma gravação contínua por classe, treino
  e teste saem do mesmo registro e o classificador separa por características da
  gravação, não pela falha. Deu 1,0 no baseline e em todas as taxas. Continua
  calculada e reportada, como diagnóstico.
- **O SNR de BPFI/BPFO no espectro de envelope.** O método pressupõe assinatura
  impulsiva; no subconjunto 0 Nm a diferença é tonal. Também segue como
  diagnóstico.

Uso
---
    python scripts/exploration/compare_decimation_rates.py
    python scripts/exploration/compare_decimation_rates.py --seconds 10 --no-registry
    python scripts/exploration/compare_decimation_rates.py --synthetic --no-registry

Por padrão NÃO grava PCM: quem produz os PCM de trabalho é o `02`. Use
`--write-bin` se quiser os decimados de todas as taxas para inspeção.

Dependências: numpy, scipy, matplotlib, scikit-learn
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
import pcm_io
from experimentos import (REGISTRY_COLUMNS, append_registry, git_short_hash,
                          kv, next_exp_number)
from dsp import (FirSpec, aliasing_energy_db, design_decimation,
                 envelope_spectrum, mfcc, peak_snr, psd, resample_clip, usable_band)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.fft import rfft, rfftfreq

# --------------------------------------------------------------------------- #
# Constantes físicas do dataset (Jung et al., 2023 — DOI 10.17632/ztmf3m7h5x.6)
# --------------------------------------------------------------------------- #
# Aliases locais das constantes centralizadas. Mantidos com os nomes antigos
# para não espalhar `config.` pelo arquivo inteiro; a fonte única é o config.py.
FS_ORIG = float(config.FS_ORIGINAL)
INT16_FULL = config.INT16_FULL

F_SHAFT = 50.0              # Hz — rotação do eixo
F_BPFO = 179.0              # Hz — Ball Pass Frequency, Outer race
F_BPFI = 272.0              # Hz — Ball Pass Frequency, Inner race

# Taxas candidatas. 51200/M inteiro: 25600 (M=2), 12800 (M=4), 6400 (M=8).
# 8000 e 16000 entram como L/M para quantificar o que se ganha/perde ao insistir
# numa taxa "redonda".
DEFAULT_RATES = [25_600, 16_000, 12_800, 8_000, 6_400]

# Parâmetros de MFCC (os mesmos que serão portados para o STM32)
FRAME_MS = float(config.MFCC_WINDOW_MS)
HOP_MS = float(config.MFCC_HOP_MS)
N_MELS = config.MFCC_N_MELS
N_MFCC = config.MFCC_N_COEFS
SEGMENT_S = float(config.SEGMENTO_S)    # unidade de classificação
TRAIN_FRACTION = 0.70       # split temporal contíguo, sem embaralhar

# Alvos de pico no espectro de envelope
ENV_TARGETS = {
    "eixo (1x)": F_SHAFT,
    "BPFO (1x)": F_BPFO,
    "BPFO (2x)": 2 * F_BPFO,
    "BPFO (3x)": 3 * F_BPFO,
    "BPFI (1x)": F_BPFI,
    "BPFI (2x)": 2 * F_BPFI,
    "BPFI (3x)": 3 * F_BPFI,
}

SNR_ACCEPT_DB = 6.0         # critério: pico 6 dB acima do piso local conta como presente
STOPBAND_TARGET_DB = float(config.FIR_ATTENUATION)   # atenuação mínima do anti-aliasing


# --------------------------------------------------------------------------- #
# Dataset sintético (auto-teste do pipeline, sem os dados reais)
# --------------------------------------------------------------------------- #
def synthetic_clips(seconds: float, seed: int = 7) -> list["pcm_io.Clip"]:
    """
    Gera 5 clipes com a mesma estrutura física do caso real: ruído de fundo +
    ressonância de 4,2 kHz excitada por um trem de impulsos na frequência de
    falha, modulado pela rotação do eixo. Serve para verificar que o pipeline
    roda e que a detecção de picos acha o que deveria achar.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS_ORIG)
    t = np.arange(n) / FS_ORIG

    def ring(f_fault: float, sev: float) -> np.ndarray:
        # trem de impulsos com jitter de 1 % e modulação de 50 Hz
        period = FS_ORIG / f_fault
        centers = np.arange(0, n, period)
        idx = centers + rng.normal(0, 0.01 * period, centers.size)
        idx = idx[(idx >= 0) & (idx < n)].astype(int)
        imp = np.zeros(n)
        amp = 1.0 + 0.6 * np.sin(2 * np.pi * F_SHAFT * idx / FS_ORIG)
        imp[idx] = amp * sev
        # ressonância: 2ª ordem em 4,2 kHz, Q ~ 18
        f0, q = 4200.0, 18.0
        b, a = sg.iirpeak(f0 / (FS_ORIG / 2), q)
        return sg.lfilter(b, a, imp)

    base = lambda: 0.02 * rng.standard_normal(n) + 0.004 * np.sin(2 * np.pi * F_SHAFT * t)
    specs = [
        ("normal.bin", "normal", "normal", None, 0.0),
        ("bpfi_0.3mm.bin", "bpfi_0.3mm", "falha", F_BPFI, 0.35),
        ("bpfi_1.0mm.bin", "bpfi_1.0mm", "falha", F_BPFI, 1.00),
        ("bpfo_0.3mm.bin", "bpfo_0.3mm", "falha", F_BPFO, 0.30),
        ("bpfo_1.0mm.bin", "bpfo_1.0mm", "falha", F_BPFO, 0.90),
    ]
    clips = []
    for name, label, binary, f_fault, sev in specs:
        x = base()
        if f_fault is not None:
            r = ring(f_fault, sev)
            x = x + r / (np.max(np.abs(r)) + 1e-12) * (0.25 * sev + 0.05)
        x = x / (np.max(np.abs(x)) * 1.02)
        clips.append(pcm_io.Clip(rotulo=label, binario=binary, fs=FS_ORIG, x_float=x))
    return clips


# --------------------------------------------------------------------------- #
# Etapa A — onde mora a informação: PSD por classe e busca da banda ressonante
# --------------------------------------------------------------------------- #
def discriminative_curves(clips: list[pcm_io.Clip], nperseg: int = 16384) -> tuple[np.ndarray, dict]:
    """
    Curva acumulada da energia discriminante de cada classe de falha.

    Para cada classe, D(f) = max(0, PSD_falha(f) - PSD_normal(f)) é a energia
    que ela tem A MAIS que a condição normal em cada frequência. A integral
    acumulada normalizada diz, para qualquer frequência de corte, que fração
    dessa diferença sobrevive — que é exatamente a pergunta da decimação.

    É este o critério primário de validação. O espectro de envelope (BPFI/BPFO)
    continua sendo calculado, mas como diagnóstico: ele pressupõe assinatura
    impulsiva, e no subconjunto 0 Nm do dataset a diferença entre normal e
    falha é tonal, não impulsiva — as classes de falha têm curtose ABAIXO de 3
    e fator de crista MENOR que o da classe normal. Validar a decimação pelo
    envelope, nesse caso, mediria a preservação de uma estrutura que o sinal
    não tem.
    """
    normais = [c for c in clips if c.binario == "normal"]
    falhas = [c for c in clips if c.binario != "normal"]
    if not normais or not falhas:
        return np.array([0.0]), {}

    f, _ = sg.welch(normais[0].x, fs=FS_ORIG, nperseg=min(nperseg, len(normais[0].x)),
                    window="hann")
    p_norm = np.mean([sg.welch(c.x, fs=FS_ORIG, nperseg=min(nperseg, len(c.x)),
                               window="hann")[1] for c in normais], axis=0)

    curvas = {}
    for c in falhas:
        _, p = sg.welch(c.x, fs=FS_ORIG, nperseg=min(nperseg, len(c.x)), window="hann")
        d = np.maximum(0.0, p - p_norm)
        total = float(np.trapezoid(d, f))
        if total <= 0:
            continue
        acum = np.concatenate([[0.0], np.cumsum(np.diff(f) * (d[:-1] + d[1:]) / 2)]) / total
        curvas[c.rotulo] = acum
    return f, curvas


def preservacao_em(freq: np.ndarray, curvas: dict, corte_hz: float) -> dict:
    """Fração preservada por classe para um dado Nyquist."""
    return {lab: round(float(np.interp(corte_hz, freq, acum)), 4)
            for lab, acum in curvas.items()}


def energy_sweep(clips: list[pcm_io.Clip], band_hz: float = 1000.0) -> list[dict]:
    """Varredura de energia por sub-banda — material de diagnóstico para o relatório."""
    f, p_norm = psd(np.concatenate([c.x for c in clips if c.binario == "normal"]), FS_ORIG)
    _, p_fault = psd(np.concatenate([c.x for c in clips if c.binario == "falha"]), FS_ORIG)

    rows, nyq, lo = [], FS_ORIG / 2, 200.0
    while lo + band_hz <= nyq:
        hi = lo + band_hz
        m = (f >= lo) & (f < hi)
        e_n = float(np.trapezoid(p_norm[m], f[m]))
        e_f = float(np.trapezoid(p_fault[m], f[m]))
        rows.append({"lo_hz": lo, "hi_hz": hi, "energia_normal": e_n, "energia_falha": e_f,
                     "razao_db": 10 * math.log10((e_f + 1e-30) / (e_n + 1e-30))})
        lo = hi
    return rows


def ref_clip(clips: list[pcm_io.Clip], kind: str) -> "pcm_io.Clip | None":
    """Clipe de referência de um tipo de falha — prefere a severidade maior."""
    cands = [c for c in clips if kind in c.rotulo.lower()]
    if not cands:
        return None
    return sorted(cands, key=lambda c: ("1.0" not in c.rotulo, c.rotulo))[0]


def select_demod_band(
    clips: list[pcm_io.Clip], energia: list[dict], piso_energia: float = 0.01,
    excerpt_s: float = 15.0,
) -> tuple[tuple[float, float], list[dict]]:
    """
    Escolhe a banda de demodulação pelo critério que de fato interessa: a banda
    cujo espectro de envelope mostra os picos BPFI/BPFO mais destacados.

    Por que não pela razão de energia falha/normal, que é o proxy óbvio: essa
    razão premia bandas onde a classe normal está no piso de ruído, mesmo que
    não haja energia nenhuma ali. Na primeira rodada sobre o 0 Nm, ela elegeu
    15,2–16,2 kHz (razão 21,9 dB) contra 2,2–3,2 kHz (razão 21,5 dB) — com mil
    vezes MENOS energia absoluta. A banda vencedora ficava acima de Nyquist de
    todas as taxas candidatas e invalidava a comparação inteira.

    Duas travas: (a) só concorrem bandas com pelo menos `piso_energia` da
    energia total de falha; (b) a nota é o SNR dos picos no envelope, medido
    nos clipes de maior severidade de cada tipo de falha.
    """
    total = sum(r["energia_falha"] for r in energia) + 1e-30
    nyq = FS_ORIG / 2
    c_bpfi, c_bpfo = ref_clip(clips, "bpfi"), ref_clip(clips, "bpfo")
    refs = [(c, f) for c, f in ((c_bpfi, F_BPFI), (c_bpfo, F_BPFO)) if c is not None]
    if not refs:
        raise RuntimeError("nenhum clipe de falha identificável por rótulo (bpfi/bpfo)")

    def energia_da_banda(lo, hi):
        return sum(r["energia_falha"] for r in energia
                   if r["lo_hz"] >= lo - 1 and r["hi_hz"] <= hi + 1)

    n_exc = int(excerpt_s * FS_ORIG)
    ranking = []
    for largura in (1000.0, 2000.0, 3000.0):
        lo = 200.0
        while lo + largura <= nyq:
            hi = lo + largura
            frac = energia_da_banda(lo, hi) / total
            if frac >= piso_energia:
                notas = {}
                for c, f_alvo in refs:
                    fr, mg = envelope_spectrum(c.x[:n_exc], FS_ORIG, (lo, hi), seg_s=4.0)
                    r = peak_snr(fr, mg, f_alvo, SNR_ACCEPT_DB)
                    notas[c.rotulo] = r["snr_db"] if r["snr_db"] is not None else -99.0
                ranking.append({"lo_hz": lo, "hi_hz": hi, "largura_hz": largura,
                                "fracao_energia": round(frac, 4),
                                "snr_por_clipe": notas,
                                "nota_db": round(float(np.mean(list(notas.values()))), 2)})
            lo += 500.0

    if not ranking:
        raise RuntimeError(
            f"nenhuma banda com ao menos {piso_energia:.0%} da energia de falha — "
            "reveja o piso ou o dataset")
    ranking.sort(key=lambda r: r["nota_db"], reverse=True)
    best = ranking[0]
    return (best["lo_hz"], best["hi_hz"]), ranking


# --------------------------------------------------------------------------- #
# Etapa B — espectro de envelope
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Etapa C — decimação
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Etapa D — MFCC e separabilidade
# --------------------------------------------------------------------------- #
def segment_features(clips: list[pcm_io.Clip], fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features de 1 s: média e desvio dos MFCC. Retorna (X, y_binário, t_início)."""
    X, y, t0 = [], [], []
    n_seg = int(SEGMENT_S * fs)
    for c in clips:
        n_full = (len(c.x) // n_seg) * n_seg
        for k in range(n_full // n_seg):
            seg = c.x[k * n_seg : (k + 1) * n_seg]
            m = mfcc(seg, fs)
            if m.shape[0] == 0:
                continue
            X.append(np.concatenate([m.mean(axis=0), m.std(axis=0)]))
            y.append(0 if c.binario == "normal" else 1)
            t0.append(k * SEGMENT_S)
    return np.asarray(X), np.asarray(y), np.asarray(t0)


def separability(X: np.ndarray, y: np.ndarray, t0: np.ndarray) -> dict:
    """LDA com split temporal contíguo + razão de Fisher média das features."""
    if len(np.unique(y)) < 2 or len(X) < 12:
        return {"acuracia_balanceada": None, "fisher_medio": None, "n_treino": 0, "n_teste": 0}

    cut = np.quantile(t0, TRAIN_FRACTION)
    tr, te = t0 <= cut, t0 > cut
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        tr = np.zeros(len(y), bool)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            tr[idx[: int(TRAIN_FRACTION * len(idx))]] = True
        te = ~tr

    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xs[tr], y[tr])
    acc = balanced_accuracy_score(y[te], clf.predict(Xs[te]))

    m0, m1 = Xs[y == 0], Xs[y == 1]
    fisher = (m0.mean(0) - m1.mean(0)) ** 2 / (m0.var(0) + m1.var(0) + 1e-12)
    return {
        "acuracia_balanceada": round(float(acc), 4),
        "fisher_medio": round(float(np.mean(fisher)), 4),
        "fisher_max": round(float(np.max(fisher)), 4),
        "n_treino": int(tr.sum()),
        "n_teste": int(te.sum()),
    }


# --------------------------------------------------------------------------- #
# Etapa E — custo embarcado
# --------------------------------------------------------------------------- #
def embedded_cost(fs: float, spec: FirSpec | None) -> dict:
    n_frame = int(round(FRAME_MS / 1000 * fs))
    n_fft = 1 << (n_frame - 1).bit_length()
    ram_1s_int16 = int(fs) * 2
    # FFT real N pontos ~ (N/2)log2(N/2) borboletas complexas ~ 6 flops cada
    fft_flops = 6 * (n_fft / 2) * math.log2(max(n_fft / 2, 2))
    frames_per_s = 1000.0 / HOP_MS
    fir_mac_per_s = (spec.numtaps * fs) if spec else 0.0
    return {
        "fs_hz": int(fs),
        "nyquist_hz": fs / 2,
        "amostras_por_janela_25ms": n_frame,
        "n_fft": n_fft,
        "ram_buffer_1s_int16_bytes": ram_1s_int16,
        "ram_buffer_1s_kib": round(ram_1s_int16 / 1024, 1),
        "flops_fft_por_janela": int(fft_flops),
        "mflops_mfcc_estimado": round(fft_flops * frames_per_s / 1e6, 2),
        "mac_por_s_fir_antialiasing": int(fir_mac_per_s),
        "fator_inteiro": bool(spec.integer_factor) if spec else True,
        "L_up": spec.up if spec else 1,
        "M_down": spec.down if spec else 1,
        "taps_fir": spec.numtaps if spec else 0,
    }


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_psd(clips: list[pcm_io.Clip], band: tuple[float, float], out: Path) -> None:
    plt.figure(figsize=(9, 5))
    for c in clips:
        f, p = psd(c.x, FS_ORIG)
        plt.semilogy(f / 1000, p, lw=0.9, label=c.rotulo)
    plt.axvspan(band[0] / 1000, band[1] / 1000, color="0.85", zorder=0,
                label=f"banda ressonante {band[0]:.0f}–{band[1]:.0f} Hz")
    plt.xlabel("frequência (kHz)")
    plt.ylabel("PSD")
    plt.title("Densidade espectral por classe — sinal original a 51,2 kHz")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_envelope(env_by_rate: dict, out: Path, clip_label: str) -> None:
    rates = sorted(env_by_rate, reverse=True)
    fig, axes = plt.subplots(len(rates), 1, figsize=(9, 2.1 * len(rates)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rates):
        f, m = env_by_rate[r]
        ax.plot(f, m, lw=0.8, color="#1f4e79")
        for name, ft in ENV_TARGETS.items():
            if ft < (f[-1] if len(f) > 1 else 0):
                ax.axvline(ft, color="#c0392b" if "BPFI" in name else "#2e7d32",
                           ls="--", lw=0.7, alpha=0.8)
        ax.set_xlim(0, 900)
        ax.set_ylabel(f"{r/1000:.1f} kHz", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frequência do envelope (Hz)  — tracejado: BPFO (verde) e BPFI (vermelho)")
    axes[0].set_title(f"Espectro de envelope por taxa de amostragem — classe {clip_label}")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_fir(specs: dict, out: Path) -> None:
    plt.figure(figsize=(9, 5))
    for fs_out, spec in sorted(specs.items(), reverse=True):
        fs_work = FS_ORIG * spec.up
        nyq = fs_work / 2
        numtaps, beta = sg.kaiserord(STOPBAND_TARGET_DB, 2 * spec.transition_hz / nyq)
        taps = sg.firwin(int(numtaps) | 1, spec.cutoff_hz / nyq, window=("kaiser", beta))
        w, h = sg.freqz(taps, worN=8192, fs=fs_work)
        plt.plot(w / 1000, 20 * np.log10(np.abs(h) + 1e-12), lw=1.0,
                 label=f"{fs_out/1000:.1f} kHz ({spec.numtaps} taps, L={spec.up}/M={spec.down})")
    plt.axhline(-STOPBAND_TARGET_DB, color="k", ls=":", lw=0.8, label=f"-{STOPBAND_TARGET_DB:.0f} dB")
    # Eixo limitado ao Nyquist da taxa ORIGINAL. Os filtros de fator inteiro operam
    # a 51,2 kHz e só têm resposta definida até 25,6 kHz; os de fator racional operam
    # a 256 kHz (interpolação por 5) e teriam resposta até 128 kHz. Mostrar além de
    # 25,6 kHz faria as curvas de fator inteiro parecerem truncadas, e a região extra
    # é irrelevante: ali o sinal interpolado não tem conteúdo.
    plt.xlim(0, FS_ORIG / 2000)
    plt.ylim(-120, 5)
    plt.xlabel("frequência (kHz) — eixo até o Nyquist da taxa original (25,6 kHz)")
    plt.ylabel("|H(f)| (dB)")
    plt.title("Filtros anti-aliasing projetados (Kaiser)")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_summary(metrics: dict, out: Path) -> None:
    """A figura que justifica a decisão: preservação por classe contra custo."""
    rates = sorted(metrics["por_taxa"], key=float)
    xs = [float(r) / 1000 for r in rates]
    ram = [metrics["por_taxa"][r]["custo"]["ram_buffer_1s_kib"] for r in rates]
    classes = sorted(metrics["por_taxa"][rates[0]]["preservacao"].keys())
    min_pres = metrics.get("min_preservacao_exigida", 0.95)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    for lab in classes:
        ys = [metrics["por_taxa"][r]["preservacao"][lab] * 100 for r in rates]
        # classes incipientes (0.3 mm) em linha cheia: são elas que decidem
        estilo = "-o" if "0.3" in lab else "--s"
        ax1.plot(xs, ys, estilo, lw=1.3, ms=4, label=lab)
    ax1.axhline(min_pres * 100, color="#c0392b", ls=":", lw=1.0,
                label=f"critério {min_pres:.0%}")
    ax1.set_xlabel("taxa de amostragem (kHz)")
    ax1.set_ylabel("energia discriminante preservada (%)")
    ax1.set_ylim(min(80, min_pres * 100 - 5), 101)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, ram, "v--", color="#7d6608", lw=1.0, label="RAM buffer 1 s (KiB)")
    ax2.set_ylabel("RAM do buffer de 1 s (KiB)")

    rec = metrics.get("recomendacao", {}).get("fs_hz")
    if rec:
        ax1.axvline(rec / 1000, color="#1f4e79", lw=1.2, alpha=0.5)
        ax1.text(rec / 1000, ax1.get_ylim()[0] + 0.5, f" escolhida: {rec/1000:.1f} kHz",
                 fontsize=7, color="#1f4e79")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    plt.title("Energia discriminante preservada × custo, por taxa")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


# --------------------------------------------------------------------------- #
# Registro de experimentos — CONVENTIONS.md, seção 4
# --------------------------------------------------------------------------- #

def build_registry_rows(metrics: dict, args, start_n: int, commit: str) -> list[dict]:
    hoje = _dt.date.today().isoformat()
    rec = metrics["recomendacao"]["fs_hz"]
    trecho = f";trecho_s={args.seconds:g}" if args.seconds else ""
    dataset = ("sintetico_autoteste" if args.synthetic
               else f"jung2023_acustico_{args.condicao}_5classes_60s{trecho}")
    baseline_ok = metrics.get("baseline_aprovado", True)

    def linha(n, fs, filtro, met, notas):
        return {
            "id": f"exp{n:03d}",
            "data": hoje,
            "etapa": "decimacao",
            "script": Path(__file__).name,
            "git_commit": commit,
            "parametros": kv(filtro),
            "dataset": dataset,
            "metricas": kv(met),
            "responsavel": args.responsavel,
            "notas": notas,
        }

    rows, n = [], start_n
    b = metrics["baseline"]
    rows.append(linha(
        n, FS_ORIG,
        {"taxa_hz": int(FS_ORIG), "fator": 1, "janela_ms": FRAME_MS, "hop_ms": HOP_MS,
         "n_mels": N_MELS, "n_mfcc": N_MFCC,
         "banda_envelope_hz": f"{metrics['banda_demodulacao_hz'][0]:.0f}-"
                              f"{metrics['banda_demodulacao_hz'][1]:.0f}"},
        {"snr_bpfi_db": b["picos"].get("BPFI (1x)", {}).get("melhor_snr_db"),
         "snr_bpfo_db": b["picos"].get("BPFO (1x)", {}).get("melhor_snr_db"),
         "acuracia_balanceada_diagnostica": b["separabilidade"]["acuracia_balanceada"],
         "ram_1s_kib": b["custo"]["ram_buffer_1s_kib"],
         "n_fft": b["custo"]["n_fft"]},
        "baseline sem decimacao; referencia de comparacao"
        + ("" if baseline_ok else "; BASELINE REPROVADO - assinatura ausente antes de decimar"),
    ))
    n += 1

    for r in sorted(metrics["por_taxa"], key=float):
        d = metrics["por_taxa"][r]
        f, c = d["filtro"], d["custo"]
        notas = []
        if rec is not None and int(float(r)) == rec:
            notas.append("TAXA RECOMENDADA")
        if not baseline_ok:
            notas.append("envelope BPFI/BPFO nao aplicavel - assinatura tonal, nao impulsiva")
        notas.append("fator inteiro" if f["integer_factor"]
                     else f"reamostragem racional L={f['up']}/M={f['down']}")
        if d.get("banda_substituta"):
            notas.append("banda ressonante de referencia acima de Nyquist")
        elif d.get("banda_truncada"):
            notas.append("banda ressonante truncada em Nyquist")
        rows.append(linha(
            n, float(r),
            {"taxa_hz": int(float(r)),
             "fator": (f"1/{f['down']}" if f["integer_factor"] else f"{f['up']}/{f['down']}"),
             "filtro": f"fir_kaiser_{f['numtaps']}taps",
             "corte_hz": round(f["cutoff_hz"]),
             "atenuacao_db": round(f["stopband_atten_db"], 1),
             "janela_ms": FRAME_MS, "hop_ms": HOP_MS,
             "n_mels": N_MELS, "n_mfcc": N_MFCC},
            {"preservacao_pior_classe": d["preservacao_pior_classe"],
             "pior_classe": d["pior_classe"],
             "snr_bpfi_db_diagnostico": d["snr_bpfi_db"],
             "snr_bpfo_db_diagnostico": d["snr_bpfo_db"],
             "aliasing_db": d["aliasing_db"],
             "acuracia_balanceada_diagnostica": d["separabilidade"]["acuracia_balanceada"],
             "ram_1s_kib": c["ram_buffer_1s_kib"],
             "n_fft": c["n_fft"],
             "amostras_janela": c["amostras_por_janela_25ms"]},
            "; ".join(notas),
        ))
        n += 1
    return rows


# --------------------------------------------------------------------------- #
# Relatório
# --------------------------------------------------------------------------- #
def write_report(metrics: dict, out: Path) -> None:
    m = metrics
    L = []
    L.append("# Decimação — definição da taxa e validação da assinatura BPFI/BPFO\n")
    L.append(f"- Sinal de origem: {m['fs_origem_hz']:.0f} Hz, {m['n_clipes']} clipes, "
             f"{m['duracao_s']:.1f} s por clipe.")
    L.append(f"- Condição de carga: **{m.get('condicao', 'n/d')}**.")
    L.append(f"- Critério primário: preservar ≥ {m['min_preservacao_exigida']:.0%} da energia "
             "discriminante em **todas** as classes de falha.")
    L.append(f"- Diagnóstico secundário (envelope): banda "
             f"{m['banda_demodulacao_hz'][0]:.0f}–{m['banda_demodulacao_hz'][1]:.0f} Hz, "
             f"pico considerado presente com SNR ≥ {SNR_ACCEPT_DB:.0f} dB.\n")

    L.append("## Critério primário — energia discriminante preservada\n")
    L.append("Para cada classe de falha, `D(f) = max(0, PSD_falha(f) − PSD_normal(f))` é a "
             "energia que ela tem a mais que a condição normal. A tabela dá a fração dessa "
             "diferença que sobrevive a cada taxa. Por classe, e não na média: as classes "
             "severas têm muito mais energia e esconderiam a classe incipiente, que é "
             "justamente a que decide.\n")
    classes_falha = sorted(next(iter(m["por_taxa"].values()))["preservacao"].keys())
    L.append("| taxa | " + " | ".join(classes_falha) + " | pior caso |")
    L.append("|---" * (len(classes_falha) + 2) + "|")
    for r in sorted(m["por_taxa"], key=float):
        d = m["por_taxa"][r]
        L.append(f"| {float(r)/1000:.1f} kHz | "
                 + " | ".join(f"{d['preservacao'][c]:.1%}" for c in classes_falha)
                 + f" | **{d['preservacao_pior_classe']:.1%}** ({d['pior_classe']}) |")
    L.append("")
    L.append("Frequência que concentra 95 % e 99 % da energia discriminante, por classe:\n")
    for lab in classes_falha:
        f95 = m["energia_discriminante_f95_hz"].get(lab)
        f99 = m["energia_discriminante_f99_hz"].get(lab)
        L.append(f"- **{lab}**: 95 % → {f95:.0f} Hz · 99 % → {f99:.0f} Hz")
    L.append("")

    if not m.get("baseline_aprovado", True):
        L.append("> **Nota sobre o espectro de envelope.** BPFI e/ou BPFO não atingem o\n"
                 "> critério de SNR nem no sinal original de 51,2 kHz. Neste recorte do\n"
                 "> dataset a diferença entre normal e falha é tonal, não impulsiva — as\n"
                 "> classes de falha têm curtose abaixo de 3 e fator de crista menor que o da\n"
                 "> classe normal. O espectro de envelope pressupõe impactos periódicos, e\n"
                 "> portanto não é o instrumento adequado aqui. Os números de envelope abaixo\n"
                 "> ficam como diagnóstico; a validação da decimação é a tabela acima.\n")

    L.append("## Baseline (51,2 kHz)\n")
    L.append("| alvo | melhor SNR (dB) | clipe | f do pico (Hz) | presente |")
    L.append("|---|---|---|---|---|")
    for name, r in m["baseline"]["picos"].items():
        L.append(f"| {name} | {r.get('melhor_snr_db')} | {r.get('melhor_clipe')} | "
                 f"{r.get('f_pico_hz')} | {'sim' if r.get('presente') else 'não'} |")
    L.append("")
    L.append("SNR por clipe, para separar severidades:\n")
    for name in ("BPFI (1x)", "BPFO (1x)"):
        por = m["baseline"]["picos"].get(name, {}).get("por_clipe", {})
        if por:
            L.append(f"- **{name}**: " + ", ".join(
                f"{k} = {v['snr_db']} dB" for k, v in por.items()))
    L.append("")

    L.append("## Comparativo por taxa\n")
    L.append("| taxa | fator | taps | atenuação (dB) | aliasing (dB) | SNR BPFI(1x) | "
             "SNR BPFO(1x) | harm. BPFI | harm. BPFO | amostras/25 ms | N FFT | RAM 1 s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(m["por_taxa"], key=float):
        d = m["por_taxa"][r]
        c, f = d["custo"], d["filtro"]
        fator = f"/{f['down']}" if f["integer_factor"] else f"×{f['up']}/{f['down']}"
        L.append(
            f"| {float(r)/1000:.1f} kHz | {fator} | {f['numtaps']} | "
            f"{f['stopband_atten_db']:.1f} | {d['aliasing_db']:.2f} | "
            f"{d['snr_bpfi_db']:.1f} | {d['snr_bpfo_db']:.1f} | "
            f"{d['harmonicas_bpfi_presentes']}/3 | {d['harmonicas_bpfo_presentes']}/3 | "
            f"{c['amostras_por_janela_25ms']} | {c['n_fft']} | {c['ram_buffer_1s_kib']} KiB |"
        )
    L.append("")
    L.append("## Decisão\n")
    rec = m["recomendacao"]
    if rec.get("fs_hz"):
        L.append(f"**Taxa recomendada: {rec['fs_hz']} Hz** "
                 f"(faixa-alvo {m['faixa_alvo_hz'][0]:.0f}–{m['faixa_alvo_hz'][1]:.0f} Hz; "
                 f"critérios atendidos: {rec.get('nivel')}).\n")
    else:
        L.append(f"**Sem recomendação — {rec.get('nivel')}.**\n")
    for reason in rec["justificativa"]:
        L.append(f"- {reason}")
    L.append("")

    sep = m["baseline"]["separabilidade"]
    L.append("## Acurácia do classificador — por que não entra na decisão\n")
    L.append(f"A acurácia balanceada (MFCC + LDA, split temporal contíguo) deu "
             f"{sep['acuracia_balanceada']} no baseline e "
             + ", ".join(f"{float(r)/1000:.1f} kHz → "
                         f"{m['por_taxa'][r]['separabilidade']['acuracia_balanceada']}"
                         for r in sorted(m["por_taxa"], key=float))
             + ".")
    L.append("")
    L.append("São 5 gravações contínuas, uma por classe, e o split temporal tira treino e teste "
             "da **mesma** gravação. O classificador pode separar por características do "
             "registro — ganho, ruído de fundo, ponto de operação — sem usar a assinatura de "
             "falha. Métrica que não varia entre condições não discrimina nada, e um número "
             "assim não deve ir para o relatório como evidência de desempenho. Fica registrada "
             "como diagnóstico, e a avaliação honesta do classificador depende de validação "
             "entre registros distintos (ou de aumento de dados com partição por registro).")
    L.append("")
    L.append("## Figuras\n")
    for f in m["figuras"]:
        L.append(f"- `{f}`")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Recomendação
# --------------------------------------------------------------------------- #
def recommend(metrics: dict, target: tuple[float, float]) -> dict:
    """
    Critérios de aceitação de uma taxa candidata:
      (a) preserva ao menos `min_preservacao` da energia discriminante de
          **todas** as classes de falha — inclusive a mais fraca;
      (b) energia espúria por aliasing ≤ +1 dB na banda útil;
      (c) fator de decimação inteiro — requisito de `arm_fir_decimate_f32`;
      (d) dentro da faixa-alvo acordada no projeto — **nunca relaxada**: sair da
          faixa é decisão de equipe, não de script.
    Entre as que passam, escolhe a menor taxa (menor custo embarcado). Se
    nenhuma passa, relaxa (c) e diz o que isso custa.

    Duas métricas ficam fora dos critérios, de propósito:

    *A acurácia do classificador.* Com 5 gravações contínuas, uma por classe,
    treino e teste saem do mesmo registro: o classificador separa por
    características da gravação — ganho, ruído de fundo, ponto de operação —
    e não pela falha. Deu 1,0 no baseline e em todas as taxas. Métrica que não
    varia não decide nada.

    *O SNR de BPFI/BPFO no espectro de envelope.* Esse método pressupõe
    assinatura impulsiva. No 0 Nm as classes de falha têm curtose abaixo de 3 e
    fator de crista menor que o da normal — a diferença é tonal. Medir
    preservação de uma estrutura ausente não valida nada. Continua no relatório
    como diagnóstico, e a caracterização da assinatura virou tarefa própria.
    """
    min_pres = metrics.get("min_preservacao_exigida", 0.95)
    cands = []
    for r, d in metrics["por_taxa"].items():
        fs = int(float(r))
        pres = d.get("preservacao", {})
        pior = min(pres.values()) if pres else 0.0
        cands.append(
            {
                "fs_hz": fs,
                "preservacao_pior_classe": round(pior, 4),
                "pior_classe": (min(pres, key=pres.get) if pres else None),
                "snr_bpfi_db": d["snr_bpfi_db"],
                "snr_bpfo_db": d["snr_bpfo_db"],
                "acc_diagnostica": d["separabilidade"]["acuracia_balanceada"],
                "inteiro": bool(d["filtro"]["integer_factor"]),
                "na_faixa": bool(target[0] <= fs <= target[1]),
                "ok_pres": bool(pior >= min_pres),
                "ok_alias": bool(d["aliasing_db"] <= 1.0),
            }
        )

    def pick_first(pred):
        sel = sorted([c for c in cands if pred(c)], key=lambda c: c["fs_hz"])
        return sel[0] if sel else None

    niveis = [
        (lambda c: c["na_faixa"] and c["ok_pres"] and c["ok_alias"] and c["inteiro"],
         "todos os critérios", []),
        (lambda c: c["na_faixa"] and c["ok_pres"] and c["ok_alias"],
         "fator inteiro relaxado",
         ["Exige reamostragem racional L/M: `arm_fir_decimate_f32` não atende, "
          "seria preciso interpolador + decimador em cascata no STM32."]),
    ]

    pick, nivel, extra = None, None, []
    for pred, nome, obs in niveis:
        pick = pick_first(pred)
        if pick:
            nivel, extra = nome, obs
            break

    if pick is None:
        melhor = max(cands, key=lambda c: c["preservacao_pior_classe"], default=None)
        just = [f"Nenhuma taxa dentro da faixa-alvo preserva {min_pres:.0%} da energia "
                "discriminante de todas as classes."]
        if melhor:
            just.append(
                f"A melhor candidata foi {melhor['fs_hz']} Hz, com "
                f"{melhor['preservacao_pior_classe']:.1%} na classe "
                f"{melhor['pior_classe']}. Subir a taxa acima da faixa ou aceitar um "
                "critério menor é decisão da equipe, e vai ao Registro de Decisões.")
        return {"fs_hz": None, "nivel": "nenhuma candidata atinge o critério",
                "justificativa": just, "candidatos": cands}

    just = [
        f"Menor taxa dentro da faixa-alvo que preserva ao menos {min_pres:.0%} da energia "
        f"discriminante em todas as classes — o pior caso é "
        f"{pick['pior_classe']} com {pick['preservacao_pior_classe']:.1%}.",
        "Energia espúria por aliasing dentro de +1 dB na banda útil.",
    ]
    if pick["inteiro"]:
        just.append(
            f"51.200 / {int(FS_ORIG) // pick['fs_hz']} é decimação por fator inteiro: um único FIR "
            f"passa-baixa seguido de descarte de amostras — exatamente o que "
            f"`arm_fir_decimate_f32` (CMSIS-DSP) implementa no STM32F411, sem estágio de interpolação."
        )
    just += extra
    return {"fs_hz": pick["fs_hz"], "nivel": nivel, "justificativa": just, "candidatos": cands}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", default="data/processed/pcm_raw", type=Path)
    ap.add_argument("--manifest", default=None, type=Path)
    ap.add_argument("--out-dir", default="data/processed/pcm_decimated", type=Path)
    ap.add_argument("--report-dir", default="reports/decimation", type=Path)
    ap.add_argument("--rates", nargs="*", type=int, default=DEFAULT_RATES)
    ap.add_argument("--seconds", type=float, default=None, help="usar só os primeiros N s de cada clipe")
    ap.add_argument("--target-min", type=float, default=8_000.0, help="piso da faixa-alvo (Hz)")
    ap.add_argument("--target-max", type=float, default=16_000.0, help="teto da faixa-alvo (Hz)")
    ap.add_argument("--condicao", default="0Nm",
                    help="condição de carga do registro, para o relatório e o registry")
    ap.add_argument("--min-preservacao", type=float, default=0.95,
                    help="fração mínima da energia discriminante que a taxa deve preservar "
                         "em TODAS as classes de falha (critério primário)")
    ap.add_argument("--piso-energia", type=float, default=0.01,
                    help="fração mínima da energia de falha para uma banda concorrer")
    ap.add_argument("--registry", default="experiments/registry.csv", type=Path,
                    help="registro de experimentos (CONVENTIONS.md, seção 4)")
    ap.add_argument("--responsavel", default="Clara", help="coluna `responsavel` do registry")
    ap.add_argument("--no-registry", action="store_true",
                    help="não escrever no registry (use em rodadas de teste)")
    ap.add_argument("--synthetic", action="store_true", help="auto-teste com sinal sintético")
    ap.add_argument("--write-bin", action="store_true",
                    help="gravar os .bin de todas as taxas (quem produz os de "
                         "trabalho é o 02_decimate_pcm.py)")
    args = ap.parse_args(argv)

    args.report_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        secs = args.seconds or 12.0
        print(f"[synthetic] gerando 5 clipes de {secs:.0f} s com assinatura conhecida")
        clips = synthetic_clips(secs)
    else:
        clips = pcm_io.carregar_clipes(args.pcm_dir, args.manifest, args.seconds,
                                       fs_esperado=FS_ORIG)
        # com --seconds o tamanho não bate com o manifest de propósito
        pcm_io.relatar_integridade(
            pcm_io.verificar_integridade(clips, truncado=args.seconds is not None))
    print(f"[load] {len(clips)} clipes, {len(clips[0].x)/FS_ORIG:.1f} s cada, fs = {FS_ORIG:.0f} Hz")

    # --- Etapa A: onde demodular ------------------------------------------ #
    energia = energy_sweep(clips)
    (lo, hi), ranking = select_demod_band(clips, energia, piso_energia=args.piso_energia)
    print(f"[banda] demodulação em {lo:.0f}–{hi:.0f} Hz "
          f"(nota {ranking[0]['nota_db']:.1f} dB, "
          f"{ranking[0]['fracao_energia']:.1%} da energia de falha)")
    for r in ranking[1:4]:
        print(f"        alternativa: {r['lo_hz']:.0f}–{r['hi_hz']:.0f} Hz "
              f"(nota {r['nota_db']:.1f} dB)")
    fig1 = args.report_dir / "fig1_psd_por_classe.png"
    fig_psd(clips, (lo, hi), fig1)

    # --- Baseline --------------------------------------------------------- #
    def picos_para(cs: list["pcm_io.Clip"], fs: float, band: tuple[float, float]) -> dict:
        """
        SNR por alvo, clipe a clipe. Não faz média entre severidades: misturar
        um 0,3 mm quase indetectável com um 1,0 mm forte produz um número que
        não descreve nenhum dos dois. `presente` olha o melhor clipe, que é o
        que responde "a assinatura sobreviveu à decimação?".
        """
        out = {}
        for name, ft in ENV_TARGETS.items():
            por_clipe = {}
            for c in cs:
                if "BPFI" in name and "bpfi" not in c.rotulo.lower():
                    continue
                if "BPFO" in name and "bpfo" not in c.rotulo.lower():
                    continue
                if "eixo" in name and c.binario == "normal":
                    continue
                f, m = envelope_spectrum(c.x, fs, band)
                r = peak_snr(f, m, ft, SNR_ACCEPT_DB)
                por_clipe[c.rotulo] = {"snr_db": r["snr_db"], "f_pico_hz": r.get("f_pico_hz")}
            if not por_clipe:
                out[name] = {"presente": False, "melhor_snr_db": None,
                             "melhor_clipe": None, "f_pico_hz": None, "por_clipe": {}}
                continue
            melhor = max(por_clipe.items(),
                         key=lambda kv: (kv[1]["snr_db"] is not None, kv[1]["snr_db"] or -99))
            snr = melhor[1]["snr_db"]
            out[name] = {
                "presente": bool(snr is not None and snr >= SNR_ACCEPT_DB),
                "melhor_snr_db": snr,
                "melhor_clipe": melhor[0],
                "f_pico_hz": melhor[1]["f_pico_hz"],
                "por_clipe": por_clipe,
            }
        return out

    print("[baseline] espectro de envelope a 51,2 kHz")
    base_pk = picos_para(clips, FS_ORIG, (lo, hi))
    for alvo in ("BPFI (1x)", "BPFO (1x)", "eixo (1x)"):
        d = base_pk.get(alvo, {})
        print(f"           {alvo:<10} {d.get('melhor_snr_db')} dB "
              f"({d.get('melhor_clipe')}) → {'OK' if d.get('presente') else 'ABAIXO DO CRITÉRIO'}")

    baseline_ok = bool(base_pk.get("BPFI (1x)", {}).get("presente")
                       and base_pk.get("BPFO (1x)", {}).get("presente"))
    if not baseline_ok:
        print("\n" + "-" * 72)
        print(f"NOTA: BPFI e/ou BPFO não atingem {SNR_ACCEPT_DB:.0f} dB nem no sinal original")
        print("de 51,2 kHz. Neste dataset a diferença entre normal e falha é tonal, não")
        print("impulsiva, então o espectro de envelope procura uma estrutura que o sinal")
        print("não tem. Isso NÃO invalida a decimação: a validação usa a preservação da")
        print("energia discriminante (abaixo). O envelope fica como diagnóstico.")
        print("-" * 72 + "\n")

    # --- Critério primário: energia discriminante ------------------------- #
    print("[discriminante] curvas acumuladas por classe de falha")
    freq_disc, curvas_disc = discriminative_curves(clips)
    if not curvas_disc:
        raise RuntimeError("não consegui calcular a energia discriminante "
                           "(faltou classe normal ou classe de falha)")
    for lab, acum in curvas_disc.items():
        f95 = float(np.interp(0.95, acum, freq_disc))
        f99 = float(np.interp(0.99, acum, freq_disc))
        print(f"                {lab:<12} 95 % → {f95:7.0f} Hz | 99 % → {f99:7.0f} Hz")

    Xb, yb, tb = segment_features(clips, FS_ORIG)
    base_sep = separability(Xb, yb, tb)
    print(f"[baseline] acurácia balanceada: {base_sep['acuracia_balanceada']} "
          f"(diagnóstica — ver aviso no relatório, não entra na decisão)")

    metrics = {
        "fs_origem_hz": FS_ORIG,
        "n_clipes": len(clips),
        "duracao_s": len(clips[0].x) / FS_ORIG,
        "sintetico": bool(args.synthetic),
        "condicao": args.condicao,
        "classes": {c.rotulo: c.binario for c in clips},
        "min_preservacao_exigida": args.min_preservacao,
        "energia_discriminante_f95_hz": {
            lab: round(float(np.interp(0.95, acum, freq_disc)), 1)
            for lab, acum in curvas_disc.items()},
        "energia_discriminante_f99_hz": {
            lab: round(float(np.interp(0.99, acum, freq_disc)), 1)
            for lab, acum in curvas_disc.items()},
        "banda_demodulacao_hz": [lo, hi],
        "banda_demodulacao_nota_db": ranking[0]["nota_db"],
        "banda_demodulacao_fracao_energia": ranking[0]["fracao_energia"],
        "ranking_bandas": ranking[:10],
        "varredura_energia": energia,
        "baseline_aprovado": baseline_ok,
        "baseline": {
            "picos": base_pk,
            "separabilidade": base_sep,
            "custo": embedded_cost(FS_ORIG, None),
        },
        "por_taxa": {},
        "figuras": [],
    }

    # --- Etapas C/D/E por taxa -------------------------------------------- #
    specs: dict[int, FirSpec] = {}
    env_example: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    ref_label = next((c.rotulo for c in clips if "bpfi_1.0" in c.rotulo), clips[-1].rotulo)
    ref_clip = next(c for c in clips if c.rotulo == ref_label)
    fe, me = envelope_spectrum(ref_clip.x, FS_ORIG, (lo, hi))
    env_example[FS_ORIG] = (fe, me)

    for rate in sorted(args.rates, reverse=True):
        spec = design_decimation(FS_ORIG, float(rate))
        specs[rate] = spec
        fator = f"/{spec.down}" if spec.integer_factor else f"×{spec.up}/{spec.down}"
        print(f"[{rate:>6} Hz] fator {fator}, {spec.numtaps} taps, "
              f"atenuação {spec.stopband_atten_db:.1f} dB")

        dec_clips, alias = [], []
        for c in clips:
            y = resample_clip(c.x, FS_ORIG, float(rate), spec)
            dec_clips.append(c.derivado(y, float(rate)))
            alias.append(aliasing_energy_db(c.x, FS_ORIG, y, float(rate)))

        band_dec, truncada, substituta = usable_band(lo, hi, float(rate))
        if substituta:
            print(f"           banda de demodulação ficou acima de Nyquist; "
                  f"usando a banda disponível {band_dec[0]:.0f}–{band_dec[1]:.0f} Hz")
        pk = picos_para(dec_clips, float(rate), band_dec)

        def snr_1x(prefixo: str):
            v = pk.get(f"{prefixo} (1x)", {}).get("melhor_snr_db")
            return float(v) if v is not None else float("nan")

        s_bpfi, s_bpfo = snr_1x("BPFI"), snr_1x("BPFO")
        pres = preservacao_em(freq_disc, curvas_disc, float(rate) / 2)
        pior_lab = min(pres, key=pres.get)
        Xd, yd, td = segment_features(dec_clips, float(rate))
        sep = separability(Xd, yd, td)
        marca = "OK " if pres[pior_lab] >= args.min_preservacao else "abaixo do critério"
        print(f"           preservação: " + "  ".join(
            f"{k}={v:.1%}" for k, v in sorted(pres.items())))
        print(f"           pior classe {pior_lab} = {pres[pior_lab]:.1%} → {marca}"
              f" | BPFI(1x) {s_bpfi:.1f} dB, BPFO(1x) {s_bpfo:.1f} dB (diagnóstico)")

        metrics["por_taxa"][str(rate)] = {
            "filtro": asdict(spec),
            "banda_envelope_hz": list(band_dec),
            "banda_truncada": bool(truncada),
            "banda_substituta": bool(substituta),
            "aliasing_db": round(float(np.mean(alias)), 3),
            "preservacao": pres,
            "preservacao_pior_classe": round(pres[pior_lab], 4),
            "pior_classe": pior_lab,
            "picos": pk,
            "bpfi_presente": bool(pk.get("BPFI (1x)", {}).get("presente")),
            "bpfo_presente": bool(pk.get("BPFO (1x)", {}).get("presente")),
            "snr_bpfi_db": round(s_bpfi, 2) if s_bpfi == s_bpfi else float("nan"),
            "snr_bpfo_db": round(s_bpfo, 2) if s_bpfo == s_bpfo else float("nan"),
            "harmonicas_bpfi_presentes": sum(
                1 for k in pk if k.startswith("BPFI") and pk[k]["presente"]),
            "harmonicas_bpfo_presentes": sum(
                1 for k in pk if k.startswith("BPFO") and pk[k]["presente"]),
            "separabilidade": sep,
            "custo": embedded_cost(float(rate), spec),
        }

        dref = next(c for c in dec_clips if c.rotulo == ref_label)
        env_example[float(rate)] = envelope_spectrum(dref.x, float(rate), band_dec)

        if args.write_bin:
            pcm_io.gravar_clipes(dec_clips, args.out_dir / f"{rate}", float(rate),
                                 extras={"fonte": str(args.pcm_dir),
                                         "decimacao": asdict(spec)})

    # --- Figuras e recomendação ------------------------------------------- #
    fig2 = args.report_dir / "fig2_envelope_por_taxa.png"
    fig_envelope(env_example, fig2, ref_label)
    fig3 = args.report_dir / "fig3_filtros_antialiasing.png"
    fig_fir(specs, fig3)
    metrics["faixa_alvo_hz"] = [args.target_min, args.target_max]
    metrics["recomendacao"] = recommend(metrics, (args.target_min, args.target_max))
    fig4 = args.report_dir / "fig4_assinatura_vs_custo.png"
    fig_summary(metrics, fig4)
    metrics["figuras"] = [str(p) for p in (fig1, fig2, fig3, fig4)]

    (args.report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_report(metrics, args.report_dir / "decimation_report.md")

    if not args.no_registry:
        commit = git_short_hash()
        rows = build_registry_rows(metrics, args, next_exp_number(args.registry), commit)
        append_registry(args.registry, rows)
        print(f"[registry] {len(rows)} linhas ({rows[0]['id']}–{rows[-1]['id']}) "
              f"em {args.registry}, commit {commit}")
        if commit == "sem-git":
            print("           ATENÇÃO: não consegui ler o hash do commit — rode dentro do repo, "
                  "senão a rodada não é reproduzível como manda a convenção.")

    rec = metrics["recomendacao"]
    print("\n" + "=" * 72)
    if rec.get("fs_hz"):
        print(f"TAXA RECOMENDADA: {rec['fs_hz']} Hz  ({rec.get('nivel')})")
    else:
        print(f"SEM RECOMENDAÇÃO — {rec.get('nivel')}")
    for j in rec["justificativa"]:
        print(f"  - {j}")
    print("=" * 72)
    print(f"relatório: {args.report_dir/'decimation_report.md'}")
    print(f"métricas:  {args.report_dir/'metrics.json'}")
    # código 2 = rodou, produziu diagnóstico, mas não validou nada
    return 0 if rec.get("fs_hz") else 2


if __name__ == "__main__":
    sys.exit(main())