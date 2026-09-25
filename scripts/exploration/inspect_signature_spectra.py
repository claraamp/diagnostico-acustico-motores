#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_signature_spectra.py — como cada falha difere da normal, com sinal
=========================================================================

Tarefa do Notion: "Caracterizar a natureza da assinatura acústica no dataset
(tonal × impulsiva) e o caso bpfo_0.3mm", passo 1 do plano. A identificação
dos picos fica em `identify_tonal_peaks.py`.

A pergunta
----------
No Protocolo B (exp009/exp010), o classificador detecta 100 % dos segmentos de
`bpfi_0.3mm`, `bpfi_1.0mm` e `bpfo_1.0mm` quando cada uma fica fora do treino, e
0 % dos de `bpfo_0.3mm`. Uma explicação possível é um **déficit de energia**:
a `bpfo_0.3mm` difere da normal por ter energia A MENOS em alguma faixa. O
critério da decimação, D(f) = max(0, falha − normal), não enxergava isso. Se for
o caso, a fração de déficit da `bpfo_0.3mm` sai alta aqui, e as das outras
falhas, baixas.

As outras duas explicações — diferença diluída pelos filtros largos do Mel e
diferença numa direção que o eixo do classificador não pega — dependem do MFCC
oficial e ficam para depois da tarefa "Implementar a extração de MFCC em Python".
Este script não usa MFCC.

O que é medido
--------------
PSD assinada, a 51,2 kHz, banda inteira. Cada gravação é cortada em segmentos
de 1 s, e cada segmento tem a sua PSD de Welch. A diferença entre falha e
normal é medida em dB **sem cortar em zero**, e comparada com a dispersão entre
segmentos da mesma gravação:

    z(f) = (média_dB_falha − média_dB_normal) / σ_combinado(f)

|z| > 2 marca as frequências em que a diferença supera a variação natural da
gravação. Por faixa, reporta-se a diferença de potência, o excesso e o déficit
(lineares) e a fração de déficit = déficit / (excesso + déficit).

Uso
---
    python scripts/exploration/inspect_signature_spectra.py
    python scripts/exploration/inspect_signature_spectra.py --sem-registro    # conferir figuras
    python scripts/exploration/inspect_signature_spectra.py --sintetico       # auto-teste, sem data/

Saída em reports/signature/: psd_metrics.json, psd_faixas.csv e a figura.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import dsp
import experimentos
import pcm_io

ALVO = "bpfo_0.3mm"
FALHAS = [c for c in config.CLASSES if config.BINARIO[c] == "falha"]

# Faixas do resumo. O corte em 6.400 Hz é o Nyquist da taxa de trabalho.
FAIXAS_HZ = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
             (4000, 6400), (6400, 12800), (12800, 25600)]

NPERSEG = 8192          # Welch dentro de cada segmento de 1 s a 51,2 kHz: 6,25 Hz, 11 médias
Z_LIMIAR = 2.0          # |z| acima disto: diferença maior que a variação da gravação


# --------------------------------------------------------------------------- #
# Cálculo
# --------------------------------------------------------------------------- #
def psd_por_segmento(x: np.ndarray, fs: float, seg_s: float = config.SEGMENTO_S,
                     nperseg: int = NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """PSD de Welch de cada segmento de `seg_s` segundos. Devolve (f, P[n_seg, n_f])."""
    n = int(round(seg_s * fs))
    k = len(x) // n
    ps = []
    for i in range(k):
        f, p = dsp.psd(x[i * n:(i + 1) * n], fs, nperseg=nperseg)
        ps.append(p)
    return f, np.vstack(ps)


def comparar_psd(f: np.ndarray, P_falha: np.ndarray, P_normal: np.ndarray) -> dict:
    """Diferença assinada falha − normal, por bin e por faixa."""
    eps = 1e-30
    m_f, m_n = P_falha.mean(axis=0), P_normal.mean(axis=0)
    dB_f, dB_n = 10 * np.log10(P_falha + eps), 10 * np.log10(P_normal + eps)
    delta_db = 10 * np.log10((m_f + eps) / (m_n + eps))
    sigma = np.sqrt((dB_f.var(axis=0, ddof=1) + dB_n.var(axis=0, ddof=1)) / 2) + 1e-6
    z = (dB_f.mean(axis=0) - dB_n.mean(axis=0)) / sigma

    df = f[1] - f[0]
    excesso = np.maximum(0.0, m_f - m_n) * df
    deficit = np.maximum(0.0, m_n - m_f) * df
    faixas = []
    for lo, hi in FAIXAS_HZ:
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            continue
        e, d = float(excesso[sel].sum()), float(deficit[sel].sum())
        faixas.append({
            "faixa_hz": f"{lo}-{hi}",
            "delta_potencia_db": float(10 * np.log10((m_f[sel].sum() + eps) / (m_n[sel].sum() + eps))),
            "excesso": e,
            "deficit": d,
            "fracao_deficit": d / (e + d) if e + d > 0 else float("nan"),
            "bins_z_pos": float(np.mean(z[sel] > Z_LIMIAR)),
            "bins_z_neg": float(np.mean(z[sel] < -Z_LIMIAR)),
        })
    E, D = float(excesso.sum()), float(deficit.sum())
    return {
        "delta_db": delta_db, "z": z, "faixas": faixas,
        "fracao_deficit_total": D / (E + D) if E + D > 0 else float("nan"),
        "delta_potencia_total_db": float(10 * np.log10((m_f.sum() + eps) / (m_n.sum() + eps))),
    }


# --------------------------------------------------------------------------- #
# Figura
# --------------------------------------------------------------------------- #
def suavizar(v: np.ndarray, n: int = 9) -> np.ndarray:
    return np.convolve(v, np.ones(n) / n, mode="same")


def fig_psd_assinada(f, comps: dict, caminho: Path) -> None:
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(11, 2.3 * len(FALHAS)), sharex=True)
    for ax, falha in zip(axes, FALHAS):
        c = comps[falha]
        ax.plot(f / 1000, suavizar(c["delta_db"]), color="0.25", lw=0.7)
        pos, neg = c["z"] > Z_LIMIAR, c["z"] < -Z_LIMIAR
        lim = np.nanmax(np.abs(suavizar(c["delta_db"]))) * 1.05
        ax.fill_between(f / 1000, 0, lim, where=pos, color="tab:red", alpha=0.15, lw=0)
        ax.fill_between(f / 1000, -lim, 0, where=neg, color="tab:blue", alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(config.FS_TRABALHO / 2000, color="k", ls=":", lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Δ (dB)")
        ax.set_title(f"{falha} − normal   (fração de déficit: "
                     f"{c['fracao_deficit_total']:.1%})", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frequência (kHz) — pontilhado: Nyquist da taxa de trabalho")
    fig.suptitle("Diferença assinada de PSD, com o sombreado onde |z| > 2 "
                 "(vermelho: excesso; azul: déficit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dados sintéticos
# --------------------------------------------------------------------------- #
def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Cinco gravações artificiais com resposta conhecida, para conferir o script
    sem `data/`: as falhas "fáceis" somam tons em 1–2,5 kHz; a `bpfo_0.3mm` é a
    normal com um DÉFICIT em 3–5 kHz e um tom fraco a 4,1 kHz. O script tem que
    reportar fração de déficit alta para ela, na faixa de 2–4 kHz.
    """
    from scipy import signal as sg
    rng = np.random.default_rng(seed)
    fs, dur = config.FS_ORIGINAL, 60.0
    t = np.arange(int(fs * dur)) / fs

    def base():
        x = sg.lfilter([1.0], [1.0, -0.95], rng.standard_normal(t.size)) * 0.002
        for k in range(1, 30):
            x += 0.01 / k * np.sin(2 * np.pi * 50.17 * k * t + rng.uniform(0, 6.28))
        return x

    clipes = []
    for c in config.CLASSES:
        x = base()
        if c in ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_1.0mm"):
            g = 3.0 if c.endswith("1.0mm") else 1.0
            for fk in (1250.0, 1680.0, 2210.0):
                x += g * 0.02 * np.sin(2 * np.pi * fk * t)
        if c == ALVO:
            b, a = sg.butter(4, [3000, 5000], btype="bandstop", fs=fs)
            x = sg.lfilter(b, a, x) + 0.001 * np.sin(2 * np.pi * 4100.0 * t)
        clipes.append(pcm_io.Clip(rotulo=c, binario=config.BINARIO[c], fs=fs, x_float=x))
    return clipes


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pcm-raw", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/signature"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true", help="não escreve no registry")
    ap.add_argument("--sintetico", action="store_true",
                    help="auto-teste com sinais artificiais (implica --sem-registro)")
    ap.add_argument("--responsavel", default="Clara")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        clipes = sintetico()
        print(f"Modo sintético: resposta esperada — fração de déficit alta para {ALVO} "
              "na faixa de 2.000–4.000 Hz, baixa para as outras falhas.")
    else:
        clipes = pcm_io.carregar_clipes(args.pcm_raw, fs_esperado=config.FS_ORIGINAL)
        pcm_io.relatar_integridade(pcm_io.verificar_integridade(clipes))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    por_rotulo = {c.rotulo: c for c in clipes}

    print(f"\nPSD assinada a {config.FS_ORIGINAL} Hz, segmentos de {config.SEGMENTO_S:g} s "
          f"(|z| > {Z_LIMIAR:g} = acima da variação da gravação)")
    f, P_n = psd_por_segmento(por_rotulo["normal"].x, por_rotulo["normal"].fs)
    comps = {}
    for falha in FALHAS:
        _, P_f = psd_por_segmento(por_rotulo[falha].x, por_rotulo[falha].fs)
        comps[falha] = comparar_psd(f, P_f, P_n)

    linhas_faixa = []
    for falha in FALHAS:
        c = comps[falha]
        marca = "   ← caso difícil" if falha == ALVO else ""
        print(f"\n  {falha}: potência total {c['delta_potencia_total_db']:+.1f} dB · "
              f"fração de déficit {c['fracao_deficit_total']:.1%}{marca}")
        print(f"    {'faixa (Hz)':<13}{'Δ pot. (dB)':>12}{'déficit':>10}{'z>+2':>8}{'z<−2':>8}")
        for fx in c["faixas"]:
            print(f"    {fx['faixa_hz']:<13}{fx['delta_potencia_db']:>+12.1f}"
                  f"{fx['fracao_deficit']:>10.1%}{fx['bins_z_pos']:>8.0%}{fx['bins_z_neg']:>8.0%}")
            linhas_faixa.append({"classe": falha, **fx})

    fig_psd_assinada(f, comps, args.out_dir / "fig_psd_diferenca_assinada.png")
    with (args.out_dir / "psd_faixas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas_faixa[0]))
        w.writeheader()
        w.writerows(linhas_faixa)
    metrics = {
        "data": date.today().isoformat(),
        "parametros": {"fs_hz": config.FS_ORIGINAL, "nperseg": NPERSEG,
                       "segmento_s": config.SEGMENTO_S, "z_limiar": Z_LIMIAR},
        "psd": {falha: {"delta_potencia_total_db": comps[falha]["delta_potencia_total_db"],
                        "fracao_deficit_total": comps[falha]["fracao_deficit_total"],
                        "faixas": comps[falha]["faixas"]} for falha in FALHAS},
    }
    (args.out_dir / "psd_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    print(f"\nSaída em {args.out_dir}/")

    if args.sem_registro:
        print("(sem registro)")
        return
    linha = {
        "id": f"exp{experimentos.next_exp_number(args.registry):03d}",
        "data": date.today().isoformat(),
        "etapa": "caracterizacao_assinatura",
        "script": "exploration/inspect_signature_spectra.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "fs_hz": config.FS_ORIGINAL, "nperseg": NPERSEG, "segmento_s": config.SEGMENTO_S,
            "z_limiar": Z_LIMIAR}),
        "dataset": "jung2023_acustico_0Nm",
        "metricas": experimentos.kv({
            **{f"deficit_{fa}": f"{comps[fa]['fracao_deficit_total']:.3f}" for fa in FALHAS},
            **{f"dpot_db_{fa}": f"{comps[fa]['delta_potencia_total_db']:.2f}" for fa in FALHAS}}),
        "responsavel": args.responsavel,
        "notas": args.notas or "PSD assinada (sem corte em zero); diagnóstico do exp009",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")


if __name__ == "__main__":
    main()