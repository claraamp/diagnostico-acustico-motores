#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_class_spectra.py — onde mora a informação que distingue falha de normal
=========================================================================

Passo 3 do roteiro, e o que de fato valida a decimação.

A pergunta
----------
Decimar para 12,8 kHz joga fora tudo acima de 6,4 kHz (Nyquist da nova taxa).
Validar a decimação é mostrar que o que foi jogado fora **não era** o que
distingue motor bom de motor com falha. Então basta responder:

    que fração da informação discriminante está abaixo de cada Nyquist candidato?

Como isso é medido aqui
-----------------------
Para cada classe de falha, define-se a *densidade discriminante*

    D(f) = max(0, PSD_falha(f) - PSD_normal(f))

ou seja, a energia que aquela classe tem **a mais** que a condição normal, em
cada frequência. O máximo com zero descarta as faixas em que a falha tem menos
energia que a normal — ali não há o que preservar.

Integrando D(f) de 0 até um limite e dividindo pela integral total, obtém-se a
fração da informação discriminante que sobrevive a uma decimação com aquele
Nyquist. É um número por classe, direto, sem depender de escolha de banda, de
janela, nem da hipótese de que a falha seja impulsiva.

Por que por classe e não na média
---------------------------------
As classes de 1,0 mm têm 6–7× mais energia que a normal e dominariam qualquer
média, escondendo a classe difícil. O `bpfo_0.3mm`, que no passo 2 apareceu
estatisticamente indistinguível da condição normal, precisa da sua própria
linha — é ele que decide se a decimação é segura no caso que interessa.

Uso
---
    python scripts/exploration/inspect_class_spectra.py
    python scripts/exploration/inspect_class_spectra.py --nperseg 32768   # mais resolução
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg

FS = 51_200.0
INT16_FULL = 32767.0

# Taxas candidatas → o Nyquist de cada uma é o limite do que sobrevive
CANDIDATAS = [6_400, 8_000, 12_800, 16_000, 25_600]


def carregar(pcm_dir: Path, manifest_path: Path):
    """Mesmo formato de manifest do 01: dicionário classe → metadados."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clipes = []
    for rotulo, meta in manifest.items():
        if not isinstance(meta, dict):
            continue
        caminho = Path(meta.get("arquivo_pcm", ""))
        if not caminho.exists():
            caminho = pcm_dir / caminho.name
        if not caminho.exists():
            raise FileNotFoundError(f"não achei {caminho} — rode 01_convert_mat_to_pcm.py")
        cru = np.fromfile(caminho, dtype="<i2")
        clipes.append({
            "rotulo": str(rotulo),
            "binario": str(meta.get("rotulo_binario", "falha")),
            "x": cru.astype(np.float64) / INT16_FULL,
        })
    return clipes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/exploration"))
    ap.add_argument("--nperseg", type=int, default=16384,
                    help="tamanho do segmento de Welch (resolução = FS/nperseg)")
    args = ap.parse_args()

    manifest = args.manifest or (args.pcm_dir / "manifest.json")
    clipes = carregar(args.pcm_dir, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(clipes)} clipes | resolução espectral = {FS/args.nperseg:.2f} Hz\n")

    # ------------------------------------------------------------------ #
    # PSD de cada classe
    # ------------------------------------------------------------------ #
    psds = {}
    for c in clipes:
        f, p = sg.welch(c["x"], fs=FS, nperseg=args.nperseg, window="hann")
        psds[c["rotulo"]] = p
    freq = f

    normais = [c["rotulo"] for c in clipes if c["binario"] == "normal"]
    falhas = [c["rotulo"] for c in clipes if c["binario"] != "normal"]
    if not normais:
        raise RuntimeError("nenhuma classe normal no manifest")
    p_normal = np.mean([psds[r] for r in normais], axis=0)

    # ------------------------------------------------------------------ #
    # Fração da informação discriminante abaixo de cada Nyquist
    # ------------------------------------------------------------------ #
    print("Fração da energia discriminante — D(f) = max(0, PSD_falha - PSD_normal) —")
    print("preservada ao decimar para cada taxa candidata:\n")
    cab = "classe".ljust(12) + "".join(f"{t/1000:>9.1f}k" for t in CANDIDATAS)
    print(cab)
    print("-" * len(cab))

    acumuladas = {}
    for r in falhas:
        d = np.maximum(0.0, psds[r] - p_normal)
        total = np.trapezoid(d, freq)
        if total <= 0:
            print(f"{r:<12}" + "   —  sem excesso de energia sobre a classe normal")
            continue
        acum = np.concatenate([[0.0], np.cumsum(np.diff(freq) * (d[:-1] + d[1:]) / 2)]) / total
        acumuladas[r] = acum
        linha = r.ljust(12)
        for taxa in CANDIDATAS:
            frac = float(np.interp(taxa / 2, freq, acum))
            linha += f"{frac*100:>9.1f}%"
        print(linha)

    print("\n(a coluna de uma taxa é a fração preservada ao decimar para ela;")
    print(" o corte é o Nyquist, metade da taxa)")

    # frequência abaixo da qual está 95 % / 99 % da informação, por classe
    print("\nFrequência que contém 95 % e 99 % da energia discriminante:\n")
    for r, acum in acumuladas.items():
        f95 = float(np.interp(0.95, acum, freq))
        f99 = float(np.interp(0.99, acum, freq))
        print(f"  {r:<12} 95 % → {f95:8.0f} Hz    99 % → {f99:8.0f} Hz")

    # ------------------------------------------------------------------ #
    # Figura 1 — PSD por classe
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(10, 5.5))
    for c in clipes:
        estilo = dict(lw=1.4, color="k") if c["binario"] == "normal" else dict(lw=0.9)
        plt.semilogy(freq / 1000, psds[c["rotulo"]], label=c["rotulo"], **estilo)
    for taxa in CANDIDATAS:
        plt.axvline(taxa / 2000, color="0.6", ls=":", lw=0.8)
        plt.text(taxa / 2000, plt.ylim()[1], f" {taxa/1000:.1f}k", fontsize=6,
                 rotation=90, va="top", color="0.4")
    plt.xlabel("frequência (kHz)   — pontilhado: Nyquist de cada taxa candidata")
    plt.ylabel("PSD")
    plt.title("Densidade espectral por classe (normal em preto)")
    plt.legend(fontsize=7, ncol=3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    f1 = args.out_dir / "fig_psd_por_classe.png"
    plt.savefig(f1, dpi=140)
    plt.close()

    # ------------------------------------------------------------------ #
    # Figura 2 — acumulada da energia discriminante
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(10, 5.5))
    for r, acum in acumuladas.items():
        plt.plot(freq / 1000, acum * 100, lw=1.3, label=r)
    for taxa in CANDIDATAS:
        plt.axvline(taxa / 2000, color="0.6", ls=":", lw=0.8)
        plt.text(taxa / 2000, 2, f" {taxa/1000:.1f} kHz", fontsize=6, rotation=90, color="0.4")
    plt.axhline(99, color="#c0392b", ls="--", lw=0.8, label="99 %")
    plt.ylim(0, 102)
    plt.xlabel("frequência de corte (kHz)")
    plt.ylabel("energia discriminante preservada (%)")
    plt.title("Quanto da diferença falha × normal sobrevive a cada Nyquist")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    f2 = args.out_dir / "fig_energia_discriminante_acumulada.png"
    plt.savefig(f2, dpi=140)
    plt.close()

    print(f"\nfiguras:\n  {f1}\n  {f2}")
    print("\nLeitura: se a coluna de 12.8k estiver perto de 100 % em TODAS as classes —")
    print("inclusive na mais fraca — decimar para 12,8 kHz está validado por este critério.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())