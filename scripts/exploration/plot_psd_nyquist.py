"""
Densidade espectral por classe, com o Nyquist de cada taxa candidata marcado.

Substitui a fig1_psd_por_classe.png do estudo de decimação, que destacava a
banda de demodulação do envelope antigo (descartado como medição inválida) e
não marcava os Nyquist. Só lê os clipes e desenha; não registra rodada.

    python scripts/exploration/plot_psd_nyquist.py
    python scripts/exploration/plot_psd_nyquist.py --fmax 13000 --out reports/decimation/fig1_psd_nyquist.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
import pcm_io
from dsp import psd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[2]
TAXAS = [25_600, 16_000, 12_800, 8_000, 6_400]
TAXA_TRABALHO = 12_800


def desenhar(clipes: list, fs: float, fmax: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in clipes:
        f, p = psd(c.x, fs)
        m = f <= fmax
        ax.semilogy(f[m] / 1000, p[m], lw=0.8, label=c.rotulo)
    for taxa in TAXAS:
        nyq = taxa / 2
        if nyq > fmax:
            continue
        trabalho = taxa == TAXA_TRABALHO
        ax.axvline(nyq / 1000, color="k", ls="-" if trabalho else ":",
                   lw=1.2 if trabalho else 0.9, zorder=0)
        ax.text(nyq / 1000, 0.98, f"$f_s$ = {taxa / 1000:g} kHz ".replace(".", ","),
                transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7,
                fontweight="bold" if trabalho else "normal",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlabel("frequência (kHz) — linhas verticais: Nyquist de cada taxa candidata")
    ax.set_ylabel("PSD")
    ax.set_title("Densidade espectral por classe — sinal original a 51,2 kHz")
    ax.set_xlim(0, fmax / 1000)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pcm-dir", type=Path, default=RAIZ / "data/processed/pcm_raw")
    ap.add_argument("--fmax", type=float, default=13_000.0,
                    help="limite do eixo em Hz (padrão: logo acima do Nyquist de 25,6 kHz)")
    ap.add_argument("--out", type=Path, default=RAIZ / "reports/decimation/fig1_psd_nyquist.png")
    args = ap.parse_args(argv)

    fs = float(config.FS_ORIGINAL)
    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=fs)
    desenhar(clipes, fs, args.fmax, args.out)
    print(f"figura salva em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
