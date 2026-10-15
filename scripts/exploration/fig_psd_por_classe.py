#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_psd_por_classe.py — PSD por classe, sinal original a 51,2 kHz, em Pa
===========================================================================

Gera reports/decimation/fig_psd_por_classe.png: PSD por classe com as mesmas
curvas (Welch, nperseg=16384) que entram em discriminative_curves() de
scripts/exploration/compare_decimation_rates.py, no estilo visual de
_plt() em scripts/validation/run_consolidacao.py.

Script avulso de inspeção: não roda pipeline, não grava registry, não
versiona nada além da própria figura. Lê os 5 .mat em data/raw/ diretamente
em Pa (não usa os .bin/manifest do pipeline).

Uso
---
    python scripts/exploration/fig_psd_por_classe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal as sg

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data/raw"
REPORT_DIR = REPO / "reports/decimation"
OUT_PATH = REPORT_DIR / "fig_psd_por_classe.png"
METRICS_PATH = REPORT_DIR / "metrics.json"

FS_EXPECTED = 51200.0
N_EXPECTED = 3_072_000   # 60 s a 51.200 Hz
NPERSEG = 16384
TOL_HZ = NPERSEG and (FS_EXPECTED / NPERSEG)   # 3,125 Hz = 1 bin

CLASS_FILES = {
    "0Nm_Normal.mat":  ("normal",   "normal"),
    "0Nm_BPFI_03.mat": ("bpfi_0.3mm", "falha"),
    "0Nm_BPFI_10.mat": ("bpfi_1.0mm", "falha"),
    "0Nm_BPFO_03.mat": ("bpfo_0.3mm", "falha"),
    "0Nm_BPFO_10.mat": ("bpfo_1.0mm", "falha"),
}


# --- load_acoustic: copiado de scripts/pipeline/01_convert_mat_to_pcm.py -----
# (não importado: aquele módulo roda a conversão no nível do módulo e
# regravaria os .bin/manifest versionados se fosse importado)
def load_acoustic(path):
    data = sio.loadmat(path, struct_as_record=True, squeeze_me=False)
    sig = data["Signal"][0, 0]
    x, y, fr = sig["x_values"][0, 0], sig["y_values"][0, 0], sig["function_record"][0, 0]

    grupo = "".join(str(c) for c in np.atleast_1d(fr["TL_export_properties_annotation"][0, 0]["channel_group"]).ravel())
    assert grupo == "Acoustic", f"{path}: esperava canal 'Acoustic', achei {grupo!r}"

    fs = 1.0 / float(x["increment"][0, 0])
    n = int(x["number_of_values"][0, 0])
    values = y["values"].astype(np.float64).ravel()

    assert abs(fs - 51200.0) < 1.0, f"{path}: fs inesperada ({fs} Hz)"
    assert values.shape[0] == n, f"{path}: número de amostras não bate"

    return values, fs, y


def checar_unidade_pa(y, path) -> str:
    """Acha a unidade em y_values.quantity.label (Test.Lab) e confere que é Pa."""
    if "quantity" not in (y.dtype.names or ()):
        raise SystemExit(
            f"Abortado: {path} — y_values não tem campo 'quantity'. "
            f"Campos disponíveis: {y.dtype.names}. Estrutura inesperada, inspecione manualmente."
        )
    q = y["quantity"][0, 0]
    if "label" not in (q.dtype.names or ()):
        raise SystemExit(
            f"Abortado: {path} — quantity não tem campo 'label'. "
            f"Campos disponíveis: {q.dtype.names}. Estrutura inesperada, inspecione manualmente."
        )
    label = str(np.atleast_1d(q["label"]).ravel()[0])
    if label != "Pa":
        raise SystemExit(f"Abortado: {path} — unidade é {label!r}, esperava 'Pa'.")
    return label


def main() -> int:
    if OUT_PATH.exists():
        raise SystemExit(f"Abortado: {OUT_PATH} já existe — não sobrescrevo. Apague/mova antes de rodar de novo.")

    print("=" * 78)
    print("1) Leitura em Pa + conferências (canal, fs, duração, unidade)")
    print("=" * 78)

    sinais: dict[str, np.ndarray] = {}
    binario: dict[str, str] = {}
    verif_rows = []
    for fname, (label5, label_bin) in CLASS_FILES.items():
        path = RAW_DIR / fname
        values, fs, y = load_acoustic(path)

        n = values.shape[0]
        dur = n / fs
        ok_dur = (n == N_EXPECTED)
        if not ok_dur:
            raise SystemExit(
                f"Abortado: {fname} tem {n} amostras ({dur:.4f} s), esperava "
                f"{N_EXPECTED} amostras (60 s a {FS_EXPECTED:.0f} Hz)."
            )

        unidade = checar_unidade_pa(y, fname)

        sinais[label5] = values
        binario[label5] = label_bin
        verif_rows.append({
            "classe": label5, "arquivo": fname, "canal": "Acoustic (ok)",
            "fs_hz": fs, "n_amostras": n, "duracao_s": dur, "unidade": unidade,
            "pico_pa": float(np.abs(values).max()),
        })
        print(f"  {label5:12s} ({fname}): canal=Acoustic  fs={fs:.1f} Hz  "
              f"n={n} ({dur:.1f} s)  unidade={unidade}  pico={np.abs(values).max():.6f} Pa")

    print("\nTodas as conferências passaram: canal Acoustic, fs=51.200 Hz, "
          "duração=60 s (3.072.000 amostras), unidade Pa.\n")

    # --- 2) PSD igual a discriminative_curves() -----------------------------
    print("=" * 78)
    print("2) PSD — scipy.signal.welch(fs=51200, nperseg=16384, window='hann')")
    print("=" * 78)
    psd = {}
    freq = None
    for label5, x in sinais.items():
        f, p = sg.welch(x, fs=FS_EXPECTED, nperseg=min(NPERSEG, len(x)), window="hann")
        psd[label5] = p
        freq = f
    print(f"  resolução Δf = {freq[1] - freq[0]:.4f} Hz (esperado {FS_EXPECTED / NPERSEG:.4f} Hz)\n")

    # --- 3) Conferência contra metrics.json, ANTES de gravar a figura -------
    print("=" * 78)
    print("3) Conferência contra reports/decimation/metrics.json "
          "(energia_discriminante_f95_hz)")
    print("=" * 78)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(f"  metrics.json: sintetico={metrics.get('sintetico')}  "
          f"duracao_s={metrics.get('duracao_s')}  n_clipes={metrics.get('n_clipes')}")
    print("  -> rodada real (não sintética), 60 s inteiros por clipe: comparável "
          "com a leitura completa em Pa feita aqui.")
    print("  -> fração acumulada normalizada é invariante a um fator de escala comum "
          "às 5 classes: a quantização int16 do pipeline usa um ÚNICO fator de escala "
          "para as 5 classes (ver 01_convert_mat_to_pcm.py), então Pa vs. int16/32767 "
          "não deveria mudar o f95 alem do ruído de quantização (desprezível p/ 16 bits).\n")

    p_norm = psd["normal"]
    f95_relatorio = metrics["energia_discriminante_f95_hz"]
    tabela = []
    diffs_ok = True
    for label5 in ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm"):
        p = psd[label5]
        d = np.maximum(0.0, p - p_norm)
        total = float(np.trapezoid(d, freq))
        acum = np.concatenate([[0.0], np.cumsum(np.diff(freq) * (d[:-1] + d[1:]) / 2)]) / total
        f95_recalc = float(np.interp(0.95, acum, freq))
        f95_rel = f95_relatorio[label5]
        diff = abs(f95_recalc - f95_rel)
        ok = diff <= TOL_HZ
        diffs_ok &= ok
        tabela.append((label5, f95_rel, f95_recalc, diff, ok))

    print(f"  {'classe':<12} | {'relatório (Hz)':>14} | {'recalculado (Hz)':>17} | "
          f"{'diferença (Hz)':>14} | ok (<= {TOL_HZ:.3f} Hz)")
    print("  " + "-" * 78)
    for label5, rel, rec, diff, ok in tabela:
        print(f"  {label5:<12} | {rel:>14.1f} | {rec:>17.1f} | {diff:>14.3f} | {'sim' if ok else 'NAO'}")
    print()

    if not diffs_ok:
        print("ABORTADO: diferença acima de 1 bin (3,125 Hz) em pelo menos uma classe.")
        print("Causa provável mais comum: metrics.json foi gerado com --seconds (recorte) "
              "ou --synthetic (sinal sintético), e não com os 60 s completos do dataset real "
              "usados aqui. Verificado acima: sintetico=" + str(metrics.get("sintetico")) +
              ", duracao_s=" + str(metrics.get("duracao_s")) + ".")
        print("A figura NÃO foi gravada.")
        return 2

    print("Conferência OK (todas as diferenças dentro de 1 bin). Prosseguindo para a figura.\n")

    # --- 4) Figura ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    TINTA, TINTA_2, GRADE = "#1f1f1e", "#5f5e58", "#d9d8d2"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2,
        "ytick.color": TINTA_2, "text.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    })
    virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    CORES = {
        "normal": (TINTA, 1.3),
        "bpfi_0.3mm": ("#86b6ef", 0.8),
        "bpfi_1.0mm": ("#2a78d6", 0.8),
        "bpfo_0.3mm": ("#f2a37f", 0.8),
        "bpfo_1.0mm": ("#eb6834", 0.8),
    }
    ORDEM = ["normal", "bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm"]

    fig, ax = plt.subplots(figsize=(9, 5))
    f_khz = freq[1:] / 1000.0   # sem o bin DC
    for label5 in ORDEM:
        cor, lw = CORES[label5]
        p_db = 10 * np.log10(psd[label5][1:])
        ax.plot(f_khz, p_db, color=cor, lw=lw, label=label5)

    ax.set_xlim(0, 25.6)
    ax.xaxis.set_major_formatter(virgula)

    ymin, ymax = ax.get_ylim()
    folga = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin, ymax + folga)
    ytopo = ymax + folga * 0.97

    NYQUIST_LINHAS = [
        (3.2, "6.400 Hz", False),
        (4.0, "8.000 Hz", False),
        (6.4, "12.800 Hz (adotada)", True),
        (8.0, "16.000 Hz", False),
        (12.8, "25.600 Hz", False),
    ]
    for x_khz, rotulo, destaque in NYQUIST_LINHAS:
        cor = "#c0392b" if destaque else TINTA_2
        lw = 1.4 if destaque else 0.8
        ax.axvline(x_khz, color=cor, ls="--", lw=lw, zorder=1)
        ax.text(x_khz, ytopo, rotulo, color=cor, fontsize=7.5,
                 fontweight="bold" if destaque else "normal",
                 rotation=90, ha="right", va="top")

    ax.set_xlabel("frequência (kHz) — tracejado: Nyquist de cada taxa candidata")
    ax.set_ylabel("PSD (dB re 1 Pa²/Hz)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)

    print(f"Figura gravada em: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
