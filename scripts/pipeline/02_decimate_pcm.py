#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_decimate_pcm.py — decima os PCM do dataset para a taxa de trabalho
======================================================================

Segunda etapa do pipeline. Lê os PCM a 51.200 Hz produzidos pelo
`01_convert_mat_to_pcm.py` e grava a versão decimada na taxa de trabalho
definida em `config.FS_TRABALHO`, com o filtro anti-aliasing projetado para ela.

Entra: data/processed/pcm_raw/         (.bin + manifest.json)
Sai:   data/processed/pcm_decimated/<fs>/   (.bin + manifest.json)

Este script **não decide** a taxa nem valida nada: ele aplica a decisão já
tomada. A comparação entre taxas candidatas, com as métricas e figuras que a
justificam, é um estudo pontual e está em
`scripts/exploration/compare_decimation_rates.py`.

A separação segue a distinção do CONVENTIONS.md entre `pipeline/` e
`exploration/`: isto aqui é transformação reprodutível, que roda de novo toda
vez que o dado mudar; aquilo respondeu uma pergunta uma vez. Por não variar
parâmetro nem produzir métrica comparável, esta etapa não escreve em
`experiments/registry.csv`.

Por que o fator tem que ser inteiro
-----------------------------------
51.200 / 12.800 = 4, exato. Isso não é detalhe de implementação: o
`arm_fir_decimate_f32` da CMSIS-DSP implementa exatamente "filtrar e descartar
amostras" e só aceita fator inteiro. Uma taxa que não divida a original exigiria
interpolação antes da decimação, com o filtro operando a uma taxa muito mais
alta. O `config.py` tem um `assert` que trava se alguém mudar `FS_TRABALHO` para
um valor não divisível.

Uso
---
    python scripts/pipeline/02_decimate_pcm.py
    python scripts/pipeline/02_decimate_pcm.py --fs 6400     # outra taxa, para teste
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
import pcm_io
from dsp import design_decimation, resample_clip

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/pcm_decimated"))
    ap.add_argument("--fs", type=float, default=float(config.FS_TRABALHO),
                    help="taxa de trabalho (padrão: config.FS_TRABALHO)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="usar só os primeiros N s de cada clipe (teste)")
    args = ap.parse_args(argv)

    fs_orig = float(config.FS_ORIGINAL)

    clipes = pcm_io.carregar_clipes(args.pcm_dir, args.manifest, args.seconds,
                                    fs_esperado=fs_orig)
    pcm_io.relatar_integridade(
        pcm_io.verificar_integridade(clipes, truncado=args.seconds is not None))
    print(f"[load] {len(clipes)} clipes, {clipes[0].duracao_s:.1f} s cada, "
          f"fs = {fs_orig:.0f} Hz")

    spec = design_decimation(fs_orig, args.fs)
    fator = f"/{spec.down}" if spec.integer_factor else f"×{spec.up}/{spec.down}"
    print(f"[filtro] {fs_orig/1000:.1f} → {args.fs/1000:.1f} kHz, fator {fator}, "
          f"{spec.numtaps} taps, corte {spec.cutoff_hz:.0f} Hz, "
          f"atenuação {spec.stopband_atten_db:.1f} dB")
    if not spec.integer_factor:
        print("           ATENÇÃO: fator não inteiro — incompatível com "
              "arm_fir_decimate_f32 no porte embarcado")

    decimados = [c.derivado(resample_clip(c.x, fs_orig, args.fs, spec), args.fs)
                 for c in clipes]

    destino = args.out_dir / f"{int(args.fs)}"
    manifest = pcm_io.gravar_clipes(
        decimados, destino, args.fs,
        extras={"fonte": str(args.pcm_dir), "decimacao": asdict(spec)})

    n = decimados[0].n
    janela = int(round(config.MFCC_WINDOW_MS / 1000 * args.fs))
    print(f"[grava] {len(decimados)} clipes em {destino}")
    print(f"         {n} amostras por clipe | {n * 2 / 1024:.1f} KiB por clipe")
    print(f"         {janela} amostras por janela de {config.MFCC_WINDOW_MS:.0f} ms | "
          f"buffer de 1 s = {args.fs * 2 / 1024:.1f} KiB")
    print(f"[manifest] {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())