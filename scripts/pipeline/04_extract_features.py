#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extract_features.py — extrai e salva o MFCC de todos os segmentos
====================================================================

Percorre os segmentos do `splits.json` na ordem dos ids (`particao.segmentos_de`)
e salva uma matriz `X` em que a linha i é o segmento i: média e desvio de cada
coeficiente MFCC ao longo dos quadros. Segmento é definido num lugar só — a
partição —, e o `run_protocol.py` confere o hash dela antes de usar as features.

Saídas (não versionadas, reconstruíveis):
  data/processed/features/mfcc_features.npz        matriz X
  data/processed/features/manifest_features.json   parâmetros da extração

Opções
------
--norm-clipe   normaliza cada segmento pelo próprio RMS antes do MFCC (ablação)
--ref-c        exporta `reports/c_reference/reference_data.h` com a entrada int16
               e o MFCC do primeiro segmento da classe `normal`, para validar o
               porte em C da Fase 2. Só sem `--norm-clipe`.

Uso
---
    python scripts/pipeline/04_extract_features.py
    python scripts/pipeline/04_extract_features.py --norm-clipe
    python scripts/pipeline/04_extract_features.py --ref-c
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import dsp
import experimentos
import pcm_io
from validation import particao


def exportar_referencia_c(sinal_bruto_int16: np.ndarray, mfcc_matriz: np.ndarray, destino: Path):
    """Gera um arquivo .h com o vetor de entrada em int16 e a matriz MFCC exata."""
    linhas = [
        "// Referencia gerada para validacao do porte CMSIS-DSP",
        "#ifndef REFERENCE_DATA_H",
        "#define REFERENCE_DATA_H",
        "",
        "#include <stdint.h>",
        "#include <arm_math.h>",
        "",
        f"#define REF_SAMPLES_LEN {len(sinal_bruto_int16)}",
        f"#define REF_MFCC_FRAMES {mfcc_matriz.shape[0]}",
        f"#define REF_MFCC_COEFS {mfcc_matriz.shape[1]}",
        "",
        "static const int16_t ref_signal[REF_SAMPLES_LEN] = {"
    ]
    linhas.append("    " + ", ".join(str(x) for x in sinal_bruto_int16))
    linhas.append("};")
    linhas.append("")
    linhas.append("static const float32_t ref_mfcc_matrix[REF_MFCC_FRAMES * REF_MFCC_COEFS] = {")
    # .9g preserva todos os dígitos significativos de um float32
    linhas.append("    " + ", ".join(f"{x:.9g}f" for x in mfcc_matriz.flatten()))
    linhas.append("};")
    linhas.append("")
    linhas.append("#endif // REFERENCE_DATA_H")
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--ref-dir", type=Path, default=Path("reports/c_reference"))
    ap.add_argument("--norm-clipe", action="store_true",
                    help="normaliza cada segmento pelo próprio RMS antes do MFCC")
    ap.add_argument("--ref-c", action="store_true",
                    help="exporta o header C de referência (primeiro segmento 'normal')")
    args = ap.parse_args()

    if args.ref_c and args.norm_clipe:
        # o .h guarda a entrada bruta; o MFCC dele tem que ser o dessa entrada
        ap.error("--ref-c não pode ser combinado com --norm-clipe")

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: os PCM não batem com o manifest versionado.")
        return 1

    particoes = particao.carregar(args.splits)
    problemas = particao.conferir_compatibilidade(particoes, clipes)
    if problemas:
        print("\nAbortado — splits.json incompatível com estes dados:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    hash_splits = particao.hash_arquivo(args.splits)
    segmentos = particao.segmentos_de(particoes)
    por_rotulo = {c.rotulo: c for c in clipes}

    print(f"Extraindo features de {len(segmentos)} segmentos (norm_clipe={args.norm_clipe})...")
    linhas_X = []
    referencia_salva = False
    for seg in segmentos:
        c = por_rotulo[seg.rotulo]
        sinal_int16 = c.pcm[seg.inicio:seg.fim]   # int16 original, para a referência em C
        trecho = c.x[seg.inicio:seg.fim]          # float em [-1, 1], escala config.INT16_FULL
        if args.norm_clipe:
            trecho = dsp.normalizar_rms_clipe(trecho)

        m = dsp.mfcc(trecho, config.FS_TRABALHO)
        if args.ref_c and seg.rotulo == "normal" and not referencia_salva:
            args.ref_dir.mkdir(parents=True, exist_ok=True)
            exportar_referencia_c(sinal_int16, m, args.ref_dir / "reference_data.h")
            referencia_salva = True
        linhas_X.append(np.concatenate([m.mean(axis=0), m.std(axis=0)]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destino_features = args.out_dir / "mfcc_features.npz"
    destino_manifesto = args.out_dir / "manifest_features.json"
    np.savez_compressed(destino_features, X=np.vstack(linhas_X))

    manifesto = {
        "git_commit": experimentos.git_short_hash(),
        "splits_hash": hash_splits,
        "n_segmentos": len(segmentos),
        "fs_hz": config.FS_TRABALHO,
        "norm_clipe": args.norm_clipe,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS,
    }
    destino_manifesto.write_text(json.dumps(manifesto, indent=2) + "\n", encoding="utf-8")
    print(f"\nfeatures:  {destino_features}")
    print(f"manifesto: {destino_manifesto}")
    if referencia_salva:
        print(f"referência C: {args.ref_dir / 'reference_data.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
