#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extract_features.py — extrai e salva o MFCC de todos os segmentos
====================================================================

Corta cada gravação decimada em clipes rígidos de 1 segundo sem sobreposição
entre janelas e blocos vizinhos. Extrai o MFCC e agrupa a média e desvio.

Durante a extração, salva os coeficientes do primeiro segmento de um clipe 
de referência para comparação e validação futura da versão em C.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
import dsp
import pcm_io


def exportar_referencia_c(sinal_bruto: np.ndarray, mfcc_matriz: np.ndarray, destino: Path):
    """Gera um arquivo .h com o vetor de entrada e a matriz MFCC esperada."""
    linhas = [
        "// Referencia gerada pelo 04_extract_features.py para validacao do porte em C",
        "#ifndef REFERENCE_DATA_H",
        "#define REFERENCE_DATA_H",
        "",
        "#include <stdint.h>",
        "#include <arm_math.h>",
        "",
        f"#define REF_SAMPLES_LEN {len(sinal_bruto)}",
        f"#define REF_MFCC_FRAMES {mfcc_matriz.shape[0]}",
        f"#define REF_MFCC_COEFS {mfcc_matriz.shape[1]}",
        "",
        "const float32_t ref_signal[REF_SAMPLES_LEN] = {"
    ]
    
    linhas.append("    " + ", ".join(f"{x:.6f}" for x in sinal_bruto))
    linhas.append("};")
    linhas.append("")
    
    linhas.append("const float32_t ref_mfcc_matrix[REF_MFCC_FRAMES * REF_MFCC_COEFS] = {")
    linhas.append("    " + ", ".join(f"{x:.6f}" for x in mfcc_matriz.flatten()))
    linhas.append("};")
    linhas.append("")
    linhas.append("#endif // REFERENCE_DATA_H")
    
    destino.write_text("\n".join(linhas), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--ref-dir", type=Path, default=Path("reports/c_reference"))
    ap.add_argument("--norm-clipe", action="store_true", help="aplica normalização RMS isolada por segmento")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ref_dir.mkdir(parents=True, exist_ok=True)
    
    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: dados PCM corrompidos ou não batem com o manifest.")
        return 1

    amostras_por_seg = int(config.FS_TRABALHO * config.SEGMENTO_S)
    features = {}
    referencia_salva = False
    
    print(f"Extraindo MFCC (1s por segmento, norm_clipe={args.norm_clipe})...")
    
    for clipe in clipes:
        n_segmentos = len(clipe.x) // amostras_por_seg
        matriz_segmentos = []
        
        for i in range(n_segmentos):
            inicio = i * amostras_por_seg
            fim = inicio + amostras_por_seg
            # Clip.x já é float64 em [-1, 1], astype apenas assegura a cópia de segurança
            trecho = clipe.x[inicio:fim].astype(np.float64)
            
            if args.norm_clipe:
                trecho = dsp.normalizar_rms_clipe(trecho)
                
            # Extração de MFCC isolada no trecho
            m = dsp.mfcc(trecho, clipe.fs, norm_cepstral=False)
            
            if clipe.rotulo == "normal" and not referencia_salva:
                exportar_referencia_c(trecho, m, args.ref_dir / "reference_data.h")
                referencia_salva = True
                
            resumo = np.concatenate([m.mean(axis=0), m.std(axis=0)])
            matriz_segmentos.append(resumo)
            
        features[clipe.rotulo] = np.vstack(matriz_segmentos)
        print(f"  {clipe.rotulo:<12}: {n_segmentos} segmentos extraídos.")

    destino_features = args.out_dir / "mfcc_features.npz"
    destino_manifesto = args.out_dir / "manifest_features.json"
    
    np.savez_compressed(destino_features, **features)
    
    # Salva o manifesto para rastreabilidade no run_protocol.py
    manifesto = {"norm_clipe": args.norm_clipe}
    destino_manifesto.write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    
    print(f"\nFeatures salvas em: {destino_features}")
    print(f"Manifesto salvo em: {destino_manifesto}")

    return 0

if __name__ == "__main__":
    sys.exit(main())