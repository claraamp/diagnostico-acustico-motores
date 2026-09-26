#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extract_features.py — extrai e salva o MFCC de todos os segmentos
====================================================================

Corta cada gravação decimada em clipes rígidos de 1 segundo sem sobreposição
entre janelas e blocos vizinhos[cite: 7]. Aplica a normalização e extrai o 
MFCC.

Durante a extração, salva os coeficientes do primeiro segmento de um clipe 
de referência para comparação e validação futura da versão em C[cite: 7].
"""

import argparse
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
    ap.add_argument("--sem-norm", action="store_true", help="desliga a normalização temporal (ablação)")
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
    
    print(f"Extraindo MFCC (1s por segmento, norm={'desligada' if args.sem_norm else config.TIPO_NORMALIZACAO})...")
    
    for clipe in clipes:
        n_segmentos = len(clipe.x) // amostras_por_seg
        matriz_segmentos = []
        
        for i in range(n_segmentos):
            inicio = i * amostras_por_seg
            fim = inicio + amostras_por_seg
            trecho = clipe.x[inicio:fim].astype(np.float64)
            
            # Normalização por clipe, sem estado[cite: 7]
            if not args.sem_norm and getattr(config, "TIPO_NORMALIZACAO", None) == "clipe":
                trecho = dsp.normalizar_rms_clipe(trecho)
                
            if getattr(config, "APLICAR_HANNING_TEMPO", False):
                trecho = dsp.aplicar_janela_hanning(trecho)
                
            # Extração de MFCC isolada no trecho[cite: 7]
            m = dsp.mfcc(trecho, clipe.fs, norm_cepstral=False)
            
            # Guarda os coeficientes do primeiro clipe normal da iteração[cite: 7]
            if clipe.rotulo == "normal" and not referencia_salva:
                exportar_referencia_c(trecho, m, args.ref_dir / "reference_data.h")
                print(f"  [Ref] C reference exportada para: {args.ref_dir}/reference_data.h")
                referencia_salva = True
                
            # Representação final para o classificador
            resumo = np.concatenate([m.mean(axis=0), m.std(axis=0)])
            matriz_segmentos.append(resumo)
            
        features[clipe.rotulo] = np.vstack(matriz_segmentos)
        print(f"  {clipe.rotulo:<12}: {n_segmentos} segmentos extraídos.")

    destino = args.out_dir / "mfcc_features.npz"
    np.savez_compressed(destino, **features)
    print(f"\nFeatures salvas em: {destino}")

    return 0

if __name__ == "__main__":
    sys.exit(main())