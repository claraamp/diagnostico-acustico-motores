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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import dsp
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
    # Usa .9g para preservar precisão máxima do float32 na geração C
    linhas.append("    " + ", ".join(f"{x:.9g}f" for x in mfcc_matriz.flatten()))
    linhas.append("};")
    linhas.append("")
    linhas.append("#endif // REFERENCE_DATA_H")
    
    destino.write_text("\n".join(linhas), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrai MFCC dos segmentos definidos na partição.")
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/features"))
    ap.add_argument("--ref-dir", type=Path, default=Path("reports/c_reference"))
    ap.add_argument("--norm-clipe", action="store_true", help="aplica normalização RMS isolada por segmento")
    ap.add_argument("--ref-c", action="store_true", help="exporta header C de referência no primeiro segmento")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.ref_c:
        args.ref_dir.mkdir(parents=True, exist_ok=True)

    particoes = particao.carregar(args.splits)
    hash_splits = particao.hash_arquivo(args.splits)
    segmentos = particao.segmentos_de(particoes)

    # Dicionário de buffers PCM brutos na memória para acesso indexado
    buffers_pcm = {}
    for classe in config.CLASSES:
        caminho = args.pcm_dir / f"{classe}.bin"
        if caminho.exists():
            # Carrega como int16 original para extrair a referência em C corretamente
            buffers_pcm[classe] = np.fromfile(caminho, dtype=np.int16)

    matriz_X = []
    referencia_salva = False
    
    print(f"Extraindo features de {len(segmentos)} segmentos (norm_clipe={args.norm_clipe})...")
    
    for seg in segmentos:
        # Usa seg.rotulo que carrega o nome exato do arquivo/classe na partição
        sinal_int16 = buffers_pcm[seg.rotulo][seg.inicio:seg.fim]
        
        # Converte para float e escala para o domínio [-1.0, 1.0] para processamento matemático
        trecho = sinal_int16.astype(np.float64) / 32768.0
        
        if args.norm_clipe:
            trecho = dsp.normalizar_rms_clipe(trecho)
            
        m = dsp.mfcc(trecho, config.FS_TRABALHO)
        
        if args.ref_c and seg.rotulo == "normal" and not referencia_salva:
            exportar_referencia_c(sinal_int16, m, args.ref_dir / "reference_data.h")
            referencia_salva = True
            
        resumo = np.concatenate([m.mean(axis=0), m.std(axis=0)])
        matriz_X.append(resumo)

    destino_features = args.out_dir / "mfcc_features.npz"
    destino_manifesto = args.out_dir / "manifest_features.json"
    
    # Salva uma matriz unificada, onde a linha i bate com o segmento i
    X = np.vstack(matriz_X)
    np.savez_compressed(destino_features, X=X)
    
    manifesto = {
        "norm_clipe": args.norm_clipe,
        "splits_hash": hash_splits,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS
    }
    destino_manifesto.write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    
    print(f"\nFeatures salvas em: {destino_features}")
    print(f"Manifesto salvo em: {destino_manifesto}")
    return 0

if __name__ == "__main__":
    sys.exit(main())