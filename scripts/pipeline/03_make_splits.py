#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_make_splits.py — gera a partição dos protocolos A e B, uma única vez
======================================================================

Aplica o protocolo de validação registrado em 24/09 (Notion, Registro de
Decisões). Lê os PCM decimados, corta cada gravação em segmentos de
`config.SEGMENTO_S`, agrupa em blocos e grava todos os folds em
`data/processed/splits/splits.json` — arquivo VERSIONADO, lido por todo
experimento de classificação.

Por que uma vez só: se cada experimento gerasse a própria partição, os números
de rodadas diferentes deixariam de ser comparáveis. A partição é determinística,
então rodar de novo com os mesmos dados e o mesmo config dá o mesmo arquivo. Se
o arquivo existente for diferente, o script para, em vez de sobrescrever: uma
partição nova invalida a comparação com tudo o que foi registrado antes, e isso
tem que ser uma decisão, não um efeito colateral. Use `--sobrescrever` só
depois de registrar essa decisão.

Uso
---
    python scripts/pipeline/04_make_splits.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
import pcm_io
from validation import particao


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--saida", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--sobrescrever", action="store_true",
                    help="substitui um splits.json diferente (ver docstring)")
    args = ap.parse_args()

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: a partição seria gerada sobre dados que não batem com o manifest.")
        return 1

    particoes = particao.gerar_particoes(clipes)
    erros = particao.verificar(particoes)
    if erros:
        print("\nAbortado — a partição gerada viola o protocolo:")
        for e in erros:
            print(f"  - {e}")
        return 1

    # ------------------------------------------------------------ resumo
    segmentos = particao.segmentos_de(particoes)
    print(f"\nSegmentos de {config.SEGMENTO_S:g} s ({config.AMOSTRAS_POR_SEGMENTO} amostras), "
          f"blocos de {config.SEGMENTOS_POR_BLOCO}, descarte de {config.SEGMENTOS_DESCARTE}:\n")
    for rotulo, g in particoes["gravacoes"].items():
        blocos = Counter(s.bloco for s in segmentos if s.rotulo == rotulo)
        sobra = g["n_amostras"] - g["n_segmentos"] * config.AMOSTRAS_POR_SEGMENTO
        print(f"  {rotulo:<12} {g['binario']:<7} {g['n_segmentos']:>3} segmentos  "
              f"blocos {[blocos[b] for b in sorted(blocos)]}  sobra descartada: {sobra} amostras")

    for protocolo in ("A", "B"):
        folds = particao.folds_de(particoes, protocolo)
        n_tr = [len(f.treino) for f in folds]
        n_te = [len(f.teste) for f in folds]
        print(f"\n  Protocolo {protocolo}: {len(folds)} folds · treino {min(n_tr)}–{max(n_tr)} "
              f"· teste {min(n_te)}–{max(n_te)} segmentos")
        if protocolo == "B":
            normal_tr = [sum(segmentos[i].binario == "normal" for i in f.treino) for f in folds]
            falha_tr = [sum(segmentos[i].binario == "falha" for i in f.treino) for f in folds]
            print(f"    treino por fold: {min(falha_tr)}–{max(falha_tr)} de falha, "
                  f"{min(normal_tr)}–{max(normal_tr)} normais")

    # ------------------------------------------------------------ gravação
    texto = particao.para_texto(particoes)
    if args.saida.exists():
        atual = args.saida.read_text(encoding="utf-8")
        if atual == texto:
            print(f"\n{args.saida} já existe e é idêntico — nada a fazer.")
            print(f"hash: {particao.hash_arquivo(args.saida)}")
            return 0
        if not args.sobrescrever:
            print(f"\nAbortado: {args.saida} existe e é DIFERENTE da partição gerada agora.")
            print("Uma partição nova invalida a comparação com as rodadas já registradas.")
            print("Se a mudança é intencional, registre a decisão e rode com --sobrescrever.")
            return 1

    particao.salvar(particoes, args.saida)
    print(f"\ngravado: {args.saida}  (hash {particao.hash_arquivo(args.saida)})")
    print("Esse arquivo é versionado: commit junto com o código que o gerou.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())