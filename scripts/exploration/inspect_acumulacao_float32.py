#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_acumulacao_float32.py — média e desvio do MFCC em float32, por método
=============================================================================

Pergunta: como o firmware deve acumular a média e o desvio de cada coeficiente
ao longo dos 98 quadros de um segmento, em float32, sem se afastar do Python?
Compara os três métodos de `dsp.media_desvio_float32` (Welford, soma e soma dos
quadrados, dois passos com a matriz guardada) com a referência em float64, nos
segmentos da partição.

Isola o erro da acumulação: os quadros de MFCC são os do Python, arredondados
para float32 (o que o C entregaria ao acumulador se o MFCC saísse igual), e a
referência é a média e o desvio populacional (ddof=0) desses mesmos valores em
float64. O erro do MFCC em float32 é outro assunto (tarefas B e A4).

Métricas, por método (máximo sobre os segmentos e os 13 coeficientes):
  erro absoluto da média, erro absoluto e relativo do desvio, e o efeito no
  escore da LDA final (|pesos · Δx|), para comparar com a menor margem do escore.

Saída: reports/exploration/acumulacao_float32.json (com --sem-registro, só imprime)

Uso
---
    python scripts/exploration/inspect_acumulacao_float32.py --responsavel <nome>
    python scripts/exploration/inspect_acumulacao_float32.py --sem-registro
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
import dsp
import experimentos
import pcm_io
from validation import particao


def medir(segmentos, por_rotulo: dict, pesos: np.ndarray) -> dict:
    """Erros de cada método contra a referência em float64, segmento a segmento."""
    n = config.MFCC_N_COEFS
    acc = {m: {"media": [], "desvio": [], "desvio_rel": [], "escore": []}
           for m in dsp.METODOS_RESUMO}
    razao = []
    for s in segmentos:
        q32 = dsp.mfcc(por_rotulo[s.rotulo].x[s.inicio:s.fim], config.FS_TRABALHO).astype(np.float32)
        q = q32.astype(np.float64)
        media, desvio = q.mean(axis=0), q.std(axis=0)
        razao.append(float(np.max(np.abs(media) / desvio)))
        for metodo in dsp.METODOS_RESUMO:
            m32, d32 = dsp.media_desvio_float32(q32, metodo)
            dm, dd = m32.astype(np.float64) - media, d32.astype(np.float64) - desvio
            a = acc[metodo]
            a["media"].append(float(np.max(np.abs(dm))))
            a["desvio"].append(float(np.max(np.abs(dd))))
            a["desvio_rel"].append(float(np.max(np.abs(dd) / desvio)))
            a["escore"].append(float(abs(pesos[:n] @ dm + pesos[n:] @ dd)))
    return {
        "metodos": {m: {f"max_dif_{k}": max(v) for k, v in a.items()} for m, a in acc.items()},
        "max_razao_media_desvio": max(razao),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/exploration/acumulacao_float32.json"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    ap.add_argument("--sem-registro", action="store_true",
                    help="teste: só imprime; não grava o JSON nem escreve no registry")
    args = ap.parse_args()
    if not args.sem_registro and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada")

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, clipes, config.FS_TRABALHO)
                 + particao.verificar(particoes))
    if divergencias or problemas:
        print("Abortado: PCM ou partição inválidos:\n  " + "\n  ".join(problemas))
        return 1
    segmentos = particao.segmentos_de(particoes)
    modelo = json.loads(args.modelo.read_text(encoding="utf-8"))
    pesos = np.asarray(modelo["dobrada"]["pesos"], dtype=float)
    escores = np.abs([float(l.split(",")[4]) for l in
                      args.modelo.with_name("escores_referencia.csv").read_text(encoding="utf-8")
                      .splitlines()[1:]])

    print(f"{len(segmentos)} segmentos; métodos: {', '.join(dsp.METODOS_RESUMO)}")
    r = medir(segmentos, {c.rotulo: c for c in clipes}, pesos)
    r["min_abs_escore"] = float(escores.min())
    for metodo, m in r["metodos"].items():
        print(f"  {metodo:<15} " + "  ".join(f"{k}={v:.2e}" for k, v in m.items()))
    print(f"  maior |média|/desvio de um coeficiente: {r['max_razao_media_desvio']:.1f}; "
          f"menor |escore|: {r['min_abs_escore']:.0f}")
    if args.sem_registro:
        print("(rodada de teste: nada gravado)")
        return 0

    exp_id = f"exp{experimentos.next_exp_number(args.registry):03d}"
    saida = {"id": exp_id, "splits": particao.hash_arquivo(args.splits),
             "modelo": modelo.get("id"), "n_segmentos": len(segmentos),
             "quadros_por_segmento": 98, "n_coefs": config.MFCC_N_COEFS, **r}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "contrato_numerico",
        "script": "exploration/inspect_acumulacao_float32.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "precisao": "float32", "metodos": "+".join(dsp.METODOS_RESUMO),
            "splits": saida["splits"], "modelo": saida["modelo"], "fs_hz": config.FS_TRABALHO,
            "quadros_por_segmento": 98, "n_coefs": config.MFCC_N_COEFS,
        }),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            f"{m}_{k.removeprefix('max_dif_')}": f"{v:.1e}"
            for m, d in r["metodos"].items() for k, v in d.items()
            if k in ("max_dif_desvio_rel", "max_dif_escore")
        }),
        "responsavel": args.responsavel,
        "notas": args.notas or ("erro da média e do desvio do MFCC acumulados em float32, por "
                                "método, contra float64; isola a acumulação (A3 do contrato numérico)"),
    }])
    print(f"resultado: {args.out}\nregistrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
