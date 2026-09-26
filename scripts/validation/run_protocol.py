#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_protocol.py — roda o Protocolo A ou B sobre a partição do splits.json
=========================================================================

Treina e avalia um classificador em cada fold da partição gerada pelo
`pipeline/03_make_splits.py`, e grava o resultado em
`reports/validation/<id>_<descrição>/` e no `experiments/registry.csv`.

Leitura dos resultados (Registro de Decisões, 24/09):
  A — limite superior otimista. Treino e teste vêm das mesmas gravações; NÃO
      conta para a meta.
  B — resultado principal. A gravação de falha testada nunca foi vista no
      treino. Meta: acurácia balanceada média ≥ 85 %, reportada com
      sensibilidade, especificidade e o pior caso.

Features
--------
Carrega a matriz pré-calculada pelo `04_extract_features.py` (média e desvio
dos coeficientes MFCC por segmento).

Regras do protocolo que este script cumpre
------------------------------------------
- Lê a partição do splits.json; não sorteia nada. Aborta se o arquivo violar
  o protocolo ou não corresponder aos dados carregados.
- O escalonamento (`StandardScaler`) é ajustado só no treino de cada fold.
- Sem aumento de dados nesta versão.

Controles
---------
--sem-c0     descarta o coeficiente 0 (energia): ablação do ganho
--permutar   embaralha os rótulos DE TREINO por bloco (gravação × bloco), com a
             semente do config, e avalia nos rótulos verdadeiros; o resultado
             deve cair para ~50 % — se não cair, há vazamento no pipeline

Uso
---
    python scripts/validation/run_protocol.py --protocolo A
    python scripts/validation/run_protocol.py --protocolo A --tarefa multiclasse
    python scripts/validation/run_protocol.py --protocolo B
    python scripts/validation/run_protocol.py --protocolo B --sem-c0
    python scripts/validation/run_protocol.py --protocolo B --sem-registro   # teste
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import experimentos
import pcm_io
from validation import metricas, particao

FEATURES_ID = "mfcc_dsp_media_desvio"


# --------------------------------------------------------------------------- #
# Features e Rótulos
# --------------------------------------------------------------------------- #
def rotulos(segmentos: list[particao.Segmento], tarefa: str) -> np.ndarray:
    return np.array([s.binario if tarefa == "binario" else s.rotulo for s in segmentos])


def permutar_treino_por_bloco(segmentos: list[particao.Segmento], ids: list[int],
                              y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Rótulos de TREINO sorteados de novo por grupo (gravação, bloco), mantendo
    quantos grupos há de cada classe. Todos os segmentos de um grupo recebem o
    mesmo rótulo: a estrutura temporal fica igual, só a relação rótulo–sinal
    some. O teste continua com os rótulos verdadeiros — um modelo treinado sem
    relação rótulo–sinal tem que acertar ~50 % deles. Se acertar mais, alguma
    informação do teste está chegando ao treino por outro caminho.
    """
    grupos: dict[tuple[str, int], list[int]] = {}
    for pos, i in enumerate(ids):
        s = segmentos[i]
        grupos.setdefault((s.rotulo, s.bloco), []).append(pos)
    chaves = list(grupos)
    y_tr = y[ids].copy()
    novos = rng.permutation([y_tr[grupos[k][0]] for k in chaves])
    for k, r in zip(chaves, novos):
        y_tr[grupos[k]] = r
    return y_tr


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def novo_modelo(nome: str, n_classes: int):
    if nome == "lda":
        # priors uniformes: a LDA não aceita pesos de classe, e o treino do B
        # tem ~177 segmentos de falha para ~48 normais. Com priors iguais, a
        # fronteira não é empurrada para a classe majoritária.
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(priors=np.full(n_classes, 1.0 / n_classes)),
        )
    raise ValueError(f"modelo desconhecido: {nome}")


def rodar_folds(folds, X, y, tarefa: str, modelo: str,
                segmentos: list[particao.Segmento] | None = None,
                permutar: bool = False) -> list[dict]:
    classes = np.unique(y)
    rng = np.random.default_rng(config.SEMENTE)
    resultados = []
    for f in folds:
        y_tr = (permutar_treino_por_bloco(segmentos, f.treino, y, rng) if permutar
                else y[f.treino])
        mdl = novo_modelo(modelo, len(classes))
        mdl.fit(X[f.treino], y_tr)               # scaler ajustado só no treino
        pred = mdl.predict(X[f.teste])
        if tarefa == "binario":
            m = metricas.metricas_binarias(y[f.teste], pred)
        else:
            m = {"acuracia_balanceada": metricas.acuracia_balanceada(y[f.teste], pred),
                 "recall_por_classe": metricas.recall_por_classe(y[f.teste], pred)}
        resultados.append({"nome": f.nome, "info": f.info, "n_treino": len(f.treino),
                           "n_teste": len(f.teste), **m})
    return resultados


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
def imprimir(resumo: dict, resultados: list[dict]) -> None:
    print(f"\nProtocolo {resumo['protocolo']} — {resumo['leitura']}")
    if resumo["protocolo"] == "A":
        for r in resultados:
            print(f"  {r['nome']:<10} acurácia balanceada {r['acuracia_balanceada']:.3f}")
        print(f"\n  média {resumo['acuracia_balanceada_media']:.3f} ± "
              f"{resumo['acuracia_balanceada_desvio']:.3f}  "
              f"(mín. {resumo['acuracia_balanceada_min']:.3f})")
        return
    print(f"\n  {'falha deixada de fora':<22} {'sensib.':>8} {'especif.':>9} {'acc. bal.':>10}")
    for falha, t in resumo["por_falha"].items():
        print(f"  {falha:<22} {t['sensibilidade']:>8.3f} {t['especificidade']:>9.3f} "
              f"{t['acuracia_balanceada']:>10.3f}")
    print(f"  {'média':<22} {resumo['sensibilidade_media']:>8.3f} "
          f"{resumo['especificidade_media']:>9.3f} {resumo['acuracia_balanceada_media']:>10.3f}")
    print(f"\n  pior falha: {resumo['pior_falha']} "
          f"({resumo['pior_falha_acuracia_balanceada']:.3f})")
    print(f"  pior fold:  {resumo['pior_fold']} ({resumo['pior_fold_acuracia_balanceada']:.3f})")
    print(f"  meta de 85 %: {'ATINGIDA' if resumo['meta_atingida'] else 'não atingida'}")


def gravar_folds_csv(caminho: Path, resultados: list[dict]) -> None:
    campos = [k for k in resultados[0] if k not in ("info", "recall_por_classe")]
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(campos + ["info"])
        for r in resultados:
            w.writerow([r[k] for k in campos] + [json.dumps(r["info"], ensure_ascii=False)])


def metricas_registry(resumo: dict) -> dict:
    if resumo["protocolo"] == "A":
        return {"acc_bal_media": f"{resumo['acuracia_balanceada_media']:.4f}",
                "acc_bal_desvio": f"{resumo['acuracia_balanceada_desvio']:.4f}",
                "acc_bal_min": f"{resumo['acuracia_balanceada_min']:.4f}"}
    m = {"acc_bal_media": f"{resumo['acuracia_balanceada_media']:.4f}",
         "sens_media": f"{resumo['sensibilidade_media']:.4f}",
         "espec_media": f"{resumo['especificidade_media']:.4f}",
         "acc_bal_pior_falha": f"{resumo['pior_falha_acuracia_balanceada']:.4f}",
         "acc_bal_pior_fold": f"{resumo['pior_fold_acuracia_balanceada']:.4f}"}
    for falha, t in resumo["por_falha"].items():
        m[f"acc_bal_{falha}"] = f"{t['acuracia_balanceada']:.4f}"
    return m


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocolo", choices=["A", "B"], required=True)
    ap.add_argument("--tarefa", choices=["binario", "multiclasse"], default="binario")
    ap.add_argument("--modelo", choices=["lda"], default="lda")
    ap.add_argument("--sem-c0", action="store_true", help="ablação do ganho")
    ap.add_argument("--permutar", action="store_true", help="controle de permutação por bloco")
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--features", type=Path, default=Path("data/processed/features/mfcc_features.npz"))
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/validation"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true",
                    help="rodada de teste: não escreve no registry")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.protocolo == "B" and args.tarefa == "multiclasse":
        ap.error("o Protocolo B é binário: cada falha testada não aparece no treino, "
                 "então não há como acertar a classe dela")

    # ------------------------------------------------ dados e partição
    # O carregamento do PCM agora serve estritamente para manter a verificação de integridade
    # do manifesto e a compatibilidade do splits.json.
    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: os PCM não batem com o manifest versionado.")
        return 1

    particoes = particao.carregar(args.splits)
    problemas = particao.conferir_compatibilidade(particoes, clipes) + particao.verificar(particoes)
    if problemas:
        print("\nAbortado — splits.json inválido para estes dados:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    hash_splits = particao.hash_arquivo(args.splits)

    segmentos = particao.segmentos_de(particoes)
    folds = particao.folds_de(particoes, args.protocolo)
    print(f"\n{len(segmentos)} segmentos, {len(folds)} folds do Protocolo {args.protocolo} "
          f"(splits {hash_splits})")

    # ------------------------------------------------ features e rótulos
    print(f"Carregando features extraídas ({args.features.name})...")
    if not args.features.exists():
        print(f"Abortado: arquivo de features não encontrado em {args.features}")
        print("Execute o 04_extract_features.py primeiro.")
        return 1

    dados_extraidos = np.load(args.features)
    
    linhas_X = []
    for s in segmentos:
        idx = s.inicio // config.AMOSTRAS_POR_SEGMENTO
        resumo = dados_extraidos[s.rotulo][idx].copy()
        
        # Ablação do ganho: remove o coeficiente de energia c0 da média e do desvio[cite: 7]
        if args.sem_c0:
            resumo = np.delete(resumo, [0, config.MFCC_N_COEFS])
            
        linhas_X.append(resumo)
        
    X = np.vstack(linhas_X)
    y = rotulos(segmentos, args.tarefa)

    # ------------------------------------------------ folds
    resultados = rodar_folds(folds, X, y, args.tarefa, args.modelo,
                             segmentos=segmentos, permutar=args.permutar)
    resumo = (metricas.resumo_protocolo_a(resultados) if args.protocolo == "A"
              else metricas.resumo_protocolo_b(resultados))
    imprimir(resumo, resultados)

    # ------------------------------------------------ gravação
    descricao = "_".join(p for p in (
        f"protocolo{args.protocolo}", args.tarefa, args.modelo,
        "semc0" if args.sem_c0 else None, "permutado" if args.permutar else None,
    ) if p)
    exp_id = "teste" if args.sem_registro else f"exp{experimentos.next_exp_number(args.registry):03d}"
    destino = args.out_dir / f"{exp_id}_{descricao}"
    destino.mkdir(parents=True, exist_ok=True)

    parametros = {
        "protocolo": args.protocolo,
        "tarefa": args.tarefa,
        "modelo": args.modelo,
        "features": FEATURES_ID,
        "sem_c0": args.sem_c0,
        "permutado": args.permutar,
        "splits": hash_splits,
        "fs_hz": config.FS_TRABALHO,
        "segmento_s": config.SEGMENTO_S,
        "segmentos_por_bloco": config.SEGMENTOS_POR_BLOCO,
        "segmentos_descarte": config.SEGMENTOS_DESCARTE,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS,
        "semente": config.SEMENTE if args.permutar else None,
    }
    (destino / "metrics.json").write_text(json.dumps(
        {"id": exp_id, "parametros": parametros, "resumo": resumo, "folds": resultados},
        ensure_ascii=False, indent=2), encoding="utf-8")
    gravar_folds_csv(destino / "folds.csv", resultados)
    print(f"\nresultados: {destino}/")

    if args.sem_registro:
        print("(rodada de teste: registry não alterado)")
        return 0

    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "validacao_classificador",
        "script": "validation/run_protocol.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv(parametros),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv(metricas_registry(resumo)),
        "responsavel": args.responsavel,
        "notas": args.notas or (
            "limite otimista, não conta para a meta" if args.protocolo == "A"
            else "resultado principal (meta = acc_bal_media do B)"),
    }])
    print(f"registrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())