#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_classifier.py — treina o modelo final (LDA) com todos os segmentos
============================================================================

Treina uma única vez o classificador escolhido na tarefa 3/6 (padronização + LDA
com priors uniformes, o mesmo `novo_modelo` do run_protocol) com os 295 segmentos
da partição, na tarefa binária e sem aumento de dados, e grava os parâmetros num
formato neutro, sem depender do sklearn, para o porte na Fase 2.

**Não mede desempenho.** Medir desempenho é papel do `run_protocol.py`, que treina
e descarta um modelo por fold; o número da meta é o do Protocolo B (exp015). Este
script só produz o modelo que vai para o firmware (CONVENTIONS, seção 2).

Saídas (em reports/modelo_final/)
--------------------------------
lda_final.json           parâmetros nas formas padronizada e dobrada (ver
                         `models/lda.py`), a ordem das 26 features, a convenção do
                         escore (≥ 0 → falha), a cadeia que produz as features
                         (escala, janela, FFT, Mel, log, DCT) e o que foi usado no treino
escores_referencia.csv   escore e previsão de cada segmento pela forma dobrada:
                         referência para conferir a implementação em C. Os
                         segmentos são os do próprio treino; não é desempenho.

Antes de gravar, confere que o JSON reproduz o sklearn (mesmos escores e mesmas
previsões em todos os segmentos) e aborta se não reproduzir.

Pré-requisitos: os PCM decimados, o splits.json versionado e as features do `04`
sem `--norm-clipe` (as mesmas do exp015). Aborta se o manifesto das features não
bater com a partição e os parâmetros de MFCC do config.

Uso
---
    python scripts/pipeline/05_train_classifier.py --responsavel <nome>
    python scripts/pipeline/05_train_classifier.py --sem-registro     # teste: grava em teste/
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

import config
import experimentos
import pcm_io
from models import lda
from validation import particao
from validation.run_protocol import FEATURES_ID, rotulos


def carregar_features(features: Path, hash_splits: str, n_segmentos: int) -> tuple[np.ndarray, dict]:
    """X do 04, com o manifesto conferido contra a partição e o MFCC do config."""
    manifesto_path = features.with_name("manifest_features.json")
    for caminho in (features, manifesto_path):
        if not caminho.exists():
            raise SystemExit(f"Abortado: {caminho} não encontrado; rode o 04_extract_features.py.")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    esperado = {
        "splits_hash": hash_splits,
        "fs_hz": config.FS_TRABALHO,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS,
    }
    erros = [f"{k}: features={manifesto.get(k)}, esperado={v}"
             for k, v in esperado.items() if manifesto.get(k) != v]
    if manifesto.get("norm_clipe"):
        erros.append("features extraídas com --norm-clipe; o modelo final usa as de referência")
    if erros:
        raise SystemExit("Abortado: features extraídas com outra configuração; rode o 04 "
                         "sem opções.\n  " + "\n  ".join(erros))
    X = np.load(features)["X"]
    if len(X) != n_segmentos:
        raise SystemExit(f"Abortado: o .npz tem {len(X)} linhas e a partição tem {n_segmentos} segmentos.")
    if X.shape[1] != 2 * config.MFCC_N_COEFS:
        raise SystemExit(f"Abortado: esperava {2 * config.MFCC_N_COEFS} colunas, o .npz tem {X.shape[1]}.")
    return X, manifesto


def montar_saida(p: dict, conferencia: dict, exp_id: str, hash_splits: str,
                 manifesto: dict, y: np.ndarray) -> dict:
    n_falha = int(np.sum(y == lda.CLASSE_POSITIVA))
    return {
        "id": exp_id,
        **p,
        "entrada": {
            "mfcc_janela_ms": config.MFCC_WINDOW_MS,
            "mfcc_hop_ms": config.MFCC_HOP_MS,
            "mfcc_n_mels": config.MFCC_N_MELS,
            "mfcc_n_coefs": config.MFCC_N_COEFS,
            **lda.cadeia_entrada(config.FS_TRABALHO),
        },
        "uso_no_firmware": {
            "decisao": "usar só o sinal do escore (regra em `escore`); o tamanho do escore não é "
                       "confiança: vem de o treino conter as cinco gravações",
            "conferencia_com_c": "comparar o escore do C com o escores_referencia.csv com tolerância "
                                 "relativa (~1e-3): o MFCC em float32 difere um pouco do Python; "
                                 "a previsão tem que ser idêntica",
            "referencia_mfcc": "reports/c_reference/reference_data.h (entrada int16 e MFCC esperado)",
        },
        "treino": {
            "n_segmentos": int(len(y)),
            "n_falha": n_falha,
            "n_normal": int(len(y) - n_falha),
            "splits": hash_splits,
            "features": FEATURES_ID,
            "features_commit": manifesto.get("git_commit"),
            "aumento": "nenhum",
        },
        "conferencia_sklearn": conferencia,
    }


def gravar_escores(caminho: Path, segmentos: list, y: np.ndarray, p: dict, X: np.ndarray) -> None:
    escores = lda.escore_dobrado(p, X)
    previsao = lda.prever(p, X)
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segmento", "gravacao", "indice", "binario", "escore", "previsao"])
        for s, yi, e, pr in zip(segmentos, y, escores, previsao):
            w.writerow([s.id, s.rotulo, s.indice, yi, f"{e:.9g}", pr])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path,
                    default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/modelo_final"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    ap.add_argument("--sem-registro", action="store_true",
                    help="teste: grava em <out-dir>/teste/ e não escreve no registry")
    args = ap.parse_args()
    if not args.sem_registro and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada")

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: os PCM não batem com o manifest versionado.")
        return 1
    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, clipes, config.FS_TRABALHO)
                 + particao.verificar(particoes))
    if problemas:
        print("Abortado — splits.json inválido para estes dados:\n  " + "\n  ".join(problemas))
        return 1
    hash_splits = particao.hash_arquivo(args.splits)
    segmentos = particao.segmentos_de(particoes)

    X, manifesto = carregar_features(args.features, hash_splits, len(segmentos))
    y = rotulos(segmentos, "binario")
    print(f"{len(segmentos)} segmentos ({int(np.sum(y == 'falha'))} de falha, "
          f"{int(np.sum(y == 'normal'))} normais), splits {hash_splits}")

    modelo = lda.treinar(X, y)
    p = lda.parametros(modelo, config.MFCC_N_COEFS)
    conferencia = lda.conferir(modelo, p, X)
    print("conferência com o sklearn: " + ", ".join(f"{k}={v}" for k, v in conferencia.items()))
    if not conferencia["previsoes_iguais"] or conferencia["max_dif_relativa"] > 1e-9:
        print("Abortado: os parâmetros exportados não reproduzem o sklearn.")
        return 1

    exp_id = "teste" if args.sem_registro else f"exp{experimentos.next_exp_number(args.registry):03d}"
    destino = args.out_dir / "teste" if args.sem_registro else args.out_dir
    destino.mkdir(parents=True, exist_ok=True)
    saida = montar_saida(p, conferencia, exp_id, hash_splits, manifesto, y)
    (destino / "lda_final.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    gravar_escores(destino / "escores_referencia.csv", segmentos, y, p, X)
    print(f"modelo final: {destino}/lda_final.json")

    if args.sem_registro:
        print("(rodada de teste: registry não alterado)")
        return 0
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "treino_classificador",
        "script": "pipeline/05_train_classifier.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "tarefa": "binario", "modelo": "lda", "priors": "uniformes",
            "features": FEATURES_ID, "splits": hash_splits, "fs_hz": config.FS_TRABALHO,
            "segmento_s": config.SEGMENTO_S, "mfcc_janela_ms": config.MFCC_WINDOW_MS,
            "mfcc_hop_ms": config.MFCC_HOP_MS, "mfcc_n_mels": config.MFCC_N_MELS,
            "mfcc_n_coefs": config.MFCC_N_COEFS, "features_commit": manifesto.get("git_commit"),
            "aumento": "nenhum", "versao_formato": lda.VERSAO_FORMATO,
        }),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            "n_treino": len(y), "n_falha": saida["treino"]["n_falha"],
            "n_normal": saida["treino"]["n_normal"],
            "max_dif_escore_dobrado": f"{conferencia['max_dif_escore_dobrado']:.1e}",
        }),
        "responsavel": args.responsavel,
        "notas": args.notas or ("modelo final para o firmware (todos os segmentos, sem aumento); "
                                "não é medida de desempenho, que é a do exp015"),
    }])
    print(f"registrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())