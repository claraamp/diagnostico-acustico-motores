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
- Aborta se o `manifest_features.json` faltar, se as features vierem de outra
  partição ou de outros parâmetros de MFCC, ou se o número de linhas do `.npz`
  não bater com o de segmentos. Os parâmetros gravados são os do manifesto.
- O escalonamento (`StandardScaler`) é ajustado só no treino de cada fold.
- Aumento de dados (`--aumento`) entra SÓ no treino. As variantes vêm do
  `mfcc_aumento.npz` gerado pelo 04; em cada fold, uma variante só é aceita se
  todas as amostras que ela lê caírem em segmentos de treino daquele fold
  (`augmentation.variantes.mascara_fold`). Variante de um segmento de teste,
  ou que encosta no teste ou na faixa de descarte, fica de fora. O teste é
  sempre o segmento original.

Controles
---------
--sem-c0     descarta o coeficiente 0 (energia): ablação do ganho
--permutar   embaralha os rótulos DE TREINO por bloco (gravação × bloco), com a
             semente do config, e avalia nos rótulos verdadeiros; o resultado
             deve cair para ~50 % — se não cair, há vazamento no pipeline.
             Com --aumento, cada variante herda o rótulo embaralhado do seu
             segmento de origem.
--aumento    soma ao treino de cada fold as variantes aceitas do mfcc_aumento.npz
--segundos-treino N
             curva de aprendizado: em cada fold, sorteia N segundos de treino
             por classe (N segmentos de SEGMENTO_S), repartidos entre as
             gravações da classe, e descarta o resto do treino. O folds.csv
             registra quantos segmentos vieram de cada gravação. O teste é o mesmo. Se poucos segundos por classe já
             acertam tudo, o modelo está separando por um atalho, não pela
             falha. Mínimo de 2 segmentos por classe: com 1, a LDA não tem
             covariância dentro da classe.
--fs N       roda na taxa N (padrão: config.FS_TRABALHO), com o PCM, a partição
             e as features dessa taxa (`02`, `03` e `04` com `--fs N`), nos
             caminhos de `config`. Serve para comparar taxas com o mesmo
             protocolo e o mesmo modelo, sem mudar a taxa de produção. A
             taxa entra no nome da pasta (`_<N>hz`) quando não é a de trabalho.
--semente    semente dos sorteios (permutação e curva de aprendizado); padrão
             config.SEMENTE, que reproduz as rodadas já registradas

Uso
---
    python scripts/validation/run_protocol.py --protocolo A
    python scripts/validation/run_protocol.py --protocolo A --tarefa multiclasse
    python scripts/validation/run_protocol.py --protocolo B
    python scripts/validation/run_protocol.py --protocolo B --sem-c0
    python scripts/validation/run_protocol.py --protocolo B --aumento
    python scripts/validation/run_protocol.py --protocolo B --aumento --permutar
    python scripts/validation/run_protocol.py --protocolo B --permutar --semente 1
    python scripts/validation/run_protocol.py --protocolo B --segundos-treino 5
    python scripts/validation/run_protocol.py --protocolo B --fs 25600
    python scripts/validation/run_protocol.py --protocolo B --sem-registro   # teste
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
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
from augmentation import variantes
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


def cotas_por_gravacao(disponiveis: dict[str, int], n: int,
                       rng: np.random.Generator) -> dict[str, int]:
    """
    Reparte `n` segmentos entre as gravações de uma classe o mais igual
    possível. Quando não divide exato, as gravações que levam um a mais são
    sorteadas; uma gravação sem segmentos suficientes entra inteira e o que
    falta passa para as outras. No binário a classe "falha" junta 3 (B) ou 4 (A)
    gravações: sem isso, os poucos segmentos de um ponto da curva podiam vir
    todos da mesma.
    """
    ordem = [str(g) for g in rng.permutation(sorted(disponiveis))]
    cotas = {g: 0 for g in ordem}
    restante = min(n, sum(disponiveis.values()))
    while restante:
        abertas = [g for g in ordem if cotas[g] < disponiveis[g]]
        for g in abertas[:restante]:
            cotas[g] += 1
            restante -= 1
    return cotas


def subamostrar_treino(treino: list[int], y: np.ndarray, gravacao: np.ndarray,
                       n_por_classe: int, rng: np.random.Generator) -> list[int]:
    """
    Curva de aprendizado: sorteia `n_por_classe` segmentos de treino de cada
    classe, sem reposição, repartidos entre as gravações da classe
    (`cotas_por_gravacao`). Classe com menos segmentos que isso entra inteira.
    Só escolhe entre os ids de treino, então o teste e a faixa de descarte
    continuam fora.
    """
    treino = np.asarray(treino)
    escolhidos = []
    for c in np.unique(y[treino]):
        da_classe = treino[y[treino] == c]
        por_grav = {str(g): da_classe[gravacao[da_classe] == g] for g in np.unique(gravacao[da_classe])}
        cotas = cotas_por_gravacao({g: len(ids) for g, ids in por_grav.items()}, n_por_classe, rng)
        for g, k in cotas.items():
            escolhidos.extend(rng.choice(por_grav[g], size=k, replace=False).tolist())
    return sorted(int(i) for i in escolhidos)


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


def montar_treino(f, X: np.ndarray, y_tr: np.ndarray, aumento: dict | None):
    """
    Matriz e rótulos de treino do fold: os segmentos de treino e, com aumento,
    as variantes aceitas neste fold. Cada variante recebe o rótulo de treino do
    seu segmento de origem — o verdadeiro, ou o embaralhado no --permutar.
    """
    if aumento is None:
        return X[f.treino], y_tr, 0
    aceitas = variantes.mascara_fold(f.treino, aumento["lidos"])
    posicao = {int(i): k for k, i in enumerate(f.treino)}
    origem = [posicao[int(o)] for o in aumento["origem"][aceitas]]
    X_fit = np.vstack([X[f.treino], aumento["X"][aceitas]])
    y_fit = np.concatenate([y_tr, y_tr[origem]])
    return X_fit, y_fit, int(aceitas.sum())


def rodar_folds(folds, X, y, tarefa: str, modelo: str,
                segmentos: list[particao.Segmento] | None = None,
                permutar: bool = False, aumento: dict | None = None,
                semente: int = config.SEMENTE,
                segmentos_treino: int | None = None) -> list[dict]:
    classes = np.unique(y)
    rng = np.random.default_rng(semente)
    gravacao = np.array([s.rotulo for s in segmentos]) if segmentos is not None else None
    resultados = []
    for f in folds:
        extra = {}
        if segmentos_treino is not None:
            f = dataclasses.replace(f, treino=subamostrar_treino(f.treino, y, gravacao,
                                                                 segmentos_treino, rng))
            g, n = np.unique(gravacao[f.treino], return_counts=True)
            extra["treino_por_gravacao"] = json.dumps(dict(zip(g.tolist(), n.tolist())),
                                                      ensure_ascii=False)
        y_tr = (permutar_treino_por_bloco(segmentos, f.treino, y, rng) if permutar
                else y[f.treino])
        X_fit, y_fit, n_aumento = montar_treino(f, X, y_tr, aumento)
        mdl = novo_modelo(modelo, len(classes))
        mdl.fit(X_fit, y_fit)                    # scaler ajustado só no treino
        pred = mdl.predict(X[f.teste])
        if tarefa == "binario":
            m = metricas.metricas_binarias(y[f.teste], pred)
        else:
            m = {"acuracia_balanceada": metricas.acuracia_balanceada(y[f.teste], pred),
                 "recall_por_classe": metricas.recall_por_classe(y[f.teste], pred)}
        resultados.append({"nome": f.nome, "info": f.info, "n_treino": len(f.treino),
                           "n_treino_aumento": n_aumento, "n_teste": len(f.teste), **extra, **m})
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

_SIGLAS = {"deslocamento": "desl", "estiramento": "estir", "ruido": "ruido"}


def rotulo_aumento(info: dict) -> str:
    """
    Trecho do nome da pasta da rodada, ex. `aum4-desl-estir-ruido`. O que não
    é o padrão ganha sufixo, para duas rodadas diferentes não terem pastas com
    o mesmo nome: `-vel` para o estiramento no modo `velocidade`, e `-s<N>`
    para uma semente de aumento diferente de `config.SEMENTE`.
    """
    partes = [f"aum{info['copias']}"] + [_SIGLAS[t] for t in info["tecnicas"]]
    if "estiramento" in info["tecnicas"] and info["modo_estiramento"] == "velocidade":
        partes.append("vel")
    semente = info.get("semente", config.SEMENTE)
    if semente != config.SEMENTE:
        partes.append(f"s{semente}")
    return "-".join(partes)


def parametros_aumento(info: dict | None) -> dict:
    """Colunas do aumento em `parametros` (registry e metrics.json)."""
    if not info:
        return {"aumento": "nenhum"}
    return {
        "aumento": "+".join(info["tecnicas"]),
        "aumento_copias": info["copias"],
        "aumento_modo_estir": info["modo_estiramento"],
        "aumento_desloc_max_s": info["desloc_max_s"],
        "aumento_estir_taxas": "-".join(str(t) for t in info["estir_taxas"]),
        "aumento_snr_db": "-".join(str(t) for t in info["snr_db"]),
        "aumento_pv_nfft": info["pv_nfft"],
        "aumento_pv_hop": info["pv_hop"],
        # rodadas anteriores a este campo (exp016–exp022) usaram config.SEMENTE
        "aumento_semente": info.get("semente", config.SEMENTE),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocolo", choices=["A", "B"], required=True)
    ap.add_argument("--tarefa", choices=["binario", "multiclasse"], default="binario")
    ap.add_argument("--modelo", choices=["lda"], default="lda")
    ap.add_argument("--sem-c0", action="store_true", help="ablação do ganho")
    ap.add_argument("--permutar", action="store_true", help="controle de permutação por bloco")
    ap.add_argument("--aumento", action="store_true",
                    help="soma ao treino as variantes aceitas do mfcc_aumento.npz")
    ap.add_argument("--segundos-treino", type=float, default=None, metavar="N",
                    help="curva de aprendizado: N segundos de treino por classe em cada fold")
    ap.add_argument("--semente", type=int, default=None,
                    help=f"semente da permutação e da curva de aprendizado (padrão {config.SEMENTE})")
    ap.add_argument("--fs", type=int, default=config.FS_TRABALHO,
                    help="taxa da rodada (padrão: config.FS_TRABALHO)")
    ap.add_argument("--splits", type=Path, default=None,
                    help="padrão: a partição da taxa (config.arquivo_splits)")
    ap.add_argument("--features", type=Path, default=None,
                    help="padrão: mfcc_features.npz em config.dir_features(fs)")
    ap.add_argument("--pcm-dir", type=Path, default=None,
                    help="padrão: data/processed/pcm_decimated/<fs>")
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
    segmentos_treino = None
    if args.segundos_treino is not None:
        if args.aumento or args.permutar:
            ap.error("--segundos-treino não se combina com --aumento nem com --permutar: "
                     "a curva de aprendizado mede só o efeito da quantidade de treino")
        segmentos_treino = int(round(args.segundos_treino / config.SEGMENTO_S))
        if segmentos_treino < 2:
            # com 1 segmento por classe, a LDA não tem covariância dentro da classe
            # (o sklearn exige mais amostras que classes)
            ap.error(f"--segundos-treino precisa de pelo menos 2 segmentos por classe "
                     f"({2 * config.SEGMENTO_S:g} s) para a LDA")
    semente = config.SEMENTE if args.semente is None else args.semente
    args.splits = args.splits or Path(config.arquivo_splits(args.fs))
    args.features = args.features or Path(config.dir_features(args.fs)) / "mfcc_features.npz"
    args.pcm_dir = args.pcm_dir or Path(config.dir_pcm_decimado(args.fs))

    clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=args.fs)
    divergencias = pcm_io.verificar_integridade(clipes)
    pcm_io.relatar_integridade(divergencias)
    if divergencias:
        print("Abortado: os PCM não batem com o manifest versionado.")
        return 1

    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, clipes, args.fs)
                 + particao.verificar(particoes))
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
    manifesto_path = args.features.with_name("manifest_features.json")
    for caminho in (args.features, manifesto_path):
        if not caminho.exists():
            print(f"Abortado: {caminho} não encontrado; rode o 04_extract_features.py.")
            return 1
    manifesto_features = json.loads(manifesto_path.read_text(encoding="utf-8"))

    esperado = {
        "splits_hash": hash_splits,
        "fs_hz": args.fs,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS,
    }
    divergentes = [f"{k}: features={manifesto_features.get(k)}, atual={v}"
                   for k, v in esperado.items() if manifesto_features.get(k) != v]
    if divergentes:
        print("Abortado: features extraídas com outra configuração; rode o 04 de novo.\n  "
              + "\n  ".join(divergentes))
        return 1

    norm_clipe = bool(manifesto_features.get("norm_clipe", False))
    X = np.load(args.features)["X"]
    if len(X) != len(segmentos):
        print(f"Abortado: o .npz tem {len(X)} linhas e a partição tem {len(segmentos)} segmentos.")
        return 1
    info_aumento = manifesto_features.get("aumento")
    aumento = None
    if args.aumento:
        caminho_aumento = args.features.with_name("mfcc_aumento.npz")
        if not info_aumento or not caminho_aumento.exists():
            print("Abortado: --aumento pedido, mas as features foram extraídas sem "
                  "aumento; rode o 04 com --aumento N.")
            return 1
        dados = np.load(caminho_aumento)
        if len(dados["X"]) != info_aumento["n_variantes"] or dados["origem"].max() >= len(segmentos):
            print("Abortado: mfcc_aumento.npz não corresponde ao manifesto ou à partição.")
            return 1
        aumento = {"X": dados["X"], "origem": dados["origem"],
                   "lidos": variantes.ids_lidos(segmentos, dados["origem"],
                                                dados["inicio_lido"], dados["fim_lido"])}
        print(f"Aumento: {info_aumento['n_variantes']} variantes "
              f"({info_aumento['copias']} por segmento; {', '.join(info_aumento['tecnicas'])})")

    if args.sem_c0:
        # colunas da média e do desvio do c0
        X = np.delete(X, [0, config.MFCC_N_COEFS], axis=1)
        if aumento:
            aumento["X"] = np.delete(aumento["X"], [0, config.MFCC_N_COEFS], axis=1)
    y = rotulos(segmentos, args.tarefa)

    # ------------------------------------------------ folds
    resultados = rodar_folds(folds, X, y, args.tarefa, args.modelo,
                             segmentos=segmentos, permutar=args.permutar, aumento=aumento,
                             semente=semente, segmentos_treino=segmentos_treino)
    resumo = (metricas.resumo_protocolo_a(resultados) if args.protocolo == "A"
              else metricas.resumo_protocolo_b(resultados))
    imprimir(resumo, resultados)

    # ------------------------------------------------ gravação
    descricao = "_".join(p for p in (
        f"protocolo{args.protocolo}", args.tarefa, args.modelo,
        "normclipe" if norm_clipe else None,
        rotulo_aumento(info_aumento) if aumento else None,
        "semc0" if args.sem_c0 else None, "permutado" if args.permutar else None,
        f"treino{segmentos_treino * config.SEGMENTO_S:g}s" if segmentos_treino is not None else None,
        f"semente{semente}" if args.semente is not None else None,
        f"{args.fs}hz" if args.fs != config.FS_TRABALHO else None,
    ) if p)
    exp_id = "teste" if args.sem_registro else f"exp{experimentos.next_exp_number(args.registry):03d}"
    destino = args.out_dir / f"{exp_id}_{descricao}"
    destino.mkdir(parents=True, exist_ok=True)

    parametros = {
        "protocolo": args.protocolo,
        "tarefa": args.tarefa,
        "modelo": args.modelo,
        "features": FEATURES_ID + ("_normclipe" if norm_clipe else ""),
        "norm_clipe": norm_clipe,
        "sem_c0": args.sem_c0,
        "permutado": args.permutar,
        "splits": hash_splits,
        "fs_hz": manifesto_features["fs_hz"],
        "segmento_s": config.SEGMENTO_S,
        "segmentos_por_bloco": config.SEGMENTOS_POR_BLOCO,
        "segmentos_descarte": config.SEGMENTOS_DESCARTE,
        "mfcc_janela_ms": manifesto_features["mfcc_janela_ms"],
        "mfcc_hop_ms": manifesto_features["mfcc_hop_ms"],
        "mfcc_n_mels": manifesto_features["mfcc_n_mels"],
        "mfcc_n_coefs": manifesto_features["mfcc_n_coefs"],
        "features_commit": manifesto_features.get("git_commit"),
        "semente": semente if (args.permutar or aumento or segmentos_treino) else None,
        "segundos_treino": (segmentos_treino * config.SEGMENTO_S
                            if segmentos_treino is not None else None),
        # rodadas da curva anteriores a este campo (exp051–exp058) sorteavam
        # na classe inteira, sem repartir entre as gravações
        "amostragem_treino": "por_gravacao" if segmentos_treino is not None else None,
        **parametros_aumento(info_aumento if aumento else None),
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
        "dataset": f"jung2023_acustico_0Nm_{args.fs}Hz",
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
