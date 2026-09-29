#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_classifiers.py — estudo da escolha do classificador (Classificador 3/6, parte 1)
=========================================================================================

Compara LDA (referência), MLP raso e duas CNNs pequenas usando SÓ validação
interna ao treino de cada fold do Protocolo B. O teste do B (a gravação de falha
deixada de fora) nunca é lido para treinar, escolher ou reportar: este script não
calcula nenhuma métrica sobre ele.

Validação interna ("B dentro do B")
-----------------------------------
Para cada fold externo do B, o treino tem 3 gravações de falha e 5 blocos da
normal. A validação interna repete a lógica do B dentro desse treino: cada falha
de treino é deixada de fora uma vez, combinada com cada bloco normal de treino,
com a mesma faixa de descarte (1 segmento) do protocolo. São 3 × 5 = 15 treinos
internos por fold externo. O número reportado é a acurácia balanceada média e o
pior caso desses treinos internos, agregados sobre os folds externos.

Modelos
-------
lda     referência — média e desvio dos 13 MFCC (26 features), priors uniformes
mlp     26 → 16 → n_classes, ReLU, pesos de classe na perda
cnn1d   MFCC 13 × 98 quadros; convolução no tempo (13 coeficientes = canais)
cnn2d   log-Mel 20 × 98 quadros; convolução 2D tempo × frequência

Custo no STM32F411CEU6 (parâmetros, MACs, pico de ativação) é calculado a partir
da arquitetura e impresso junto; não depende dos dados.

Estudo, não etapa do pipeline: responde uma vez qual modelo usar, e fica
versionado para a decisão continuar auditável (CONVENTIONS, seção 2). Como o
resultado vai para o relatório, escreve a própria linha no registry, com etapa
`escolha_classificador`. Não é protocolo de validação: o número que conta para a
meta continua sendo o do `run_protocol.py` com o modelo escolhido.

Features: recalculadas aqui com `dsp.mfcc`/`dsp.log_mel` sobre os segmentos da
partição, porque as CNNs precisam da matriz por quadro, que o `04` não guarda.
O resumo de 26 valores é o mesmo do `04` (média e desvio dos 13 MFCC); a LDA
daqui reproduz o B oficial (0,875, exp015).

Uso
---
    python scripts/exploration/compare_classifiers.py --responsavel Clara      # dados reais, registra
    python scripts/exploration/compare_classifiers.py --sintetico              # auto-teste, sem registro
    python scripts/exploration/compare_classifiers.py --blocos-externos 1 --sem-registro   # rápido: 4 folds externos
"""

from __future__ import annotations

import argparse
import json
from datetime import date
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
import torch
from torch import nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import dsp
import experimentos
import pcm_io
from validation import metricas, particao

import os
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def clipes_sinteticos() -> list[pcm_io.Clip]:
    """5 gravações de 60 s com resposta conhecida: harmônicos do eixo + séries de falha."""
    rng = np.random.default_rng(0)
    fs, n = config.FS_TRABALHO, 767982
    t = np.arange(n) / fs
    base = sum(0.02 / k * np.sin(2 * np.pi * 50.2 * k * t) for k in range(1, 8))
    serie = {"bpfi_0.3mm": (268.3, 0.010), "bpfi_1.0mm": (268.3, 0.030),
             "bpfo_0.3mm": (182.7, 0.004), "bpfo_1.0mm": (183.4, 0.030)}
    clipes = []
    for rot in config.CLASSES:
        x = base + 0.01 * rng.standard_normal(n)
        if rot in serie:
            f0, a = serie[rot]
            x = x + sum(a / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 12))
        clipes.append(pcm_io.Clip(rotulo=rot, binario=config.BINARIO[rot], fs=fs, x_float=x))
    return clipes


def extrair(clipes, segmentos):
    """Por segmento: resumo (26), matriz MFCC (98 × 13) e log-Mel (98 × 20)."""
    por = {c.rotulo: c for c in clipes}
    resumo, mf, lm = [], [], []
    for s in segmentos:
        x = por[s.rotulo].x[s.inicio:s.fim]
        lmel = dsp.log_mel(x, config.FS_TRABALHO)
        m = dsp.mfcc(x, config.FS_TRABALHO)
        resumo.append(np.concatenate([m.mean(0), m.std(0)]))
        mf.append(m.T)          # (13, 98): coeficientes como canais
        lm.append(lmel.T)       # (20, 98)
    return (np.asarray(resumo, np.float32), np.asarray(mf, np.float32),
            np.asarray(lm, np.float32))


# --------------------------------------------------------------------------- #
# Folds internos
# --------------------------------------------------------------------------- #
def folds_internos(segmentos, treino: list[int], descarte: int) -> list[tuple[list[int], list[int], str]]:
    """B dentro do treino externo: cada falha de treino × cada bloco normal de treino."""
    sub = [segmentos[i] for i in treino]
    falhas = sorted({s.rotulo for s in sub if s.binario == "falha"})
    blocos = sorted({s.bloco for s in sub if s.binario == "normal"})
    saida = []
    for f in falhas:
        for b in blocos:
            teste = [s.id for s in sub if s.rotulo == f or (s.binario == "normal" and s.bloco == b)]
            por_grav: dict[str, list[int]] = {}
            for i in teste:
                por_grav.setdefault(segmentos[i].rotulo, []).append(segmentos[i].indice)
            tr = []
            for s in sub:
                if s.id in teste:
                    continue
                idx = por_grav.get(s.rotulo)
                if idx and min(abs(s.indice - k) for k in idx) <= descarte:
                    continue
                tr.append(s.id)
            saida.append((tr, teste, f))
    return saida


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int, oculto: int = 16):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(n_in, oculto), nn.ReLU(), nn.Linear(oculto, n_classes))

    def forward(self, x):
        return self.rede(x)


class CNN1D(nn.Module):
    """Entrada (C=13, T=98). Conv no tempo, pooling global: invariante à posição."""
    def __init__(self, n_in: int, n_classes: int, f: int = 16, k: int = 5):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv1d(n_in, f, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(f, f, k, padding=k // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(f, n_classes))

    def forward(self, x):
        return self.rede(x)


class CNN2D(nn.Module):
    """Entrada (1, 20 bandas, 98 quadros)."""
    def __init__(self, n_classes: int, f1: int = 8, f2: int = 16):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv2d(1, f1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(f1, f2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(f2, n_classes))

    def forward(self, x):
        return self.rede(x.unsqueeze(1))


def construir(nome: str, n_classes: int) -> nn.Module:
    if nome == "mlp":
        return MLP(2 * config.MFCC_N_COEFS, n_classes)
    if nome == "cnn1d":
        return CNN1D(config.MFCC_N_COEFS, n_classes)
    if nome == "cnn2d":
        return CNN2D(n_classes)
    raise ValueError(nome)


def padronizar(X_tr, X_te, eixo_canal: bool):
    """Média/desvio ajustados SÓ no treino (por feature, ou por canal nas CNNs)."""
    eixos = (0, 2) if eixo_canal else 0
    mu = X_tr.mean(axis=eixos, keepdims=True)
    sd = X_tr.std(axis=eixos, keepdims=True) + 1e-6
    return (X_tr - mu) / sd, (X_te - mu) / sd


def treinar_nn(nome, X_tr, y_tr, X_te, n_classes, semente, epocas, lr=3e-3, wd=1e-3):
    torch.manual_seed(semente)
    X_tr, X_te = padronizar(X_tr, X_te, eixo_canal=(X_tr.ndim == 3))
    mdl = construir(nome, n_classes)
    cont = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    peso = torch.tensor(cont.sum() / (n_classes * np.maximum(cont, 1)))   # pesos de classe
    perda = nn.CrossEntropyLoss(weight=peso)
    opt = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()
    g = torch.Generator().manual_seed(semente)
    mdl.train()
    for _ in range(epocas):
        for idx in torch.randperm(len(Xt), generator=g).split(32):
            opt.zero_grad()
            perda(mdl(Xt[idx]), yt[idx]).backward()
            opt.step()
    mdl.eval()
    with torch.no_grad():
        return mdl(torch.from_numpy(X_te)).argmax(1).numpy()


def treinar_lda(X_tr, y_tr, X_te, n_classes):
    mdl = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(priors=np.full(n_classes, 1 / n_classes)))
    return mdl.fit(X_tr, y_tr).predict(X_te)


# --------------------------------------------------------------------------- #
# Custo embarcado (independe dos dados)
# --------------------------------------------------------------------------- #
def custo(nome: str, n_classes: int = 2) -> dict:
    """Parâmetros, MACs e maior par entrada+saída de camada, em nº de valores (float32 = ×4 bytes).

    ReLU e Flatten contam como in-place. Sem fusão conv+pool: é o pico pessimista.
    """
    T, C, M = 98, config.MFCC_N_COEFS, config.MFCC_N_MELS
    if nome == "lda":
        return {"parametros": 2 * C * n_classes + n_classes + 4 * C, "macs": 2 * C * n_classes,
                "pico_ativacao": 2 * C + n_classes, "entrada": 2 * C, "acumula_quadros": False}
    mdl = construir(nome, n_classes)
    params = sum(p.numel() for p in mdl.parameters())
    x = torch.zeros((1, 2 * C) if nome == "mlp" else (1, C, T) if nome == "cnn1d" else (1, M, T))
    macs, pico = 0, 0
    h = x.unsqueeze(1) if nome == "cnn2d" else x
    for camada in mdl.rede:
        out = camada(h)
        if isinstance(camada, nn.Linear):
            macs += camada.in_features * camada.out_features
        elif isinstance(camada, nn.Conv1d):
            macs += out.numel() * camada.in_channels * camada.kernel_size[0]
        elif isinstance(camada, nn.Conv2d):
            macs += out.numel() * camada.in_channels * camada.kernel_size[0] * camada.kernel_size[1]
        if not isinstance(camada, (nn.ReLU, nn.Flatten)):   # ReLU/Flatten: in-place no X-CUBE-AI
            pico = max(pico, h.numel() + out.numel())
        h = out
    return {"parametros": int(params), "macs": int(macs), "pico_ativacao": int(pico),
            "entrada": int(x.numel()), "acumula_quadros": nome != "mlp"}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--pcm-dir", type=Path,
                    default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--modelos", default="lda,mlp,cnn1d,cnn2d")
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--blocos-externos", type=int, default=None,
                    help="usa só os N primeiros blocos normais de cada falha nos folds externos "
                         "(ex.: 1 → 4 folds externos em vez de 24; ~6× mais rápido)")
    ap.add_argument("--saida", type=Path, default=Path("reports/classifier/compare_classifiers.json"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        clipes = clipes_sinteticos()
        particoes = particao.gerar_particoes(clipes)
        args.saida = args.saida.with_name("sintetico") / args.saida.name
    else:
        clipes = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
        div = pcm_io.verificar_integridade(clipes)
        if div:
            pcm_io.relatar_integridade(div)
            return 1
        particoes = particao.carregar(args.splits)
        prob = particao.conferir_compatibilidade(particoes, clipes)
        if prob:
            print("splits.json incompatível:", *prob, sep="\n  ")
            return 1
    erros = particao.verificar(particoes)
    if erros:
        print("partição inválida:", *erros, sep="\n  ")
        return 1

    segmentos = particao.segmentos_de(particoes)
    folds_b = particao.folds_de(particoes, "B")
    if args.blocos_externos is not None:
        folds_b = [f for f in folds_b if f.info["bloco_normal"] < args.blocos_externos]
    descarte = int(particoes["parametros"]["segmentos_descarte"])
    print(f"Extraindo MFCC e log-Mel de {len(segmentos)} segmentos...")
    resumo, mf, lm = extrair(clipes, segmentos)
    classes = sorted({s.binario for s in segmentos})
    y = np.array([classes.index(s.binario) for s in segmentos])
    n_classes = len(classes)
    entrada = {"lda": resumo, "mlp": resumo, "cnn1d": mf, "cnn2d": lm}
    modelos = args.modelos.split(",")

    res = {m: [] for m in modelos}   # uma linha por (fold externo, fold interno, semente)
    t0 = time.time()
    for k, fe in enumerate(folds_b):
        # o teste externo (fe.teste) não é usado em nenhum ponto daqui para baixo
        internos = folds_internos(segmentos, fe.treino, descarte)
        for tr, te, falha_int in internos:
            for m in modelos:
                X = entrada[m]
                sementes = [0] if m == "lda" else range(args.sementes)
                for sem in sementes:
                    if m == "lda":
                        pred = treinar_lda(X[tr], y[tr], X[te], n_classes)
                    else:
                        pred = treinar_nn(m, X[tr], y[tr], X[te], n_classes,
                                          config.SEMENTE + sem, args.epocas)
                    yt = np.array(classes)[y[te]]
                    mb = metricas.metricas_binarias(yt, np.array(classes)[pred])
                    res[m].append({"externo": fe.nome, "falha_interna": falha_int, "semente": sem,
                                   **{c: mb[c] for c in ("sensibilidade", "especificidade",
                                                         "acuracia_balanceada")}})
        print(f"  fold externo {k + 1:2d}/{len(folds_b)} ({fe.nome}) "
              f"— {time.time() - t0:5.0f} s")

    resumo_final = {"n_folds_externos": len(folds_b), "blocos_externos": args.blocos_externos, "epocas": args.epocas,
                    "sementes": args.sementes, "sintetico": args.sintetico, "modelos": {}}
    print(f"\n{'modelo':<7} {'acc.bal. interna':>17} {'pior falha int.':>16} "
          f"{'sensib.':>8} {'especif.':>9} {'params':>7} {'MACs':>8} {'pico ativ.':>11}")
    for m in modelos:
        linhas = res[m]
        ab = np.array([r["acuracia_balanceada"] for r in linhas])
        por_falha = {}
        for r in linhas:
            por_falha.setdefault(r["falha_interna"], []).append(r["acuracia_balanceada"])
        por_falha = {f: float(np.mean(v)) for f, v in sorted(por_falha.items())}
        # dispersão entre sementes: média por semente, desvio entre elas
        por_sem = {}
        for r in linhas:
            por_sem.setdefault(r["semente"], []).append(r["acuracia_balanceada"])
        desv_sem = float(np.std([np.mean(v) for v in por_sem.values()]))
        c = custo(m, n_classes)
        resumo_final["modelos"][m] = {
            "acuracia_balanceada_media": float(ab.mean()),
            "acuracia_balanceada_desvio_folds": float(ab.std()),
            "desvio_entre_sementes": desv_sem,
            "sensibilidade_media": float(np.mean([r["sensibilidade"] for r in linhas])),
            "especificidade_media": float(np.mean([r["especificidade"] for r in linhas])),
            "por_falha_interna": por_falha,
            "pior_falha_interna": min(por_falha, key=por_falha.get),
            "custo": c,
        }
        s = resumo_final["modelos"][m]
        print(f"{m:<7} {s['acuracia_balanceada_media']:>11.3f} ± {desv_sem:.3f} "
              f"{min(por_falha.values()):>16.3f} {s['sensibilidade_media']:>8.3f} "
              f"{s['especificidade_media']:>9.3f} {c['parametros']:>7d} {c['macs']:>8d} "
              f"{c['pico_ativacao']:>11d}")
        print("        por falha interna: " +
              ", ".join(f"{f} {v:.3f}" for f, v in por_falha.items()))

    # Escolha aninhada: para o fold externo da falha F, só valem os treinos internos
    # dos folds externos de F (onde F está inteira fora). Se o vencedor for o mesmo
    # para as quatro falhas, uma escolha única equivale à escolha aninhada.
    externas = sorted({r["externo"].rsplit("_bloco", 1)[0] for r in res[modelos[0]]})
    vencedor = {}
    print("\nescolha aninhada (acc. bal. interna média, só folds externos de cada falha):")
    for e in externas:
        medias = {m: float(np.mean([r["acuracia_balanceada"] for r in res[m]
                                    if r["externo"].rsplit("_bloco", 1)[0] == e]))
                  for m in modelos}
        vencedor[e] = max(medias, key=medias.get)
        print(f"  {e:<18} " + "  ".join(f"{m} {v:.3f}" for m, v in medias.items())
              + f"  → {vencedor[e]}")
    resumo_final["escolha_aninhada"] = vencedor
    resumo_final["escolha_unica_equivale"] = len(set(vencedor.values())) == 1

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps({"resumo": resumo_final, "linhas": res},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGravado em {args.saida}")

    if args.sem_registro:
        print("(sem registro)")
        return 0
    mm = resumo_final["modelos"]
    linha = {
        "id": f"exp{experimentos.next_exp_number(args.registry):03d}",
        "data": date.today().isoformat(),
        "etapa": "escolha_classificador",
        "script": "exploration/compare_classifiers.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "validacao": "interna_B", "tarefa": "binario" if n_classes == 2 else "multiclasse",
            "modelos": "/".join(modelos), "folds_externos": len(folds_b),
            "blocos_externos": args.blocos_externos if args.blocos_externos is not None else "todos",
            "sementes": args.sementes, "epocas": args.epocas,
            "splits": particao.hash_arquivo(args.splits), "fs_hz": config.FS_TRABALHO,
            "segmento_s": config.SEGMENTO_S, "mfcc_janela_ms": config.MFCC_WINDOW_MS,
            "mfcc_hop_ms": config.MFCC_HOP_MS, "mfcc_n_mels": config.MFCC_N_MELS,
            "mfcc_n_coefs": config.MFCC_N_COEFS, "aumento": "nenhum"}),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            **{f"acc_bal_int_{m}": f"{mm[m]['acuracia_balanceada_media']:.4f}" for m in modelos},
            **{f"desvio_sementes_{m}": f"{mm[m]['desvio_entre_sementes']:.4f}" for m in modelos},
            **{f"vencedor_{e.removeprefix('B_')}": v for e, v in vencedor.items()},
            "escolha_unica": "sim" if resumo_final["escolha_unica_equivale"] else "nao"}),
        "responsavel": args.responsavel,
        "notas": args.notas or "validação interna ao treino do B; não é desempenho",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())