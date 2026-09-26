"""
metricas.py — métricas do protocolo de validação.

Por que acurácia balanceada, e não acurácia simples: no Protocolo B cada teste
tem 59 segmentos de falha e só 9 ou 10 normais. Um modelo que chamasse tudo de
falha teria ~86 % de acurácia e passaria na meta sem detectar nada. A acurácia
balanceada é a média do acerto por classe, e nesse caso daria 50 %.

No binário ela é a média de:
    sensibilidade  = fração dos segmentos de falha detectados como falha
    especificidade = fração dos segmentos normais classificados como normal
                     (1 − taxa de alarme falso)

A meta do projeto (Registro de Decisões, 24/09) é a acurácia balanceada média
do Protocolo B ≥ 85 %, sempre reportada junto com sensibilidade, especificidade
e o pior caso — a média pode esconder uma falha que o modelo não detecta.

Este módulo não é executável: é importado.
"""

from __future__ import annotations

import numpy as np


def recall_por_classe(y_true, y_pred) -> dict[str, float]:
    """Acerto por classe presente em `y_true`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {str(c): float(np.mean(y_pred[y_true == c] == c)) for c in np.unique(y_true)}


def acuracia_balanceada(y_true, y_pred) -> float:
    """Média do acerto por classe. Vale para binário e multiclasse."""
    return float(np.mean(list(recall_por_classe(y_true, y_pred).values())))


def metricas_binarias(y_true, y_pred, positivo: str = "falha",
                      negativo: str = "normal") -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == positivo
    neg = y_true == negativo
    if not pos.any() or not neg.any():
        raise ValueError("o teste binário precisa ter as duas classes")
    sens = float(np.mean(y_pred[pos] == positivo))
    espec = float(np.mean(y_pred[neg] == negativo))
    return {
        "sensibilidade": sens,
        "especificidade": espec,
        "acuracia_balanceada": (sens + espec) / 2,
        "n_teste_falha": int(pos.sum()),
        "n_teste_normal": int(neg.sum()),
    }


def resumo_protocolo_a(resultados: list[dict]) -> dict:
    """Média ± desvio da acurácia balanceada entre os folds. Limite otimista."""
    acc = np.array([r["acuracia_balanceada"] for r in resultados])
    return {
        "protocolo": "A",
        "leitura": "limite superior otimista — não conta para a meta",
        "n_folds": len(resultados),
        "acuracia_balanceada_media": float(acc.mean()),
        "acuracia_balanceada_desvio": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "acuracia_balanceada_min": float(acc.min()),
    }


def resumo_protocolo_b(resultados: list[dict], meta: float = 0.85) -> dict:
    """
    Agrega os folds do B por falha deixada de fora (média sobre os blocos
    normais), depois entre falhas. Reporta a média geral e os dois piores casos:
    a pior falha (média dos seus blocos) e o pior fold individual.
    """
    por_falha: dict[str, list[dict]] = {}
    for r in resultados:
        por_falha.setdefault(r["info"]["falha_de_fora"], []).append(r)

    tabela = {}
    for falha, rs in por_falha.items():
        tabela[falha] = {
            k: float(np.mean([r[k] for r in rs]))
            for k in ("sensibilidade", "especificidade", "acuracia_balanceada")
        }
        tabela[falha]["n_folds"] = len(rs)

    def media(k):
        return float(np.mean([t[k] for t in tabela.values()]))

    pior_falha = min(tabela, key=lambda f: tabela[f]["acuracia_balanceada"])
    pior_fold = min(resultados, key=lambda r: r["acuracia_balanceada"])
    acc_media = media("acuracia_balanceada")
    return {
        "protocolo": "B",
        "leitura": "resultado principal — meta: acurácia balanceada média ≥ "
                   f"{meta:.0%}",
        "n_folds": len(resultados),
        "acuracia_balanceada_media": acc_media,
        "sensibilidade_media": media("sensibilidade"),
        "especificidade_media": media("especificidade"),
        "pior_falha": pior_falha,
        "pior_falha_acuracia_balanceada": tabela[pior_falha]["acuracia_balanceada"],
        "pior_fold": pior_fold["nome"],
        "pior_fold_acuracia_balanceada": float(pior_fold["acuracia_balanceada"]),
        "meta_atingida": bool(acc_media >= meta),
        "por_falha": tabela,
    }