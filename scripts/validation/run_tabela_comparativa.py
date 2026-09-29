#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_comparativa.py — tabela da rodada provisória × rodada oficial
========================================================================

Monta, a partir dos `metrics.json` já gravados em `reports/validation/`, a
tabela que compara os Protocolos A e B com o MFCC provisório (rodada de 24/09)
e com o MFCC oficial do `dsp.py`. Não treina nada nem escreve no registry: só lê
rodadas que já têm linha própria, e os números saem delas, não de cópia manual.

A: acurácia balanceada média ± desvio e mínimo entre os 6 folds.
B: sensibilidade, especificidade e acurácia balanceada por falha deixada de
   fora, média e pior fold.

Uso
---
    python scripts/validation/run_tabela_comparativa.py
    python scripts/validation/run_tabela_comparativa.py \
        --a-binario exp007 exp027 --a-multiclasse exp008 exp028 --b exp009 exp015
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def carregar(pasta: Path, exp_id: str) -> dict:
    achados = sorted(pasta.glob(f"{exp_id}_*/metrics.json"))
    if len(achados) != 1:
        raise SystemExit(f"Abortado: esperava uma pasta {exp_id}_* em {pasta}, achei {len(achados)}.")
    return json.loads(achados[0].read_text(encoding="utf-8"))


IGUAIS = ("splits", "modelo", "mfcc_janela_ms", "mfcc_hop_ms", "mfcc_n_mels", "mfcc_n_coefs")


def conferir_comparaveis(ids: dict, rodadas: dict) -> None:
    """
    As rodadas só entram na tabela se forem a mesma rodada de referência: mesma
    partição, modelo e MFCC; sem aumento, sem normalização, sem ablação nem
    controle; e cada par no protocolo e na tarefa da sua linha.
    """
    esperado = {"A binário": ("A", "binario"), "A multiclasse": ("A", "multiclasse"),
                "B": ("B", "binario")}
    erros = []
    todas = [(e, r["parametros"]) for k in ids for e, r in zip(ids[k], rodadas[k])]
    for campo in IGUAIS:
        valores = {str(p.get(campo)) for _, p in todas}
        if len(valores) != 1:
            erros.append(f"{campo} difere entre as rodadas: {', '.join(sorted(valores))}")
    for linha, (protocolo, tarefa) in esperado.items():
        for e, r in zip(ids[linha], rodadas[linha]):
            p = r["parametros"]
            if (p.get("protocolo"), p.get("tarefa")) != (protocolo, tarefa):
                erros.append(f"{e}: é {p.get('protocolo')}/{p.get('tarefa')}, "
                             f"a linha '{linha}' pede {protocolo}/{tarefa}")
            if p.get("aumento") not in (None, "nenhum"):
                erros.append(f"{e}: tem aumento ({p['aumento']})")
            if p.get("norm_clipe") or p.get("sem_c0") or p.get("permutado"):
                erros.append(f"{e}: é ablação ou controle (norm_clipe/sem_c0/permutado)")
    if erros:
        raise SystemExit("Abortado: rodadas não comparáveis:\n  " + "\n  ".join(erros))


def linha_a(rotulo: str, prov: dict, ofic: dict) -> str:
    def cel(d):
        r = d["resumo"]
        return (f"{r['acuracia_balanceada_media']:.3f} ± {r['acuracia_balanceada_desvio']:.3f} "
                f"(mín. {r['acuracia_balanceada_min']:.3f})")
    return f"| {rotulo} | {cel(prov)} | {cel(ofic)} |"


def tabela_b(prov: dict, ofic: dict) -> list[str]:
    rp, ro = prov["resumo"], ofic["resumo"]
    linhas = ["| falha deixada de fora | sensib. prov. | especif. prov. | acc. bal. prov. "
              "| sensib. ofic. | especif. ofic. | acc. bal. ofic. |",
              "|---|---:|---:|---:|---:|---:|---:|"]

    def tripla(t):
        return f"{t['sensibilidade']:.3f} | {t['especificidade']:.3f} | {t['acuracia_balanceada']:.3f}"

    for falha in rp["por_falha"]:
        linhas.append(f"| {falha} | {tripla(rp['por_falha'][falha])} | {tripla(ro['por_falha'][falha])} |")
    media = lambda r: (f"{r['sensibilidade_media']:.3f} | {r['especificidade_media']:.3f} | "
                       f"{r['acuracia_balanceada_media']:.3f}")
    linhas.append(f"| **média** | {media(rp)} | {media(ro)} |")
    linhas.append(f"| pior fold | | | {rp['pior_fold_acuracia_balanceada']:.3f} ({rp['pior_fold']}) "
                  f"| | | {ro['pior_fold_acuracia_balanceada']:.3f} ({ro['pior_fold']}) |")
    return linhas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-binario", nargs=2, default=["exp007", "exp027"], metavar=("PROV", "OFIC"))
    ap.add_argument("--a-multiclasse", nargs=2, default=["exp008", "exp028"], metavar=("PROV", "OFIC"))
    ap.add_argument("--b", nargs=2, default=["exp009", "exp015"], metavar=("PROV", "OFIC"))
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--saida", type=Path,
                    default=Path("reports/validation/tabela_provisorio_vs_oficial.md"))
    args = ap.parse_args()

    ids = {"A binário": args.a_binario, "A multiclasse": args.a_multiclasse, "B": args.b}
    rodadas = {k: [carregar(args.pasta, e) for e in v] for k, v in ids.items()}
    conferir_comparaveis(ids, rodadas)

    feats = {k: [r["parametros"]["features"] for r in par] for k, par in rodadas.items()}
    texto = [
        "# Protocolos A e B: MFCC provisório × MFCC oficial",
        "",
        "Gerado por `scripts/validation/run_tabela_comparativa.py` a partir dos `metrics.json`. "
        f"Partição `{rodadas['B'][0]['parametros']['splits']}`, LDA, sem aumento de dados.",
        "",
        "| rodada | provisório | oficial |",
        "|---|---|---|",
        *(f"| {k} | {v[0]} (`{feats[k][0]}`) | {v[1]} (`{feats[k][1]}`) |" for k, v in ids.items()),
        "",
        "## Protocolo A — limite otimista, não conta para a meta",
        "",
        "| tarefa | provisório | oficial |",
        "|---|---|---|",
        linha_a("binário", *rodadas["A binário"]),
        linha_a("multiclasse", *rodadas["A multiclasse"]),
        "",
        "## Protocolo B — resultado principal (meta: acc. bal. média ≥ 0,85)",
        "",
        *tabela_b(*rodadas["B"]),
        "",
    ]
    args.saida.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))
    print(f"tabela: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
