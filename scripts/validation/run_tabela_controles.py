#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_controles.py — tabela dos controles obrigatórios do classificador
============================================================================

Monta, a partir dos `metrics.json` de `reports/validation/`, a tabela dos três
controles que dizem se o classificador aprende a falha ou a gravação:

  ablação do ganho       A e B sem o c0 do MFCC, contra a referência com c0
  permutação por bloco   distribuição do B com rótulos embaralhados, uma rodada
                         por semente; o critério é não ficar acima de 0,5
  curva de aprendizado   A e B com N segundos de treino por classe

As rodadas são achadas pelos parâmetros gravados, não por id: MFCC oficial, a
partição do splits.json atual, LDA, sem aumento e sem normalização por clipe.
Não treina nada nem escreve no registry.

Uso
---
    python scripts/validation/run_tabela_controles.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

from validation import particao

FEATURES_OFICIAL = "mfcc_dsp_media_desvio"


def carregar_rodadas(pasta: Path, splits: str) -> list[dict]:
    """Rodadas registradas (exp*) da referência: MFCC oficial, LDA, sem aumento."""
    rodadas = []
    for caminho in sorted(pasta.glob("exp*/metrics.json")):
        d = json.loads(caminho.read_text(encoding="utf-8"))
        p = d["parametros"]
        if (p.get("features") == FEATURES_OFICIAL and p.get("splits") == splits
                and p.get("modelo") == "lda" and p.get("tarefa") == "binario"
                and p.get("aumento") in (None, "nenhum") and not p.get("norm_clipe")):
            rodadas.append(d)
    return rodadas


def acc(d: dict) -> float:
    return d["resumo"]["acuracia_balanceada_media"]


def e_referencia(p: dict) -> bool:
    return not (p.get("sem_c0") or p.get("permutado") or p.get("segundos_treino") is not None)


def ultima(rodadas: list[dict]) -> dict | None:
    return max(rodadas, key=lambda d: d["id"]) if rodadas else None


def celula_b(d: dict) -> str:
    r = d["resumo"]
    return (f"{r['acuracia_balanceada_media']:.3f} (sens. {r['sensibilidade_media']:.3f}, "
            f"espec. {r['especificidade_media']:.3f}; bpfo_0.3mm "
            f"{r['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.3f})")


def celula(d: dict | None) -> str:
    if d is None:
        return "—"
    if d["resumo"]["protocolo"] == "A":
        r = d["resumo"]
        return f"{r['acuracia_balanceada_media']:.3f} ± {r['acuracia_balanceada_desvio']:.3f}"
    return celula_b(d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--saida", type=Path, default=Path("reports/validation/tabela_controles.md"))
    args = ap.parse_args()

    splits = particao.hash_arquivo(args.splits)
    rodadas = carregar_rodadas(args.pasta, splits)
    por = lambda prot, cond: [d for d in rodadas
                              if d["parametros"]["protocolo"] == prot and cond(d["parametros"])]

    ref = {prot: ultima(por(prot, e_referencia)) for prot in ("A", "B")}
    semc0 = {prot: ultima(por(prot, lambda p: p.get("sem_c0") and not p.get("permutado")
                              and p.get("segundos_treino") is None)) for prot in ("A", "B")}
    perm = sorted(por("B", lambda p: p.get("permutado") and not p.get("sem_c0")),
                  key=lambda d: d["id"])
    curva = {prot: sorted(por(prot, lambda p: p.get("segundos_treino") is not None
                              and not p.get("sem_c0")),
                          key=lambda d: d["parametros"]["segundos_treino"])
             for prot in ("A", "B")}

    ids = lambda ds: ", ".join(d["id"] for d in ds if d) or "—"
    texto = [
        "# Controles obrigatórios do classificador",
        "",
        "Gerado por `scripts/validation/run_tabela_controles.py` a partir dos `metrics.json`. "
        f"MFCC oficial, partição `{splits}`, LDA, binário, sem aumento de dados. "
        "B: acurácia balanceada média (sensibilidade e especificidade médias; acurácia "
        "balanceada da `bpfo_0.3mm`).",
        "",
        "## Ablação do ganho (sem o c0)",
        "",
        "| protocolo | com c0 | sem c0 |",
        "|---|---|---|",
        *(f"| {prot} | {celula(ref[prot])} ({ref[prot]['id'] if ref[prot] else '—'}) "
          f"| {celula(semc0[prot])} ({semc0[prot]['id'] if semc0[prot] else '—'}) |"
          for prot in ("A", "B")),
        "",
        "## Permutação por bloco (Protocolo B)",
        "",
    ]
    if perm:
        a = np.array([acc(d) for d in perm])
        texto += [
            f"{len(perm)} sementes ({ids(perm)}).",
            "",
            "| média ± desvio | mediana | mín. | máx. | sementes acima de 0,5 |",
            "|---|---|---|---|---|",
            f"| {a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0:.3f} | {np.median(a):.3f} "
            f"| {a.min():.3f} | {a.max():.3f} | {int((a > 0.5).sum())} de {len(a)} |",
            "",
            "| semente | acc. bal. média | pior fold |",
            "|---|---|---|",
            *(f"| {d['parametros']['semente']} | {acc(d):.3f} "
              f"| {d['resumo']['pior_fold_acuracia_balanceada']:.3f} |" for d in perm),
        ]
    else:
        texto.append("Nenhuma rodada encontrada.")
    texto += [
        "",
        "## Curva de aprendizado",
        "",
        "| segundos de treino por classe | A | B |",
        "|---|---|---|",
    ]
    segundos = sorted({d["parametros"]["segundos_treino"] for prot in curva for d in curva[prot]})
    for s in segundos:
        cel = {prot: next((d for d in curva[prot] if d["parametros"]["segundos_treino"] == s), None)
               for prot in ("A", "B")}
        texto.append(f"| {s:g} | {celula(cel['A'])} | {celula(cel['B'])} |")
    texto.append(f"| todo o treino | {celula(ref['A'])} | {celula(ref['B'])} |")
    texto += ["", f"Rodadas da curva: A {ids(curva['A'])}; B {ids(curva['B'])}.", ""]

    args.saida.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))
    print(f"tabela: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
