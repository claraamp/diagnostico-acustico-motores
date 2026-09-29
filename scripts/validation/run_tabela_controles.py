#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_controles.py — tabela dos controles obrigatórios do classificador
============================================================================

Monta, a partir dos `metrics.json` de `reports/validation/`, a tabela dos três
controles que dizem se o classificador aprende a falha ou a gravação:

  ablação do ganho       A e B sem o c0 do MFCC e com normalização RMS por
                         segmento, contra a referência
  permutação por bloco   distribuição do B com rótulos embaralhados, uma rodada
                         por semente. Critério: p empírico da referência contra
                         essa distribuição, (1 + nº de sementes ≥ referência) /
                         (n + 1), e a média comparada com 0,5 (teste t)
  curva de aprendizado   A e B com N segundos de treino por classe, repartidos
                         entre as gravações; média ± desvio entre as sementes

As rodadas são achadas pelos parâmetros gravados, não por id: MFCC oficial, a
partição do splits.json atual, LDA, binário e sem aumento. Os pontos da curva
anteriores à amostragem por gravação (sem `amostragem_treino`) ficam de fora.
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
from scipy import stats

from validation import particao

FEATURES_OFICIAL = ("mfcc_dsp_media_desvio", "mfcc_dsp_media_desvio_normclipe")


def carregar_rodadas(pasta: Path, splits: str) -> list[dict]:
    """Rodadas registradas (exp*) com o MFCC oficial, LDA, binário, sem aumento."""
    rodadas = []
    for caminho in sorted(pasta.glob("exp*/metrics.json")):
        d = json.loads(caminho.read_text(encoding="utf-8"))
        p = d["parametros"]
        if (p.get("features") in FEATURES_OFICIAL and p.get("splits") == splits
                and p.get("modelo") == "lda" and p.get("tarefa") == "binario"
                and p.get("aumento") in (None, "nenhum")):
            rodadas.append(d)
    return rodadas


def acc(d: dict) -> float:
    return d["resumo"]["acuracia_balanceada_media"]


def e_curva(p: dict) -> bool:
    return p.get("segundos_treino") is not None


def ultima(rodadas: list[dict]) -> dict | None:
    return max(rodadas, key=lambda d: d["id"]) if rodadas else None


def celula(d: dict | None) -> str:
    if d is None:
        return "—"
    r = d["resumo"]
    if r["protocolo"] == "A":
        texto = f"{r['acuracia_balanceada_media']:.3f} ± {r['acuracia_balanceada_desvio']:.3f}"
    else:
        texto = (f"{r['acuracia_balanceada_media']:.3f} (sens. {r['sensibilidade_media']:.3f}, "
                 f"espec. {r['especificidade_media']:.3f}; bpfo_0.3mm "
                 f"{r['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.3f})")
    return f"{texto} ({d['id']})"


def ms(v) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.3f}"


def intervalo_ids(ds: list[dict]) -> str:
    ids = sorted(d["id"] for d in ds)
    return f"{ids[0]}–{ids[-1]}" if ids else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--saida", type=Path, default=Path("reports/validation/tabela_controles.md"))
    args = ap.parse_args()

    splits = particao.hash_arquivo(args.splits)
    rodadas = carregar_rodadas(args.pasta, splits)

    def por(prot, cond):
        return [d for d in rodadas if d["parametros"]["protocolo"] == prot and cond(d["parametros"])]

    def so(sem_c0=False, norm=False, perm=False):
        return lambda p: (bool(p.get("sem_c0")) == sem_c0 and bool(p.get("norm_clipe")) == norm
                          and bool(p.get("permutado")) == perm and not e_curva(p))

    ref = {prot: ultima(por(prot, so())) for prot in ("A", "B")}
    semc0 = {prot: ultima(por(prot, so(sem_c0=True))) for prot in ("A", "B")}
    norm = {prot: ultima(por(prot, so(norm=True))) for prot in ("A", "B")}
    perm = por("B", so(perm=True))
    curva = [d for d in rodadas if e_curva(d["parametros"])
             and d["parametros"].get("amostragem_treino") == "por_gravacao"
             and not d["parametros"].get("sem_c0") and not d["parametros"].get("norm_clipe")]

    texto = [
        "# Controles obrigatórios do classificador",
        "",
        "Gerado por `scripts/validation/run_tabela_controles.py` a partir dos `metrics.json`. "
        f"MFCC oficial, partição `{splits}`, LDA, binário, sem aumento de dados. "
        "A: acurácia balanceada média ± desvio entre os folds. B: acurácia balanceada média "
        "(sensibilidade e especificidade médias; acurácia balanceada da `bpfo_0.3mm`).",
        "",
        "## Ablação do ganho",
        "",
        "| protocolo | referência | sem c0 | normalização RMS por segmento |",
        "|---|---|---|---|",
        *(f"| {prot} | {celula(ref[prot])} | {celula(semc0[prot])} | {celula(norm[prot])} |"
          for prot in ("A", "B")),
        "",
        "## Permutação por bloco (Protocolo B)",
        "",
    ]
    if perm and ref["B"]:
        a = np.array([acc(d) for d in perm])
        obs = acc(ref["B"])
        p_emp = (1 + int((a >= obs).sum())) / (len(a) + 1)
        t = stats.ttest_1samp(a, 0.5)
        texto += [
            f"{len(perm)} sementes ({intervalo_ids(perm)}), contra a referência "
            f"{obs:.3f} ({ref['B']['id']}).",
            "",
            "| média ± desvio | mediana | mín. | máx. | sementes ≥ referência | p empírico "
            "| média × 0,5 (teste t) |",
            "|---|---|---|---|---|---|---|",
            f"| {ms(a)} | {np.median(a):.3f} | {a.min():.3f} | {a.max():.3f} "
            f"| {int((a >= obs).sum())} de {len(a)} | {p_emp:.4f} "
            f"| t = {t.statistic:.2f}, p = {t.pvalue:.2g} |",
            "",
            "p empírico = (1 + nº de sementes com acurácia ≥ a da referência) / (n + 1): a "
            "probabilidade de um modelo sem relação rótulo–sinal chegar ao resultado da "
            "referência. Com n sementes, o menor valor possível é 1 / (n + 1).",
        ]
        if t.pvalue < 0.05 and a.mean() < 0.5:
            texto += [
                "",
                "A média fica abaixo de 0,5 de forma significativa. A permutação por bloco "
                "mantém quantos blocos há de cada rótulo, e há uma única gravação normal: a "
                "maior parte dos blocos dela recebe o rótulo \"falha\" no treino, e parte dos "
                "blocos de falha recebe \"normal\". O modelo aprende uma regra invertida em "
                "relação ao teste, que usa os rótulos verdadeiros. Vazamento empurraria o "
                "resultado para cima, não para baixo.",
            ]
    else:
        texto.append("Nenhuma rodada encontrada.")

    texto += ["", "## Curva de aprendizado", ""]
    if curva:
        grupos: dict[tuple[str, float], list[dict]] = {}
        for d in curva:
            grupos.setdefault((d["parametros"]["protocolo"], d["parametros"]["segundos_treino"]), []).append(d)
        segundos = sorted({s for _, s in grupos})
        n_sem = sorted({len(v) for v in grupos.values()})
        texto += [
            f"Segmentos de treino repartidos entre as gravações de cada classe; "
            f"{'/'.join(map(str, n_sem))} sementes por ponto ({intervalo_ids(curva)}). "
            "Média ± desvio entre as sementes.",
            "",
            "| s por classe | A | B (acc. bal.) | B, sensib. | B, especif. | B, bpfo_0.3mm (acc. bal.) |",
            "|---|---|---|---|---|---|",
        ]
        for s in segundos:
            ga, gb = grupos.get(("A", s), []), grupos.get(("B", s), [])
            cel_a = ms([acc(d) for d in ga]) if ga else "—"
            if gb:
                rb = [d["resumo"] for d in gb]
                cel_b = (f"{ms([r['acuracia_balanceada_media'] for r in rb])} "
                         f"| {ms([r['sensibilidade_media'] for r in rb])} "
                         f"| {ms([r['especificidade_media'] for r in rb])} "
                         f"| {ms([r['por_falha']['bpfo_0.3mm']['acuracia_balanceada'] for r in rb])}")
            else:
                cel_b = "— | — | — | —"
            texto.append(f"| {s:g} | {cel_a} | {cel_b} |")
        if ref["A"] and ref["B"]:
            rb = ref["B"]["resumo"]
            texto.append(
                f"| todo o treino | {acc(ref['A']):.3f} | {rb['acuracia_balanceada_media']:.3f} "
                f"| {rb['sensibilidade_media']:.3f} | {rb['especificidade_media']:.3f} "
                f"| {rb['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.3f} |")
    else:
        texto.append("Nenhuma rodada encontrada.")
    texto.append("")

    args.saida.write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))
    print(f"tabela: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
