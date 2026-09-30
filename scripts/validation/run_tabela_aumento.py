#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_aumento.py — Protocolo B com e sem aumento de dados, numa tabela
===========================================================================

Monta, a partir dos `metrics.json` já gravados em `reports/validation/`, a
tabela do Protocolo B sem aumento e com aumento aplicado só no treino de cada
fold: sensibilidade, especificidade e acurácia balanceada (média), o pior caso
(pior falha deixada de fora e pior fold) e o fator de expansão do treino. Não
treina nada nem escreve no registry: só lê rodadas que já têm linha própria.

Linhas da tabela
----------------
--referencia   a rodada sem aumento
--completo     uma ou mais rodadas com o aumento completo, que só podem diferir
               pela semente do aumento; com mais de uma, a linha traz média ±
               desvio entre as sementes e o pior caso entre elas
--ablacoes     rodadas com uma técnica isolada ou outro modo de estiramento,
               uma linha cada

Fator de expansão
-----------------
Nominal: 1 + cópias por segmento (cada segmento de treino e as suas variantes).
Efetivo: média, entre os folds, de (segmentos de treino + variantes aceitas) /
segmentos de treino. É menor que o nominal porque o filtro por fold recusa as
variantes que encostam no teste ou na faixa de descarte.

Uso
---
    python scripts/validation/run_tabela_aumento.py
    python scripts/validation/run_tabela_aumento.py --referencia exp015 \\
        --completo exp016 --ablacoes exp019 exp020 exp021 exp022
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config

# a mesma rodada de referência: nada disso pode variar entre as linhas
IGUAIS = ("splits", "modelo", "mfcc_janela_ms", "mfcc_hop_ms", "mfcc_n_mels", "mfcc_n_coefs")
_NOMES = {"deslocamento": "deslocamento", "estiramento": "estiramento", "ruido": "ruído"}


def carregar(pasta: Path, exp_id: str) -> dict:
    achados = sorted(pasta.glob(f"{exp_id}_*/metrics.json"))
    if len(achados) != 1:
        raise SystemExit(f"Abortado: esperava uma pasta {exp_id}_* em {pasta}, achei {len(achados)}.")
    return json.loads(achados[0].read_text(encoding="utf-8"))


def semente_aumento(p: dict) -> int:
    """Rodadas anteriores ao campo `aumento_semente` (exp016–exp022) usaram a do config."""
    return int(p.get("aumento_semente", config.SEMENTE))


def conferir(referencia: tuple[str, dict], completo: list[tuple[str, dict]],
             ablacoes: list[tuple[str, dict]]) -> None:
    """Só entram rodadas do B binário comparáveis entre si; aborta dizendo o motivo."""
    erros = []
    todas = [referencia] + completo + ablacoes
    for campo in IGUAIS:
        valores = {str(r["parametros"].get(campo)) for _, r in todas}
        if len(valores) != 1:
            erros.append(f"{campo} difere entre as rodadas: {', '.join(sorted(valores))}")
    for e, r in todas:
        p = r["parametros"]
        if (p.get("protocolo"), p.get("tarefa")) != ("B", "binario"):
            erros.append(f"{e}: é {p.get('protocolo')}/{p.get('tarefa')}; a tabela é do B binário")
        if p.get("permutado") or p.get("sem_c0") or p.get("norm_clipe") or p.get("segundos_treino"):
            erros.append(f"{e}: é controle ou ablação do protocolo "
                         "(permutado/sem_c0/norm_clipe/segundos_treino)")
    e, r = referencia
    if r["parametros"].get("aumento", "nenhum") != "nenhum":
        erros.append(f"{e}: a referência tem aumento ({r['parametros']['aumento']})")
    for e, r in completo + ablacoes:
        if r["parametros"].get("aumento", "nenhum") == "nenhum":
            erros.append(f"{e}: não tem aumento")
    if completo:
        chaves = ("aumento", "aumento_copias", "aumento_modo_estir", "aumento_desloc_max_s",
                  "aumento_estir_taxas", "aumento_snr_db")
        assinaturas = {tuple(str(r["parametros"].get(k)) for k in chaves) for _, r in completo}
        if len(assinaturas) != 1:
            erros.append("as rodadas de --completo têm parâmetros de aumento diferentes; "
                         "elas só podem diferir pela semente")
        sementes = [semente_aumento(r["parametros"]) for _, r in completo]
        if len(set(sementes)) != len(sementes):
            erros.append(f"--completo repete semente de aumento: {sementes}")
    if erros:
        raise SystemExit("Abortado: rodadas não comparáveis:\n  " + "\n  ".join(erros))


def resumo(r: dict) -> dict:
    """Números de uma rodada: médias, pior caso e expansão do treino."""
    s, folds = r["resumo"], r["folds"]
    fatores = [(f["n_treino"] + f.get("n_treino_aumento", 0)) / f["n_treino"] for f in folds]
    aceitas = [f.get("n_treino_aumento", 0) for f in folds]
    return {
        "sens": s["sensibilidade_media"],
        "espec": s["especificidade_media"],
        "ab": s["acuracia_balanceada_media"],
        "pior_falha": s["pior_falha"],
        "ab_pior_falha": s["pior_falha_acuracia_balanceada"],
        "pior_fold": s["pior_fold"],
        "ab_pior_fold": s["pior_fold_acuracia_balanceada"],
        "fator_nominal": 1 + int(r["parametros"].get("aumento_copias") or 0),
        "fator_efetivo": sum(fatores) / len(fatores),
        "aceitas_min": min(aceitas),
        "aceitas_max": max(aceitas),
        "por_falha": {k: v["acuracia_balanceada"] for k, v in s["por_falha"].items()},
    }


def descrever(p: dict) -> str:
    """Rótulo da linha a partir dos parâmetros, não de um nome digitado à mão."""
    if p.get("aumento", "nenhum") == "nenhum":
        return "sem aumento"
    tecnicas = p["aumento"].split("+")
    texto = ("três técnicas" if len(tecnicas) == 3
             else "só " + " + ".join(_NOMES.get(t, t) for t in tecnicas))
    if "estiramento" in tecnicas and p.get("aumento_modo_estir") == "velocidade":
        texto += ", estiramento *velocidade*"
    return texto


def _pm(valores: list[float]) -> str:
    if len(valores) == 1:
        return f"{valores[0]:.3f}"
    return f"{statistics.mean(valores):.3f} ± {statistics.stdev(valores):.3f}"


def linha(ids: list[str], rodadas: list[dict]) -> str:
    rs = [resumo(r) for r in rodadas]
    p = rodadas[0]["parametros"]
    pior = min(rs, key=lambda x: x["ab_pior_falha"])
    pior_fold = min(rs, key=lambda x: x["ab_pior_fold"])
    fator = ("1" if rs[0]["fator_nominal"] == 1 else
             f"{rs[0]['fator_nominal']} / {statistics.mean(x['fator_efetivo'] for x in rs):.2f}")
    aceitas = ("—" if rs[0]["fator_nominal"] == 1 else
               f"{min(x['aceitas_min'] for x in rs)}–{max(x['aceitas_max'] for x in rs)}")
    rotulo = descrever(p) + (f" ({len(ids)} sementes)" if len(ids) > 1 else "")
    rodada = ids[0] if len(ids) == 1 else f"{ids[0]}–{ids[-1]}" if _seguidos(ids) else ", ".join(ids)
    return (f"| {rotulo} | {rodada} | {_pm([x['sens'] for x in rs])} | {_pm([x['espec'] for x in rs])} "
            f"| **{_pm([x['ab'] for x in rs])}** | {pior['ab_pior_falha']:.3f} ({pior['pior_falha']}) "
            f"| {pior_fold['ab_pior_fold']:.3f} | {fator} | {aceitas} |")


def _seguidos(ids: list[str]) -> bool:
    try:
        n = [int(i.removeprefix("exp")) for i in ids]
    except ValueError:
        return False
    return n == list(range(n[0], n[0] + len(n)))


def linha_por_falha(ids: list[str], rodadas: list[dict], falhas: list[str]) -> str:
    rs = [resumo(r) for r in rodadas]
    rotulo = descrever(rodadas[0]["parametros"]) + (f" ({len(ids)} sementes)" if len(ids) > 1 else "")
    return "| " + rotulo + " | " + " | ".join(_pm([x["por_falha"][f] for x in rs]) for f in falhas) + " |"


def montar(ids_ref: str, ids_completo: list[str], ids_ablacoes: list[str],
           pasta: Path) -> str:
    ref = (ids_ref, carregar(pasta, ids_ref))
    completo = [(e, carregar(pasta, e)) for e in ids_completo]
    ablacoes = [(e, carregar(pasta, e)) for e in ids_ablacoes]
    conferir(ref, completo, ablacoes)

    p = ref[1]["parametros"]
    grupos = [([ref[0]], [ref[1]])]
    if completo:
        grupos.append(([e for e, _ in completo], [r for _, r in completo]))
    grupos += [([e], [r]) for e, r in ablacoes]
    falhas = list(ref[1]["resumo"]["por_falha"])
    copias = {r["parametros"].get("aumento_copias") for _, r in completo + ablacoes}

    texto = [
        "# Protocolo B com e sem aumento de dados",
        "",
        "Gerado por `scripts/validation/run_tabela_aumento.py` a partir dos `metrics.json`. "
        f"Partição `{p['splits']}`, modelo `{p['modelo']}`, MFCC de "
        f"{p['mfcc_n_coefs']} coeficientes (média e desvio por segmento). O aumento entra "
        "só no treino de cada fold; o teste é sempre o segmento original.",
        "",
        "## Médias e pior caso",
        "",
        "| aumento | rodada | sensib. | especif. | acc. bal. | pior falha (acc. bal.) "
        "| pior fold | fator de expansão (nominal / efetivo) | variantes aceitas por fold |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
        *(linha(ids, rs) for ids, rs in grupos),
        "",
        "## Acurácia balanceada por falha deixada de fora",
        "",
        "| aumento | " + " | ".join(falhas) + " |",
        "|---|" + "---:|" * len(falhas),
        *(linha_por_falha(ids, rs, falhas) for ids, rs in grupos),
        "",
        "Leitura: acurácia balanceada = (sensibilidade + especificidade) / 2. Um 0,500 "
        "numa falha é sensibilidade 0 com especificidade 1 — a falha não é detectada —, "
        "e não detecção parcial.",
        "",
        "Fator de expansão: nominal = 1 + cópias por segmento"
        + (f" ({', '.join(str(c) for c in sorted(copias))} cópias)" if copias else "")
        + "; efetivo = média, entre os folds, de (segmentos de treino + variantes aceitas) / "
        "segmentos de treino. A diferença são as variantes recusadas por encostarem no teste "
        "ou na faixa de descarte.",
        "",
    ]
    if len(completo) > 1:
        texto += [
            "Com mais de uma semente, a linha traz média ± desvio entre as sementes do "
            "aumento, e o pior caso é o pior entre elas.",
            "",
        ]
    return "\n".join(texto)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--referencia", default="exp015")
    # exp224–exp233: aumento completo com 10 sementes, no mesmo commit (o exp224
    # usa a semente do config e reproduz o exp016)
    ap.add_argument("--completo", nargs="+", default=[f"exp{n}" for n in range(224, 234)])
    ap.add_argument("--ablacoes", nargs="*", default=["exp019", "exp020", "exp021", "exp022"])
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--saida", type=Path, default=Path("reports/validation/tabela_aumento.md"))
    args = ap.parse_args()

    texto = montar(args.referencia, args.completo, args.ablacoes, args.pasta)
    args.saida.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"tabela: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
