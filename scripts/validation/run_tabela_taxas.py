#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_taxas.py — Protocolo B em duas taxas de amostragem, numa tabela
===========================================================================

Responde, com o classificador, a ressalva do estudo de decimação: "fração de
energia não é fração de informação". Monta, a partir dos `metrics.json` já
gravados em `reports/validation/`, a tabela do Protocolo B com o mesmo modelo e
a mesma partição em taxas diferentes: sensibilidade, especificidade e acurácia
balanceada (média), o pior caso, e a acurácia balanceada e a sensibilidade por
falha deixada de fora — com destaque para as de 0,3 mm. Não treina nada nem
escreve no registry: só lê rodadas que já têm linha própria.

Comparabilidade
---------------
Aborta se as rodadas não forem do B binário com o mesmo modelo, as mesmas
features extraídas no mesmo commit (`features_commit`), a mesma segmentação e
o mesmo MFCC (em milissegundos), sem controles nem aumento, ou se repetirem a
taxa. A partição de cada taxa é um splits.json
próprio (`config.arquivo_splits`), então o hash difere; o script confere, em
vez disso, que as partições descrevem os MESMOS segmentos em segundos e os
mesmos folds, e que o hash de cada uma é o que a rodada registrou.

Uso
---
    python scripts/validation/run_tabela_taxas.py --rodadas expA expB
        (a primeira é a referência; as outras são comparadas com ela)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import config
from validation import particao
from validation.run_tabela_aumento import carregar

# tudo o que define a rodada além da taxa: nada disso pode variar entre as linhas
# (o features_commit garante o mesmo código de extração, `dsp.mfcc` inclusive:
# com outro commit, a diferença entre as linhas poderia não ser da taxa)
IGUAIS = ("protocolo", "tarefa", "modelo", "features", "features_commit", "segmento_s",
          "segmentos_por_bloco", "segmentos_descarte", "mfcc_janela_ms", "mfcc_hop_ms",
          "mfcc_n_mels", "mfcc_n_coefs")
CONTROLES = ("permutado", "sem_c0", "norm_clipe", "segundos_treino")


def conferir(rodadas: list[tuple[str, dict]]) -> None:
    """Só entram rodadas do B binário comparáveis entre si; aborta dizendo o motivo."""
    erros = []
    for campo in IGUAIS:
        valores = {str(r["parametros"].get(campo)) for _, r in rodadas}
        if len(valores) != 1:
            erros.append(f"{campo} difere entre as rodadas: {', '.join(sorted(valores))}")
    for e, r in rodadas:
        p = r["parametros"]
        if (p.get("protocolo"), p.get("tarefa")) != ("B", "binario"):
            erros.append(f"{e}: é {p.get('protocolo')}/{p.get('tarefa')}; a tabela é do B binário")
        if any(p.get(c) for c in CONTROLES):
            erros.append(f"{e}: é controle ou ablação ({'/'.join(CONTROLES)})")
        if p.get("aumento", "nenhum") != "nenhum":
            erros.append(f"{e}: tem aumento ({p['aumento']}); a comparação de taxas é sem aumento")
    taxas = [r["parametros"].get("fs_hz") for _, r in rodadas]
    if len(set(taxas)) != len(taxas):
        erros.append(f"taxa repetida entre as rodadas: {taxas}")
    if erros:
        raise SystemExit("Abortado: rodadas não comparáveis:\n  " + "\n  ".join(erros))


def _em_segundos(particoes: dict) -> tuple:
    """A partição sem as amostras: segmentos em segundos e folds por id."""
    fs = particoes["parametros"]["fs_hz"]
    segs = tuple((s["id"], s["rotulo"], s["indice"], s["bloco"],
                  s["inicio"] / fs, s["fim"] / fs) for s in particoes["segmentos"])
    folds = tuple((f["protocolo"], f["nome"], tuple(f["treino"]), tuple(f["teste"]),
                   tuple(f["descartados"])) for f in particoes["folds"])
    return segs, folds


def conferir_particoes(rodadas: list[tuple[str, dict]], arquivos: list[Path]) -> None:
    """Cada rodada leu a partição da sua taxa, e todas descrevem os mesmos segmentos e folds."""
    erros, formas = [], []
    for (e, r), arq in zip(rodadas, arquivos):
        if not arq.exists():
            erros.append(f"{e}: não achei a partição {arq}")
            continue
        h = particao.hash_arquivo(arq)
        if h != r["parametros"].get("splits"):
            erros.append(f"{e}: registrou splits {r['parametros'].get('splits')}, "
                         f"mas {arq} tem hash {h}")
        formas.append(_em_segundos(particao.carregar(arq)))
    if not erros and any(f != formas[0] for f in formas[1:]):
        erros.append("as partições das taxas não descrevem os mesmos segmentos e folds em segundos")
    if erros:
        raise SystemExit("Abortado: partições não conferem:\n  " + "\n  ".join(erros))


def corte_fir(fs: int, raiz: Path = Path(".")) -> str:
    """Corte do anti-aliasing usado no 02, lido do manifest versionado do PCM decimado."""
    caminho = raiz / config.dir_pcm_decimado(fs) / "manifest.json"
    try:
        dec = json.loads(caminho.read_text(encoding="utf-8"))["_conjunto"]["decimacao"]
        return f"{dec['cutoff_hz']:.0f} Hz ({dec['numtaps']} taps)"
    except (OSError, KeyError, ValueError):
        return "—"


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _dif(v: float) -> str:
    return "0" if abs(v) < 5e-4 else f"{v:+.3f}"


def montar(ids: list[str], pasta: Path, raiz: Path = Path(".")) -> str:
    """`raiz` é a raiz do repositório, de onde saem as partições e os manifests do PCM."""
    rodadas = [(e, carregar(pasta, e)) for e in ids]
    conferir(rodadas)
    taxas = [int(r["parametros"]["fs_hz"]) for _, r in rodadas]
    conferir_particoes(rodadas, [raiz / config.arquivo_splits(fs) for fs in taxas])

    p = rodadas[0][1]["parametros"]
    falhas = list(rodadas[0][1]["resumo"]["por_falha"])
    ref = rodadas[0][1]["resumo"]

    texto = [
        "# Protocolo B em duas taxas de amostragem",
        "",
        "Gerado por `scripts/validation/run_tabela_taxas.py` a partir dos `metrics.json`. "
        f"Modelo `{p['modelo']}`, MFCC de {p['mfcc_n_coefs']} coeficientes (média e desvio "
        f"por segmento; janela de {p['mfcc_janela_ms']} ms, passo de {p['mfcc_hop_ms']} ms, "
        f"{p['mfcc_n_mels']} bandas Mel até o Nyquist de cada taxa), sem aumento. As partições "
        "das taxas descrevem os mesmos segmentos de "
        f"{p['segmento_s']:g} s e os mesmos folds; só o número de amostras muda.",
        "",
        "## Médias e pior caso",
        "",
        "| taxa | rodada | Nyquist | corte do FIR | sensib. | especif. | acc. bal. "
        "| pior falha (acc. bal.) | pior fold |",
        "|---|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for (e, r), fs in zip(rodadas, taxas):
        s = r["resumo"]
        texto.append(
            f"| {fs / 1000:g} kHz | {e} | {fs / 2:.0f} Hz | {corte_fir(fs, raiz)} "
            f"| {_fmt(s['sensibilidade_media'])} | {_fmt(s['especificidade_media'])} "
            f"| **{_fmt(s['acuracia_balanceada_media'])}** "
            f"| {_fmt(s['pior_falha_acuracia_balanceada'])} ({s['pior_falha']}) "
            f"| {_fmt(s['pior_fold_acuracia_balanceada'])} |")
    for (e, r), fs in list(zip(rodadas, taxas))[1:]:
        s = r["resumo"]
        texto.append(
            f"| diferença ({fs / 1000:g} − {taxas[0] / 1000:g} kHz) | | | "
            f"| {_dif(s['sensibilidade_media'] - ref['sensibilidade_media'])} "
            f"| {_dif(s['especificidade_media'] - ref['especificidade_media'])} "
            f"| **{_dif(s['acuracia_balanceada_media'] - ref['acuracia_balanceada_media'])}** "
            f"| {_dif(s['pior_falha_acuracia_balanceada'] - ref['pior_falha_acuracia_balanceada'])} "
            f"| {_dif(s['pior_fold_acuracia_balanceada'] - ref['pior_fold_acuracia_balanceada'])} |")

    for metrica, titulo in (("acuracia_balanceada", "Acurácia balanceada"),
                            ("sensibilidade", "Sensibilidade")):
        cab = [f"**{f}**" if "0.3mm" in f else f for f in falhas]
        texto += [
            "",
            f"## {titulo} por falha deixada de fora",
            "",
            "| taxa | " + " | ".join(cab) + " |",
            "|---|" + "---:|" * len(falhas),
        ]
        for (e, r), fs in zip(rodadas, taxas):
            pf = r["resumo"]["por_falha"]
            texto.append(f"| {fs / 1000:g} kHz | "
                         + " | ".join(_fmt(pf[f][metrica]) for f in falhas) + " |")
        for (e, r), fs in list(zip(rodadas, taxas))[1:]:
            pf = r["resumo"]["por_falha"]
            texto.append(f"| diferença | " + " | ".join(
                _dif(pf[f][metrica] - ref["por_falha"][f][metrica]) for f in falhas) + " |")

    texto += [
        "",
        "Leitura: acurácia balanceada = (sensibilidade + especificidade) / 2. Um 0,500 numa "
        "falha é sensibilidade 0 com especificidade 1 — a falha não é detectada. As falhas de "
        "0,3 mm (em negrito) são as que o estudo de decimação apontava como dependentes da "
        "banda alta: se a informação delas estivesse acima do Nyquist da taxa de trabalho, "
        "é nelas que a taxa maior faria diferença.",
        "",
    ]
    return "\n".join(texto)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # sem padrão: os ids dependem da ordem em que as rodadas foram registradas
    ap.add_argument("--rodadas", nargs="+", required=True, metavar="EXP",
                    help="ids das rodadas; a primeira é a referência")
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--saida", type=Path, default=Path("reports/validation/tabela_taxas.md"))
    args = ap.parse_args()
    if len(args.rodadas) < 2:
        ap.error("--rodadas precisa de pelo menos duas rodadas")

    texto = montar(args.rodadas, args.pasta)
    args.saida.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"tabela: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
