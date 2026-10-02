#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_consolidacao.py — resultados do classificador consolidados para o relatório
===============================================================================

Fecha a tarefa "Classificador 6/6". Lê os `metrics.json` já gravados em
`reports/validation/` e o `experiments/registry.csv`, e grava em
`reports/validation/consolidacao/` as tabelas (LaTeX e Markdown) e as figuras
que entram nas seções Particionamento, Desempenho do Classificador e Discussão
do relatório. Não treina nada nem escreve no registry.

O que sai
---------
tab_protocoloB_folds.tex     acurácia balanceada dos 24 folds do B (falha de fora
                             × bloco normal), com sensibilidade, especificidade e
                             AB por falha e a média — o número da meta
tab_protocoloA_folds.tex     os 6 folds do A, binário e multiclasse, rotulados
                             como limite superior otimista
tab_matriz_confusao_B.tex    matriz de confusão do B somada nos 24 folds, por
fig_matriz_confusao_B.png    classe verdadeira (normal e as 4 falhas) × previsão
tab_controles.tex            ablação do ganho e permutação (a curva fica na figura)
tab_taxas_B.tex              B a 12,8 × 25,6 kHz (tarefa 5/6)
tab_aumento_B.tex            B com e sem aumento, três técnicas em 10 sementes (tarefa 4/6)
tab_harmonicos.tex           intervenção nas bandas dos harmônicos do eixo (controle)
fig_controles.png            permutação (histograma) e curva de aprendizado
rastreabilidade.md           cada número das tabelas → rodada → linha do registry,
                             com o valor do registry conferido contra o metrics.json
consolidacao.md              tudo acima em Markdown, para leitura no GitHub

Rastreabilidade
---------------
Toda rodada usada precisa ter linha no registry com o mesmo script, e as
métricas da linha precisam bater com o `resumo` do metrics.json (o registry grava
4 casas). Se alguma não bater, o script aborta antes de gravar qualquer saída:
uma tabela do relatório não pode citar um número sem rodada correspondente.

As rodadas são passadas por id, e não achadas por busca, para que a tabela diga
exatamente de onde veio cada número. O script confere, pelos `parametros`, que
cada id é o que o argumento diz (protocolo, tarefa, controle) e que todas usam a
mesma partição, o mesmo modelo e as mesmas features.

Comparação de taxas (tarefa 5/6)
--------------------------------
Entra também a tabela do B a 12,8 kHz × 25,6 kHz (`exp234` × `exp235`, as duas
no mesmo commit). A comparabilidade é conferida pelas mesmas funções do
`run_tabela_taxas.py` (`conferir` e `conferir_particoes`): mesmo modelo, mesmas
features no mesmo commit e partições que descrevem os mesmos segmentos em
segundos. A rodada a 12,8 kHz tem que ter a partição e a taxa da referência do
B. `--sem-taxas` tira essa tabela.

Uso
---
    python scripts/validation/run_consolidacao.py
    python scripts/validation/run_consolidacao.py --sem-taxas
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
from validation import run_tabela_aumento, run_tabela_taxas

FALHAS = ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm")
CLASSES = ("normal",) + FALHAS

# O que tem que ser igual entre todas as rodadas consolidadas (exceto a da taxa
# alternativa, conferida pelo run_tabela_taxas).
IGUAIS = ("splits", "modelo", "features", "fs_hz", "segmento_s", "segmentos_por_bloco",
          "segmentos_descarte", "mfcc_janela_ms", "mfcc_hop_ms", "mfcc_n_mels", "mfcc_n_coefs")

# Chave do registry → caminho no `resumo` do metrics.json.
METRICAS_B = {
    "acc_bal_media": ("acuracia_balanceada_media",),
    "sens_media": ("sensibilidade_media",),
    "espec_media": ("especificidade_media",),
    "acc_bal_pior_falha": ("pior_falha_acuracia_balanceada",),
    "acc_bal_pior_fold": ("pior_fold_acuracia_balanceada",),
    **{f"acc_bal_{f}": ("por_falha", f, "acuracia_balanceada") for f in FALHAS},
}
METRICAS_A = {
    "acc_bal_media": ("acuracia_balanceada_media",),
    "acc_bal_desvio": ("acuracia_balanceada_desvio",),
    "acc_bal_min": ("acuracia_balanceada_min",),
}
TOL_REGISTRY = 6e-5      # o registry grava 4 casas


def _intervalo(a: int, b: int) -> list[str]:
    return [f"exp{n:03d}" for n in range(a, b + 1)]


# Padrões: as rodadas oficiais citadas no README e nas tabelas já geradas.
PADRAO = {
    "ref_b": "exp015",                    # B sem aumento (mesmo commit das rodadas com aumento)
    "reproducoes_b": ["exp013"],          # mesma configuração, outro commit: mesmos números
    "ref_a": "exp027",
    "ref_a_multi": "exp028",
    "semc0_a": "exp029", "semc0_b": "exp030",
    "norm_a": "exp219", "norm_b": "exp014",
    "permutacao": _intervalo(31, 50) + _intervalo(59, 138),
    "curva": _intervalo(139, 218),
    "taxa_ref": "exp234",                 # 12,8 kHz, mesmo commit da rodada a 25,6 kHz
    "taxa_alt": "exp235",                 # 25,6 kHz (tarefa 5/6)
    # aumento de dados (tarefa 4/6): referência é o ref_b
    "aum_completo": _intervalo(224, 233),  # três técnicas, 10 sementes do sorteio das variantes
    "aum_ablacoes": ["exp019", "exp020", "exp021", "exp022"],   # técnica isolada e modo velocidade
    "aum_permutado": "exp017",             # três técnicas com rótulos permutados (controle)
    # harmônicos do eixo (controle): pastas sem o prefixo expNNN
    "harm_decomp": "exp220",
    "harm_interv": "exp221",
}
PASTAS_HARMONICOS = {"exp220": "controle_harmonicos", "exp221": "controle_harmonicos/intervencao"}
SCRIPT_HARMONICOS = "exploration/inspect_lda_harmonicos.py"

# Registry do controle dos harmônicos → caminho no metrics.json. A decomposição grava
# 3 casas no registry; a intervenção, 4.
METRICAS_HARM_DECOMP = {
    "frac_sep_harm_mediana": ("resumo", "fracao_separacao_harm_mediana"),
    "frac_sep_harm_min": ("resumo", "fracao_separacao_harm_min"),
    "frac_sep_harm_max": ("resumo", "fracao_separacao_harm_max"),
    "frac_bandas_mediana": ("resumo", "fracao_de_bandas_mediana"),
    "frac_desvio_mfcc_mediana": ("resumo", "fracao_desvio_mfcc_mediana"),
    **{f"frac_sep_harm_{f}": ("resumo", "por_falha", f, "fracao_bandas_harmonicos") for f in FALHAS},
}
METRICAS_HARM_INTERV = {
    "acc_bal_referencia": ("intervencao", "referencia", "acuracia_balanceada_media"),
    "acc_bal_sem_harm": ("intervencao", "sem_harmonicos", "acuracia_balanceada_media"),
    "sens_sem_harm": ("intervencao", "sem_harmonicos", "sensibilidade_media"),
    "espec_sem_harm": ("intervencao", "sem_harmonicos", "especificidade_media"),
    "acc_bal_sorteios_min": ("intervencao", "acc_bal_sorteios_min"),
    "acc_bal_sorteios_max": ("intervencao", "acc_bal_sorteios_max"),
}
TOL_REGISTRY_3 = 6e-4    # métricas gravadas com 3 casas


# ---------------------------------------------------------------------------
# leitura e conferência
# ---------------------------------------------------------------------------

def carregar(pasta: Path, exp_id: str) -> dict:
    achados = sorted(pasta.glob(f"{exp_id}_*/metrics.json"))
    if len(achados) != 1:
        raise SystemExit(f"Abortado: esperava uma pasta {exp_id}_* em {pasta}, achei {len(achados)}.")
    d = json.loads(achados[0].read_text(encoding="utf-8"))
    d.setdefault("id", exp_id)
    d["_pasta"] = achados[0].parent.name
    return d


def ler_registry(caminho: Path) -> dict[str, dict]:
    with caminho.open(encoding="utf-8", newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def _kv(texto: str) -> dict[str, str]:
    return dict(par.split("=", 1) for par in (texto or "").split(";") if "=" in par)


def _pegar(resumo: dict, caminho: tuple[str, ...]):
    v = resumo
    for k in caminho:
        v = v[k]
    return v


@dataclass
class Conferencia:
    """Uma linha da rastreabilidade: o número, de onde veio e se o registry bate."""
    exp_id: str
    chave: str
    valor_json: float
    valor_registry: float | None
    ok: bool


@dataclass
class Rodadas:
    ref_b: dict
    reproducoes_b: list[dict]
    ref_a: dict
    ref_a_multi: dict
    semc0_a: dict
    semc0_b: dict
    norm_a: dict
    norm_b: dict
    permutacao: list[dict]
    curva: list[dict]
    taxa_ref: dict | None = None
    taxa_alt: dict | None = None
    aum_completo: list[dict] = field(default_factory=list)
    aum_ablacoes: list[dict] = field(default_factory=list)
    aum_permutado: dict | None = None
    harm_decomp: dict | None = None
    harm_interv: dict | None = None
    conferencias: list[Conferencia] = field(default_factory=list)

    def todas(self) -> list[dict]:
        """Rodadas do run_protocol (os harmônicos têm conferência própria)."""
        r = [self.ref_b, *self.reproducoes_b, self.ref_a, self.ref_a_multi, self.semc0_a,
             self.semc0_b, self.norm_a, self.norm_b, *self.permutacao, *self.curva,
             *self.aum_completo, *self.aum_ablacoes]
        return r + [d for d in (self.aum_permutado, self.taxa_ref, self.taxa_alt) if d]


def _comparavel(p: dict, campo: str) -> str:
    """A ablação da normalização grava as features com o sufixo `_normclipe`; o MFCC é o mesmo."""
    v = str(p.get(campo))
    return v.removesuffix("_normclipe") if campo == "features" else v


def conferir_parametros(rod: Rodadas) -> None:
    """Cada id é o que o argumento diz, e as rodadas são comparáveis entre si."""
    erros = []

    def esperar(d, **campos):
        p = d["parametros"]
        for k, v in campos.items():
            atual = p.get(k, "nenhum" if k == "aumento" else None)   # exp013 é anterior ao campo
            if isinstance(v, bool):
                atual = bool(atual)
            if atual != v:
                erros.append(f"{d['id']}: {k}={p.get(k)!r}, esperado {v!r}")

    base = dict(permutado=False, sem_c0=False, norm_clipe=False)
    for d in [rod.ref_b, *rod.reproducoes_b]:
        esperar(d, protocolo="B", tarefa="binario", aumento="nenhum", **base)
        if d["parametros"].get("segundos_treino") is not None:
            erros.append(f"{d['id']}: é ponto da curva de aprendizado")
    esperar(rod.ref_a, protocolo="A", tarefa="binario", aumento="nenhum", **base)
    esperar(rod.ref_a_multi, protocolo="A", tarefa="multiclasse", aumento="nenhum", **base)
    esperar(rod.semc0_a, protocolo="A", tarefa="binario", sem_c0=True, permutado=False)
    esperar(rod.semc0_b, protocolo="B", tarefa="binario", sem_c0=True, permutado=False)
    esperar(rod.norm_a, protocolo="A", tarefa="binario", norm_clipe=True, permutado=False)
    esperar(rod.norm_b, protocolo="B", tarefa="binario", norm_clipe=True, permutado=False)
    for d in rod.permutacao:
        esperar(d, protocolo="B", tarefa="binario", permutado=True, aumento="nenhum")
    for d in rod.curva:
        esperar(d, tarefa="binario", permutado=False, aumento="nenhum")
        p = d["parametros"]
        if p.get("segundos_treino") is None or p.get("amostragem_treino") != "por_gravacao":
            erros.append(f"{d['id']}: não é ponto da curva estratificada por gravação")

    consolidadas = [d for d in rod.todas() if d is not rod.taxa_alt]
    for campo in IGUAIS:
        valores = {_comparavel(d["parametros"], campo) for d in consolidadas}
        if len(valores) != 1:
            erros.append(f"{campo} difere entre as rodadas: {', '.join(sorted(valores))}")
    ids = [d["id"] for d in rod.todas()]
    if len(set(ids)) != len(ids):
        erros.append("o mesmo id aparece em mais de um papel")

    if rod.aum_completo:
        # mesmas conferências do run_tabela_aumento: referência sem aumento, completo só
        # difere pela semente, ablações sem controles
        try:
            run_tabela_aumento.conferir((rod.ref_b["id"], rod.ref_b),
                                        [(d["id"], d) for d in rod.aum_completo],
                                        [(d["id"], d) for d in rod.aum_ablacoes])
        except SystemExit as e:
            erros.append(str(e))
        if rod.aum_permutado:
            esperar(rod.aum_permutado, protocolo="B", tarefa="binario", permutado=True)
            chaves = ("aumento", "aumento_copias", "aumento_modo_estir", "aumento_desloc_max_s",
                      "aumento_estir_taxas", "aumento_snr_db")
            ref_aum = rod.aum_completo[0]["parametros"]
            for k in chaves:
                if str(rod.aum_permutado["parametros"].get(k)) != str(ref_aum.get(k)):
                    erros.append(f"{rod.aum_permutado['id']}: {k} difere do aumento completo")

    if rod.taxa_alt:
        # mesma partição e taxa da referência do B, e não um controle
        esperar(rod.taxa_ref, protocolo="B", tarefa="binario", aumento="nenhum", **base)
        try:
            run_tabela_taxas.conferir([(d["id"], d) for d in (rod.taxa_ref, rod.taxa_alt)])
        except SystemExit as e:
            erros.append(str(e))
    if erros:
        raise SystemExit("Abortado: rodadas não são as esperadas:\n  " + "\n  ".join(erros))


def conferir_registry(rod: Rodadas, registry: dict[str, dict]) -> None:
    """Toda rodada tem linha no registry, do mesmo script, com as mesmas métricas."""
    erros = []
    for d in rod.todas():
        linha = registry.get(d["id"])
        if linha is None:
            erros.append(f"{d['id']}: sem linha no registry")
            continue
        if linha["script"] != "validation/run_protocol.py":
            erros.append(f"{d['id']}: registry diz script {linha['script']}")
        p_reg = _kv(linha["parametros"])
        if p_reg.get("splits") != str(d["parametros"].get("splits")):
            erros.append(f"{d['id']}: splits do registry ({p_reg.get('splits')}) ≠ metrics.json")
        m_reg = _kv(linha["metricas"])
        mapa = METRICAS_B if d["parametros"]["protocolo"] == "B" else METRICAS_A
        for chave, caminho in mapa.items():
            v_json = float(_pegar(d["resumo"], caminho))
            v_reg = float(m_reg[chave]) if chave in m_reg else None
            ok = v_reg is not None and abs(v_reg - v_json) <= TOL_REGISTRY
            rod.conferencias.append(Conferencia(d["id"], chave, v_json, v_reg, ok))
            if not ok:
                erros.append(f"{d['id']}: {chave} registry={v_reg} metrics.json={v_json:.4f}")
    if erros:
        raise SystemExit("Abortado: registry e metrics.json não batem:\n  " + "\n  ".join(erros))


def carregar_harmonicos(pasta: Path, exp_id: str, pastas: dict[str, str]) -> dict:
    caminho = pasta / pastas[exp_id] / "metrics.json"
    if not caminho.exists():
        raise SystemExit(f"Abortado: não achei {caminho} ({exp_id})")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    d["id"], d["_pasta"] = exp_id, pastas[exp_id]
    return d


def conferir_harmonicos(rod: Rodadas, registry: dict[str, dict]) -> None:
    """
    Controle dos harmônicos do eixo (inspect_lda_harmonicos.py), que não é rodada do
    run_protocol: confere a linha do registry (script, partição, commit das features,
    intervenção ligada só no exp da intervenção) e as métricas contra o metrics.json,
    e que a referência da intervenção é o mesmo B da meta.
    """
    if not rod.harm_decomp:
        return
    erros = []
    for d, interv in ((rod.harm_decomp, False), (rod.harm_interv, True)):
        linha = registry.get(d["id"])
        if linha is None:
            erros.append(f"{d['id']}: sem linha no registry")
            continue
        if linha["script"] != SCRIPT_HARMONICOS:
            erros.append(f"{d['id']}: registry diz script {linha['script']}")
        p_reg = _kv(linha["parametros"])
        if p_reg.get("splits") != str(d.get("splits")) or d.get("splits") != rod.ref_b["parametros"]["splits"]:
            erros.append(f"{d['id']}: partição diferente da referência do B")
        if p_reg.get("features_commit") != str(d.get("features_commit")):
            erros.append(f"{d['id']}: features_commit do registry ≠ metrics.json")
        if (p_reg.get("intervencao") == "True") != interv or (d.get("intervencao") is not None) != interv:
            erros.append(f"{d['id']}: {'deveria' if interv else 'não deveria'} ser a intervenção")
            continue
        m_reg = _kv(linha["metricas"])
        mapas = [(METRICAS_HARM_DECOMP, TOL_REGISTRY_3)]
        if interv:
            mapas.append((METRICAS_HARM_INTERV, TOL_REGISTRY))
        for mapa, tol in mapas:
            for chave, caminho in mapa.items():
                v_json = float(_pegar(d, caminho))
                v_reg = float(m_reg[chave]) if chave in m_reg else None
                ok = v_reg is not None and abs(v_reg - v_json) <= tol
                rod.conferencias.append(Conferencia(d["id"], chave, v_json, v_reg, ok))
                if not ok:
                    erros.append(f"{d['id']}: {chave} registry={v_reg} metrics.json={v_json:.4f}")
    if rod.harm_interv and not erros:
        iv = rod.harm_interv["intervencao"]
        if p := _kv(registry[rod.harm_interv["id"]]["parametros"]).get("n_sorteios"):
            if int(p) != len(iv["sorteios"]):
                erros.append(f"{rod.harm_interv['id']}: n_sorteios do registry ≠ metrics.json")
        a, b = iv["referencia"], rod.ref_b["resumo"]
        for k in ("acuracia_balanceada_media", "sensibilidade_media", "especificidade_media"):
            if abs(a[k] - b[k]) > 1e-9:
                erros.append(f"{rod.harm_interv['id']}: referência da intervenção ≠ {rod.ref_b['id']} ({k})")
    if erros:
        raise SystemExit("Abortado: controle dos harmônicos não confere:\n  " + "\n  ".join(erros))


def montar_rodadas(pasta: Path, ids: dict, taxas: bool = True) -> Rodadas:
    def um(k):
        return carregar(pasta, ids[k])

    def varios(k):
        return [carregar(pasta, e) for e in ids[k]]

    return Rodadas(
        ref_b=um("ref_b"), reproducoes_b=varios("reproducoes_b"), ref_a=um("ref_a"),
        ref_a_multi=um("ref_a_multi"), semc0_a=um("semc0_a"), semc0_b=um("semc0_b"),
        norm_a=um("norm_a"), norm_b=um("norm_b"), permutacao=varios("permutacao"),
        curva=varios("curva"),
        taxa_ref=um("taxa_ref") if taxas else None, taxa_alt=um("taxa_alt") if taxas else None,
        # os grupos abaixo são opcionais: sem a chave em `ids`, ficam de fora
        aum_completo=[carregar(pasta, e) for e in ids.get("aum_completo", [])],
        aum_ablacoes=[carregar(pasta, e) for e in ids.get("aum_ablacoes", [])],
        aum_permutado=carregar(pasta, ids["aum_permutado"]) if ids.get("aum_permutado") else None,
        harm_decomp=(carregar_harmonicos(pasta, ids["harm_decomp"], ids.get("harm_pastas", PASTAS_HARMONICOS))
                     if ids.get("harm_decomp") else None),
        harm_interv=(carregar_harmonicos(pasta, ids["harm_interv"], ids.get("harm_pastas", PASTAS_HARMONICOS))
                     if ids.get("harm_interv") else None),
    )


# ---------------------------------------------------------------------------
# números
# ---------------------------------------------------------------------------

def folds_b(d: dict) -> dict[tuple[str, int], dict]:
    """(falha de fora, bloco normal) → fold."""
    return {(f["info"]["falha_de_fora"], int(f["info"]["bloco_normal"])): f for f in d["folds"]}


def _contagem(fracao: float, n: int, onde: str) -> int:
    c = fracao * n
    if abs(c - round(c)) > 1e-6:
        raise SystemExit(f"Abortado: {onde}: {fracao} × {n} não é inteiro")
    return int(round(c))


def matriz_confusao_b(d: dict) -> dict[str, dict[str, int]]:
    """
    Classe verdadeira → {"normal": n, "falha": n}, somando os 24 folds do B.

    Reconstruída de sensibilidade × n_teste_falha e especificidade × n_teste_normal
    de cada fold. A soma conta cada segmento tantas vezes quantas ele é testado:
    cada segmento de falha, 6 vezes (uma por bloco normal); cada segmento normal,
    4 vezes (uma por falha deixada de fora).
    """
    m = {c: {"normal": 0, "falha": 0} for c in CLASSES}
    for f in d["folds"]:
        falha = f["info"]["falha_de_fora"]
        vp = _contagem(f["sensibilidade"], f["n_teste_falha"], f["nome"])
        vn = _contagem(f["especificidade"], f["n_teste_normal"], f["nome"])
        m[falha]["falha"] += vp
        m[falha]["normal"] += f["n_teste_falha"] - vp
        m["normal"]["normal"] += vn
        m["normal"]["falha"] += f["n_teste_normal"] - vn
    return m


def resumo_permutacao(perm: list[dict], ref: dict) -> dict:
    a = np.array([x["resumo"]["acuracia_balanceada_media"] for x in perm])
    obs = ref["resumo"]["acuracia_balanceada_media"]
    acima = int((a >= obs - 1e-12).sum())
    return {"valores": a, "media": float(a.mean()), "desvio": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max()), "n": len(a),
            "acima": acima, "p_emp": (1 + acima) / (len(a) + 1)}


def resumo_curva(curva: list[dict]) -> dict[str, dict[float, dict]]:
    """protocolo → segundos → {media, desvio, n, sens, espec, bpfo03}."""
    grupos: dict[tuple[str, float], list[dict]] = {}
    for d in curva:
        p = d["parametros"]
        grupos.setdefault((p["protocolo"], float(p["segundos_treino"])), []).append(d["resumo"])
    out: dict[str, dict[float, dict]] = {}
    for (prot, s), rs in sorted(grupos.items()):
        def ms(k, rs=rs):
            v = [r[k] for r in rs]
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        e = {"n": len(rs), "ab": ms("acuracia_balanceada_media")}
        if prot == "B":
            e["sens"] = ms("sensibilidade_media")
            e["espec"] = ms("especificidade_media")
            e["bpfo03"] = (statistics.mean(r["por_falha"]["bpfo_0.3mm"]["acuracia_balanceada"] for r in rs),)
        out.setdefault(prot, {})[s] = e
    return out


# ---------------------------------------------------------------------------
# formatação
# ---------------------------------------------------------------------------

def br(x: float, casas: int = 3) -> str:
    """Número com vírgula decimal, como no relatório."""
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "$-$")


def br_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tt(classe: str) -> str:
    return r"\texttt{" + classe.replace("_", r"\_") + "}"


def _faixa(ids: list[str]) -> str:
    """exp031–exp050, exp059–exp138: ids contíguos viram intervalos."""
    if not ids:
        return "—"
    ns = sorted(int(i.removeprefix("exp")) for i in ids)
    partes, ini = [], ns[0]
    for a, b in zip(ns, ns[1:] + [None]):
        if b != a + 1:
            partes.append(f"exp{ini:03d}" if ini == a else f"exp{ini:03d}–exp{a:03d}")
            ini = b
    return ", ".join(partes)


def _tex_faixa(ids: list[str]) -> str:
    return r"\texttt{" + _faixa(ids).replace("–", "--") + "}"


def tex_protocolo_b(rod: Rodadas) -> str:
    d = rod.ref_b
    fb = folds_b(d)
    blocos = sorted({b for _, b in fb})
    r = d["resumo"]
    linhas = []
    for falha in FALHAS:
        pf = r["por_falha"][falha]
        cel = " & ".join(br(fb[(falha, b)]["acuracia_balanceada"]) for b in blocos)
        linhas.append(f"{tt(falha)} & {cel} & {br(pf['sensibilidade'])} & "
                      f"{br(pf['especificidade'])} & {br(pf['acuracia_balanceada'])} \\\\")
    media_bloco = " & ".join(
        br(statistics.mean(fb[(f, b)]["acuracia_balanceada"] for f in FALHAS)) for b in blocos)
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Protocolo B, \emph{fold} a \emph{fold}: acurácia balanceada por falha deixada "
        r"de fora e bloco normal no teste.}",
        r"\label{tab:protocoloB-folds}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X " + "C{0.95cm} " * len(blocos)
        + r"C{1.15cm} C{1.15cm} C{1.15cm}}",
        r"\toprule",
        r" & \multicolumn{" + str(len(blocos)) + r"}{c}{\textbf{Bloco normal no teste}} & "
        r"\multicolumn{3}{c}{\textbf{Média dos blocos}} \\",
        r"\cmidrule(lr){2-" + str(1 + len(blocos)) + r"}\cmidrule(lr){"
        + f"{2 + len(blocos)}-{4 + len(blocos)}" + "}",
        r"\textbf{Falha de fora} & " + " & ".join(f"\\textbf{{{b + 1}}}" for b in blocos)
        + r" & \textbf{Sens.} & \textbf{Espec.} & \textbf{AB} \\",
        r"\midrule",
        *linhas,
        r"\midrule",
        f"\\textbf{{Média}} & {media_bloco} & \\textbf{{{br(r['sensibilidade_media'])}}} & "
        f"\\textbf{{{br(r['especificidade_media'])}}} & "
        f"\\textbf{{{br(r['acuracia_balanceada_media'])}}} \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def tex_protocolo_a(rod: Rodadas) -> str:
    a, m = rod.ref_a, rod.ref_a_multi
    fm = {f["info"]["bloco"]: f for f in m["folds"]}
    linhas = []
    for f in sorted(a["folds"], key=lambda f: f["info"]["bloco"]):
        b = f["info"]["bloco"]
        g = fm[b]
        menor = min(g["recall_por_classe"].values())
        linhas.append(f"{b + 1} & {f['n_teste']} & {br(f['sensibilidade'])} & "
                      f"{br(f['especificidade'])} & {br(f['acuracia_balanceada'])} & "
                      f"{br(g['acuracia_balanceada'])} & {br(menor)} \\\\")
    ra, rm = a["resumo"], m["resumo"]
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Protocolo A, \emph{fold} a \emph{fold} (limite superior otimista, não usado "
        r"para aferir a meta).}",
        r"\label{tab:protocoloA-folds}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X C{1.4cm} C{1.4cm} C{1.4cm} C{2.4cm} C{2.4cm} C{1.9cm}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{Binário}} & \multicolumn{2}{c}{\textbf{Multiclasse}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"\textbf{Bloco no teste} & \textbf{Teste} & \textbf{Sens.} & \textbf{Espec.} & "
        r"\textbf{AB} & \textbf{AB} & \textbf{Menor acerto} \\",
        r"\midrule",
        *linhas,
        r"\midrule",
        f"\\textbf{{Média $\\pm$ desvio}} & & & & {br(ra['acuracia_balanceada_media'])} $\\pm$ "
        f"{br(ra['acuracia_balanceada_desvio'])} & {br(rm['acuracia_balanceada_media'])} $\\pm$ "
        f"{br(rm['acuracia_balanceada_desvio'])} & \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def tex_matriz(rod: Rodadas, m: dict) -> str:
    linhas = []
    for c in CLASSES:
        n = m[c]["normal"] + m[c]["falha"]
        cel = [f"{br_int(m[c][p])} ({br(100 * m[c][p] / n, 1)}\\%)" for p in ("normal", "falha")]
        linhas.append(f"{tt(c)} & {br_int(n)} & {cel[0]} & {cel[1]} \\\\")
        if c == "normal":
            linhas.append(r"\midrule")
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Matriz de confusão do Protocolo B, somada nos 24 \emph{folds} (porcentagens "
        r"por linha).}",
        r"\label{tab:confusaoB}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X C{2.4cm} C{3.4cm} C{3.4cm}}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Previsto}} \\",
        r"\cmidrule(lr){3-4}",
        r"\textbf{Classe verdadeira} & \textbf{Testes} & \textbf{normal} & \textbf{falha} \\",
        r"\midrule",
        *linhas,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def tex_controles(rod: Rodadas, perm: dict) -> str:
    """Ablação do ganho e permutação. A curva de aprendizado tem tabela própria no relatório
    (`tab:curva`) e está na figura dos controles; os ids das rodadas ficam no
    rastreabilidade.md e no texto."""
    def b(d):
        r = d["resumo"]
        return (f"{br(r['acuracia_balanceada_media'])} & {br(r['sensibilidade_media'])} & "
                f"{br(r['especificidade_media'])} & "
                f"{br(r['por_falha']['bpfo_0.3mm']['acuracia_balanceada'])}")

    def a(d):
        return br(d["resumo"]["acuracia_balanceada_media"])

    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Ablação do ganho e permutação de rótulos (AB: acurácia balanceada).}",
        r"\label{tab:controles-completa}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X C{1.5cm} C{2.3cm} C{1.3cm} "
        r"C{1.3cm} C{1.7cm}}",
        r"\toprule",
        r" & \textbf{A} & \multicolumn{4}{c}{\textbf{B}} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-6}",
        r"\textbf{Rodada} & \textbf{AB} & \textbf{AB} & \textbf{Sens.} & \textbf{Espec.} & "
        r"\textbf{BPFO 0,3~mm} \\",
        r"\midrule",
        f"Referência (MFCC completo) & {a(rod.ref_a)} & {b(rod.ref_b)} \\\\",
        f"Sem o coeficiente $c_0$ & {a(rod.semc0_a)} & {b(rod.semc0_b)} \\\\",
        f"Normalização RMS por segmento & {a(rod.norm_a)} & {b(rod.norm_b)} \\\\",
        f"Rótulos permutados por bloco ({perm['n']} sorteios) & -- & {br(perm['media'])} $\\pm$ "
        f"{br(perm['desvio'])} & -- & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def tex_taxas(rod: Rodadas) -> str:
    def linha(d):
        r = d["resumo"]
        por = " & ".join(br(r["por_falha"][f]["acuracia_balanceada"]) for f in FALHAS)
        return (f"{br_int(int(d['parametros']['fs_hz']))}~Hz & {br(r['acuracia_balanceada_media'])} & "
                f"{br(r['sensibilidade_media'])} & {br(r['especificidade_media'])} & {por} \\\\")

    def dif(a, b):
        v = b - a
        return "0" if abs(v) < 5e-4 else br(v)

    ra, rb = rod.taxa_ref["resumo"], rod.taxa_alt["resumo"]
    difs = " & ".join([dif(ra[k], rb[k]) for k in ("acuracia_balanceada_media", "sensibilidade_media",
                                                   "especificidade_media")]
                      + [dif(ra["por_falha"][f]["acuracia_balanceada"], rb["por_falha"][f]["acuracia_balanceada"])
                         for f in FALHAS])
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Protocolo B a duas taxas de amostragem. Colunas por falha: AB com aquela "
        r"falha deixada de fora.}",
        r"\label{tab:taxasB}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X C{1.15cm} C{1.15cm} C{1.2cm} "
        r"C{1.35cm} C{1.35cm} C{1.35cm} C{1.35cm}}",
        r"\toprule",
        r"\textbf{Taxa} & \textbf{AB} & \textbf{Sens.} & \textbf{Espec.} & \textbf{BPFI 0,3~mm} & "
        r"\textbf{BPFI 1,0~mm} & \textbf{BPFO 0,3~mm} & \textbf{BPFO 1,0~mm} \\",
        r"\midrule",
        linha(rod.taxa_ref),
        linha(rod.taxa_alt),
        r"\midrule",
        f"Diferença & {difs} \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def resumo_aumento(rs: list[dict]) -> dict:
    """Média entre rodadas (sementes) e o maior desvio entre elas, em qualquer coluna."""
    def col(f):
        v = [f(r["resumo"]) for r in rs]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    cols = {"ab": col(lambda r: r["acuracia_balanceada_media"]),
            "sens": col(lambda r: r["sensibilidade_media"]),
            "espec": col(lambda r: r["especificidade_media"]),
            **{f: col(lambda r, f=f: r["por_falha"][f]["acuracia_balanceada"]) for f in FALHAS}}
    return {"media": {k: v[0] for k, v in cols.items()}, "desvio_max": max(v[1] for v in cols.values()),
            "n": len(rs)}


def tex_aumento(rod: Rodadas) -> str:
    """
    Mesma tabela do relatório (tab:res-aumento), com as três técnicas em 10 sementes. Uma
    coluna por falha em 2 casas; em negrito, a falha que muda ≥ 0,1 em relação à referência
    numa rodada que não é controle.
    """
    ref = resumo_aumento([rod.ref_b])["media"]

    def linha(rotulo, rs, negrito_ab=False, controle=False):
        x = resumo_aumento(rs)["media"]
        ab = br(x["ab"])
        cel = []
        for f in FALHAS:
            v = br(x[f], 2)
            if not controle and abs(x[f] - ref[f]) >= 0.1:
                v = f"\\textbf{{{v}}}"
            cel.append(v)
        return (f"{rotulo} & {f'\\textbf{{{ab}}}' if negrito_ab else ab} & {br(x['sens'])} & "
                f"{br(x['espec'])} & " + " & ".join(cel) + " \\\\")

    nomes = {"deslocamento": "Só deslocamento", "estiramento": r"Só estiramento (\emph{tempo})",
             "ruido": "Só ruído"}

    def rotulo(d):
        p = d["parametros"]
        t = p["aumento"].split("+")
        if len(t) == 1:
            return nomes.get(t[0], t[0])
        return r"Três técnicas, estiramento \emph{velocidade}" if p.get("aumento_modo_estir") == "velocidade" \
            else "Três técnicas"

    completo = resumo_aumento(rod.aum_completo)
    linhas = [linha("Sem aumento (referência)", [rod.ref_b]),
              linha(f"\\textbf{{Três técnicas}} ({completo['n']} sementes)", rod.aum_completo, negrito_ab=True)]
    so = [d for d in rod.aum_ablacoes if len(d["parametros"]["aumento"].split("+")) == 1]
    outras = [d for d in rod.aum_ablacoes if d not in so]
    linhas += [linha(rotulo(d), [d]) for d in so]
    if rod.aum_permutado:
        linhas.append(linha("Três técnicas, rótulos permutados", [rod.aum_permutado], controle=True))
    linhas += [linha(rotulo(d), [d]) for d in outras]
    copias = rod.aum_completo[0]["parametros"].get("aumento_copias")
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{Protocolo B com e sem aumento de dados (LDA, {'quatro' if str(copias) == '4' else copias} "
        r"variantes por segmento). Colunas por falha: acurácia balanceada com aquela falha deixada de "
        r"fora.}",
        r"\label{tab:res-aumento}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{X C{1.1cm} C{1.1cm} C{1.1cm} C{1.3cm} C{1.3cm} C{1.3cm} C{1.3cm}}",
        r"\toprule",
        r"\textbf{Rodada} & \textbf{AB} & \textbf{Sens.} & \textbf{Espec.} & \textbf{BPFI 0,3~mm} & "
        r"\textbf{BPFI 1,0~mm} & \textbf{BPFO 0,3~mm} & \textbf{BPFO 1,0~mm} \\",
        r"\midrule",
        *linhas,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


def tex_harmonicos(rod: Rodadas) -> str:
    """Mesma tabela do relatório (tab:harmonicos), a partir do exp da intervenção."""
    iv = rod.harm_interv["intervencao"]

    def linha(rotulo, r):
        return (f"{rotulo} & {br(r['acuracia_balanceada_media'])} & {br(r['sensibilidade_media'])} & "
                f"{br(r['especificidade_media'])} & "
                f"{br(r['por_falha']['bpfo_0.3mm']['acuracia_balanceada'])} \\\\")

    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Intervenção nas bandas dos harmônicos do eixo (Protocolo B).}",
        r"\label{tab:harmonicos}",
        r"\begin{tabularx}{\textwidth}{X C{1.8cm} C{1.6cm} C{1.6cm} C{2.2cm}}",
        r"\toprule",
        r"\textbf{Rodada} & \textbf{AB} & \textbf{Sens.} & \textbf{Espec.} & \textbf{\texttt{bpfo\_0.3mm}} \\",
        r"\midrule",
        linha("Referência", iv["referencia"]),
        linha("Bandas dos harmônicos apagadas", iv["sem_harmonicos"]),
        f"Mesmo número de bandas sorteadas fora dos harmônicos ({len(iv['sorteios'])} sorteios) & "
        f"{br(iv['acc_bal_sorteios_min'])} a {br(iv['acc_bal_sorteios_max'])} & -- & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ])


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------

AZUL, LARANJA = "#2a78d6", "#eb6834"          # paleta categórica validada (slots 1 e 2)
TINTA, TINTA_2, GRADE = "#1f1f1e", "#5f5e58", "#d9d8d2"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2,
        "ytick.color": TINTA_2, "text.color": TINTA, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    })
    virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    return plt, virgula


def fig_matriz(m: dict, caminho: Path) -> None:
    plt, _ = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    frac = np.array([[m[c]["normal"], m[c]["falha"]] for c in CLASSES], dtype=float)
    cont = frac.copy()
    frac /= frac.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(len(CLASSES)):
        for j in range(2):
            ax.text(j, i, f"{int(cont[i, j]):,}".replace(",", ".") + f"\n{100 * frac[i, j]:.1f} %".replace(".", ","),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if frac[i, j] > 0.55 else TINTA)
    ax.set_xticks([0, 1], ["normal", "falha"])
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("previsto")
    ax.set_ylabel("classe verdadeira")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.axhline(0.5, color="white", linewidth=2)
    ax.axvline(0.5, color="white", linewidth=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def fig_controles(rod: Rodadas, perm: dict, curva: dict, caminho: Path) -> None:
    plt, virgula = _plt()
    from matplotlib.ticker import MaxNLocator
    ref_b = rod.ref_b["resumo"]["acuracia_balanceada_media"]
    ref_a = rod.ref_a["resumo"]["acuracia_balanceada_media"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) permutação
    bordas = np.arange(0.30, 0.95, 0.025)
    ax1.hist(perm["valores"], bins=bordas, color=AZUL, edgecolor="white", linewidth=1)
    ax1.axvline(0.5, color=TINTA_2, linestyle=":", linewidth=1.2)
    ax1.axvline(ref_b, color=LARANJA, linewidth=2)
    ymax = ax1.get_ylim()[1]
    ax1.text(0.5, ymax * 0.97, " acaso", color=TINTA_2, va="top", fontsize=8)
    ax1.text(ref_b, ymax * 0.97, f"referência\n{ref_b:.3f} ".replace(".", ","), color=TINTA,
             ha="right", va="top", fontsize=8)
    ax1.set_xlabel("acurácia balanceada (Protocolo B)")
    ax1.set_ylabel("sorteios")
    ax1.set_title(f"(a) Rótulos permutados por bloco, {perm['n']} sorteios", loc="left")
    ax1.xaxis.set_major_formatter(virgula)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(axis="x", visible=False)

    # (b) curva de aprendizado
    for prot, cor, ref, marca in (("A", LARANJA, ref_a, "s"), ("B", AZUL, ref_b, "o")):
        pts = curva.get(prot, {})
        if not pts:
            continue
        s = np.array(sorted(pts))
        mu = np.array([pts[x]["ab"][0] for x in s])
        sd = np.array([pts[x]["ab"][1] for x in s])
        ax2.fill_between(s, mu - sd, np.minimum(mu + sd, 1.0), color=cor, alpha=0.15, linewidth=0)
        ax2.plot(s, mu, color=cor, linewidth=2, marker=marca, markersize=5,
                 label=f"Protocolo {prot}" + (" (limite otimista)" if prot == "A" else ""))
        ax2.plot([s[-1], 59], [mu[-1], ref], color=cor, linewidth=1.2, linestyle="--")
        ax2.plot([59], [ref], color=cor, marker=marca, markersize=6, markerfacecolor="white",
                 markeredgewidth=1.6)
    ax2.axhline(0.85, color=TINTA_2, linestyle=":", linewidth=1.2)
    ax2.text(12, 0.857, "meta 0,85", color=TINTA_2, fontsize=8, va="bottom")
    ax2.set_xscale("log")
    ticks = sorted(set(curva.get("B", {})) | set(curva.get("A", {}))) + [59]
    ax2.set_xticks(ticks, [f"{t:g}" for t in ticks[:-1]] + ["todo\ntreino"])
    ax2.minorticks_off()
    ax2.set_ylim(0.6, 1.02)
    ax2.yaxis.set_major_formatter(virgula)
    ax2.set_xlabel("segundos de treino por classe")
    ax2.set_ylabel("acurácia balanceada")
    ax2.set_title("(b) Curva de aprendizado (média ± desvio)", loc="left")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def md_rastreabilidade(rod: Rodadas) -> str:
    papeis = [
        ("Protocolo B, referência (meta)", [rod.ref_b]),
        ("Protocolo B, reprodução", rod.reproducoes_b),
        ("Protocolo A binário (limite otimista)", [rod.ref_a]),
        ("Protocolo A multiclasse (limite otimista)", [rod.ref_a_multi]),
        ("Ablação: sem c0", [rod.semc0_a, rod.semc0_b]),
        ("Ablação: normalização RMS por segmento", [rod.norm_a, rod.norm_b]),
        ("Permutação por bloco", rod.permutacao),
        ("Curva de aprendizado", rod.curva),
    ] + ([("Comparação de taxas (5/6): 12,8 × 25,6 kHz", [rod.taxa_ref, rod.taxa_alt])]
         if rod.taxa_alt else []) + [
        ("Aumento (4/6): três técnicas, 10 sementes", rod.aum_completo),
        ("Aumento (4/6): técnica isolada e modo velocidade", rod.aum_ablacoes),
        ("Aumento (4/6): rótulos permutados", [d for d in (rod.aum_permutado,) if d]),
        ("Harmônicos do eixo: decomposição (inspect_lda_harmonicos.py)", [d for d in (rod.harm_decomp,) if d]),
        ("Harmônicos do eixo: intervenção (inspect_lda_harmonicos.py)", [d for d in (rod.harm_interv,) if d]),
    ]
    por_id: dict[str, list[Conferencia]] = {}
    for c in rod.conferencias:
        por_id.setdefault(c.exp_id, []).append(c)
    texto = [
        "# Rastreabilidade dos números consolidados",
        "",
        "Gerado por `scripts/validation/run_consolidacao.py`. Cada rodada usada nas tabelas e "
        "figuras de `reports/validation/consolidacao/` tem linha no `experiments/registry.csv`, "
        "com o mesmo script e a mesma partição, e as métricas da linha batem com o `metrics.json` "
        "(tolerância de 6e-5 para as gravadas com 4 casas e de 6e-4 para as de 3 casas da "
        "decomposição dos harmônicos). As rodadas do `run_protocol.py` são conferidas pelo "
        "`resumo`; o controle dos harmônicos (`inspect_lda_harmonicos.py`, que grava fora das "
        "pastas `expNNN_*`), pelo `resumo` e pelo bloco `intervencao`, e a referência da "
        "intervenção tem que ser o B da meta. Se alguma não bater, o script aborta sem gravar nada.",
        "",
        "## Rodadas por papel",
        "",
        "| papel | rodadas | métricas conferidas | resultado |",
        "|---|---|---:|---|",
    ]
    for papel, ds in papeis:
        if not ds:
            continue
        cs = [c for d in ds for c in por_id.get(d["id"], [])]
        texto.append(f"| {papel} | {_faixa([d['id'] for d in ds])} | {len(cs)} | "
                     f"{'todas batem' if all(c.ok for c in cs) else 'DIVERGE'} |")
    texto += [
        "",
        "## Rodadas citadas uma a uma (exceto permutação, curva e as 10 sementes do aumento)",
        "",
        "| rodada | pasta | métrica | registry | metrics.json |",
        "|---|---|---|---:|---:|",
    ]
    soltas = [rod.ref_b, *rod.reproducoes_b, rod.ref_a, rod.ref_a_multi, rod.semc0_a,
              rod.semc0_b, rod.norm_a, rod.norm_b] + [d for d in (rod.taxa_ref, rod.taxa_alt) if d] \
        + rod.aum_ablacoes + [d for d in (rod.aum_permutado, rod.harm_decomp, rod.harm_interv) if d]
    for d in soltas:
        for c in por_id.get(d["id"], []):
            texto.append(f"| {c.exp_id} | `{d['_pasta']}` | `{c.chave}` | {c.valor_registry:.4f} "
                         f"| {c.valor_json:.4f} |")
    texto.append("")
    return "\n".join(texto)


def md_consolidacao(rod: Rodadas, m: dict, perm: dict, curva: dict) -> str:
    r = rod.ref_b["resumo"]
    fb = folds_b(rod.ref_b)
    blocos = sorted({b for _, b in fb})
    vp = sum(m[f]["falha"] for f in FALHAS)
    fn = sum(m[f]["normal"] for f in FALHAS)
    texto = [
        "# Resultados consolidados do classificador",
        "",
        "Gerado por `scripts/validation/run_consolidacao.py` a partir dos `metrics.json` e do "
        "`registry.csv`; não treina nem registra. LDA sobre média e desvio dos 13 MFCC, "
        f"partição `{rod.ref_b['parametros']['splits']}`, "
        f"{rod.ref_b['parametros']['fs_hz']} Hz, sem aumento. "
        "Versões LaTeX das tabelas nos `.tex` desta pasta; origem de cada número em "
        "`rastreabilidade.md`.",
        "",
        "## O número da meta",
        "",
        f"Acurácia balanceada média do Protocolo B: **{r['acuracia_balanceada_media']:.3f}** "
        f"(meta ≥ 0.85: {'atingida' if r['acuracia_balanceada_media'] >= 0.85 else 'não atingida'}), "
        f"com sensibilidade média {r['sensibilidade_media']:.3f}, especificidade média "
        f"{r['especificidade_media']:.3f}, pior falha `{r['pior_falha']}` "
        f"({r['pior_falha_acuracia_balanceada']:.3f}) e pior fold `{r['pior_fold']}` "
        f"({r['pior_fold_acuracia_balanceada']:.3f}). Rodada {rod.ref_b['id']}"
        + (f", reproduzida por {', '.join(x['id'] for x in rod.reproducoes_b)}." if rod.reproducoes_b else "."),
        "",
        "## Protocolo B, fold a fold (acurácia balanceada)",
        "",
        "| falha de fora | " + " | ".join(f"bloco {b + 1}" for b in blocos) + " | sensib. | especif. | acc. bal. |",
        "|---|" + "---:|" * (len(blocos) + 3),
    ]
    for f in FALHAS:
        pf = r["por_falha"][f]
        texto.append(f"| {f} | " + " | ".join(f"{fb[(f, b)]['acuracia_balanceada']:.3f}" for b in blocos)
                     + f" | {pf['sensibilidade']:.3f} | {pf['especificidade']:.3f} "
                       f"| {pf['acuracia_balanceada']:.3f} |")
    texto.append("| **média** | " + " | ".join(
        f"{statistics.mean(fb[(f, b)]['acuracia_balanceada'] for f in FALHAS):.3f}" for b in blocos)
        + f" | **{r['sensibilidade_media']:.3f}** | **{r['especificidade_media']:.3f}** "
          f"| **{r['acuracia_balanceada_media']:.3f}** |")
    ra, rm = rod.ref_a["resumo"], rod.ref_a_multi["resumo"]
    texto += [
        "",
        "## Protocolo A — limite superior otimista (não conta para a meta)",
        "",
        f"Binário ({rod.ref_a['id']}): {ra['acuracia_balanceada_media']:.3f} ± "
        f"{ra['acuracia_balanceada_desvio']:.3f} (mín. {ra['acuracia_balanceada_min']:.3f}). "
        f"Multiclasse ({rod.ref_a_multi['id']}): {rm['acuracia_balanceada_media']:.3f} ± "
        f"{rm['acuracia_balanceada_desvio']:.3f} (mín. {rm['acuracia_balanceada_min']:.3f}). "
        "Treino e teste vêm das mesmas gravações. Fold a fold em `tab_protocoloA_folds.tex`.",
        "",
        "## Matriz de confusão do B (somada nos 24 folds)",
        "",
        "Cada segmento de falha é testado 6 vezes (uma por bloco normal) e cada segmento normal "
        "4 vezes (uma por falha deixada de fora).",
        "",
        "| classe verdadeira | testes | previsto normal | previsto falha |",
        "|---|---:|---:|---:|",
    ]
    for c in CLASSES:
        n = m[c]["normal"] + m[c]["falha"]
        texto.append(f"| {c} | {n} | {m[c]['normal']} ({100 * m[c]['normal'] / n:.1f} %) "
                     f"| {m[c]['falha']} ({100 * m[c]['falha'] / n:.1f} %) |")
    texto += [
        "",
        f"VP = {vp}, FN = {fn}, VN = {m['normal']['normal']}, FP = {m['normal']['falha']}. "
        "Figura: `fig_matriz_confusao_B.png`.",
        "",
        "## Controles",
        "",
        "| rodada | A (acc. bal.) | B (acc. bal.) | B, sensib. | B, especif. | B, bpfo_0.3mm | registro |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for nome, a, b in (("referência", rod.ref_a, rod.ref_b), ("sem c0", rod.semc0_a, rod.semc0_b),
                       ("normalização RMS por segmento", rod.norm_a, rod.norm_b)):
        rb = b["resumo"]
        texto.append(f"| {nome} | {a['resumo']['acuracia_balanceada_media']:.3f} "
                     f"| {rb['acuracia_balanceada_media']:.3f} | {rb['sensibilidade_media']:.3f} "
                     f"| {rb['especificidade_media']:.3f} "
                     f"| {rb['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.3f} | {a['id']}, {b['id']} |")
    texto.append(f"| permutação por bloco ({perm['n']} sorteios) | — | {perm['media']:.3f} ± "
                 f"{perm['desvio']:.3f} | — | — | — | {_faixa([x['id'] for x in rod.permutacao])} |")
    faixa_curva = _faixa([x["id"] for x in rod.curva])
    for s in sorted(set(curva.get("A", {})) | set(curva.get("B", {}))):
        ea, eb = curva.get("A", {}).get(s), curva.get("B", {}).get(s)
        cel_a = "{:.3f} ± {:.3f}".format(*ea["ab"]) if ea else "—"
        if eb:
            cel_b = "{:.3f} ± {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(
                *eb["ab"], eb["sens"][0], eb["espec"][0], eb["bpfo03"][0])
        else:
            cel_b = "— | — | — | —"
        texto.append(f"| curva: {s:g} s por classe | {cel_a} | {cel_b} | {faixa_curva} |")
    texto += [
        "",
        f"Permutação: p empírico da referência {perm['p_emp']:.4f} ({perm['acima']} de {perm['n']} "
        f"sorteios ≥ {r['acuracia_balanceada_media']:.3f}); mín. {perm['min']:.3f}, máx. {perm['max']:.3f}. "
        "Figura: `fig_controles.png`.",
        "",
    ]
    if rod.aum_completo:
        ra = resumo_aumento(rod.aum_completo)
        x = ra["media"]
        texto += ["## Aumento de dados (tarefa 4/6)", "",
                  f"Três técnicas (deslocamento, estiramento por phase vocoder e ruído) em "
                  f"{ra['n']} sementes do sorteio das variantes ({_faixa([d['id'] for d in rod.aum_completo])}): "
                  f"acc. bal. {x['ab']:.3f}, sensib. {x['sens']:.3f}, especif. {x['espec']:.3f}, "
                  f"bpfo_0.3mm {x['bpfo_0.3mm']:.3f}; desvio máximo entre sementes {ra['desvio_max']:.3f}. "
                  f"Referência sem aumento: {rod.ref_b['id']}. Conferido pelo `conferir` do "
                  "`run_tabela_aumento.py` (tabela completa em `reports/validation/tabela_aumento.md`).",
                  "",
                  "| rodada | id | acc. bal. | sensib. | especif. | " + " | ".join(FALHAS) + " |",
                  "|---|---|---:|---:|---:|" + "---:|" * len(FALHAS)]
        grupos = [("sem aumento", [rod.ref_b]), (f"três técnicas ({ra['n']} sementes)", rod.aum_completo)]
        grupos += [(d["parametros"]["aumento"] + ("" if d["parametros"].get("aumento_modo_estir") != "velocidade"
                    else ", estiramento velocidade"), [d]) for d in rod.aum_ablacoes]
        if rod.aum_permutado:
            grupos.append(("três técnicas, rótulos permutados", [rod.aum_permutado]))
        for nome, ds in grupos:
            y = resumo_aumento(ds)["media"]
            texto.append(f"| {nome} | {_faixa([d['id'] for d in ds])} | {y['ab']:.3f} | {y['sens']:.3f} "
                         f"| {y['espec']:.3f} | " + " | ".join(f"{y[f]:.3f}" for f in FALHAS) + " |")
        texto.append("")
    if rod.harm_interv:
        iv = rod.harm_interv["intervencao"]
        rd = rod.harm_decomp["resumo"] if rod.harm_decomp else rod.harm_interv["resumo"]
        texto += ["## Harmônicos do eixo (controle)", "",
                  f"Intervenção ({rod.harm_interv['id']}): com as bandas dos harmônicos apagadas, o B fica em "
                  f"{iv['sem_harmonicos']['acuracia_balanceada_media']:.3f} (referência "
                  f"{iv['referencia']['acuracia_balanceada_media']:.3f}); apagando o mesmo número de bandas "
                  f"sorteadas fora delas ({len(iv['sorteios'])} sorteios), {iv['acc_bal_sorteios_min']:.3f} a "
                  f"{iv['acc_bal_sorteios_max']:.3f}. Decomposição por banda "
                  f"({rod.harm_decomp['id'] if rod.harm_decomp else rod.harm_interv['id']}): as bandas dos "
                  f"harmônicos somam {rd['fracao_separacao_harm_mediana']:.2f} da separação (mediana dos folds), "
                  f"contra {rd['fracao_de_bandas_mediana']:.0%} das bandas; o desvio dos coeficientes responde "
                  f"por {rd['fracao_desvio_mfcc_mediana']:.0%}.",
                  ""]
    if rod.taxa_alt:
        texto += ["## Taxa de amostragem (tarefa 5/6)", "",
                  "Mesma LDA e mesmo MFCC, features no mesmo commit; as 20 bandas Mel vão até o "
                  "Nyquist de cada taxa. Partições conferidas em segundos pelo "
                  "`run_tabela_taxas.py` (tabela completa em `reports/validation/tabela_taxas.md`).",
                  "",
                  "| taxa | rodada | acc. bal. | sensib. | especif. | " + " | ".join(FALHAS) + " |",
                  "|---|---|---:|---:|---:|" + "---:|" * len(FALHAS)]
        for d in (rod.taxa_ref, rod.taxa_alt):
            x = d["resumo"]
            texto.append(f"| {d['parametros']['fs_hz']} Hz | {d['id']} | {x['acuracia_balanceada_media']:.3f} "
                         f"| {x['sensibilidade_media']:.3f} | {x['especificidade_media']:.3f} | "
                         + " | ".join(f"{x['por_falha'][f]['acuracia_balanceada']:.3f}" for f in FALHAS) + " |")
        texto.append("")
    return "\n".join(texto)


# ---------------------------------------------------------------------------

def gerar(pasta: Path, registry: Path, saida: Path, ids: dict, taxas: bool = True,
          figuras: bool = True, raiz: Path = Path(".")) -> Rodadas:
    """`raiz` é a raiz do repositório, de onde saem as partições versionadas das taxas."""
    rod = montar_rodadas(pasta, ids, taxas)
    conferir_parametros(rod)
    if rod.taxa_alt:
        run_tabela_taxas.conferir_particoes(
            [(d["id"], d) for d in (rod.taxa_ref, rod.taxa_alt)],
            [raiz / config.arquivo_splits(int(d["parametros"]["fs_hz"])) for d in (rod.taxa_ref, rod.taxa_alt)])
    reg = ler_registry(registry)
    conferir_registry(rod, reg)
    conferir_harmonicos(rod, reg)

    m = matriz_confusao_b(rod.ref_b)
    perm = resumo_permutacao(rod.permutacao, rod.ref_b)
    curva = resumo_curva(rod.curva)

    saida.mkdir(parents=True, exist_ok=True)
    arquivos = {
        "tab_protocoloB_folds.tex": tex_protocolo_b(rod),
        "tab_protocoloA_folds.tex": tex_protocolo_a(rod),
        "tab_matriz_confusao_B.tex": tex_matriz(rod, m),
        "tab_controles.tex": tex_controles(rod, perm),
        "rastreabilidade.md": md_rastreabilidade(rod),
        "consolidacao.md": md_consolidacao(rod, m, perm, curva),
    }
    if rod.taxa_alt:
        arquivos["tab_taxas_B.tex"] = tex_taxas(rod)
    if rod.aum_completo:
        arquivos["tab_aumento_B.tex"] = tex_aumento(rod)
    if rod.harm_interv:
        arquivos["tab_harmonicos.tex"] = tex_harmonicos(rod)
    for nome, texto in arquivos.items():
        (saida / nome).write_text(texto, encoding="utf-8")
    if figuras:
        fig_matriz(m, saida / "fig_matriz_confusao_B.png")
        fig_controles(rod, perm, curva, saida / "fig_controles.png")
    return rod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--saida", type=Path, default=Path("reports/validation/consolidacao"))
    ap.add_argument("--ref-b", default=PADRAO["ref_b"])
    ap.add_argument("--ref-a", default=PADRAO["ref_a"])
    ap.add_argument("--ref-a-multi", default=PADRAO["ref_a_multi"])
    ap.add_argument("--taxa-referencia", default=PADRAO["taxa_ref"],
                    help="rodada do B à taxa de trabalho, no mesmo commit da outra taxa")
    ap.add_argument("--taxa-alternativa", default=PADRAO["taxa_alt"],
                    help="rodada do B a outra taxa (tarefa 5/6)")
    ap.add_argument("--sem-taxas", action="store_true", help="não inclui a comparação de taxas")
    args = ap.parse_args()

    ids = {**PADRAO, "ref_b": args.ref_b, "ref_a": args.ref_a, "ref_a_multi": args.ref_a_multi,
           "taxa_ref": args.taxa_referencia, "taxa_alt": args.taxa_alternativa}
    rod = gerar(args.pasta, args.registry, args.saida, ids, taxas=not args.sem_taxas)
    print((args.saida / "consolidacao.md").read_text(encoding="utf-8"))
    n = len({c.exp_id for c in rod.conferencias})
    print(f"{len(rod.conferencias)} métricas de {n} rodadas conferidas contra o registry.")
    print(f"saídas: {args.saida}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())