#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tabela_protocolo_c.py — fechamento do Protocolo C numa tabela única
=======================================================================

Fecha o card "Validação externa no HUSTmotor multi-modal (Protocolo C)": junta,
para o relatório final, as leituras do Protocolo C que já têm rodada no
registry. Lê os três `metrics.json` do `run_protocol_c.py` em
`reports/validation/protocolo_c/` e grava, na mesma pasta:

tabela_protocolo_c.md    a tabela com o alarme falso e a detecção por rotação,
                         o total, a decisão e a rodada de cada leitura
tab_protocoloC.tex       a mesma tabela no formato do relatório (vírgula decimal,
                         `tabularx`/`booktabs`), sem a coluna do total, com as
                         leituras que decidem em negrito e a nota do retreino

Na detecção, a tabela segue a regra da página "Adoção do D2": numa rotação em
que o detector alarma os normais (`deteccao_lida` falso no metrics.json), ela
sai entre parênteses e fica fora do total; nas etapas sem esse campo, que só
têm a decisão do alarme no total, toda a detecção da leitura sai entre
parênteses e o total sai como traço quando essa decisão reprova. Os alarmes em extrapolação
(5 e 30 Hz na rotação deixada de fora) saem entre colchetes.

Não treina nada nem escreve no registry.

Leituras, na ordem do protocolo
-------------------------------
1. Sem retreino: a LDA final do Jung aplicada ao HUST (`exp274`).
2. C1, transferência: cada detector com o modelo e o limiar do Jung (`exp275`–`exp278`).
3. C2, comissionamento: a normal do próprio HUST calibra, aquisições 1 e 2
   (`exp279`–`exp281`).
4. Recalibração da LDA pela normal da máquina nova (`exp287`–`exp291`).
5. Rotação deixada de fora: o D2 calibra com as outras três rotações (`exp296`–`exp300`).

A checagem do retreino da LDA (Amanda, 29/10) não tem rodada no registry: está
só na página "Retreinar a LDA no HUSTmotor" do Registro de Decisões, descartada.
Entra como nota abaixo da tabela, dita como fora do registry, e não como linha.

Rastreabilidade
---------------
Cada linha tem que ter rodada no registry, do `run_protocol_c.py`, com o mesmo
detector, não substituída por outra, e com as mesmas contagens do metrics.json:
alarme e detecção no total e a decisão do alarme; na recalibração, também a
detecção a 10–30 Hz, se passa no critério, e a de 5 Hz; na rotação deixada de
fora, também alarme, detecção e decisão por rotação e a decisão da página do D2.
Se alguma não bater, o script aborta antes de gravar qualquer saída. O registry
só grava por rotação a rotação deixada de fora: nas outras etapas, as colunas
por rotação saem só do metrics.json, conferidas pela soma com o total.

Os itens de leitura abaixo da tabela não citam números, que ficam só nas células
conferidas; a exceção é a nota do retreino, dita como fora do registry.

Uso
---
    python scripts/validation/run_tabela_protocolo_c.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROTACOES = (5, 10, 20, 30)
SCRIPT = "validation/run_protocol_c.py"


@dataclass(frozen=True)
class Linha:
    exp_id: str
    arquivo: str          # subpasta do metrics.json ("" = o do C1 e do C2)
    chave: str            # leitura dentro de "leituras"
    bloco: str            # "sozinho", "ou_lda" ou "combinado"
    etapa: str
    nome: str
    papel: str


LINHAS = (
    Linha("exp274", "", "LDA_sem_retreino", "sozinho", "Sem retreino", "LDA do Jung", "transferência"),
    Linha("exp275", "", "C1_D2", "sozinho", "C1, limiar do Jung", "D2", "reporte"),
    Linha("exp275", "", "C1_D2", "ou_lda", "C1, limiar do Jung", "D2 OU LDA", "reporte"),
    Linha("exp276", "", "C1_D1_nominal", "sozinho", "C1, limiar do Jung", "D1, rotação nominal", "reporte"),
    Linha("exp277", "", "C1_D1_estimado", "sozinho", "C1, limiar do Jung", "D1, rotação estimada", "reporte"),
    Linha("exp278", "", "C1_D4", "sozinho", "C1, limiar do Jung", "D4", "reporte"),
    Linha("exp279", "", "C2_D2", "sozinho", "C2, normal do HUST", "D2", "reporte"),
    Linha("exp279", "", "C2_D2", "ou_lda", "C2, normal do HUST", "D2 OU LDA", "reporte"),
    Linha("exp280", "", "C2_D1_nominal", "sozinho", "C2, normal do HUST", "D1, rotação nominal", "reporte"),
    Linha("exp281", "", "C2_D1_estimado", "sozinho", "C2, normal do HUST", "D1, rotação estimada", "reporte"),
    Linha("exp287", "recalibracao", "C2_LDA_recal_medias13", "sozinho", "LDA recalibrada (C2)",
          "só as 13 médias", "decide"),
    Linha("exp288", "recalibracao", "C2_LDA_recal_sem_c0", "sozinho", "LDA recalibrada (C2)",
          "sem o c0", "decide junto"),
    Linha("exp289", "recalibracao", "C2_LDA_recal_medias26", "sozinho", "LDA recalibrada (C2)",
          "as 26 médias", "reporte"),
    Linha("exp290", "recalibracao", "C2_LDA_recal_desvio", "sozinho", "LDA recalibrada (C2)",
          "com o desvio", "reporte"),
    Linha("exp291", "recalibracao", "C2_D2_ou_LDA_recal", "combinado", "LDA recalibrada (C2)",
          "D2 OU LDA recalibrada", "reporte"),
    Linha("exp296", "rotacao_fora", "RF_D2", "sozinho", "Rotação deixada de fora", "D2",
          "decide (adoção do D2)"),
    Linha("exp297", "rotacao_fora", "RF_D2_sem_c0", "sozinho", "Rotação deixada de fora",
          "D2 sem o c0", "reporte"),
    Linha("exp298", "rotacao_fora", "RF_D2_ou_LDA", "combinado", "Rotação deixada de fora",
          "D2 OU LDA", "reporte"),
    Linha("exp299", "rotacao_fora", "RF_D1_nominal", "sozinho", "Rotação deixada de fora",
          "D1, rotação nominal", "reporte"),
    Linha("exp300", "rotacao_fora", "RF_D1_estimado", "sozinho", "Rotação deixada de fora",
          "D1, rotação estimada", "reporte"),
)

RETREINO = ("Checagem do retreino da LDA (Amanda, 29/10), fora do registry: está só no Registro de "
            "Decisões, na página \"Retreinar a LDA no HUSTmotor\" (descartada) e na decisão 4 da página "
            "\"Recalibrar a LDA pela normal da máquina nova\". Rodou antes de o protocolo ser aprovado e "
            "serviu só para descartar o retreino. Acurácia balanceada com uma rotação deixada de fora: "
            "LDA 0,65 (1,00 / 1,00 / 0,50 / 0,10 a 5 / 10 / 20 / 30 Hz), SVM linear 0,88. No controle com "
            "rótulos trocados por arquivo, a LDA tem média 0,48 e percentil 95 de 0,82; nas 14 trocas "
            "possíveis, 4 igualam ou superam a LDA (valor-p ≈ 0,33) e 2 igualam o SVM (≈ 0,2). As trocas "
            "formam 7 pares com o mesmo valor (inverter todos os rótulos não muda a acurácia balanceada), "
            "e o valor-p nunca cai abaixo de 1/15 ≈ 0,07: o controle não teria como mostrar "
            "significância a 5 %. Dividindo dentro do "
            "arquivo, 0,94 a 1,00, que mede reconhecer a gravação.")

LEGENDA = ("Entre colchetes, alarmes em rotação fora da faixa calibrada (extrapolação): ficam fora da "
           "decisão e são reportados à parte; no D2 OU LDA não há colchetes, porque a LDA do Jung alarma "
           "em qualquer rotação, sem depender da calibração. Entre parênteses, detecção onde o detector "
           "alarma os normais (na rotação deixada de fora, rotação a rotação; nas outras etapas, pela "
           "decisão do total): não é lida como acerto.")


# ---------------------------------------------------------------------------

def ler_registry(caminho: Path) -> dict[str, dict]:
    with caminho.open(encoding="utf-8", newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def _kv(texto: str) -> dict[str, str]:
    return dict(par.split("=", 1) for par in (texto or "").split(";") if "=" in par)


def carregar(pasta: Path) -> dict[str, dict]:
    """Os três metrics.json do run_protocol_c.py, pela subpasta."""
    arquivos = {}
    for sub in sorted({ln.arquivo for ln in LINHAS}):
        caminho = pasta / sub / "metrics.json"
        if not caminho.exists():
            raise SystemExit(f"Abortado: falta {caminho} (rode o run_protocol_c.py).")
        arquivos[sub] = json.loads(caminho.read_text(encoding="utf-8"))
    return arquivos


def resultado(arquivos: dict, ln: Linha) -> dict:
    try:
        return arquivos[ln.arquivo]["leituras"][ln.chave]["resultado"][ln.bloco]
    except KeyError as e:
        raise SystemExit(f"Abortado: {ln.exp_id}: {ln.chave}/{ln.bloco} não está no metrics.json ({e}).")


def _kn(c: dict) -> str:
    return f"{c['k']}/{c['n']}"


def conferir(arquivos: dict, registry: dict[str, dict]) -> int:
    """Toda linha tem rodada no registry com as mesmas contagens; devolve quantas foram conferidas."""
    erros, n = [], 0
    for ln in LINHAS:
        reg = registry.get(ln.exp_id)
        if reg is None:
            erros.append(f"{ln.exp_id}: sem linha no registry")
            continue
        if reg["script"] != SCRIPT:
            erros.append(f"{ln.exp_id}: registry diz script {reg['script']}")
        if "substituída" in reg["notas"]:
            erros.append(f"{ln.exp_id}: substituída por outra rodada no registry")
        leitura = arquivos[ln.arquivo]["leituras"][ln.chave]
        if _kv(reg["parametros"]).get("detector") != leitura["detector"]:
            erros.append(f"{ln.exp_id}: detector do registry ≠ {leitura['detector']} do metrics.json")
        r, m = resultado(arquivos, ln), _kv(reg["metricas"])
        esperado = {f"alarme_{ln.bloco}": _kn(r["alarme"]), f"deteccao_{ln.bloco}": _kn(r["deteccao"])}
        chave_dec = f"decisao_{ln.bloco}" if f"decisao_{ln.bloco}" in m else f"decisao_total_{ln.bloco}"
        esperado[chave_dec] = r["alarme"]["decisao"]
        if "decisao_pagina_d2" in r:
            esperado["decisao_pagina_d2"] = r["decisao_pagina_d2"]
        if "deteccao_criterio" in r:            # recalibração: o critério de 12/15 a 10–30 Hz
            c = r["deteccao_criterio"]
            esperado |= {f"deteccao_10a30hz_{ln.bloco}": _kn(c),
                         f"passa_10a30hz_{ln.bloco}": "sim" if c["passa"] else "nao",
                         f"deteccao_5hz_{ln.bloco}": _kn(r["deteccao_fora_do_criterio"])}
        if ln.arquivo == "rotacao_fora":       # só aqui o registry grava por rotação
            for v, pr in r["por_rotacao"].items():
                esperado |= {f"alarme_{v}hz": _kn(pr["alarme"]), f"deteccao_{v}hz": _kn(pr["deteccao"]),
                             f"decisao_{v}hz": pr["alarme"]["decisao"]}
        for chave, valor in esperado.items():
            n += 1
            if m.get(chave) != valor:
                erros.append(f"{ln.exp_id}: {chave} registry={m.get(chave)} metrics.json={valor}")
        for campo in ("alarme", "deteccao"):    # as colunas por rotação somam o total conferido
            soma = [sum(pr[campo][x] for pr in r["por_rotacao"].values()) for x in ("k", "n")]
            if soma != [r[campo]["k"], r[campo]["n"]]:
                erros.append(f"{ln.exp_id}: {campo} por rotação soma {soma[0]}/{soma[1]}, "
                             f"total {_kn(r[campo])}")
    if erros:
        raise SystemExit("Abortado: registry e metrics.json não batem:\n  " + "\n  ".join(erros))
    return n


# ---------------------------------------------------------------------------

def _celula(pr: dict, campo: str, reprova: bool) -> str:
    """[k/n] no alarme em extrapolação; (k/n) na detecção que a página do D2 não lê.

    Sem `deteccao_lida`, a única decisão do alarme é a do total: se ela reprova,
    nenhuma rotação da leitura tem a detecção lida.
    """
    if campo == "alarme" and pr.get("papel") == "extrapolacao":
        return f"[{_kn(pr[campo])}]"
    if campo == "deteccao" and not pr.get("deteccao_lida", not reprova):
        return f"({_kn(pr[campo])})"
    return _kn(pr[campo])


def por_rotacao(r: dict, campo: str) -> list[str]:
    """k/n em 5, 10, 20 e 30 Hz; — onde a leitura não avalia a rotação (o D1 a 5 e a 10 Hz)."""
    pr, reprova = r["por_rotacao"], r["alarme"]["decisao"] == "reprova"
    return [_celula(pr[str(v)], campo, reprova) if str(v) in pr else "—" for v in ROTACOES]


def deteccao_total(r: dict) -> str:
    """Só a detecção lida: sem as rotações em que o detector alarma os normais; — se o alarme reprova."""
    pr = r["por_rotacao"].values()
    if any("deteccao_lida" in p for p in pr):
        lidas = [p["deteccao"] for p in pr if p["deteccao_lida"]]
        return _kn({x: sum(c[x] for c in lidas) for x in ("k", "n")}) if lidas else "—"
    return "—" if r["alarme"]["decisao"] == "reprova" else _kn(r["deteccao"])


def decisao(r: dict, resultado_recal: str | None = None) -> str:
    """O que a leitura decide, na regra de cada etapa; na recalibração, `resultado_recal` só na que decide."""
    if "deteccao_criterio" in r:               # recalibração: pelo menos 12/15 a 10–30 Hz, 5 Hz à parte
        criterio = f"{_kn(r['deteccao_criterio'])} a 10–30 Hz; 5 Hz à parte: {_kn(r['deteccao_fora_do_criterio'])}"
        return f"{resultado_recal}, com o controle sem o c0 ({criterio})" if resultado_recal else criterio
    pr = r["por_rotacao"]
    if "papel" not in next(iter(pr.values())):
        return r["alarme"]["decisao"]
    # rotação deixada de fora: o total é só reporte; na regra da página do D2, decidem 10 e 20 Hz
    lidas = [v for v in pr if pr[v]["papel"] == "decide"] if "decisao_pagina_d2" in r else list(pr)
    reprova = [v for v in lidas if pr[v]["alarme"]["decisao"] == "reprova"]
    if not reprova:
        return "inconclusivo"
    if len(reprova) == len(pr):
        return "reprova em todas"
    return "reprova a " + " e ".join(filter(None, (", ".join(reprova[:-1]), reprova[-1]))) + " Hz"


def _decisao(arquivos: dict, ln: Linha, r: dict) -> str:
    """A da recalibração sai uma vez, na leitura que decide, com o resultado do metrics.json."""
    recal = ln.arquivo == "recalibracao" and ln.papel == "decide"
    return decisao(r, arquivos[ln.arquivo]["resultado"] if recal else None)


def md(arquivos: dict, n_conferidas: int) -> str:
    rot = " / ".join(str(v) for v in ROTACOES)
    texto = [
        "# Protocolo C no HUSTmotor: tabela única",
        "",
        "Gerada pelo `run_tabela_protocolo_c.py` a partir dos `metrics.json` do `run_protocol_c.py`; "
        f"{n_conferidas} contagens conferidas contra o `registry.csv`. Alarme falso nos segmentos normais "
        "e detecção nos de falha, por rotação; a decisão é a do alarme (Clopper–Pearson unilateral de 95 %, "
        "`decidir()`), que só consegue reprovar.",
        "",
        f"| Etapa | Leitura | Papel | Alarme falso {rot} Hz | Detecção {rot} Hz | Alarme total "
        "| Detecção total | Decisão | Rodada |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for ln in LINHAS:
        r = resultado(arquivos, ln)
        texto.append(f"| {ln.etapa} | {ln.nome} | {ln.papel} | {' · '.join(por_rotacao(r, 'alarme'))} "
                     f"| {' · '.join(por_rotacao(r, 'deteccao'))} | {_kn(r['alarme'])} "
                     f"| {deteccao_total(r)} | {_decisao(arquivos, ln, r)} | `{ln.exp_id}` |")
    texto += [
        "",
        LEGENDA + " No total, a detecção entre parênteses fica de fora; traço quando não sobra nenhuma.",
        "",
        "- **Sem retreino e C1:** com o limiar do Jung, a LDA, o D2 e o D4 disparam nos normais do HUST; "
        "é o limite de transferência que o critério já previa. O D1 não dispara, mas detecta pouco.",
        "- **C2:** calibrado com a normal do HUST na mesma rotação, o D2 fica inconclusivo no alarme (o C2 "
        "só consegue reprovar) e detecta a 10–30 Hz. O combinado com a LDA do Jung herda os alarmes dela.",
        "- **LDA recalibrada:** zera os alarmes, mas não detecta a falha, nem sem o c0: a diferença entre "
        "as máquinas não é só de nível e microfone. A detecção do combinado com o D2 vem toda do D2.",
        "- **Rotação deixada de fora:** sem a normal da própria rotação, o D2 reprova a 20 Hz, dentro da "
        "faixa calibrada. Na decisão, só ele conta (página \"Adoção do D2\"); o resto é reporte. A 5 e "
        "a 30 Hz, os alarmes são extrapolação e não decidem; o achado é que o D2 não generaliza para fora "
        "da faixa calibrada.",
        "",
        "**Decisão (reunião de 03/11, página \"Adoção do D2\" Definida):** opção 1, o D2 não é adotado. "
        "Ele fica só como resultado em Python no relatório.",
        "",
        "## Fora do registry",
        "",
        RETREINO.replace("fora do registry", "**fora do registry**", 1),
        "",
    ]
    return "\n".join(texto)


def _tex(s: str) -> str:
    return (re.sub(r'"([^"]*)"', r"``\1''", s)
            .replace("≈ ", r"$\approx$ ").replace("—", "---").replace("–", "--").replace(" · ", r" $\cdot$ ").replace(" Hz", "~Hz")
            .replace("c0", r"$c_0$").replace("D2 OU", r"D2 \textsc{ou}").replace("D4 OU", r"D4 \textsc{ou}").replace(" %", r"~\%"))


def tex(arquivos: dict) -> str:
    linhas, etapa = [], None
    for ln in LINHAS:
        if ln.etapa != etapa:
            if etapa is not None:
                linhas.append(r"\midrule")
            linhas.append(r"\multicolumn{5}{l}{\textbf{" + _tex(ln.etapa) + r"}} \\")
            etapa = ln.etapa
        r = resultado(arquivos, ln)
        nome = _tex(ln.nome)
        if ln.papel.startswith("decide"):
            nome = r"\textbf{" + nome + "}"
        linhas.append(f"{nome} & {_tex(' · '.join(por_rotacao(r, 'alarme')))} & "
                      f"{_tex(' · '.join(por_rotacao(r, 'deteccao')))} & {_tex(_decisao(arquivos, ln, r))} & "
                      r"\texttt{" + ln.exp_id + r"} \\")
    return "\n".join([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Protocolo C no HUSTmotor: alarme falso nos segmentos normais e detecção nos de falha, "
        r"a 5, 10, 20 e 30~Hz. A decisão é a do alarme (Clopper--Pearson unilateral de 95~\%), que só "
        r"consegue reprovar. Em negrito, as leituras que decidem; as outras são reporte. "
        + _tex(LEGENDA) + "}",
        r"\label{tab:protocoloC}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.3cm} C{2.6cm} C{2.6cm} "
        r">{\raggedright\arraybackslash}X C{1.1cm}}",
        r"\toprule",
        r"\textbf{Leitura} & \textbf{Alarme falso} & \textbf{Detecção} & \textbf{Decisão} & "
        r"\textbf{Rodada} \\",
        r"\midrule",
        *linhas,
        r"\bottomrule",
        r"\end{tabularx}",
        "",
        r"\smallskip",
        r"\parbox{\textwidth}{\footnotesize " + _tex(RETREINO) + "}",
        r"\end{table}",
        "",
    ])


def gerar(pasta: Path, registry: Path) -> int:
    arquivos = carregar(pasta)
    n = conferir(arquivos, ler_registry(registry))
    saidas = {"tabela_protocolo_c.md": md(arquivos, n), "tab_protocoloC.tex": tex(arquivos)}
    for nome, texto in saidas.items():
        (pasta / nome).write_text(texto, encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=Path("reports/validation/protocolo_c"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    args = ap.parse_args()
    n = gerar(args.pasta, args.registry)
    print((args.pasta / "tabela_protocolo_c.md").read_text(encoding="utf-8"))
    print(f"{n} contagens de {len({ln.exp_id for ln in LINHAS})} rodadas conferidas contra o registry.")
    print(f"saídas: {args.pasta}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
