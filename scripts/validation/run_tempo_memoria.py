#!/usr/bin/env python3
"""
run_tempo_memoria.py — tempo de execução e ocupação de memória do firmware final

Card "Medir tempo de execução e ocupação de memória da cadeia embarcada" (Fase 3,
seção 3.5 do relatório final). Junta três medidas do firmware de `main` e escreve
a tabela da seção, o resumo e a linha do registry:

1. **Memória** (build padrão do firmware): a saída do `tools/orcamento_memoria.py
   --markdown` (flash e RAM por parte, pelo .map do linker), lado a lado com a
   Tabela `tab:res_flash` do 2º parcial.
2. **Pilha**: a conta estática do `tools/pilha_estatica.py` (pior caminho pelo
   grafo de chamadas, com as interrupções) e a marca d'água medida na placa
   (linhas `med` do build de medida), contra a reserva de 4 KB do linker.
3. **Tempo**, nos dois modos, da captura do build de medida (`-DMEDIDA=ON`):
   (a) áudio da flash, do int16 a 51,2 kHz ao escore (linha `decim ciclos` do
   boot), mais o envio do `$SEG`; (b) segmento recebido pela serial, com a E/S
   separada do processamento pelas linhas `med` (espera do PC, tempo de linha,
   lacunas, trabalho da CPU com os bytes, resposta).

O critério de tempo real (Kuo, Lee e Tian, 2013, como na proposta): tempo de
processamento mais o de E/S menor que o período de amostragem. Em (a), o período
é o da amostra a 51,2 kHz; em (b), a entrada não chega no ritmo do áudio, e sim no
da linha serial, então o período que importa é o de um byte na linha (10 bits no
baud real) e a folga do anel do DMA. O resumo diz os dois.

    python scripts/validation/run_tempo_memoria.py \\
        --captura reports/embarcado/capturas/tempo-memoria-AAAA-MM-DD.log \\
        --orcamento reports/embarcado/tempo_memoria/orcamento.md \\
        --pilha reports/embarcado/tempo_memoria/pilha_estatica.txt \\
        --firmware-commit <hash do main do firmware> --responsavel Clara

Com --sem-registro, não escreve no registry (ensaio).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import experimentos

FLASH_TOTAL = 512 * 1024
RAM_TOTAL = 128 * 1024
RESERVA_PILHA = 4096
FS_ENTRADA = 51200          # taxa da gravação original (modo a)
FS_CADEIA = 12800           # taxa da cadeia (modo b: o quadro já chega decimado)
ANEL_RX = 2048              # anel do DMA da recepção (README do firmware)

# Firmware do 2º parcial (e492ef6, PR #10, 22/10; gcc 13.2.1, Debug) pelas partes do
# tools/orcamento_memoria.py de hoje: os totais são os da Tabela tab:res_flash do
# relatório, e as partes foram conferidas recompilando o e492ef6 com o script atual
# (16/11). A linha "HAL, inicialização e aplicação base" do relatório aparece aqui
# separada em duas, porque é a HAL que cresce com a recepção por DMA.
P2 = {
    "clipes a 12,8 kHz": (256215, 0),
    "recorte a 51,2 kHz": (103284, 0),
    "cadeia (MFCC)": (1432, 10788),
    "cadeia completa na placa (cadeia_clipes.c)": (3569, 3446),
    "decimacao (decimacao.c; a FIR esta na CMSIS-DSP)": (6108, 7125),
    "marco e LDA": (37352, 0),
    "pacote da serial": (1398, 0),
    "CMSIS-DSP": (13028, 0),
    "HAL e CubeMX": (6025, 13),
    "aplicacao base": (2224, 5220),
    "biblioteca C": (9856, 408),
    "reserva de heap e pilha": (0, 4612),
    "outros e alinhamento": (177, 36),
}
P2_TOTAL = (440668, 31648)
P2_LINKER = (440676, 31648)

# Partes do script -> linha da tabela do relatório (pt-BR, como na tab:res_flash)
ROTULOS = [
    ("clipes a 12,8 kHz", "Clipes a 12,8~kHz (5 gravações $\\times$ 2~s)"),
    ("recorte a 51,2 kHz", "Recorte a 51,2~kHz (1~s, com as margens)"),
    ("cadeia (MFCC)", "Extração de características (MFCC)"),
    ("cadeia completa na placa (cadeia_clipes.c)", "Cadeia completa na placa (verificação e laço)"),
    ("decimacao (decimacao.c; a FIR esta na CMSIS-DSP)", "Decimação e inferência de ponta a ponta"),
    ("recepcao pela serial (recepcao.c e recepcao_serial.c)", "Recepção pela serial (anel do DMA e parser)"),
    ("CRC-32 comum (crc32.c)", "CRC-32"),
    ("marco e LDA", "Classificador e vetores de teste"),
    ("pacote da serial", "Pacote da serial"),
    ("CMSIS-DSP", "CMSIS-DSP"),
    ("HAL e CubeMX", "HAL e inicialização (CubeMX)"),
    ("aplicacao base", "Aplicação base (\\texttt{main.c}, FFT de teste)"),
    ("biblioteca C", "Biblioteca C"),
    ("reserva de heap e pilha", "Reserva mínima de \\emph{heap} e pilha"),
    ("outros e alinhamento", "Outros e alinhamento"),
]

RE_KV = re.compile(r"(\w+)=([^\s]+)")


# --------------------------------------------------------------------------- #
# Leitura                                                                     #
# --------------------------------------------------------------------------- #
def numero(txt: str) -> int:
    return int(txt.replace(".", "").replace("*", "").strip())


def ler_orcamento(caminho: Path) -> dict[str, tuple[int, int]]:
    """A tabela markdown do orcamento_memoria.py, já agrupada como a do relatório."""
    partes = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in linha.strip().strip("|").split("|")]
        if len(c) != 3 or not re.fullmatch(r"\**[\d.]+\**", c[1] or "x"):
            continue
        nome = c[0].strip("*")
        if nome.startswith("reserva de heap e pilha"):
            nome = "reserva de heap e pilha"
        partes[nome] = (numero(c[1]), numero(c[2]))
    if "total" not in partes:
        raise SystemExit(f"{caminho}: sem a linha de total (é a saída do orcamento_memoria.py --markdown?)")
    agrupar = {"outros e alinhamento": ("outros", "alinhamento")}
    for novo, velhos in agrupar.items():
        pares = [partes.pop(v, (0, 0)) for v in velhos]
        partes[novo] = (sum(p[0] for p in pares), sum(p[1] for p in pares))
    if "build de medida (orcamento.c; nao vai para o firmware final)" in partes:
        raise SystemExit("o orçamento é do build -DORCAMENTO=ON; meça o build padrão")
    return partes


def ler_pilha(caminho: Path) -> dict[str, int]:
    txt = caminho.read_text(encoding="utf-8")
    m = re.search(r"main (\d+) B \+ interrupcoes (\d+) B = (\d+) B", txt)
    r = re.search(r"so o modo de recepcao: (\d+) B \+ (\d+) B = (\d+) B", txt)
    if not (m and r):
        raise SystemExit(f"{caminho}: não achei o pior caso (é a saída do tools/pilha_estatica.py?)")
    return {"main": int(m.group(1)), "interrupcoes": int(m.group(2)), "pior": int(m.group(3)),
            "recepcao": int(r.group(3))}


def ler_captura(caminho: Path) -> dict:
    linhas = caminho.read_text(encoding="utf-8", errors="replace").splitlines()
    cap = {"clock_mhz": None, "decim": None, "cadeia": None, "seg_d": [], "inicio": None,
           "pares": [], "erros": 0, "estouros": 0}
    rx = None
    for l in linhas:
        if l.startswith("$CAB,"):
            campos = l.split("*")[0].split(",")
            cap["clock_mhz"] = int(campos[5])
        elif l.startswith("decim ciclos ") and cap["decim"] is None:
            kv = dict(RE_KV.findall(l))
            cap["decim"] = {k: int(kv[k]) for k in ("total", "decimacao", "cadeia")}
        elif l.startswith("cadeia resumo ") and cap["cadeia"] is None:
            m = re.search(r"ciclos min=(\d+) med=(\d+) max=(\d+)", l)
            cap["cadeia"] = tuple(int(x) for x in m.groups()) if m else None
        elif l.startswith("$SEG,") and ",D," in l:
            cap["seg_d"].append(len(l) + 2)       # + \r\n
        elif l.startswith("med inicio "):
            kv = dict(RE_KV.findall(l))
            cap["inicio"] = {k: int(v) for k, v in kv.items()}
        elif l.startswith("rx erro"):
            cap["erros"] += 1
            rx = None
        elif l.startswith("rx grupo="):
            rx = {k: v for k, v in RE_KV.findall(l)}
        elif l.startswith("$SEG,") and ",S," in l and rx is not None:
            rx["seg_bytes"] = len(l) + 2
        elif l.startswith("med grupo=") and rx is not None:
            med = {k: int(v) for k, v in RE_KV.findall(l)}
            if (str(med["grupo"]), str(med["id"])) == (rx["grupo"], rx["id"]):
                if int(rx.get("estouros", 0)) > 0:
                    cap["estouros"] += 1      # regra 7: não vale como medida
                else:
                    cap["pares"].append({"rx": rx, "med": med})
            rx = None
    if cap["inicio"] is None or not cap["pares"]:
        raise SystemExit(f"{caminho}: sem linhas med (o firmware foi gravado com -DMEDIDA=ON?)")
    if cap["decim"] is None:
        raise SystemExit(f"{caminho}: sem a linha 'decim ciclos' do boot (a captura começa no RESET?)")
    return cap


# --------------------------------------------------------------------------- #
# Contas                                                                      #
# --------------------------------------------------------------------------- #
def resumo_estat(valores: list[float]) -> dict:
    return {"min": min(valores), "med": statistics.median(valores), "max": max(valores),
            "n": len(valores)}


def calcular(cap: dict, orc: dict, pilha: dict) -> dict:
    clock = (cap["clock_mhz"] or 100) * 1_000_000
    ini = cap["inicio"]
    baud = ini["baud_real"]
    cpb = ini["ciclos_por_byte"]
    ms = lambda ciclos: 1000.0 * ciclos / clock

    # (a) áudio da flash
    d = cap["decim"]
    seg_d_bytes = statistics.median(cap["seg_d"]) if cap["seg_d"] else 45
    es_a_us = 1e6 * seg_d_bytes * 10 / baud
    a = {
        "ciclos_total": d["total"], "ciclos_decimacao": d["decimacao"], "ciclos_cadeia": d["cadeia"],
        "proc_ms": ms(d["total"]), "es_us": es_a_us, "seg_bytes": seg_d_bytes,
        "tempo_real_pct": 100.0 * (ms(d["total"]) + es_a_us / 1000) / 1000.0,
        "ciclos_por_amostra": d["total"] / FS_ENTRADA, "orcamento_por_amostra": clock / FS_ENTRADA,
        "cadeia_clipes": cap["cadeia"],
    }

    # (b) segmento pela serial
    por = {k: [] for k in ("espera", "linha", "lacunas", "lacuna_max", "consumir", "ciclos", "bytes_cpu",
                           "consumir_max", "resposta", "ciclo", "ocupacao", "pilha", "n_bytes", "ms")}
    for p in cap["pares"]:
        rx, m = p["rx"], p["med"]
        ciclos = int(rx["ciclos"])
        n_bytes = int(rx["bytes"])
        por["espera"].append(m["espera_us"] / 1000)
        por["linha"].append(m["linha_us"] / 1000)
        por["lacunas"].append(m["lacunas_us"] / 1000)
        por["lacuna_max"].append(m["lacuna_max_us"] / 1000)
        por["consumir"].append(ms(m["consumir"]))
        por["ciclos"].append(ciclos)
        # o consumir() faz os quadros do MFCC e o resumo; o ciclos da rx tem esses e a
        # LDA (~240 ciclos). A diferença é o trabalho byte a byte (cópia, CRC, amostras)
        por["bytes_cpu"].append(ms(max(m["consumir"] - ciclos, 0)))
        por["consumir_max"].append(m["consumir_max_us"] / 1000)
        por["resposta"].append(m["resposta_us"] / 1000)
        por["ciclo"].append((m["espera_us"] + m["linha_us"] + m["resposta_us"]) / 1000)
        por["ocupacao"].append(100.0 * m["consumir"] / max(m["linha_us"] * clock / 1e6, 1))
        por["pilha"].append(m["pilha"])
        por["n_bytes"].append(n_bytes)
        por["ms"].append(int(rx["ms"]))
    est = {k: resumo_estat(v) for k, v in por.items()}
    n_bytes = int(statistics.median(por["n_bytes"]))
    linha_teorica_ms = 1000.0 * n_bytes * 10 / baud
    b = {
        "est": est, "n": len(cap["pares"]), "n_bytes": n_bytes, "baud": baud, "ciclos_por_byte": cpb,
        "linha_teorica_ms": linha_teorica_ms,
        "cpu_por_byte": statistics.median(
            [p["med"]["consumir"] / int(p["rx"]["bytes"]) for p in cap["pares"]]),
        "anel_ms": 1000.0 * ANEL_RX * 10 / baud,
        "vezes_tempo_real": 1000.0 / est["ciclo"]["med"],
        "erros": cap["erros"], "estouros": cap["estouros"],
    }

    # memória e pilha
    tot = orc["total"]
    mem = {"partes": orc, "flash": tot[0], "ram": tot[1],
           "flash_livre": FLASH_TOTAL - tot[0], "ram_livre": RAM_TOTAL - tot[1],
           "p2": P2_TOTAL}
    pilha_med = max(por["pilha"] + [ini.get("pilha_boot", 0)])
    pl = dict(pilha, medida=pilha_med, medida_boot=ini.get("pilha_boot"), reserva=RESERVA_PILHA,
              pintada=ini.get("pilha_pintada"))
    return {"a": a, "b": b, "mem": mem, "pilha": pl, "clock": clock}


# --------------------------------------------------------------------------- #
# Saídas                                                                      #
# --------------------------------------------------------------------------- #
def br(x: float, casas: int = 1) -> str:
    """Número no formato do relatório: milhar com ponto, decimal com vírgula."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def tabela_memoria_tex(mem: dict) -> str:
    linhas = []
    for chave, rotulo in ROTULOS:
        f, r = mem["partes"].get(chave, (0, 0))
        f2, r2 = P2.get(chave, (None, None))
        p2f = br(f2, 0) if f2 is not None else "---"
        p2r = br(r2, 0) if r2 is not None else "---"
        linhas.append(f"{rotulo} & {p2f} & {br(f, 0)} & {p2r} & {br(r, 0)} \\\\")
    corpo = "\n".join(linhas)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{Ocupação de memória do firmware, por parte: 2\\textordmasculine\\ parcial (22/10) e firmware final, com a recepção pela serial (versão de depuração, pelo mapa de memória do \\emph{{linker}}).}}
\\label{{tab:final_memoria}}
\\small
\\begin{{tabular}}{{lrrrr}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{\\textbf{{Flash (B)}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{RAM (B)}}}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}
\\textbf{{Parte}} & \\textbf{{22/10}} & \\textbf{{Final}} & \\textbf{{22/10}} & \\textbf{{Final}} \\\\
\\midrule
{corpo}
\\midrule
\\textbf{{Total}} & \\textbf{{{br(P2_TOTAL[0], 0)}}} & \\textbf{{{br(mem['flash'], 0)}}} & \\textbf{{{br(P2_TOTAL[1], 0)}}} & \\textbf{{{br(mem['ram'], 0)}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tabela_tempo_tex(r: dict) -> str:
    a, b, e = r["a"], r["b"], r["b"]["est"]
    med = lambda k, c=1: br(e[k]["med"], c)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{Tempo por segundo de áudio nos dois modos do firmware final (DWT a 100~MHz; no modo (b), mediana dos {b['n']} segmentos da rodada pela serial).}}
\\label{{tab:final_tempo}}
\\small
\\begin{{tabular}}{{llr}}
\\toprule
\\textbf{{Modo}} & \\textbf{{Etapa}} & \\textbf{{Tempo (ms)}} \\\\
\\midrule
(a) Áudio da \\emph{{flash}} & Decimação (FIR, fator 4) & {br(1000 * a['ciclos_decimacao'] / r['clock'], 2)} \\\\
 & Cadeia (MFCC e LDA) & {br(1000 * a['ciclos_cadeia'] / r['clock'], 2)} \\\\
 & Envio do \\texttt{{\\$SEG}} ({br(a['seg_bytes'], 0)}~B) & {br(a['es_us'] / 1000, 2)} \\\\
 & \\textbf{{Total}} & \\textbf{{{br(a['proc_ms'] + a['es_us'] / 1000, 2)}}} \\\\
\\midrule
(b) Segmento pela serial & Espera do PC (do \\texttt{{\\$PED}} ao 1\\textordmasculine\\ byte) & {med('espera', 2)} \\\\
 & Linha ({br(b['n_bytes'], 0)}~B a {br(b['baud'], 0)}~bit/s; teórico {br(b['linha_teorica_ms'], 1)}) & {med('linha')} \\\\
 & \\quad das quais, linha parada entre pacotes & {med('lacunas')} \\\\
 & CPU durante a linha (MFCC e bytes) & {med('consumir')} \\\\
 & \\quad das quais, bytes (cópia, CRC, amostras) & {med('bytes_cpu', 2)} \\\\
 & Resposta (LDA, \\texttt{{rx}} e \\texttt{{\\$SEG}}) & {med('resposta', 2)} \\\\
 & \\textbf{{Ciclo por segmento}} (espera, linha e resposta) & \\textbf{{{med('ciclo')}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def resumo_md(r: dict, meta: dict) -> str:
    a, b, e, mem, pl = r["a"], r["b"], r["b"]["est"], r["mem"], r["pilha"]
    linhas_mem = "\n".join(
        f"| {chave} | {br(P2[chave][0], 0) if chave in P2 else '—'} | {br(mem['partes'].get(chave, (0, 0))[0], 0)} | "
        f"{br(P2[chave][1], 0) if chave in P2 else '—'} | {br(mem['partes'].get(chave, (0, 0))[1], 0)} |"
        for chave, _ in ROTULOS)
    t = lambda k, c=1: f"{br(e[k]['med'], c)} (mín. {br(e[k]['min'], c)}, máx. {br(e[k]['max'], c)})"
    cumpre_a = a["ciclos_por_amostra"] < a["orcamento_por_amostra"]
    cumpre_b = b["cpu_por_byte"] < b["ciclos_por_byte"] and e["consumir_max"]["max"] < b["anel_ms"]
    return f"""# Tempo de execução e ocupação de memória do firmware final

Captura `{meta['captura']}` (sha {meta['captura_sha']}), firmware `{meta['firmware_commit']}` com `-DMEDIDA=ON`
para o tempo e a pilha medida; memória e pilha estática do build padrão (Debug, `-DMEDIDA=OFF`).
Relógio de {r['clock'] // 1_000_000} MHz; baud real {br(b['baud'], 0)} bit/s ({b['ciclos_por_byte']} ciclos por byte de 10 bits).

## Memória (build padrão)

| Parte | Flash 22/10 | Flash final | RAM 22/10 | RAM final |
|---|---:|---:|---:|---:|
{linhas_mem}
| **total** | **{br(P2_TOTAL[0], 0)}** | **{br(mem['flash'], 0)}** | **{br(P2_TOTAL[1], 0)}** | **{br(mem['ram'], 0)}** |

- Flash: {br(mem['flash'], 0)} B de {br(FLASH_TOTAL, 0)} ({br(100 * mem['flash'] / FLASH_TOTAL)} %), {br(mem['flash'] - P2_TOTAL[0], 0)} B a mais que em 22/10; livres {br(mem['flash_livre'], 0)} B.
- RAM: {br(mem['ram'], 0)} B de {br(RAM_TOTAL, 0)} ({br(100 * mem['ram'] / RAM_TOTAL)} %), {br(mem['ram'] - P2_TOTAL[1], 0)} B a mais; livres {br(mem['ram_livre'], 0)} B.

## Pilha

- Estática (`tools/pilha_estatica.py`, pior caminho com as interrupções): **{pl['pior']} B** ({pl['main']} B do `main` e {pl['interrupcoes']} B de interrupções); só o modo de recepção, {pl['recepcao']} B.
- Medida na placa (marca d'água desde o boot): **{pl['medida']} B** (no fim do boot e da demonstração, {pl['medida_boot']} B).
- Reserva do linker: {pl['reserva']} B. Uso: {br(100 * max(pl['pior'], pl['medida']) / pl['reserva'], 0)} % no pior dos dois.

## (a) Áudio da flash, do int16 a 51,2 kHz ao escore

- Decimação {br(1000 * a['ciclos_decimacao'] / r['clock'], 2)} ms + cadeia {br(1000 * a['ciclos_cadeia'] / r['clock'], 2)} ms = **{br(a['proc_ms'], 2)} ms** por segundo de áudio ({br(a['ciclos_total'], 0)} ciclos).
- Saída: o `$SEG` ({br(a['seg_bytes'], 0)} B) leva {br(a['es_us'], 0)} µs na linha. Processamento e E/S: **{br(a['tempo_real_pct'])} %** do tempo real.
- Por amostra de entrada: {br(a['ciclos_por_amostra'], 0)} ciclos, contra {br(a['orcamento_por_amostra'], 0)} do período de amostragem a 51,2 kHz. **Critério de tempo real {'cumprido' if cumpre_a else 'NÃO cumprido'}.**
- Cadeia dos clipes no mesmo boot (min, mediana, máx): {', '.join(br(x, 0) for x in a['cadeia_clipes']) if a['cadeia_clipes'] else '—'} ciclos.

## (b) Segmento recebido pela serial ({b['n']} segmentos, {b['erros']} erros, {b['estouros']} com estouro descartados)

| Grandeza (ms) | mediana (mín., máx.) |
|---|---|
| espera do PC (fim do `$PED` ao 1º byte) | {t('espera', 2)} |
| linha (1º ao último byte) | {t('linha')} |
| linha teórica ({br(b['n_bytes'], 0)} B × 10 bits / {br(b['baud'], 0)}) | {br(b['linha_teorica_ms'])} |
| lacunas (linha parada) | {t('lacunas')} |
| maior lacuna | {t('lacuna_max', 2)} |
| CPU com o quadro (`consumir()`) | {t('consumir')} |
| ... dos quais, bytes (cópia, CRC, amostras) | {t('bytes_cpu', 2)} |
| maior chamada ao `consumir()` | {t('consumir_max', 2)} |
| resposta (LDA, `rx` e `$SEG`) | {t('resposta', 2)} |
| ciclo por segmento (espera + linha + resposta) | {t('ciclo')} |
| campo `ms` da `rx` (resolução de 1 ms) | {t('ms', 0)} |

- MFCC e LDA (campo `ciclos` da `rx`): mediana {br(e['ciclos']['med'], 0)} ciclos ({br(1000 * e['ciclos']['med'] / r['clock'], 2)} ms).
- A CPU fica ocupada {br(e['ocupacao']['med'])} % do tempo de linha; trabalha {br(b['cpu_por_byte'], 0)} ciclos por byte recebido, contra {b['ciclos_por_byte']} de um byte na linha.
- A maior chamada ao `consumir()` ({br(e['consumir_max']['max'], 2)} ms) fica abaixo dos {br(b['anel_ms'], 1)} ms em que o anel de 2 KB enche.
- Um segmento de 1 s de áudio a cada {br(e['ciclo']['med'])} ms: **{br(b['vezes_tempo_real'], 1)} vezes o tempo real**. **Critério {'cumprido' if cumpre_b else 'NÃO cumprido'}** no período que vale neste modo (o de um byte na linha).
"""


def linha_registry(r: dict, meta: dict, proximo: int, responsavel: str, notas: str) -> dict:
    a, b, e, mem, pl = r["a"], r["b"], r["b"]["est"], r["mem"], r["pilha"]
    return {
        "id": f"exp{proximo:03d}", "data": meta["data"], "etapa": "caracterizacao_embarcada",
        "script": "validation/run_tempo_memoria.py", "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "firmware_commit": meta["firmware_commit"], "build_memoria": "Debug_padrao",
            "build_tempo": "Debug_MEDIDA", "captura": meta["captura"], "captura_sha": meta["captura_sha"],
            "clock_mhz": r["clock"] // 1_000_000, "baud_real": b["baud"], "modelo": "exp239"}),
        "dataset": f"firmware_main;rodada_serial_{b['n']}seg;recorte_51200Hz_seg29",
        "metricas": experimentos.kv({
            "flash_b": mem["flash"], "ram_b": mem["ram"], "flash_livre_b": mem["flash_livre"],
            "ram_livre_b": mem["ram_livre"], "flash_delta_p2_b": mem["flash"] - P2_TOTAL[0],
            "ram_delta_p2_b": mem["ram"] - P2_TOTAL[1],
            "pilha_estatica_b": pl["pior"], "pilha_medida_b": pl["medida"], "pilha_reserva_b": pl["reserva"],
            "a_ciclos_total": a["ciclos_total"], "a_proc_ms": f"{a['proc_ms']:.2f}",
            "a_es_us": f"{a['es_us']:.0f}", "a_tempo_real_pct": f"{a['tempo_real_pct']:.2f}",
            "a_ciclos_por_amostra": f"{a['ciclos_por_amostra']:.0f}",
            "b_segmentos": b["n"], "b_espera_ms": f"{e['espera']['med']:.2f}",
            "b_linha_ms": f"{e['linha']['med']:.1f}", "b_linha_teorica_ms": f"{b['linha_teorica_ms']:.1f}",
            "b_lacunas_ms": f"{e['lacunas']['med']:.1f}", "b_cpu_ms": f"{e['consumir']['med']:.1f}",
            "b_bytes_cpu_ms": f"{e['bytes_cpu']['med']:.2f}", "b_mfcc_lda_ciclos": f"{e['ciclos']['med']:.0f}",
            "b_resposta_ms": f"{e['resposta']['med']:.2f}", "b_ciclo_ms": f"{e['ciclo']['med']:.1f}",
            "b_vezes_tempo_real": f"{b['vezes_tempo_real']:.2f}", "b_ocupacao_cpu_pct": f"{e['ocupacao']['med']:.1f}",
            "b_consumir_max_ms": f"{e['consumir_max']['max']:.2f}", "b_cpu_ciclos_por_byte": f"{b['cpu_por_byte']:.0f}",
            "b_erros": b["erros"], "b_estouros": b["estouros"]}),
        "responsavel": responsavel,
        "notas": ("tempo e pilha medidos no build -DMEDIDA=ON (linha med; rx e $SEG iguais ao padrão); "
                  "memória e pilha estática no build padrão. " + notas).strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--captura", type=Path, required=True, help="captura do build -DMEDIDA=ON, desde o RESET")
    ap.add_argument("--orcamento", type=Path, required=True, help="saída do orcamento_memoria.py --markdown")
    ap.add_argument("--pilha", type=Path, required=True, help="saída do tools/pilha_estatica.py")
    ap.add_argument("--firmware-commit", required=True, help="hash curto do main do firmware medido")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/embarcado/tempo_memoria"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true", help="ensaio: não escreve no registry")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    cap = ler_captura(args.captura)
    r = calcular(cap, ler_orcamento(args.orcamento), ler_pilha(args.pilha))
    meta = {"captura": args.captura.name, "firmware_commit": args.firmware_commit,
            "captura_sha": hashlib.sha256(args.captura.read_bytes()).hexdigest()[:12],
            "data": date.today().isoformat()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "resumo.md").write_text(resumo_md(r, meta), encoding="utf-8", newline="\n")
    (args.out_dir / "tabelas.tex").write_text(
        tabela_memoria_tex(r["mem"]) + "\n" + tabela_tempo_tex(r), encoding="utf-8", newline="\n")
    with (args.out_dir / "segmentos.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grupo", "id", "ms", "ciclos", "espera_us", "linha_us", "lacunas_us", "lacuna_max_us",
                    "consumir", "consumir_max_us", "resposta_us", "pilha"])
        for p in cap["pares"]:
            m = p["med"]
            w.writerow([p["rx"]["grupo"], p["rx"]["id"], p["rx"]["ms"], p["rx"]["ciclos"], m["espera_us"],
                        m["linha_us"], m["lacunas_us"], m["lacuna_max_us"], m["consumir"],
                        m["consumir_max_us"], m["resposta_us"], m["pilha"]])
    print((args.out_dir / "resumo.md").read_text(encoding="utf-8"))

    if not args.sem_registro:
        linha = linha_registry(r, meta, experimentos.next_exp_number(args.registry),
                               args.responsavel, args.notas)
        experimentos.append_registry(args.registry, [linha])
        print(f"registry: {linha['id']} em {args.registry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
