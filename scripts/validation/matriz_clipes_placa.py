#!/usr/bin/env python3
"""
matriz_clipes_placa.py -- matriz de confusão do sistema embarcado nos 10 segmentos dos clipes
=============================================================================================

É o item prometido no 2º parcial para a validação embarcada: a matriz de
confusão da placa sobre os 10 segmentos dos clipes gravados na flash, comparada
à das decisões de referência (Python, modelo exp239). Os 10 segmentos estão no
treino do exp239, então a matriz mede a FIDELIDADE do porte, e não o desempenho,
que continua sendo o do Protocolo B.

Entradas
--------
- A captura da serial da placa (por padrão, a de 22/10, com o firmware final:
  reports/embarcado/capturas/primeira-inferencia-22-10.log, cópia do
  docs/primeira-inferencia-22-10.log do repositório do firmware). Dela saem:
    * a verificação do boot (linhas "cadeia seg=..."), uma por segmento;
    * as voltas do laço (linhas "demo seg=..." e o $SEG de origem C de cada uma);
    * as inferências de ponta a ponta (linhas "ponta ..."), só como nota;
    * o $CAB, com o modelo e o hash do JSON gravados no firmware.
  As linhas "seg seg=..." do marco (classificador sozinho, com as características
  do Python, 295 segmentos) NÃO entram: aqui o assunto é a cadeia inteira.
- reports/modelo_final/escores_referencia.csv: classe verdadeira, escore em
  float64 e decisão do Python de cada segmento.

O que o script confere ANTES de gravar qualquer saída (se algo falhar, aborta)
-----------------------------------------------------------------------------
1. O CSV é o mesmo de que o firmware foi gerado (sha256, CLIPES_CSV_SHA256_12 do
   clipes_demo.h), e o $CAB tem o modelo e o JSON esperados.
2. A verificação do boot tem os 10 segmentos do clipes_demo.h, uma vez cada,
   todos com OK do próprio firmware.
3. O ref= que o firmware imprime (float32, 3 casas) é o escore do CSV.
4. Conferência independente da tolerância, contra o float64 do CSV: mesmo no
   pior arredondamento do texto (0,0005), |placa - ref| fica dentro de
   1e-5 x |ref|. A fração que vai para a tabela é a da placa (dif/tol medidos
   em float32 na própria placa, 6 casas), que é a exata; a conservadora sai no CSV.
5. Cada volta do laço tem os 10 segmentos, com o mesmo escore e a mesma decisão
   do boot (a conta é determinística).
6. Cada linha "demo" tem logo depois o $SEG de origem C com o mesmo clipe,
   segmento, escore e decisão; todos os pacotes com checksum certo e o contador
   sem salto.
7. As inferências de ponta a ponta (origem D) estão todas OK e iguais ao boot.

Saídas (em --out-dir, padrão reports/embarcado/clipes_22_10/)
---------------------------------------------------------------
- segmentos.csv   uma linha por segmento
- resumo.md       tabela, matrizes e conferências, para o card do Notion
- tabelas.tex     as duas tabelas no estilo do relatório (booktabs, vírgula decimal)
e uma linha no experiments/registry.csv (etapa validacao_embarcada), a menos de
--sem-registro.

Uso (da raiz do repositório)
----------------------------
    python scripts/validation/matriz_clipes_placa.py --sem-registro     # conferir
    python scripts/validation/matriz_clipes_placa.py --responsavel Clara

Só precisa da biblioteca padrão.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import reduce
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import experimentos

# --------------------------------------------------------------------------- #
# Constantes do firmware (Core/Inc/clipes_demo.h e cadeia_clipes.h)           #
# --------------------------------------------------------------------------- #
TOL_REL = 1e-5                      # CADEIA_TOL_ESCORE_REL
CLIPES = ("normal", "bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm")   # clipes[]
SEGMENTOS_FIRMWARE = (29, 30, 88, 89, 147, 148, 206, 207, 265, 266)          # clipes_segmentos[]
MODELO = "exp239"                   # CLIPES_MODELO_ID
JSON_SHA12 = "ee0c3e088366"         # CLIPES_JSON_SHA256_12
CSV_SHA12 = "2507e68b5489"          # CLIPES_CSV_SHA256_12 (sha256 dos bytes do CSV)

MEIA_UNIDADE = 0.5e-3               # arredondamento do texto com 3 casas
FOLGA_FLOAT32 = 1e-4                # meio ulp do float32 entre 1024 e 2048 é 6,1e-5
CLASSES = ("normal", "falha")

CAMPOS_SEG = ("versao", "n", "origem", "clipe", "segmento", "decisao", "disparou",
              "escore_lda", "cod_novidade", "pont_novidade", "lim_novidade", "ciclos")
RE_KV = re.compile(r"(\w+)=(\S+)")
RE_STATUS = re.compile(r"\bdec=(normal|falha) (OK|FALHOU)\b")


class Abortar(Exception):
    pass


# --------------------------------------------------------------------------- #
# Leitura da captura                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Leitura:
    """Uma linha "cadeia seg=..." (boot) ou "demo seg=..." (volta do laço)."""
    num: int
    id: int
    clipe: str
    k: int
    ref_txt: str
    placa_txt: str
    dif: float
    tol: float
    dec: str
    ok: bool


@dataclass
class Captura:
    cabecalhos: list[dict] = field(default_factory=list)
    boot: list[Leitura] = field(default_factory=list)
    demo: list[Leitura] = field(default_factory=list)
    ponta: list[dict] = field(default_factory=list)
    pacotes: list[dict] = field(default_factory=list)
    problemas: list[str] = field(default_factory=list)
    cadeia_ok: bool = False
    resumo_firmware: str = ""


def checksum(corpo: str) -> int:
    return reduce(lambda x, c: x ^ ord(c), corpo, 0)


def ler_pacote(linha: str) -> tuple[str, dict]:
    """$CAB ou $SEG do formato v1; Abortar se o checksum ou os campos estiverem errados."""
    corpo, sep, cs = linha[1:].rpartition("*")
    if not sep or len(cs) != 2 or checksum(corpo) != int(cs, 16):
        raise Abortar(f"checksum errado: {linha}")
    tipo, *campos = corpo.split(",")
    if tipo == "CAB":
        if len(campos) != 5 or campos[0] != "1":
            raise Abortar(f"$CAB fora do formato v1: {linha}")
        return tipo, {"modelo": campos[1], "json": campos[2]}
    if tipo == "SEG":
        if len(campos) != len(CAMPOS_SEG) or campos[0] != "1":
            raise Abortar(f"$SEG fora do formato v1: {linha}")
        return tipo, dict(zip(CAMPOS_SEG, campos))
    raise Abortar(f"pacote de tipo desconhecido: {linha}")


def ler_leitura(num: int, linha: str) -> Leitura:
    kv = dict(RE_KV.findall(linha))
    st = RE_STATUS.search(linha)
    try:
        return Leitura(num=num, id=int(kv["seg"]), clipe=kv["clipe"], k=int(kv["k"]),
                       ref_txt=kv["ref"], placa_txt=kv["placa"], dif=float(kv["dif"]),
                       tol=float(kv["tol"]), dec=st.group(1), ok=st.group(2) == "OK")
    except (KeyError, ValueError, AttributeError):
        raise Abortar(f"linha {num} fora do formato esperado: {linha}") from None


def ler_captura(linhas) -> Captura:
    cap = Captura()
    n_anterior = None
    pendente: Leitura | None = None     # linha "demo" esperando o seu $SEG
    for num, bruta in enumerate(linhas, 1):
        linha = bruta.strip("\r\n")
        if linha.startswith("cadeia seg="):
            cap.boot.append(ler_leitura(num, linha))
        elif linha.startswith("cadeia resumo"):
            cap.resumo_firmware = linha
        elif linha == "CADEIA OK":
            cap.cadeia_ok = True
        elif linha.startswith("demo seg="):
            if pendente is not None:
                cap.problemas.append(f"linha {pendente.num}: demo sem o $SEG correspondente")
            pendente = ler_leitura(num, linha)
            cap.demo.append(pendente)
        elif linha.startswith("ponta "):
            kv = dict(RE_KV.findall(linha))
            kv["ok"] = " OK " in linha
            kv["num"] = num
            cap.ponta.append(kv)
        elif linha.startswith(("$CAB,", "$SEG,")):
            tipo, d = ler_pacote(linha)
            if tipo == "CAB":
                cap.cabecalhos.append(d)
                n_anterior = None
                continue
            n = int(d["n"])
            if n_anterior is not None and n != n_anterior + 1:
                cap.problemas.append(f"linha {num}: salto no contador ({n_anterior} -> {n})")
            n_anterior = n
            cap.pacotes.append(d)
            if d["origem"] == "C":
                if pendente is None:
                    cap.problemas.append(f"linha {num}: $SEG de origem C sem a linha demo antes")
                    continue
                esperado = (str(CLIPES.index(pendente.clipe)), str(pendente.k),
                            pendente.placa_txt, "F" if pendente.dec == "falha" else "N")
                lido = (d["clipe"], d["segmento"], d["escore_lda"], d["decisao"])
                if lido != esperado:
                    cap.problemas.append(f"linha {num}: $SEG {lido} difere da linha demo "
                                         f"{pendente.num} {esperado}")
                pendente = None
    if pendente is not None:
        cap.problemas.append(f"linha {pendente.num}: demo sem o $SEG correspondente (fim da captura)")
    return cap


# --------------------------------------------------------------------------- #
# Referência do Python                                                        #
# --------------------------------------------------------------------------- #
def ler_referencia(caminho: Path) -> dict[int, dict]:
    with caminho.open(encoding="utf-8", newline="") as fh:
        return {int(r["segmento"]): {"gravacao": r["gravacao"], "classe": r["binario"],
                                     "escore": float(r["escore"]), "previsao": r["previsao"]}
                for r in csv.DictReader(fh)}


def sha12(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()[:12]


def sha12_csv(caminho: Path) -> str:
    """
    Hash do CSV com as quebras em CRLF, como o 09_export_clips o gravou.

    O CLIPES_CSV_SHA256_12 do clipes_demo.h é o dos bytes com CRLF; conforme a
    configuração do git, o arquivo pode chegar com LF, e o conteúdo é o mesmo.
    """
    b = caminho.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(b).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Conferências e resultado                                                    #
# --------------------------------------------------------------------------- #
def conferir(cap: Captura, ref: dict[int, dict], modelo: str = MODELO,
             json_sha12: str = JSON_SHA12) -> list[str]:
    p = list(cap.problemas)

    # 1. cabeçalho
    if not cap.cabecalhos:
        p.append("nenhum $CAB na captura")
    for c in cap.cabecalhos:
        if (c["modelo"], c["json"]) != (modelo, json_sha12):
            p.append(f"$CAB com modelo {c['modelo']} e JSON {c['json']}; "
                     f"esperados {modelo} e {json_sha12}")

    # 2. boot com os 10 segmentos, uma vez cada, todos OK
    ids_boot = [l.id for l in cap.boot]
    if sorted(ids_boot) != sorted(SEGMENTOS_FIRMWARE):
        p.append(f"o boot tem os segmentos {ids_boot}; esperados {list(SEGMENTOS_FIRMWARE)}")
    if not cap.cadeia_ok:
        p.append("a captura não tem o CADEIA OK do boot")
    for l in cap.boot + cap.demo:
        if not l.ok:
            p.append(f"linha {l.num}: o próprio firmware marcou FALHOU (seg {l.id})")
        if l.clipe not in CLIPES:
            p.append(f"linha {l.num}: clipe desconhecido {l.clipe}")

    # 3 e 4. contra o CSV
    for l in cap.boot + cap.demo:
        r = ref.get(l.id)
        if r is None:
            p.append(f"seg {l.id} não está no escores_referencia.csv")
            continue
        if r["gravacao"] != l.clipe:
            p.append(f"linha {l.num}: seg {l.id} é de {r['gravacao']} no CSV e de {l.clipe} na placa")
        if abs(float(l.ref_txt) - r["escore"]) > MEIA_UNIDADE + FOLGA_FLOAT32:
            p.append(f"linha {l.num}: ref={l.ref_txt} do firmware não é o escore do CSV "
                     f"({r['escore']})")
        if l.dec != r["previsao"]:
            p.append(f"linha {l.num}: seg {l.id} decidiu {l.dec}, o Python decidiu {r['previsao']}")
        if not l.dif <= l.tol:
            p.append(f"linha {l.num}: dif {l.dif} acima da tolerância {l.tol} da placa")
        if abs(float(l.placa_txt) - r["escore"]) + MEIA_UNIDADE > TOL_REL * abs(r["escore"]):
            p.append(f"linha {l.num}: seg {l.id} pode estar fora da tolerância contra o float64 "
                     f"no pior arredondamento do texto")

    # 5. voltas iguais ao boot
    boot = {l.id: l for l in cap.boot}
    voltas: dict[int, list[Leitura]] = {}
    for l in cap.demo:
        voltas.setdefault(l.id, []).append(l)
    n_voltas = {len(v) for v in voltas.values()}
    if not cap.demo:
        p.append("nenhuma volta do laço na captura")
    elif sorted(voltas) != sorted(SEGMENTOS_FIRMWARE) or len(n_voltas) != 1:
        p.append("as voltas do laço não têm os 10 segmentos o mesmo número de vezes")
    for i, lst in voltas.items():
        b = boot.get(i)
        for l in lst:
            if b and (l.placa_txt, l.dec) != (b.placa_txt, b.dec):
                p.append(f"linha {l.num}: seg {i} deu {l.placa_txt} ({l.dec}) no laço e "
                         f"{b.placa_txt} ({b.dec}) no boot")

    # 7. ponta a ponta
    for d in cap.ponta:
        if not d["ok"] or d.get("igual_boot") != "1":
            p.append(f"linha {d['num']}: inferência de ponta a ponta sem OK ou diferente do boot")
    return p


def matriz(pares) -> dict[tuple[str, str], int]:
    m = {(a, b): 0 for a in CLASSES for b in CLASSES}
    for a, b in pares:
        m[(a, b)] += 1
    return m


def resultado(cap: Captura, ref: dict[int, dict]) -> dict:
    voltas: dict[int, list[Leitura]] = {}
    for l in cap.demo:
        voltas.setdefault(l.id, []).append(l)
    linhas = []
    for l in sorted(cap.boot, key=lambda x: x.id):
        r = ref[l.id]
        tol64 = TOL_REL * abs(r["escore"])
        linhas.append({
            "segmento": l.id, "clipe": l.clipe, "k": l.k, "classe": r["classe"],
            "escore_python": r["escore"], "escore_placa": float(l.placa_txt),
            "escore_placa_txt": l.placa_txt, "dif_placa": l.dif, "tol_placa": l.tol,
            "frac_tol": l.dif / l.tol,
            "frac_tol_conservadora": (abs(float(l.placa_txt) - r["escore"]) + MEIA_UNIDADE) / tol64,
            "decisao_python": r["previsao"], "decisao_placa": l.dec,
            "passagens_iguais": 1 + len(voltas.get(l.id, [])),
        })
    pior = max(linhas, key=lambda x: x["frac_tol"])
    seg_c = [d for d in cap.pacotes if d["origem"] == "C"]
    seg_d = [d for d in cap.pacotes if d["origem"] == "D"]
    return {
        "linhas": linhas,
        "m_placa": matriz((x["classe"], x["decisao_placa"]) for x in linhas),
        "m_python": matriz((x["classe"], x["decisao_python"]) for x in linhas),
        "m_concordancia": matriz((x["decisao_python"], x["decisao_placa"]) for x in linhas),
        "concordancia": sum(x["decisao_placa"] == x["decisao_python"] for x in linhas),
        "pior": pior,
        "n_voltas": len(cap.demo) // len(linhas) if linhas else 0,
        "n_pacotes_c": len(seg_c), "n_pacotes_d": len(seg_d),
        "ponta": cap.ponta,
    }


# --------------------------------------------------------------------------- #
# Formatação                                                                  #
# --------------------------------------------------------------------------- #
def br(x: float, casas: int) -> str:
    """1234.5678 -> '1.234,568' (separador de milhar e vírgula decimal)."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tex_num(x: float, casas: int) -> str:
    s = br(abs(x), casas).replace(",", "{,}")
    return f"${'-' if x < 0 else ''}{s}$"


def pct(x: float, casas: int = 1) -> str:
    return br(100 * x, casas) + " %"


def tex_pct(x: float, casas: int = 1) -> str:
    return br(100 * x, casas) + r"\,\%"


def escrever_csv(res: dict, destino: Path) -> None:
    campos = ["segmento", "clipe", "k", "classe", "escore_python", "escore_placa", "dif_placa",
              "tol_placa", "frac_tol", "frac_tol_conservadora", "decisao_python",
              "decisao_placa", "passagens_iguais"]
    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for x in res["linhas"]:
            w.writerow({**x, "escore_python": f"{x['escore_python']:.5f}",
                        "escore_placa": x["escore_placa_txt"],
                        "frac_tol": f"{x['frac_tol']:.4f}",
                        "frac_tol_conservadora": f"{x['frac_tol_conservadora']:.4f}"})


def md_matriz(m: dict, linhas_rot: str, colunas_rot: str) -> list[str]:
    return [f"| {linhas_rot} \\ {colunas_rot} | normal | falha |", "|---|---:|---:|",
            *(f"| {a} | {m[(a, 'normal')]} | {m[(a, 'falha')]} |" for a in CLASSES)]


def escrever_md(res: dict, meta: dict, destino: Path) -> None:
    n = len(res["linhas"])
    pior = res["pior"]
    t = [f"# Matriz do sistema embarcado: 10 segmentos dos clipes ({meta['captura_nome']})", "",
         f"Modelo {meta['modelo']} (JSON {meta['json']}), referência "
         f"`escores_referencia.csv` (sha {meta['csv_sha']}), captura sha {meta['captura_sha']}. "
         f"Uma passagem do boot e {res['n_voltas']} voltas do laço.", "",
         "Os 10 segmentos estão no treino do modelo final: a matriz mede a fidelidade do "
         "porte, e não o desempenho (que é o do Protocolo B).", "",
         "## Por segmento (verificação do boot)", "",
         "| Seg. | Clipe | k | Classe | Escore Python | Escore placa | Fração da tol. | "
         "Decisão Python | Decisão placa | Passagens iguais |",
         "|---:|---|---:|---|---:|---:|---:|---|---|---:|"]
    for x in res["linhas"]:
        t.append(f"| {x['segmento']} | {x['clipe']} | {x['k']} | {x['classe']} | "
                 f"{br(x['escore_python'], 5)} | {br(x['escore_placa'], 3)} | "
                 f"{pct(x['frac_tol'])} | {x['decisao_python']} | {x['decisao_placa']} | "
                 f"{x['passagens_iguais']}/{1 + res['n_voltas']} |")
    t += ["", "## Matrizes (linhas: classe verdadeira; colunas: decisão)", "", "**Placa**", "",
          *md_matriz(res["m_placa"], "verdadeira", "placa"), "", "**Python**", "",
          *md_matriz(res["m_python"], "verdadeira", "Python"), "",
          "**Concordância (linhas: Python; colunas: placa)**", "",
          *md_matriz(res["m_concordancia"], "Python", "placa"), "",
          "## Conferências", "",
          f"- Concordância placa × Python: {res['concordancia']}/{n}.",
          f"- Pior caso: seg {pior['segmento']} ({pior['clipe']}), "
          f"{pct(pior['frac_tol'])} da tolerância encadeada (dif {br(pior['dif_placa'], 6)} de "
          f"{br(pior['tol_placa'], 6)}, medidos na placa). No pior arredondamento do texto, "
          f"contra o float64 do CSV: {pct(max(x['frac_tol_conservadora'] for x in res['linhas']))}.",
          f"- Escore e decisão iguais ao boot nas {res['n_voltas']} voltas do laço, nos {n} segmentos.",
          f"- {res['n_pacotes_c']} pacotes $SEG de origem C, cada um igual à linha demo "
          f"correspondente; contador sem salto, nenhum checksum errado.",
          f"- Referência: o sha do CSV é o CLIPES_CSV_SHA256_12 do clipes_demo.h, e o ref= "
          f"impresso pela placa é o escore do CSV em todos os segmentos.",
          f"- Ponta a ponta (origem D, fora da matriz): {len(res['ponta'])} inferências, todas OK "
          f"e iguais ao boot. É o mesmo segundo do seg 29 por outro caminho; contá-lo duplicaria "
          f"o segmento.",
          f"- Firmware: `{meta['resumo_firmware']}`", ""]
    destino.write_text("\n".join(t), encoding="utf-8")


def escrever_tex(res: dict, destino: Path) -> None:
    def clipe_tex(c):
        return r"\texttt{" + c.replace("_", r"\_") + "}"

    t = [r"\begin{table}[H]", r"\centering",
         r"\caption{Cadeia completa na placa nos 10 segmentos dos clipes (verificação na "
         r"inicialização do firmware final, 22/10). Fração da tolerância encadeada de "
         r"$10^{-5}$ relativo, com a diferença medida na placa.}",
         r"\label{tab:clipes_placa_segmentos}", r"\small",
         r"\begin{tabular}{rlrrrl}", r"\toprule",
         r"\textbf{Seg.} & \textbf{Clipe} & \textbf{Escore Python} & \textbf{Escore placa} & "
         r"\textbf{\% da tol.} & \textbf{Decisão} \\", r"\midrule"]
    for x in res["linhas"]:
        dec = x["decisao_placa"] if x["decisao_placa"] == x["decisao_python"] else (
            f"{x['decisao_placa']} (Python: {x['decisao_python']})")
        t.append(f"{x['segmento']} & {clipe_tex(x['clipe'])} & {tex_num(x['escore_python'], 3)} & "
                 f"{tex_num(x['escore_placa'], 3)} & {br(100 * x['frac_tol'], 1)} & {dec} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[H]", r"\centering",
          r"\caption{Matriz de confusão do sistema embarcado e das decisões de referência "
          r"nos 10 segmentos dos clipes, todos do treino do modelo final: mede a fidelidade "
          r"do porte, e não o desempenho.}",
          r"\label{tab:clipes_placa_matriz}", r"\small",
          r"\begin{tabular}{lrrrr}", r"\toprule",
          r" & \multicolumn{2}{c}{\textbf{Placa}} & \multicolumn{2}{c}{\textbf{Python}} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"\textbf{Classe verdadeira} & normal & falha & normal & falha \\", r"\midrule"]
    for a in CLASSES:
        mp, my = res["m_placa"], res["m_python"]
        t.append(f"{a} & {mp[(a, 'normal')]} & {mp[(a, 'falha')]} & "
                 f"{my[(a, 'normal')]} & {my[(a, 'falha')]} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    destino.write_text("\n".join(t), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captura", type=Path,
                    default=Path("reports/embarcado/capturas/primeira-inferencia-22-10.log"))
    ap.add_argument("--referencia", type=Path,
                    default=Path("reports/modelo_final/escores_referencia.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/embarcado/clipes_22_10"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true",
                    help="rodada de conferência: não escreve no registry")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    try:
        if sha12_csv(args.referencia) != CSV_SHA12:
            raise Abortar(f"{args.referencia} tem sha {sha12_csv(args.referencia)} (em CRLF); "
                          f"o firmware foi gerado do CSV com sha {CSV_SHA12} (clipes_demo.h)")
        ref = ler_referencia(args.referencia)
        with args.captura.open("rb") as fh:
            cap = ler_captura(b.decode("ascii", errors="replace") for b in fh)
        problemas = conferir(cap, ref)
    except (Abortar, OSError) as e:
        print(f"Abortado: {e}")
        return 1
    if problemas:
        print("Abortado, a captura não passou nas conferências:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    res = resultado(cap, ref)
    meta = {"captura_nome": args.captura.name, "captura_sha": sha12(args.captura),
            "csv_sha": CSV_SHA12, "modelo": MODELO, "json": JSON_SHA12,
            "resumo_firmware": cap.resumo_firmware}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    escrever_csv(res, args.out_dir / "segmentos.csv")
    escrever_md(res, meta, args.out_dir / "resumo.md")
    escrever_tex(res, args.out_dir / "tabelas.tex")

    mp, n, pior = res["m_placa"], len(res["linhas"]), res["pior"]
    print(f"{n} segmentos, {res['n_voltas']} voltas; concordância placa x Python "
          f"{res['concordancia']}/{n}; pior caso seg {pior['segmento']} com "
          f"{pct(pior['frac_tol'])} da tolerância")
    print(f"matriz da placa (verdadeira x placa): VN={mp[('normal', 'normal')]} "
          f"FP={mp[('normal', 'falha')]} FN={mp[('falha', 'normal')]} VP={mp[('falha', 'falha')]}")
    print(f"saídas: {args.out_dir}/")

    if args.sem_registro:
        print("(rodada de conferência: registry não alterado)")
        return 0
    exp_id = f"exp{experimentos.next_exp_number(args.registry):03d}"
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "validacao_embarcada",
        "script": "validation/matriz_clipes_placa.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "captura": args.captura.name, "captura_sha": meta["captura_sha"],
            "modelo": MODELO, "json": JSON_SHA12, "referencia_sha": CSV_SHA12,
            "tol_rel": TOL_REL, "passagens": f"boot+{res['n_voltas']}_voltas",
            "segmentos": "-".join(str(s) for s in SEGMENTOS_FIRMWARE)}),
        "dataset": f"clipes_demo_{n}seg_treino_{MODELO}",
        "metricas": experimentos.kv({
            "n_segmentos": n, "concordancia": f"{res['concordancia']}/{n}",
            "vn": mp[("normal", "normal")], "fp": mp[("normal", "falha")],
            "fn": mp[("falha", "normal")], "vp": mp[("falha", "falha")],
            "frac_tol_max": f"{pior['frac_tol']:.4f}", "seg_frac_tol_max": pior["segmento"],
            "iguais_entre_passagens": f"{n}/{n}", "pacotes_c": res["n_pacotes_c"],
            "pacotes_perdidos": 0, "ponta_d": f"{len(res['ponta'])}/{len(res['ponta'])}"}),
        "responsavel": args.responsavel,
        "notas": args.notas or ("fidelidade do porte, não desempenho: os 10 segmentos estão no "
                                "treino do modelo final; captura de 22/10 do firmware final"),
    }])
    print(f"registrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
