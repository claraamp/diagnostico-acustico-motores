"""
pacotes.py — leitura dos pacotes da placa (formato v1) e o estado do painel

Card "Implementar a interface de diagnóstico no computador (leitura serial)". O
formato é o da seção "Pacote da serial" do README do firmware
(`lab3-firmware-stm32`, `Core/Inc/pacote_serial.h`), e as regras de leitura são as
do leitor de referência de lá (`tests/host/pacote/ler_pacotes.py`): este módulo
aceita e recusa as mesmas linhas (os testes usam as linhas do autoteste dele). Só a
biblioteca padrão: a janela (`run_painel.py`) e os testes usam o mesmo estado.

O que o painel mostra (decisão de 17/10, "O que o painel mostra no lugar de grau de
confiança", e escopo do guia de 16–23/11):

- a classe do segmento: normal, falha ou **erro** quando o bit 4 (`disparou = 4`,
  escore não finito) vem ligado. Nunca "normal" com erro;
- o escore **cru** da LDA, como veio no texto (3 casas, falha se ≥ 0), nunca em %
  nem como confiança. A decisão da LDA se lê pelo sinal do texto ("-0.000" é
  normal; `float("-0.000") >= 0` daria falha);
- o histórico das últimas inferências (origem, clipe ou grupo, segmento, decisão,
  escore) e a consistência entre segmentos, "k de N segmentos recentes indicaram
  falha", que o pacote não traz de propósito;
- pacote perdido pelo salto do contador `n`, reinício da placa e as linhas `$ERR`
  da recepção;
- o LED que a placa tem de estar mostrando (README do firmware, "Sinalização por
  LED"): verde com `N`, vermelho com `F`, os dois com o erro de conta, e os dois
  apagados depois de um `$ERR` e antes do primeiro `$SEG`.

Só as linhas `$CAB`, `$SEG` e `$ERR` contam; o resto (`hello`, `demo`, `rx`, `ref`,
`$PED`, ...) é ignorado. A exceção são as linhas que dizem algo sobre o estado: o
`hello` (a placa reiniciou: o modelo tem de ser conferido de novo pelo próximo
`$CAB`) e, numa captura do `matriz_serial_placa.py`, as linhas `#pc envio`, que dão o
nome do segmento mandado (por exemplo `normal_seg0048_branco_0db`).

O modelo é conferido antes de mostrar qualquer decisão: o `$CAB` tem de trazer o id e
o SHA-256[:12] do `lda_final.json` do Python. Sem o `$CAB` (o painel abriu com a
placa já rodando), os segmentos são contados, mas a decisão não aparece: aperte o
RESET da placa.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

VERSAO = 1
CASAS = 3
MEIA_UNIDADE = 0.5 * 10.0 ** -CASAS

CAMPOS_SEG = ("versao", "n", "origem", "clipe", "segmento", "decisao", "disparou",
              "escore_lda", "cod_novidade", "pont_novidade", "lim_novidade", "ciclos")
CAMPOS_CAB = ("versao", "modelo", "json_sha12", "n_features", "clock_mhz")
ORIGENS = {"R": "referência", "C": "clipe", "D": "decimado", "S": "serial", "A": "ao vivo"}
NOVIDADE = {0: "não adotado", 1: "pente BPFO (dB)", 2: "pente BPFI (dB)",
            3: "Mahalanobis² MFCC", 4: "Mahalanobis² MFCC + pente"}
DISPAROU_LDA, DISPAROU_NOVIDADE, ERRO = 1, 2, 4
MOTIVOS_ERR = ("versao", "formato", "tamanho", "crc", "truncado", "magica")

# Grupos do quadro no modo de recepção (regra 4 do README do firmware)
GRUPOS = {1: "Jung", 2: "HUSTmotor", 3: "ruído branco", 4: "ruído rosa", 9: "clipes da flash"}
# Clipes da flash (clipes_demo.h): o índice do clipe e o segmento k dentro dele
CLIPES = ("normal", "bpfi_0.3mm", "bpfi_1.0mm", "bpfo_0.3mm", "bpfo_1.0mm")
SEGMENTOS_CLIPES = {(0, 0): 29, (0, 1): 30, (1, 0): 88, (1, 1): 89, (2, 0): 147, (2, 1): 148,
                    (3, 0): 206, (3, 1): 207, (4, 0): 265, (4, 1): 266}

MODELO = Path("reports/modelo_final/lda_final.json")
RE_KV = re.compile(r"(\w+)=(\S+)")


class ErroPacote(Exception):
    pass


class OrigemDesconhecida(ErroPacote):
    """Origem que este leitor não conhece: descarta só o pacote, sem contar como corrompido."""

    def __init__(self, msg: str, n: int):
        super().__init__(msg)
        self.n = n                     # o contador vale: o pacote chegou certo, só não é mostrado


def checksum(corpo: str) -> int:
    return reduce(lambda x, c: x ^ ord(c), corpo, 0)


def _int(txt: str, nome: str, vazio_ok: bool = False):
    if txt == "" and vazio_ok:
        return None
    try:
        return int(txt)
    except ValueError:
        raise ErroPacote(f"{nome} não é inteiro: {txt!r}") from None


def _float(txt: str, nome: str) -> float:
    try:
        return float(txt)
    except ValueError:
        raise ErroPacote(f"{nome} não é número: {txt!r}") from None


def ler_linha(linha: str) -> tuple[str, dict]:
    """(tipo, campos) de uma linha $CAB, $SEG ou $ERR; ErroPacote se inválida."""
    linha = linha.strip("\r\n")
    if not linha.startswith("$"):
        raise ErroPacote("não começa com '$'")
    corpo, sep, cs = linha[1:].rpartition("*")
    if not sep or len(cs) != 2:
        raise ErroPacote("sem '*HH' no fim")
    try:
        cs_lido = int(cs, 16)
    except ValueError:
        raise ErroPacote(f"checksum não é hexadecimal: {cs!r}") from None
    if checksum(corpo) != cs_lido:
        raise ErroPacote(f"checksum errado (lido {cs_lido:02X}, calculado {checksum(corpo):02X})")
    tipo, *campos = corpo.split(",")
    if tipo == "ERR":
        return tipo, _ler_err(campos)
    nomes = {"CAB": CAMPOS_CAB, "SEG": CAMPOS_SEG}.get(tipo)
    if nomes is None:
        raise ErroPacote(f"tipo desconhecido: {tipo!r}")
    if len(campos) != len(nomes):
        raise ErroPacote(f"{tipo} com {len(campos)} campos, esperados {len(nomes)}")
    d = dict(zip(nomes, campos))
    if _int(d["versao"], "versao") != VERSAO:
        raise ErroPacote(f"versão {d['versao']}; o painel lê a {VERSAO}")
    d["versao"] = VERSAO
    if tipo == "CAB":
        d["n_features"] = _int(d["n_features"], "n_features")
        d["clock_mhz"] = _int(d["clock_mhz"], "clock_mhz")
        return tipo, d

    d["n"] = _int(d["n"], "n")
    if d["origem"] not in ORIGENS:
        raise OrigemDesconhecida(f"origem desconhecida: {d['origem']!r}", d["n"])
    d["clipe"] = _int(d["clipe"], "clipe", vazio_ok=True)
    d["segmento"] = _int(d["segmento"], "segmento")
    if d["decisao"] not in ("N", "F"):
        raise ErroPacote(f"decisão tem que ser N ou F: {d['decisao']!r}")
    d["disparou"] = _int(d["disparou"], "disparou")
    if d["disparou"] not in (0, 1, 2, 3, ERRO):
        raise ErroPacote(f"disparou tem que ser 0..3 ou 4 (erro, sozinho): {d['disparou']}")
    d["escore_txt"] = d["escore_lda"]
    d["escore_lda"] = _float(d["escore_lda"], "escore da LDA")
    d["cod_novidade"] = _int(d["cod_novidade"], "código de novidade")
    if d["cod_novidade"] not in NOVIDADE:
        raise ErroPacote(f"código de novidade desconhecido: {d['cod_novidade']}")
    if d["cod_novidade"] == 0:
        if d["pont_novidade"] != "" or d["lim_novidade"] != "":
            raise ErroPacote("novidade não adotada (código 0) com pontuação ou limiar")
        d["pont_novidade"] = d["lim_novidade"] = None
        d["nov_txt"] = []
    else:
        d["nov_txt"] = [d["pont_novidade"], d["lim_novidade"]]
        d["pont_novidade"] = _float(d["pont_novidade"], "pontuação de novidade")
        d["lim_novidade"] = _float(d["lim_novidade"], "limiar de novidade")
    d["ciclos"] = _int(d["ciclos"], "ciclos")
    conferir_decisao(d)
    return tipo, d


def _ler_err(campos: list[str]) -> dict:
    """$ERR,1,<grupo>,<id>,<motivo>: grupo e id vazios quando o quadro truncou antes do cabeçalho."""
    if len(campos) != 4:
        raise ErroPacote(f"ERR com {len(campos)} campos, esperados 4")
    versao, grupo, id_, motivo = campos
    if _int(versao, "versao") != VERSAO:
        raise ErroPacote(f"versão {versao}; o painel lê a {VERSAO}")
    if motivo not in MOTIVOS_ERR:
        raise ErroPacote(f"motivo de $ERR desconhecido: {motivo!r}")
    if (grupo == "") != (id_ == ""):
        raise ErroPacote("$ERR com só um de grupo e id")
    return {"versao": VERSAO, "grupo": _int(grupo, "grupo", True), "id": _int(id_, "id", True),
            "motivo": motivo}


def conferir_decisao(d: dict) -> None:
    """Coerência entre decisão, disparou e os valores (sentidos de 17/10), como no ler_pacotes.py."""
    disp = d["disparou"]
    if (d["decisao"] == "F") != (disp != 0):
        raise ErroPacote(f"decisão {d['decisao']} com disparou={disp}")
    if d["cod_novidade"] == 0 and disp & DISPAROU_NOVIDADE:
        raise ErroPacote("disparou novidade sem detector adotado")
    txt = d["escore_txt"]
    textos = [txt] + d["nov_txt"]
    if disp == ERRO:
        if not any(t.lower() in ("nan", "inf", "-inf") for t in textos):
            raise ErroPacote("bit de erro (4) sem nenhum valor nan ou inf")
        return
    if any(t.lower() == "nan" for t in textos):
        raise ErroPacote("valor nan sem o bit de erro (4)")
    lda_falha = not txt.startswith("-")          # o sinal do TEXTO decide ("-0.000" é normal)
    if bool(disp & DISPAROU_LDA) != lda_falha:
        raise ErroPacote(f"escore {txt} com o bit da LDA = {disp & DISPAROU_LDA} "
                         "(falha se ≥ 0; negativo sai com '-')")
    if d["cod_novidade"] != 0:
        p, lim = d["pont_novidade"], d["lim_novidade"]
        if abs(p - lim) > 2 * MEIA_UNIDADE and bool(disp & DISPAROU_NOVIDADE) != (p > lim):
            raise ErroPacote(f"pontuação {p} e limiar {lim} com o bit da novidade = "
                             f"{(disp & DISPAROU_NOVIDADE) >> 1} (falha se pontuação > limiar)")


def classe(d: dict) -> str:
    """'normal', 'falha' ou 'erro' de um $SEG já conferido: o erro nunca vira normal."""
    if d["disparou"] == ERRO:
        return "erro"
    return "falha" if d["decisao"] == "F" else "normal"


def modelo_python(caminho: Path = MODELO) -> tuple[str, str]:
    """(id, sha256[:12]) do lda_final.json: o que o $CAB da placa tem de trazer."""
    bruto = Path(caminho).read_bytes()
    return str(json.loads(bruto.decode("utf-8"))["id"]), hashlib.sha256(bruto).hexdigest()[:12]


def descrever_origem(d: dict) -> tuple[str, str]:
    """(onde, segmento) para mostrar: o clipe ou o grupo, e o segmento."""
    o, clipe, seg = d["origem"], d["clipe"], d["segmento"]
    if o in ("C", "D") and clipe is not None and 0 <= clipe < len(CLIPES):
        jung = SEGMENTOS_CLIPES.get((clipe, seg))
        onde = f"clipe {CLIPES[clipe]}" + (" (decimado na placa)" if o == "D" else "")
        return onde, f"k={seg}" + (f" (seg. {jung} do Jung)" if jung is not None else "")
    if o == "S":
        return f"grupo {clipe} ({GRUPOS.get(clipe, '?')})", f"id {seg}"
    if o == "R":
        return f"gravação {clipe}", f"seg. {seg}"
    return ("ao vivo" if o == "A" else str(clipe)), str(seg)


# --------------------------------------------------------------------------- #
# Estado do painel                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """Uma linha do histórico: um segmento, um $ERR ou um aviso (pacotes perdidos, reinício)."""
    tipo: str                  # "seg", "err" ou "aviso"
    hora: float
    texto: str = ""
    n: int | None = None
    origem: str = ""
    onde: str = ""
    segmento: str = ""
    nome: str = ""             # da linha #pc envio, numa captura do matriz_serial_placa.py
    classe: str = ""           # normal, falha ou erro
    escore: str = ""           # o texto do pacote, como veio
    ms: float | None = None
    novidade: str = ""


@dataclass
class Painel:
    """
    Estado do painel, alimentado linha a linha (`linha()`), sem nada de janela.

    `modelo_esperado` é (id, sha12) do lda_final.json; None desliga a conferência (só
    nos testes). `historico` guarda as últimas `n_historico` linhas; `ultimas` guarda
    as classes dos últimos `n_consistencia` segmentos, para o "k de N".
    """
    modelo_esperado: tuple[str, str] | None
    n_historico: int = 50
    n_consistencia: int = 10
    relogio: object = time.time

    cab: dict | None = None
    modelo_ok: bool = False
    modelo_errado: str = ""
    atual: Item | None = None          # o último $SEG mostrado (ou o $ERR que apagou o LED)
    led: tuple[bool, bool] = (False, False)   # (verde, vermelho) que a placa tem de mostrar
    n_anterior: int | None = None
    contagem: dict = field(default_factory=lambda: {
        "segmentos": 0, "normal": 0, "falha": 0, "erro": 0, "err": 0, "perdidos": 0,
        "reinicios": 0, "descartadas": 0, "origem_desconhecida": 0, "sem_cab": 0})
    por_origem: dict = field(default_factory=dict)
    nomes: dict = field(default_factory=dict)      # (grupo, id) -> nome, das linhas #pc envio
    avisos: list = field(default_factory=list)     # (hora, texto), os mais recentes no fim
    historico: deque = None
    ultimas: deque = None

    def __post_init__(self):
        self.historico = deque(maxlen=self.n_historico)
        self.ultimas = deque(maxlen=self.n_consistencia)

    # -- consulta
    def consistencia(self) -> tuple[int, int]:
        """(k, n): quantos dos últimos n segmentos indicaram falha (o erro não conta como falha)."""
        return sum(c == "falha" for c in self.ultimas), len(self.ultimas)

    def _avisar(self, texto: str) -> None:
        self.avisos.append((self.relogio(), texto))
        del self.avisos[:-200]

    # -- entrada
    def linha(self, texto: str) -> str | None:
        """
        Uma linha da serial ou da captura. Devolve o que mudou ("cab", "seg", "err",
        "descartada", "reinicio", "nome") ou None se a linha foi ignorada.
        """
        texto = texto.rstrip("\r\n")
        if texto.startswith("#pc envio "):
            kv = dict(RE_KV.findall(texto))
            if "grupo" in kv and "id" in kv and "nome" in kv:
                self.nomes[(int(kv["grupo"]), int(kv["id"]))] = kv["nome"]
                return "nome"
            return None
        if texto.startswith("hello"):
            # a placa reiniciou: o próximo $CAB confere o modelo de novo
            if self.n_anterior is not None or self.cab is not None:
                self.contagem["reinicios"] += 1
                self._avisar("a placa reiniciou (hello)")
                self.historico.append(Item("aviso", self.relogio(), "a placa reiniciou"))
            self.modelo_ok, self.cab, self.n_anterior = False, None, None
            self.led = (False, False)
            self.atual = None
            return "reinicio"
        if not texto.startswith(("$CAB,", "$SEG,", "$ERR,")):
            return None
        try:
            tipo, d = ler_linha(texto)
        except OrigemDesconhecida as e:
            self.contagem["origem_desconhecida"] += 1
            self._avisar(f"pacote ignorado ({e})")
            self._contador(e.n)            # chegou certo: não é pacote perdido
            return "descartada"
        except ErroPacote as e:
            self.contagem["descartadas"] += 1
            self._avisar(f"linha descartada ({e}): {texto[:60]}")
            return "descartada"
        return {"CAB": self._cab, "SEG": self._seg, "ERR": self._err}[tipo](d)

    def _cab(self, d: dict) -> str:
        self.cab = d
        self.n_anterior = None
        if self.modelo_esperado is None or (d["modelo"], d["json_sha12"]) == tuple(self.modelo_esperado):
            self.modelo_ok, self.modelo_errado = True, ""
        else:
            self.modelo_ok = False
            self.modelo_errado = (f"a placa tem o modelo {d['modelo']} (JSON {d['json_sha12']}); "
                                  f"o Python, {self.modelo_esperado[0]} (JSON {self.modelo_esperado[1]})")
            self._avisar("MODELO DIFERENTE: " + self.modelo_errado)
        return "cab"

    def _contador(self, n: int) -> None:
        """Salto do n (pacotes perdidos ou descartados por checksum) e reinício sem $CAB."""
        agora = self.relogio()
        if self.n_anterior is not None:
            if n == 0 and self.n_anterior > 0:
                self.contagem["reinicios"] += 1
                self._avisar("a placa reiniciou (n voltou a 0) sem mandar o $CAB")
                self.historico.append(Item("aviso", agora, "a placa reiniciou (n voltou a 0)"))
            elif n > self.n_anterior + 1:
                perdidos = n - self.n_anterior - 1
                self.contagem["perdidos"] += perdidos
                txt = f"{perdidos} pacote{'s' if perdidos > 1 else ''} perdido{'s' if perdidos > 1 else ''} " \
                      f"(n {self.n_anterior} → {n})"
                self._avisar(txt)
                self.historico.append(Item("aviso", agora, txt))
            elif n <= self.n_anterior:
                self._avisar(f"n fora de ordem ({self.n_anterior} → {n})")
        self.n_anterior = n

    def _seg(self, d: dict) -> str:
        agora = self.relogio()
        self._contador(d["n"])
        if not self.modelo_ok:
            self.contagem["sem_cab"] += 1
            return "seg"
        c = classe(d)
        onde, seg = descrever_origem(d)
        nome = self.nomes.get((d["clipe"], d["segmento"]), "") if d["origem"] == "S" else ""
        clock = (self.cab or {}).get("clock_mhz") or 100
        nov = "" if d["cod_novidade"] == 0 else (
            f"{NOVIDADE[d['cod_novidade']]}: {d['nov_txt'][0]} (limiar {d['nov_txt'][1]})")
        item = Item("seg", agora, n=d["n"], origem=d["origem"], onde=onde, segmento=seg, nome=nome,
                    classe=c, escore=d["escore_txt"], novidade=nov,
                    ms=d["ciclos"] / (clock * 1000.0) if d["ciclos"] else None)   # 0: sem DWT (emulador)
        self.atual = item
        self.historico.append(item)
        self.ultimas.append(c)
        self.contagem["segmentos"] += 1
        self.contagem[c] += 1
        o = self.por_origem.setdefault(d["origem"], {"normal": 0, "falha": 0, "erro": 0})
        o[c] += 1
        self.led = {"normal": (True, False), "falha": (False, True), "erro": (True, True)}[c]
        return "seg"

    def _err(self, d: dict) -> str:
        agora = self.relogio()
        self.contagem["err"] += 1
        alvo = "" if d["grupo"] is None else f"grupo {d['grupo']}, id {d['id']}"
        nome = self.nomes.get((d["grupo"], d["id"]), "") if d["grupo"] is not None else ""
        item = Item("err", agora, texto=f"$ERR {d['motivo']}" + (f" ({alvo})" if alvo else ""),
                    onde=alvo, nome=nome, classe="", escore="")
        self.historico.append(item)
        self.atual = item
        self.led = (False, False)          # um $ERR apaga os dois LEDs (recepção)
        return "err"

    def resumo(self) -> str:
        c = self.contagem
        k, n = self.consistencia()
        modelo = (f"{self.cab['modelo']} ({self.cab['json_sha12']})" if self.cab else "sem $CAB")
        return (f"modelo {modelo}{'' if self.modelo_ok else ' NÃO CONFERIDO'}; "
                f"{c['segmentos']} segmentos ({c['normal']} normal, {c['falha']} falha, {c['erro']} erro); "
                f"{c['err']} $ERR; {c['perdidos']} pacotes perdidos; {c['reinicios']} reinícios; "
                f"{c['descartadas']} linhas descartadas; {c['origem_desconhecida']} de origem desconhecida; "
                f"{c['sem_cab']} antes do $CAB; últimos {n}: {k} falha")
