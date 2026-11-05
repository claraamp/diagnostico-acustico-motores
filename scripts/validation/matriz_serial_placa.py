#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matriz_serial_placa.py -- envio dos segmentos à placa e matriz embarcada contra o Python
======================================================================================

Card "Script do computador: envio dos segmentos à placa e matriz embarcada contra o
Python". Manda à placa, pela serial, os 295 segmentos do Jung e os 40 do HUSTmotor
(1 s a 12,8 kHz, int16), um por vez, no modo de recepção do firmware (origem S do
$SEG); salva a captura; compara cada escore da placa com o do Python em float64
pela tolerância encadeada do escore no A4 (1e-5 relativo) e cada decisão com a do
Python; e grava as matrizes de confusão embarcadas, por conjunto, e o registro.

O formato do quadro e o diálogo são os do README do firmware (seção "Recepção de
segmentos pela serial"); as regras que este script segue são as "Regras do PC na
rodada" de lá:

1. Modelo: antes do primeiro quadro, o `modelo=` e o `json=` da linha
   `recepcao inicio` (e do $CAB) têm de ser os do `lda_final.json`. Senão, nada sai.
2. Tentativas: no máximo 3 por segmento, contando $ERR, timeout (o tempo de
   transmissão mais 2 s sem resposta) e resposta rejeitada; depois, "não processado".
3. Duplicatas: vale o primeiro $SEG válido de cada segmento; os outros são contados
   e têm de ter o mesmo escore, bit a bit.
4. Grupos e ids: grupo 1 = Jung, com o id do segmento no conjunto (`splits.json`);
   grupo 2 = HUSTmotor, com o índice (a partir de 0) na lista `segmentos` do
   `divisao_rotacao.json`.
5. Reinício: uma linha da demonstração (`hello`, `demo`, `ponta`, um $SEG de origem
   C ou D) no meio da rodada é a placa que reiniciou: o PC pede o modo de novo,
   confere o modelo de novo e retoma do segmento pendente (sem gastar tentativa).
6. A linha `rx` é parte do protocolo: o escore vem do `escore_hex` dela (o $SEG só
   tem 3 casas), e ela tem de bater com o $SEG que vem logo depois (grupo, id, o
   texto do escore e a decisão).
7. Anel do DMA (firmware com o contador de voltas, PR #12 do firmware): um $SEG cuja `rx` tenha `estouros` > 0 é
   descartado e o segmento é reenviado; a tentativa conta. Num firmware sem o campo,
   ele vale 0.

A captura guarda as linhas da placa como chegaram (bytes, com o CRLF) e as do PC
começando por `#pc` (o plano, cada envio, timeouts, reinícios, desistências). A
matriz sai SÓ da captura: `--captura` refaz a análise sem a placa, e o resultado é
o mesmo da rodada.

Fidelidade, não desempenho
--------------------------
Os 295 segmentos do Jung são todos do treino do modelo final (exp239): no Jung, a
matriz "verdadeira × placa" é acurácia de treino, e o resultado é a concordância
placa × Python. No HUSTmotor, a matriz "verdadeira × placa" é o C1 do Protocolo C
(a LDA do Jung transferida sem recalibrar), e o que se mede é de novo a
concordância com o Python. As duas matrizes ficam separadas, nunca as 335 juntas.

Referência do Python
--------------------
O escore em float64 de cada segmento é recalculado do mesmo int16 que vai no
quadro (`dsp.mfcc` e a forma dobrada do `lda_final.json`, como no `09`). No Jung,
ele tem de bater com o `escores_referencia.csv` (o CSV tem 5 casas) e a decisão,
com a do CSV; no HUST, com o `escore_sem_recal` do `decisoes_por_segmento.csv` da
recalibração (6 casas). Se não bater, o script aborta antes de mandar qualquer coisa.

Uso (da raiz do repositório)
----------------------------
    python scripts/validation/matriz_serial_placa.py --simulada            # sem placa (eco simulado)
    python scripts/validation/matriz_serial_placa.py --porta COM5 --sem-registro   # ensaio com a placa
    python scripts/validation/matriz_serial_placa.py --porta COM5 --responsavel Amanda   # a rodada
    python scripts/validation/matriz_serial_placa.py --captura reports/embarcado/capturas/serial-07-11.log --sem-registro

Com a placa: feche o picocom (só um programa abre a porta) e grave o firmware com a
recepção. A porta abre a 921.600 bit/s (o padrão do firmware desde 03/11);
`--baud 115200` para um firmware antigo. O DTR e o RTS ficam desligados para não
reiniciar a placa em adaptadores que os ligam ao reset. Se a placa já estiver no
modo de recepção (de uma execução anterior), ela não manda a linha `recepcao inicio`
e o modelo não pode ser conferido: aperte o RESET quando o script pedir.

`--simulada` troca a placa por uma placa de mentira em Python (`PlacaSimulada`), que
fala o mesmo protocolo e devolve o escore do Python em float32 (com alguns ulps de
diferença): testa o envio, as regras e a matriz, não a cadeia em C. Para testar
contra a cadeia em C sem a placa, use o `emular_placa` do repositório do firmware
(`--porta /dev/pts/N`, Linux e macOS). Saídas da simulada e do `--sem-registro`
vão para pastas fora do Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import experimentos

# --------------------------------------------------------------------------- #
# Protocolo (README do firmware, "Recepção de segmentos pela serial")         #
# --------------------------------------------------------------------------- #
MAGICA = b"QSEG"
VERSAO_QUADRO = 1
FORMATO_12K8 = 1
N_AMOSTRAS = 12800
TAM_QUADRO = 18 + 2 * N_AMOSTRAS          # 25.618 B
BAUD_PADRAO = 921600
TENTATIVAS = 3                             # regra 2
MAX_REINICIOS = 5                          # regra 5: mais que isso, a rodada para
TOL_REL = 1e-5                             # tolerância encadeada do escore no A4
MEIA_UNIDADE = 0.5e-3                      # o escore do $SEG tem 3 casas
FOLGA_TEXTO = 2.5e-4                       # meio ulp do float32 até 4096, com folga

GRUPOS = {"jung": 1, "hust": 2}
NOMES_CONJUNTO = {"jung": "Jung", "hust": "HUSTmotor"}
CLASSES = ("normal", "falha")

CAMPOS_SEG = ("versao", "n", "origem", "clipe", "segmento", "decisao", "disparou",
              "escore_lda", "cod_novidade", "pont_novidade", "lim_novidade", "ciclos")
RE_RX = re.compile(r"^rx grupo=(\d+) id=(\d+) bytes=(\d+) ms=(\d+) .*?\bciclos=(\d+) "
                   r"escore_hex=0x([0-9A-F]{8})(?: |$)")
RE_KV = re.compile(r"(\w+)=(\S+)")
REINICIO = ("hello", "demo", "ponta")      # linhas que só existem fora do modo
ORIGENS_FORA_DO_MODO = ("R", "C", "D", "A")  # $SEG do marco, dos clipes, decimado e ao vivo

# Pastas padrão
REFERENCIA = Path("reports/modelo_final/escores_referencia.csv")
MODELO = Path("reports/modelo_final/lda_final.json")
SPLITS = Path("data/processed/splits/splits.json")
PCM_JUNG = Path("data/processed/pcm_decimated/12800")
PCM_HUST = Path("data/processed/hustmotor/pcm_decimated/12800")
DIVISAO_HUST = Path("data/processed/hustmotor/divisao_rotacao.json")
ESCORES_HUST = Path("reports/validation/protocolo_c/recalibracao/decisoes_por_segmento.csv")
CAPTURAS = Path("reports/embarcado/capturas")
SAIDA_TESTE = Path("reports/embarcado/teste_serial")          # fora do Git (.gitignore)
SAIDA_SIMULADA = Path("reports/embarcado/sintetico/serial")   # fora do Git (.gitignore)


class Abortar(Exception):
    pass


def checksum(corpo: str) -> int:
    x = 0
    for c in corpo:
        x ^= ord(c)
    return x


def assinar(corpo: str) -> str:
    """'SEG,1,...' -> '$SEG,1,...*HH' (o formato v1 do pacote)."""
    return f"${corpo}*{checksum(corpo):02X}"


def ler_pacote(linha: str) -> list[str] | None:
    """Os campos de uma linha $XXX,...*HH, com o tipo no primeiro; None se o checksum não bate."""
    if not linha.startswith("$"):
        return None
    corpo, sep, cs = linha[1:].rpartition("*")
    if not sep or len(cs) != 2 or not re.fullmatch(r"[0-9A-F]{2}", cs) or checksum(corpo) != int(cs, 16):
        return None
    return corpo.split(",")


def montar_quadro(amostras: np.ndarray, grupo: int, id_seg: int, versao: int = VERSAO_QUADRO,
                  formato: int = FORMATO_12K8) -> bytes:
    """O quadro do PC para a placa: QSEG, cabeçalho, CRC-32 do zlib (bytes 4..13 e corpo) e corpo."""
    a = np.asarray(amostras)
    if a.dtype != np.int16 or a.size != N_AMOSTRAS:
        raise ValueError(f"o corpo tem de ser {N_AMOSTRAS} amostras int16 (veio {a.size} {a.dtype})")
    if not 0 <= id_seg < 2 ** 31:
        raise ValueError(f"id {id_seg} fora de 0..2^31-1")
    corpo = a.astype("<i2").tobytes()
    cab = struct.pack("<BBHIH", versao, formato, grupo, id_seg, N_AMOSTRAS)
    return MAGICA + cab + struct.pack("<I", zlib.crc32(cab + corpo)) + corpo


def f32_de_hex(h: str) -> float:
    return float(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])


def hex_de_f32(x: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', x))[0]:08X}"


def decisao_de(escore: float) -> str:
    """A LDA dá falha com escore >= 0 (o mesmo sentido na placa e no Python)."""
    return "falha" if escore >= 0 else "normal"


# --------------------------------------------------------------------------- #
# Validação de uma resposta (a mesma na rodada e na análise)                   #
# --------------------------------------------------------------------------- #
@dataclass
class Resposta:
    grupo: int
    id: int
    escore: float          # o float32 da placa, pelo escore_hex
    hexa: str
    decisao: str
    n: int
    ms: int
    ciclos: int
    voltas: int | None
    estouros: int


def ler_rx(rx: str) -> dict | None:
    m = RE_RX.match(rx or "")
    if not m:
        return None
    kv = dict(RE_KV.findall(rx))
    return {"grupo": int(m.group(1)), "id": int(m.group(2)), "bytes": int(m.group(3)),
            "ms": int(m.group(4)), "ciclos": int(m.group(5)), "hexa": m.group(6),
            "voltas": int(kv["voltas"]) if "voltas" in kv else None,
            "estouros": int(kv.get("estouros", 0))}


def validar_resposta(rx: str | None, seg: list[str]) -> tuple[Resposta | None, str]:
    """
    Um $SEG de origem S (campos já com o checksum conferido) e a linha rx que veio antes
    dele. Devolve (Resposta, "") ou (None, motivo da rejeição).
    """
    if len(seg) != 1 + len(CAMPOS_SEG) or seg[0] != "SEG" or seg[1] != "1":
        return None, "seg_fora_do_formato"
    p = dict(zip(CAMPOS_SEG, seg[1:]))
    if p["origem"] != "S":
        return None, "origem"
    r = ler_rx(rx)
    if r is None:
        return None, "sem_rx"
    if (r["grupo"], r["id"]) != (int(p["clipe"]), int(p["segmento"])):
        return None, "rx_de_outro_segmento"
    if r["estouros"] > 0:
        return None, "estouro"                  # regra 7
    if p["disparou"] == "4":
        return None, "erro_de_conta"            # o firmware não conseguiu calcular
    escore = f32_de_hex(r["hexa"])
    txt = p["escore_lda"]
    if not np.isfinite(escore) or abs(escore - float(txt)) > MEIA_UNIDADE + FOLGA_TEXTO \
            or (txt.startswith("-") != (escore < 0)):
        return None, "hex_nao_bate_com_o_seg"   # rx corrompida no caminho
    dec = decisao_de(escore)
    if p["decisao"] != ("F" if dec == "falha" else "N"):
        return None, "decisao_nao_bate"
    return Resposta(r["grupo"], r["id"], escore, r["hexa"], dec, int(p["n"]), r["ms"], r["ciclos"],
                    r["voltas"], r["estouros"]), ""


def ler_inicio(linha: str) -> dict:
    """A linha 'recepcao inicio modelo=... json=... quadro=... baud=...'."""
    return dict(RE_KV.findall(linha))


def conferir_inicio(info: dict, modelo_id: str, json_sha12: str) -> list[str]:
    """Regra 1: o modelo da placa é o do Python."""
    p = []
    if info.get("modelo") != modelo_id or info.get("json") != json_sha12:
        p.append(f"a placa tem o modelo {info.get('modelo')} (JSON {info.get('json')}); "
                 f"o Python, {modelo_id} (JSON {json_sha12})")
    if info.get("quadro") not in (None, str(TAM_QUADRO)):
        p.append(f"a placa espera quadros de {info.get('quadro')} B, e o PC manda {TAM_QUADRO}")
    return p


# --------------------------------------------------------------------------- #
# Segmentos e referência do Python                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Segmento:
    conjunto: str          # "jung" ou "hust"
    grupo: int
    id: int                # o id do quadro (regra 4)
    nome: str              # normal_seg0029, h_20hz_a3, ...
    classe: str            # normal ou falha (a verdade)
    pcm: np.ndarray        # int16, 12.800 amostras: o que vai no quadro
    escore_py: float       # float64
    decisao_py: str

    @property
    def chave(self) -> tuple[int, int]:
        return (self.grupo, self.id)


def sha12(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:12]


def info_modelo(caminho: Path = MODELO) -> tuple[str, str, np.ndarray, float]:
    """(id, sha256[:12] do JSON, pesos e bias da forma dobrada): o mesmo hash do lda_modelo.h."""
    p = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return (str(p["id"]), sha12(caminho), np.asarray(p["dobrada"]["pesos"], dtype=float),
            float(p["dobrada"]["bias"]))


def escore_python(pcm: np.ndarray, pesos: np.ndarray, bias: float) -> float:
    """Escore da LDA final em float64, do int16 do quadro (como o 09_export_clips)."""
    import config
    import dsp
    m = dsp.mfcc(pcm.astype(np.float64) / config.INT16_FULL, config.FS_TRABALHO)
    return float(np.concatenate([m.mean(axis=0), m.std(axis=0)]) @ pesos + bias)


def carregar_jung(pesos, bias, referencia: Path = REFERENCIA, splits: Path = SPLITS,
                  pcm_dir: Path = PCM_JUNG) -> list[Segmento]:
    import pcm_io
    from validation import particao
    clipes = pcm_io.carregar_clipes(pcm_dir, fs_esperado=12800)
    if pcm_io.verificar_integridade(clipes):
        raise Abortar(f"os .bin de {pcm_dir} não batem com o manifest versionado")
    pcm = {c.rotulo: c.pcm for c in clipes}
    with Path(referencia).open(encoding="utf-8", newline="") as fh:
        csv_ref = {int(r["segmento"]): r for r in csv.DictReader(fh)}
    segs, problemas = [], []
    for s in particao.segmentos_de(particao.carregar(splits)):
        x = pcm[s.rotulo][s.inicio:s.fim]
        e = escore_python(x, pesos, bias)
        r = csv_ref.get(s.id)
        if r is None or abs(e - float(r["escore"])) > 1e-5 or r["previsao"] != decisao_de(e) \
                or r["binario"] != s.binario:
            problemas.append(f"Jung seg {s.id}: escore {e:.6f} ({decisao_de(e)}) e o CSV "
                             f"{r and r['escore']} ({r and r['previsao']})")
        segs.append(Segmento("jung", GRUPOS["jung"], s.id, f"{s.rotulo}_seg{s.id:04d}", s.binario,
                             np.ascontiguousarray(x, dtype=np.int16), e, decisao_de(e)))
    if problemas:
        raise Abortar("a referência do Python não bate com o escores_referencia.csv:\n  "
                      + "\n  ".join(problemas[:10]))
    return segs


def carregar_hust(pesos, bias, divisao: Path = DIVISAO_HUST, pcm_dir: Path = PCM_HUST,
                  escores: Path = ESCORES_HUST) -> list[Segmento]:
    import pcm_io
    clipes = pcm_io.carregar_clipes(pcm_dir, fs_esperado=12800)
    if pcm_io.verificar_integridade(clipes):
        raise Abortar(f"os .bin de {pcm_dir} não batem com o manifest versionado")
    pcm = {c.rotulo: c.pcm for c in clipes}
    d = json.loads(Path(divisao).read_text(encoding="utf-8"))
    if not Path(escores).exists():
        raise Abortar(f"não achei {escores} (versionado): sem ele, a referência do HUST não é conferida")
    with Path(escores).open(encoding="utf-8", newline="") as fh:
        conferencia = {r["segmento"]: float(r["escore_sem_recal"]) for r in csv.DictReader(fh)}
    segs, problemas = [], []
    for i, s in enumerate(d["segmentos"]):                     # regra 4: o índice, a partir de 0
        x = pcm[s["arquivo"]][s["amostra_inicio"]:s["amostra_fim"]]
        e = escore_python(x, pesos, bias)
        if abs(e - conferencia.get(s["id"], np.inf)) > 1e-5:
            problemas.append(f"HUST {s['id']}: escore {e:.6f} e o da recalibração "
                             f"{conferencia.get(s['id'])}")
        segs.append(Segmento("hust", GRUPOS["hust"], i, s["id"], s["binario"],
                             np.ascontiguousarray(x, dtype=np.int16), e, decisao_de(e)))
    if problemas:
        raise Abortar("a referência do HUST não bate com o decisoes_por_segmento.csv:\n  "
                      + "\n  ".join(problemas[:10]))
    return segs


# --------------------------------------------------------------------------- #
# Captura                                                                     #
# --------------------------------------------------------------------------- #
class Captura:
    """As linhas da placa como chegaram (bytes) e as do PC com '#pc '; tudo com CRLF."""

    def __init__(self, caminho: Path | None):
        self.fh = open(caminho, "wb") if caminho else None

    def placa(self, bruta: bytes) -> str:
        if self.fh:
            self.fh.write(bruta + b"\n")
        return bruta.decode("ascii", errors="replace").rstrip("\r")

    def pc(self, texto: str) -> None:
        if self.fh:
            self.fh.write(f"#pc {texto}\r\n".encode("ascii"))
            self.fh.flush()

    def fechar(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


def ler_linhas_captura(caminho: Path) -> list[str]:
    return [b.decode("ascii", errors="replace").rstrip("\r")
            for b in Path(caminho).read_bytes().split(b"\n") if b.strip()]


def faixas(ids: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2,5'."""
    out, ids = [], sorted(ids)
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        out.append(f"{ids[i]}-{ids[j]}" if j > i else f"{ids[i]}")
        i = j + 1
    return ",".join(out)


def de_faixas(texto: str) -> list[int]:
    out = []
    for parte in filter(None, texto.split(",")):
        a, _, b = parte.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# --------------------------------------------------------------------------- #
# Porta: a serial de verdade ou a placa simulada                              #
# --------------------------------------------------------------------------- #
class PortaSerial:
    """pyserial, com DTR e RTS desligados antes de abrir (não reinicia a placa)."""

    def __init__(self, porta: str, baud: int):
        try:
            import serial
        except ImportError:
            raise Abortar("precisa do pyserial: pip install pyserial") from None
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = porta, baud, 0.01
        self.s.dtr = False
        self.s.rts = False
        self.s.open()

    def write(self, dados: bytes) -> None:
        self.s.write(dados)
        self.s.flush()

    def read(self) -> bytes:
        return self.s.read(self.s.in_waiting or 1)

    def close(self) -> None:
        self.s.close()


@dataclass
class Tempos:
    """Os tempos do diálogo; os testes usam valores curtos."""
    folga_s: float = 2.0             # regra 2: tempo de transmissão mais 2 s
    silencio_s: float = 0.3          # antes de reenviar depois de um timeout (a placa espera 200 ms)
    receber_s: float = 1.0           # RECEBER a cada 1 s até o $PED
    max_modo_s: float = 90.0         # para pedir o modo (dá tempo de apertar o RESET)
    aviso_calada_s: float = 4.0      # calada esse tempo: talvez já no modo; pedir o RESET


class PlacaSimulada:
    """
    Uma placa de mentira com o protocolo do firmware, para testar sem a placa.

    Começa na "demonstração" (hello e uma linha demo de tempos em tempos); com
    RECEBER, entra no modo: manda `recepcao inicio`, as linhas `ref`, o $CAB e o $PED;
    e para cada quadro procura a mágica, confere o cabeçalho e o CRC e responde rx e
    $SEG (ou rx de erro e $ERR) e um $PED novo. O escore é o `escore(grupo, id)` em
    float32, mais `ulps(grupo, id)` ulps.

    `falhas` simula a linha e a placa: {(grupo, id, tentativa): ação}, com ação
    "crc" ($ERR crc), "ignora" (sem resposta, como uma mágica perdida), "perde_seg"
    (a rx e o $PED chegam, o $SEG não), "corrompe_seg" (checksum errado), "estouro"
    (rx com estouros=1), "hex_errado" (rx com outro escore), "reinicia" (a placa
    reinicia no meio do quadro), "erro_conta" ($SEG com disparou 4) ou "atrasa" (a
    resposta só sai quando o PC escreve de novo, depois do timeout dele: dá duplicata).
    Com `inicia_no_modo`, a placa começa calada, já no modo (de uma execução anterior).
    """

    def __init__(self, escore, modelo_id: str, json_sha12: str, refs: dict[int, float] | None = None,
                 falhas: dict | None = None, ulps=lambda g, i: 0, demo_s: float = 0.05,
                 versao_rx: int = 2, inicia_no_modo: bool = False):
        self.escore, self.ulps = escore, ulps
        self.modelo_id, self.json_sha12 = modelo_id, json_sha12
        self.refs = refs or {}
        self.falhas = dict(falhas or {})
        self.demo_s, self.versao_rx = demo_s, versao_rx
        self.saida = bytearray()
        self.entrada = bytearray()
        self.no_modo = False
        self.n = 0
        self.pedidos = 0
        self.tentativa: Counter = Counter()
        self.ultima_demo = 0.0
        self.atrasadas = b""
        if inicia_no_modo:
            self.no_modo = True
        else:
            self._linha(b"hello frente1 (simulada)")

    # -- o que a placa manda
    def _linha(self, texto: str | bytes) -> None:
        self.saida += (texto.encode("ascii") if isinstance(texto, str) else texto) + b"\r\n"

    def _escore32(self, grupo: int, id_seg: int) -> float:
        x = np.float32(self.escore(grupo, id_seg))
        for _ in range(abs(self.ulps(grupo, id_seg))):
            x = np.nextafter(x, np.float32(np.inf if self.ulps(grupo, id_seg) > 0 else -np.inf))
        return float(x)

    def _pedido(self) -> None:
        self.pedidos += 1
        self._linha(assinar(f"PED,1,{self.pedidos}"))

    def _entrar(self) -> None:
        self.no_modo = True
        self.pedidos = 0
        self.entrada.clear()
        self._linha(f"recepcao inicio modelo={self.modelo_id} json={self.json_sha12} quadro={TAM_QUADRO} "
                    f"anel=2048 baud={BAUD_PADRAO} silencio_ms=200")
        for i, e in self.refs.items():
            self._linha(f"ref seg={i} clipe=? k=0 escore={e:.3f} escore_hex=0x{hex_de_f32(np.float32(e))}")
        self._linha(assinar(f"CAB,1,{self.modelo_id},{self.json_sha12},26,100"))
        self._pedido()

    def _reiniciar(self) -> None:
        self.no_modo = False
        self.entrada.clear()
        self.n = 0
        self._linha(b"hello frente1 (simulada)")
        self._linha(b"demo (simulada)")

    def _rx_fim(self, estouros: int) -> str:
        if self.versao_rx < 2:
            return ""
        return f" voltas=12 estouros={estouros}" + (" anel=estourado" if estouros else "")

    def _responder(self, grupo: int, id_seg: int) -> None:
        self.tentativa[(grupo, id_seg)] += 1
        acao = self.falhas.get((grupo, id_seg, self.tentativa[(grupo, id_seg)]))
        if acao == "ignora":
            return
        if acao == "atrasa":
            antes = len(self.saida)
            self._responder_certo(grupo, id_seg, None)
            self.atrasadas += bytes(self.saida[antes:])
            del self.saida[antes:]
            return
        if acao == "reinicia":
            self._reiniciar()
            return
        if acao == "crc":
            self._linha(f"rx erro=crc grupo={grupo} id={id_seg} bytes={TAM_QUADRO} descartados=0 "
                        f"ignorados=0 erros_uart=0{self._rx_fim(0)}")
            self._linha(assinar(f"ERR,1,{grupo},{id_seg},crc"))
            self._pedido()
            return
        self._responder_certo(grupo, id_seg, acao)

    def _responder_certo(self, grupo: int, id_seg: int, acao: str | None) -> None:
        e = self._escore32(grupo, id_seg)
        hexa = hex_de_f32(e)
        if acao == "hex_errado":
            hexa = hex_de_f32(np.float32(e + 100.0))
        estouros = 1 if acao == "estouro" else 0
        self._linha(f"rx grupo={grupo} id={id_seg} bytes={TAM_QUADRO} ms=307 ignorados=0 a_mais=0 "
                    f"erros_uart=0 ciclos=5843000 escore_hex=0x{hexa}{self._rx_fim(estouros)}")
        falha = e >= 0
        disparou = "4" if acao == "erro_conta" else str(int(falha))
        corpo = (f"SEG,1,{self.n},S,{grupo},{id_seg},{'F' if falha or acao == 'erro_conta' else 'N'},"
                 f"{disparou},{e:.3f},0,,,5843000")
        self.n += 1
        if acao == "corrompe_seg":
            self._linha(assinar(corpo)[:-2] + "00")
        elif acao != "perde_seg":
            self._linha(assinar(corpo))
        self._pedido()

    def _processar(self) -> None:
        while True:
            k = self.entrada.find(MAGICA)
            if k < 0:
                del self.entrada[:max(0, len(self.entrada) - 3)]   # um começo de mágica pode ficar
                return
            del self.entrada[:k]
            if len(self.entrada) < 18:
                return
            versao, formato, grupo, id_seg, n_am = struct.unpack("<BBHIH", bytes(self.entrada[4:14]))
            motivo = ("versao" if versao != VERSAO_QUADRO else "formato" if formato != FORMATO_12K8
                      else "tamanho" if n_am != N_AMOSTRAS else "")
            if motivo:
                del self.entrada[:]
                self._linha(f"rx erro={motivo} grupo={grupo} id={id_seg} bytes=18 descartados=0")
                self._linha(assinar(f"ERR,1,{grupo},{id_seg},{motivo}"))
                self._pedido()
                return
            if len(self.entrada) < TAM_QUADRO:
                return
            q = bytes(self.entrada[:TAM_QUADRO])
            del self.entrada[:TAM_QUADRO]
            (crc,) = struct.unpack("<I", q[14:18])
            if crc != zlib.crc32(q[4:14] + q[18:]):
                self._linha(f"rx erro=crc grupo={grupo} id={id_seg} bytes={TAM_QUADRO}")
                self._linha(assinar(f"ERR,1,{grupo},{id_seg},crc"))
                self._pedido()
                continue
            self._responder(grupo, id_seg)
            if not self.no_modo:
                return

    # -- a interface da porta
    def write(self, dados: bytes) -> None:
        if self.atrasadas:
            self.saida += self.atrasadas
            self.atrasadas = b""
        if self.no_modo:
            self.entrada += dados
            self._processar()
        elif b"RECEBER" in dados:
            self._entrar()

    def read(self) -> bytes:
        agora = time.monotonic()
        if not self.no_modo and agora - self.ultima_demo >= self.demo_s:
            self.ultima_demo = agora
            self._linha(b"demo (simulada)")
        if not self.saida:
            time.sleep(0.001)
            return b""
        k = self.saida.find(b"\n") + 1 or len(self.saida)   # uma linha por leitura
        out = bytes(self.saida[:k])
        del self.saida[:k]
        return out

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# A rodada (envio e captura)                                                  #
# --------------------------------------------------------------------------- #
class Rodada:
    def __init__(self, porta, captura: Captura, modelo_id: str, json_sha12: str, baud: int,
                 tempos: Tempos = Tempos(), mostrar=print):
        self.porta, self.cap = porta, captura
        self.modelo_id, self.json_sha12 = modelo_id, json_sha12
        self.baud, self.t = baud, tempos
        self.mostrar = mostrar
        self.resto = b""
        self.reinicios = 0
        self.viu_ped = False
        self.ultimo_byte = time.monotonic()

    # -- linhas
    def linha(self, ate: float) -> str | None:
        """A próxima linha da placa até o instante `ate` (monotonic), gravada na captura."""
        while True:
            k = self.resto.find(b"\n")
            if k >= 0:
                bruta, self.resto = self.resto[:k], self.resto[k + 1:]
                return self.cap.placa(bruta)
            if time.monotonic() >= ate:
                return None
            dados = self.porta.read()
            if dados:
                self.resto += dados
                self.ultimo_byte = time.monotonic()

    def esperar_silencio(self) -> bool:
        """Lê até a linha ficar `silencio_s` calada (no máximo 5 s); True se a placa reiniciou."""
        reiniciou = False
        fim = time.monotonic() + 5.0
        self.ultimo_byte = time.monotonic()
        while time.monotonic() < fim and time.monotonic() - self.ultimo_byte < self.t.silencio_s:
            l = self.linha(time.monotonic() + 0.01)
            if l is not None and e_reinicio(l):
                reiniciou = True
        return reiniciou

    # -- regra 1 e entrada no modo
    def entrar_no_modo(self) -> None:
        self.cap.pc("pedir_modo")
        inicio_t = time.monotonic()
        proximo = 0.0
        avisou = False
        conferido = False
        while time.monotonic() - inicio_t < self.t.max_modo_s:
            agora = time.monotonic()
            if agora >= proximo:
                self.porta.write(b"RECEBER\n")
                proximo = agora + self.t.receber_s
            l = self.linha(time.monotonic() + 0.05)
            if l is None:
                if not avisou and time.monotonic() - self.ultimo_byte > self.t.aviso_calada_s:
                    self.mostrar("a placa está calada: se ela já está no modo de recepção (de uma execução "
                                 "anterior), aperte o RESET para ela mandar o modelo de novo")
                    avisou = True
                continue
            if l.startswith("recepcao recusada"):
                raise Abortar(f"a placa recusou o modo: {l}")
            if l.startswith("recepcao inicio"):
                problemas = conferir_inicio(ler_inicio(l), self.modelo_id, self.json_sha12)
                if problemas:                    # regra 1: nada sai
                    raise Abortar("; ".join(problemas))
                conferido = True
            elif l.startswith("$CAB") and conferido:
                c = ler_pacote(l)
                if c and (c[2], c[3]) != (self.modelo_id, self.json_sha12):
                    raise Abortar(f"o $CAB da placa tem o modelo {c[2]} (JSON {c[3]})")
            elif l.startswith("$PED") and conferido and ler_pacote(l):
                self.mostrar(f"placa no modo de recepção, modelo {self.modelo_id} conferido")
                return
        raise Abortar(f"a placa não entrou no modo de recepção em {self.t.max_modo_s:.0f} s "
                      "(firmware com a recepção gravado? picocom fechado? baud certo?)")

    # -- um envio
    def esperar_resposta(self, s: Segmento, fim: float) -> str:
        """
        'ok', 'reinicio', 'timeout', 'err:<motivo>' ou 'rejeitado:<motivo>'.

        Só conta o que é do segmento mandado: um $SEG ou um $ERR de outro id (a resposta
        atrasada de um reenvio anterior) e um $PED sem resposta antes dele (o pedido que
        veio com essa resposta atrasada) vão só para a captura. Sem isso, um $PED atrasado
        faria o PC mandar de novo enquanto a placa ainda recebe o quadro.
        """
        rx = None
        achado = None              # ("ok", Resposta) | ("err", motivo) | ("rejeitado", motivo)
        self.viu_ped = False
        while True:
            l = self.linha(fim)
            if l is None:
                break
            if e_reinicio(l):
                return "reinicio"
            if l.startswith("rx "):
                rx = l
                continue
            campos = ler_pacote(l)
            if campos is None:
                continue                             # checksum errado: a análise conta
            if campos[0] == "SEG":
                if len(campos) > 3 and campos[3] in ORIGENS_FORA_DO_MODO:
                    return "reinicio"
                resp, motivo = validar_resposta(rx, campos)
                rx = None
                if resp is None:
                    achado = achado or ("rejeitado", motivo)
                elif resp.grupo == s.grupo and resp.id == s.id:
                    achado = ("ok", resp)
                # um $SEG de outro segmento (atrasado, de um reenvio anterior) só vai para a captura
            elif campos[0] == "ERR":
                rx = None
                do_mandado = len(campos) == 5 and (campos[2:4] == ["", ""]
                                                   or campos[2:4] == [str(s.grupo), str(s.id)])
                if do_mandado and not (achado and achado[0] == "ok"):
                    achado = ("err", campos[-1])
            elif campos[0] == "PED":
                rx = None                            # uma rx velha não casa com o próximo $SEG
                if achado is not None:
                    self.viu_ped = True
                    break
        if achado is None:
            return "timeout"
        return "ok" if achado[0] == "ok" else f"{achado[0]}:{achado[1]}"

    def enviar(self, segs: list[Segmento]) -> dict:
        """Manda os segmentos, com as regras 2, 3, 5 e 7. Devolve um resumo do envio."""
        resumo = Counter()
        t_tx = TAM_QUADRO * 10.0 / self.baud if self.baud else 0.0
        precisa_modo = True
        for k, s in enumerate(segs, 1):
            tentativa = 0
            while True:
                if precisa_modo:
                    self.entrar_no_modo()
                    precisa_modo = False
                if tentativa == TENTATIVAS:
                    self.cap.pc(f"nao_processado grupo={s.grupo} id={s.id} tentativas={tentativa}")
                    self.mostrar(f"  {s.nome}: NÃO PROCESSADO depois de {tentativa} tentativas")
                    resumo["nao_processados"] += 1
                    break
                tentativa += 1
                quadro = montar_quadro(s.pcm, s.grupo, s.id)
                self.cap.pc(f"envio grupo={s.grupo} id={s.id} tentativa={tentativa} nome={s.nome}")
                self.porta.write(quadro)
                status = self.esperar_resposta(s, time.monotonic() + t_tx + self.t.folga_s)
                if status == "ok":
                    resumo["ok"] += 1
                    break
                if status == "reinicio":
                    self.reinicios += 1
                    tentativa -= 1                  # regra 5: o reinício não gasta tentativa
                    self.cap.pc(f"reinicio grupo={s.grupo} id={s.id}")
                    self.mostrar(f"  a placa reiniciou em {s.nome}: pedindo o modo de novo")
                    if self.reinicios > MAX_REINICIOS:
                        raise Abortar(f"a placa reiniciou {self.reinicios} vezes; a rodada parou")
                    precisa_modo = True
                    continue
                resumo[status.split(":")[0]] += 1
                self.cap.pc(f"falhou grupo={s.grupo} id={s.id} tentativa={tentativa} motivo={status}")
                if not self.viu_ped:
                    # sem o $PED: espera a placa calar antes de reenviar (a mágica pode ter se
                    # perdido, ou a placa ainda espera o silêncio depois de um erro)
                    if self.esperar_silencio():
                        self.reinicios += 1
                        self.cap.pc(f"reinicio grupo={s.grupo} id={s.id}")
                        self.mostrar(f"  a placa reiniciou em {s.nome}: pedindo o modo de novo")
                        if self.reinicios > MAX_REINICIOS:
                            raise Abortar(f"a placa reiniciou {self.reinicios} vezes; a rodada parou")
                        precisa_modo = True
            if k % 25 == 0 or k == len(segs):
                self.mostrar(f"  {k}/{len(segs)} ({s.nome})")
        resumo["reinicios"] = self.reinicios
        return dict(resumo)


def e_reinicio(linha: str) -> bool:
    return linha.startswith(REINICIO)


# --------------------------------------------------------------------------- #
# Análise da captura                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Analise:
    inicios: list[dict] = field(default_factory=list)
    plano: dict[int, list[int]] = field(default_factory=dict)
    respostas: dict[tuple[int, int], Resposta] = field(default_factory=dict)
    duplicatas: dict[tuple[int, int], list[Resposta]] = field(default_factory=dict)
    envios: Counter = field(default_factory=Counter)
    rejeitadas: Counter = field(default_factory=Counter)
    erros: Counter = field(default_factory=Counter)
    falhas_pc: Counter = field(default_factory=Counter)
    refs: dict[int, str] = field(default_factory=dict)
    estouros: int = 0
    checksum_errado: int = 0
    reinicios: int = 0
    nao_processados_pc: set = field(default_factory=set)
    cabecalho: dict = field(default_factory=dict)
    problemas: list[str] = field(default_factory=list)


def analisar(linhas: list[str], modelo_id: str, json_sha12: str) -> Analise:
    a = Analise()
    rx = None
    for num, l in enumerate(linhas, 1):
        if l.startswith("#pc "):
            texto = l[4:]
            kv = dict(RE_KV.findall(texto))
            if texto.startswith("rodada "):
                a.cabecalho.update(kv)
            elif texto.startswith("plano "):
                a.plano[int(kv["grupo"])] = de_faixas(kv["ids"])
            elif texto.startswith("envio "):
                a.envios[(int(kv["grupo"]), int(kv["id"]))] += 1
            elif texto.startswith("falhou "):
                a.falhas_pc[kv["motivo"].split(":")[0]] += 1
            elif texto.startswith("reinicio"):
                a.reinicios += 1
            elif texto.startswith("nao_processado "):
                a.nao_processados_pc.add((int(kv["grupo"]), int(kv["id"])))
            continue
        if l.startswith("recepcao inicio"):
            info = ler_inicio(l)
            a.inicios.append(info)
            a.problemas += [f"linha {num}: {p}" for p in conferir_inicio(info, modelo_id, json_sha12)]
        elif l.startswith("ref "):
            kv = dict(RE_KV.findall(l))
            if "seg" in kv and "escore_hex" in kv:
                a.refs[int(kv["seg"])] = kv["escore_hex"][2:].upper()
        elif l.startswith("rx "):
            rx = l
            r = ler_rx(l)
            kv = dict(RE_KV.findall(l))
            a.estouros += int(kv.get("estouros", 0)) if (r or l.startswith("rx erro")) else 0
        elif l.startswith("$"):
            campos = ler_pacote(l)
            if campos is None:
                a.checksum_errado += 1
                continue
            if campos[0] == "CAB" and len(campos) > 3 and a.inicios \
                    and (campos[2], campos[3]) != (modelo_id, json_sha12):
                a.problemas.append(f"linha {num}: $CAB com o modelo {campos[2]} (JSON {campos[3]})")
            elif campos[0] in ("ERR", "PED"):
                rx = None                            # uma rx velha não casa com o próximo $SEG
                if campos[0] == "ERR":
                    a.erros[campos[-1]] += 1
            elif campos[0] == "SEG" and len(campos) > 3 and campos[3] == "S":
                resp, motivo = validar_resposta(rx, campos)
                rx = None
                if resp is None:
                    a.rejeitadas[motivo] += 1
                    continue
                chave = (resp.grupo, resp.id)
                if chave in a.respostas:
                    a.duplicatas.setdefault(chave, []).append(resp)       # regra 3
                    if resp.hexa != a.respostas[chave].hexa:
                        a.problemas.append(f"linha {num}: duplicata de {chave} com outro escore "
                                           f"(0x{resp.hexa} e 0x{a.respostas[chave].hexa})")
                else:
                    a.respostas[chave] = resp
    if not a.inicios:
        a.problemas.append("a captura não tem a linha 'recepcao inicio': o modelo não foi conferido")
    if not a.plano:
        a.problemas.append("a captura não tem o plano do PC (#pc plano)")
    return a


def matriz(pares) -> dict[tuple[str, str], int]:
    m = {(x, y): 0 for x in CLASSES for y in CLASSES}
    for x, y in pares:
        m[(x, y)] += 1
    return m


def resultado(a: Analise, segs: list[Segmento]) -> dict:
    """Uma linha por segmento planejado e o resumo por conjunto."""
    por_chave = {s.chave: s for s in segs}
    linhas, faltam = [], []
    for grupo, ids in sorted(a.plano.items()):
        for i in ids:
            s = por_chave.get((grupo, i))
            if s is None:
                faltam.append((grupo, i))
                continue
            r = a.respostas.get(s.chave)
            x = {"conjunto": s.conjunto, "grupo": s.grupo, "id": s.id, "segmento": s.nome,
                 "classe": s.classe, "escore_python": s.escore_py, "decisao_python": s.decisao_py,
                 "tentativas": a.envios.get(s.chave, 0),
                 "duplicatas": len(a.duplicatas.get(s.chave, []))}
            if r is None:
                x.update(estado="nao_processado")
            else:
                tol = TOL_REL * abs(s.escore_py)
                dif = abs(r.escore - s.escore_py)
                x.update(estado="processado", escore_placa=r.escore, escore_hex=r.hexa, dif=dif, tol=tol,
                         frac_tol=dif / tol, decisao_placa=r.decisao, n=r.n, ms=r.ms, ciclos=r.ciclos)
            linhas.append(x)
    conjuntos = {}
    for c in ("jung", "hust"):
        ls = [x for x in linhas if x["conjunto"] == c]
        if not ls:
            continue
        ok = [x for x in ls if x["estado"] == "processado"]
        pior = max(ok, key=lambda x: x["frac_tol"]) if ok else None
        conjuntos[c] = {
            "n": len(ls), "processados": len(ok), "nao_processados": [x["segmento"] for x in ls if x["estado"] != "processado"],
            "concordancia": sum(x["decisao_placa"] == x["decisao_python"] for x in ok),
            "m_placa": matriz((x["classe"], x["decisao_placa"]) for x in ok),
            "m_python": matriz((x["classe"], x["decisao_python"]) for x in ls),
            "m_concordancia": matriz((x["decisao_python"], x["decisao_placa"]) for x in ok),
            "discordantes": [x["segmento"] for x in ok if x["decisao_placa"] != x["decisao_python"]],
            "fora_da_tolerancia": [x["segmento"] for x in ok if x["frac_tol"] > 1.0],
            "pior": pior,
            "frac_mediana": float(np.median([x["frac_tol"] for x in ok])) if ok else None,
            "ms_mediana": float(np.median([x["ms"] for x in ok])) if ok else None,
            "ciclos_mediana": float(np.median([x["ciclos"] for x in ok])) if ok else None,
            "tentativas_extras": sum(max(0, x["tentativas"] - 1) for x in ls),
            "duplicatas": sum(x["duplicatas"] for x in ls),
            "menor_abs_escore": min(abs(x["escore_python"]) for x in ls),
        }
    # os segmentos dos clipes da flash (linhas ref) estão no Jung: o escore pela serial
    # tem de ser bit a bit o da flash (o PR #11 já mostrou nos 10)
    clipes = [(i, a.refs[i], a.respostas[(GRUPOS["jung"], i)].hexa) for i in sorted(a.refs)
              if (GRUPOS["jung"], i) in a.respostas]
    return {"linhas": linhas, "conjuntos": conjuntos, "faltam_na_referencia": faltam,
            "clipes": clipes, "clipes_iguais": sum(r == s for _, r, s in clipes)}


# --------------------------------------------------------------------------- #
# Saídas                                                                      #
# --------------------------------------------------------------------------- #
def br(x: float, casas: int) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, casas: int = 1) -> str:
    return br(100 * x, casas) + " %"


def f32_txt(x: float) -> str:
    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def escrever_csv(res: dict, destino: Path) -> None:
    campos = ["conjunto", "grupo", "id", "segmento", "classe", "estado", "escore_python", "escore_placa",
              "escore_hex", "dif", "tol", "frac_tol", "decisao_python", "decisao_placa", "tentativas",
              "duplicatas", "n", "ms", "ciclos"]
    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for x in res["linhas"]:
            y = dict(x, escore_python=f"{x['escore_python']:.6f}")
            if x["estado"] == "processado":
                y.update(escore_placa=f32_txt(x["escore_placa"]), dif=f"{x['dif']:.6g}",
                         tol=f"{x['tol']:.6f}", frac_tol=f"{x['frac_tol']:.4f}")
            w.writerow(y)


def _md_matriz(m: dict, linhas_rot: str, colunas_rot: str) -> list[str]:
    return [f"| {linhas_rot} \\ {colunas_rot} | normal | falha |", "|---|---:|---:|",
            *(f"| {x} | {m[(x, 'normal')]} | {m[(x, 'falha')]} |" for x in CLASSES)]


LEITURA = {
    "jung": "Os 295 segmentos estão no treino do modelo final: a matriz \"verdadeira × placa\" é acurácia "
            "de treino; o que se mede é a fidelidade do porte (concordância e fração da tolerância).",
    "hust": "A LDA do Jung aplicada ao HUSTmotor sem recalibrar (o C1 do Protocolo C, `exp274`): a matriz "
            "\"verdadeira × placa\" repete a do Python, e o que se mede é a concordância. Os escores são "
            "menores que os do Jung, então a mesma diferença absoluta pesa mais na fração da tolerância.",
}


def escrever_md(res: dict, a: Analise, meta: dict, destino: Path) -> None:
    t = [f"# Matriz do sistema embarcado pela serial ({meta['captura_nome']})", "",
         f"Modelo {meta['modelo']} (JSON {meta['json']}); captura sha {meta['captura_sha']}; "
         f"{meta.get('baud', '?')} bit/s. Escore da placa pelo `escore_hex` da linha `rx` (float32 exato), "
         f"contra o float64 do Python; tolerância encadeada de 10⁻⁵ relativo (A4).", ""]
    for c, r in res["conjuntos"].items():
        nome = NOMES_CONJUNTO[c]
        t += [f"## {nome} ({r['processados']}/{r['n']} processados)", "", LEITURA[c], "",
              f"- Concordância placa × Python: **{r['concordancia']}/{r['processados']}**"
              + (f"; discordantes: {', '.join(r['discordantes'])}" if r["discordantes"] else "") + "."]
        if r["pior"]:
            p = r["pior"]
            t.append(f"- Fração da tolerância: pior caso {pct(p['frac_tol'])} ({p['segmento']}, dif "
                     f"{br(p['dif'], 6)} de {br(p['tol'], 6)}), mediana {pct(r['frac_mediana'])}"
                     + (f"; **fora da tolerância: {', '.join(r['fora_da_tolerancia'])}**"
                        if r["fora_da_tolerancia"] else "") + ".")
            ciclos = (f"mediana de {br(r['ciclos_mediana'], 0)} ciclos ({br(r['ciclos_mediana'] / 1e5, 1)} ms "
                      "a 100 MHz)" if r["ciclos_mediana"] else "sem medida (sem o DWT: emulador ou simulada)")
            t.append(f"- Recepção: mediana de {br(r['ms_mediana'], 0)} ms por quadro, da mágica ao último "
                     f"byte; processamento, {ciclos}.")
        t.append(f"- Menor |escore| do Python: {br(r['menor_abs_escore'], 1)} (a tolerância é "
                 f"10⁻⁵ dele).")
        t.append(f"- Não processados: {len(r['nao_processados'])}"
                 + (f" ({', '.join(r['nao_processados'])})" if r["nao_processados"] else "")
                 + f"; tentativas a mais: {r['tentativas_extras']}; duplicatas: {r['duplicatas']}.")
        t += ["", "**Placa**", "", *_md_matriz(r["m_placa"], "verdadeira", "placa"), "",
              "**Python**", "", *_md_matriz(r["m_python"], "verdadeira", "Python"), "",
              "**Concordância (linhas: Python; colunas: placa)**", "",
              *_md_matriz(r["m_concordancia"], "Python", "placa"), ""]
    erros = ", ".join(f"{k} {v}" for k, v in sorted(a.erros.items())) or "nenhum"
    rej = ", ".join(f"{k} {v}" for k, v in sorted(a.rejeitadas.items())) or "nenhuma"
    t += ["## Recepção", "",
          f"- Envios: {sum(a.envios.values())}; $ERR da placa: {erros}; respostas rejeitadas pelo PC: {rej}; "
          f"timeouts: {a.falhas_pc.get('timeout', 0)}.",
          f"- Estouros do anel: {a.estouros}; linhas com checksum errado: {a.checksum_errado}; "
          f"reinícios da placa: {a.reinicios}.",
          f"- Clipes da flash pela serial: {res['clipes_iguais']}/{len(res['clipes'])} com o escore bit a bit "
          "igual ao da linha `ref` (a mesma conta da demonstração)"
          + ("." if res["clipes_iguais"] == len(res["clipes"]) else
             "; **diferem: " + ", ".join(str(i) for i, r_, s_ in res["clipes"] if r_ != s_) + "**."), ""]
    destino.write_text("\n".join(t), encoding="utf-8", newline="\n")


def escrever_tex(res: dict, destino: Path) -> None:
    cj = res["conjuntos"]
    t = [r"\begin{table}[H]", r"\centering",
         r"\caption{Cadeia completa na placa com os segmentos mandados pela serial: concordância com o "
         r"Python e diferença do escore em fração da tolerância encadeada ($10^{-5}$ relativo, contra o "
         r"float64). No Jung, todos os segmentos são do treino do modelo final: mede a fidelidade do "
         r"porte, e não o desempenho.}",
         r"\label{tab:serial_placa_resumo}", r"\small",
         r"\begin{tabular}{lrrrrl}", r"\toprule",
         r"\textbf{Conjunto} & \textbf{Processados} & \textbf{Concordância} & "
         r"\textbf{\% da tol. (mediana)} & \textbf{\% da tol. (máx.)} & \textbf{Pior segmento} \\",
         r"\midrule"]
    for c, r in cj.items():
        p = r["pior"]
        cauda = r"--- & --- & --- \\"
        if p:
            nome = p["segmento"].replace("_", r"\_")
            cauda = (f"{br(100 * r['frac_mediana'], 1)} & {br(100 * p['frac_tol'], 1)} & "
                     f"\\texttt{{{nome}}} \\\\")
        t.append(f"{NOMES_CONJUNTO[c]} & {r['processados']}/{r['n']} & "
                 f"{r['concordancia']}/{r['processados']} & " + cauda)
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[H]", r"\centering",
          r"\caption{Matriz de confusão do sistema embarcado (segmentos pela serial) e das decisões do "
          r"Python. No HUSTmotor, a LDA do Jung sem recalibrar (C1 do Protocolo C).}",
          r"\label{tab:serial_placa_matriz}", r"\small",
          r"\begin{tabular}{llrrrr}", r"\toprule",
          r" & & \multicolumn{2}{c}{\textbf{Placa}} & \multicolumn{2}{c}{\textbf{Python}} \\",
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
          r"\textbf{Conjunto} & \textbf{Classe verdadeira} & normal & falha & normal & falha \\", r"\midrule"]
    for c, r in cj.items():
        for j, x in enumerate(CLASSES):
            mp, my = r["m_placa"], r["m_python"]
            t.append(f"{NOMES_CONJUNTO[c] if j == 0 else ''} & {x} & {mp[(x, 'normal')]} & {mp[(x, 'falha')]} & "
                     f"{my[(x, 'normal')]} & {my[(x, 'falha')]} \\\\")
        if c != list(cj)[-1]:
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    destino.write_text("\n".join(t), encoding="utf-8", newline="\n")


def linhas_registry(res: dict, a: Analise, meta: dict, responsavel: str, notas: str) -> list[dict]:
    linhas, base = [], meta["proximo_exp"]
    for c, r in res["conjuntos"].items():
        p = r["pior"]
        linhas.append({
            "id": f"exp{base + len(linhas):03d}",
            "data": date.today().isoformat(),
            "etapa": "validacao_embarcada",
            "script": "validation/matriz_serial_placa.py",
            "git_commit": meta["commit"],
            "parametros": experimentos.kv({
                "conjunto": c, "grupo": GRUPOS[c], "captura": meta["captura_nome"],
                "captura_sha": meta["captura_sha"], "modelo": meta["modelo"], "json": meta["json"],
                "referencia_sha": meta["referencia_sha"] if c == "jung" else None,
                "divisao_hust": meta["divisao_sha"] if c == "hust" else None,
                "baud": meta.get("baud"), "tol_rel": TOL_REL, "tentativas_max": TENTATIVAS}),
            "dataset": (f"jung_{r['n']}seg_treino_{meta['modelo']}" if c == "jung"
                        else f"hustmotor_{r['n']}seg_12800Hz"),
            "metricas": experimentos.kv({
                "n_segmentos": r["n"], "processados": r["processados"],
                "nao_processados": len(r["nao_processados"]),
                "concordancia": f"{r['concordancia']}/{r['processados']}",
                "vn": r["m_placa"][("normal", "normal")], "fp": r["m_placa"][("normal", "falha")],
                "fn": r["m_placa"][("falha", "normal")], "vp": r["m_placa"][("falha", "falha")],
                "frac_tol_max": f"{p['frac_tol']:.4f}" if p else None,
                "seg_frac_tol_max": p["segmento"] if p else None,
                "frac_tol_mediana": f"{r['frac_mediana']:.4f}" if p else None,
                "fora_da_tolerancia": len(r["fora_da_tolerancia"]),
                "tentativas_extras": r["tentativas_extras"], "duplicatas": r["duplicatas"],
                "ms_mediana": f"{r['ms_mediana']:.0f}" if p else None,
                "ciclos_mediana": f"{r['ciclos_mediana']:.0f}" if p else None,
                "clipes_bit_a_bit": f"{res['clipes_iguais']}/{len(res['clipes'])}" if c == "jung" else None,
                "erros_recepcao": sum(a.erros.values()), "estouros": a.estouros,
                "reinicios": a.reinicios}),
            "responsavel": responsavel,
            "notas": notas or ("fidelidade do porte, não desempenho: os segmentos estão no treino do modelo final"
                               if c == "jung" else
                               "LDA do Jung sem recalibrar (C1 do Protocolo C); mede a concordância com o Python"),
        })
    return linhas


# --------------------------------------------------------------------------- #
def caminho_livre(caminho: Path) -> Path:
    """Não sobrescreve uma captura: serial-07-11.log, serial-07-11-2.log, ..."""
    if not caminho.exists():
        return caminho
    k = 2
    while (novo := caminho.with_name(f"{caminho.stem}-{k}{caminho.suffix}")).exists():
        k += 1
    return novo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    fonte = ap.add_mutually_exclusive_group(required=True)
    fonte.add_argument("--porta", help="porta da serial (COM5, /dev/ttyUSB0, /dev/pts/N do emular_placa)")
    fonte.add_argument("--simulada", action="store_true", help="placa simulada em Python, sem a placa")
    fonte.add_argument("--captura", type=Path, help="só refaz a matriz de uma captura salva")
    ap.add_argument("--baud", type=int, default=BAUD_PADRAO, help="o mesmo do firmware (padrão 921600)")
    ap.add_argument("--conjuntos", default="jung,hust", help="jung, hust ou jung,hust")
    ap.add_argument("--limite", type=int, help="só os N primeiros de cada conjunto (ensaio; não registra)")
    ap.add_argument("--out-dir", type=Path, help="pasta das saídas")
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true", help="ensaio: saídas fora do Git, registry intocado")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args(argv)

    conjuntos = [c.strip() for c in args.conjuntos.split(",") if c.strip()]
    if not conjuntos or any(c not in GRUPOS for c in conjuntos):
        ap.error("--conjuntos: jung, hust ou jung,hust")
    registrar = not (args.sem_registro or args.simulada or args.limite)
    if registrar and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada (ou use --sem-registro)")
    hoje = datetime.now().strftime("%d-%m")
    if args.out_dir:
        out_dir = args.out_dir
    elif args.simulada:
        out_dir = SAIDA_SIMULADA
    elif not registrar:
        out_dir = SAIDA_TESTE
    else:
        out_dir = Path("reports/embarcado") / f"serial_{hoje.replace('-', '_')}"

    caminho_cap = None
    try:
        modelo_id, json_sha12, pesos, bias = info_modelo(MODELO)
        print("referência do Python: recalculando os escores do int16 dos quadros...")
        segs = []
        if "jung" in conjuntos:
            segs += carregar_jung(pesos, bias)[: args.limite]
        if "hust" in conjuntos:
            segs += carregar_hust(pesos, bias)[: args.limite]
        print(f"  {len(segs)} segmentos ({', '.join(conjuntos)}), conferidos com os CSV de referência")

        if args.captura:
            caminho_cap = args.captura
            linhas = ler_linhas_captura(caminho_cap)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            caminho_cap = caminho_livre((out_dir if not registrar else CAPTURAS) / f"serial-{hoje}.log")
            caminho_cap.parent.mkdir(parents=True, exist_ok=True)
            cap = Captura(caminho_cap)
            cap.pc(f"rodada inicio data={date.today().isoformat()} porta={args.porta or 'simulada'} "
                   f"baud={args.baud} modelo={modelo_id} json={json_sha12} commit={experimentos.git_short_hash()}")
            for c in conjuntos:
                cap.pc(f"plano grupo={GRUPOS[c]} ids={faixas([s.id for s in segs if s.conjunto == c])}")
            if args.simulada:
                ref = {s.chave: s.escore_py for s in segs}
                clipes = {s.id: s.escore_py for s in segs
                          if s.conjunto == "jung" and s.id in (29, 30, 88, 89, 147, 148, 206, 207, 265, 266)}
                porta = PlacaSimulada(lambda g, i: ref[(g, i)], modelo_id, json_sha12, refs=clipes,
                                      ulps=lambda g, i: (i % 5) - 2)
                tempos = Tempos(folga_s=0.2, silencio_s=0.02, receber_s=0.05, max_modo_s=5)
            else:
                porta = PortaSerial(args.porta, args.baud)
                tempos = Tempos()
            print(f"captura: {caminho_cap}")
            try:
                envio = Rodada(porta, cap, modelo_id, json_sha12, 0 if args.simulada else args.baud,
                               tempos).enviar(segs)
            finally:
                cap.pc("rodada fim")
                cap.fechar()
                porta.close()
            print(f"envio: {envio}")
            linhas = ler_linhas_captura(caminho_cap)

        a = analisar(linhas, modelo_id, json_sha12)
        if a.problemas:
            raise Abortar("a captura não passou nas conferências:\n  " + "\n  ".join(a.problemas))
        res = resultado(a, segs)
        if res["faltam_na_referencia"]:
            raise Abortar(f"o plano da captura tem segmentos fora da referência carregada: "
                          f"{res['faltam_na_referencia'][:5]} (use os mesmos --conjuntos e --limite)")
    except Abortar as e:
        print(f"Abortado: {e}")
        return 1
    except OSError as e:                       # porta ocupada (picocom aberto?) ou inexistente
        print(f"Abortado: não deu para usar a porta ou ler um arquivo: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrompido: a captura parcial ficou em {caminho_cap}; a matriz pode ser refeita "
              "dela com --captura (os segmentos que faltam entram como não processados)")
        return 1

    meta = {"captura_nome": caminho_cap.name, "captura_sha": sha12(caminho_cap), "modelo": modelo_id,
            "json": json_sha12, "referencia_sha": sha12(REFERENCIA), "divisao_sha": sha12(DIVISAO_HUST)[:10],
            "baud": a.cabecalho.get("baud", args.baud), "commit": experimentos.git_short_hash()}
    out_dir.mkdir(parents=True, exist_ok=True)
    escrever_csv(res, out_dir / "segmentos.csv")
    escrever_md(res, a, meta, out_dir / "resumo.md")
    escrever_tex(res, out_dir / "tabelas.tex")
    for c, r in res["conjuntos"].items():
        p = r["pior"]
        print(f"{NOMES_CONJUNTO[c]}: {r['processados']}/{r['n']} processados; concordância "
              f"{r['concordancia']}/{r['processados']}"
              + (f"; pior caso {p['segmento']} com {pct(p['frac_tol'])} da tolerância" if p else ""))
    if res["clipes_iguais"] != len(res["clipes"]):
        print(f"ATENÇÃO: só {res['clipes_iguais']}/{len(res['clipes'])} clipes da flash com o escore pela serial "
              "bit a bit igual ao da linha ref (a placa calcula os dois com a mesma conta)")
    print(f"saídas: {out_dir}/ (captura {caminho_cap})")

    completo = all(len(a.plano.get(GRUPOS[c], [])) == n for c, n in (("jung", 295), ("hust", 40))
                   if c in conjuntos)
    if not registrar:
        print("(ensaio: registry não alterado)")
        return 0
    if not completo:
        print("(a captura não tem o plano inteiro: registry não alterado)")
        return 0
    meta["proximo_exp"] = experimentos.next_exp_number(args.registry)
    novas = linhas_registry(res, a, meta, args.responsavel, args.notas)
    experimentos.append_registry(args.registry, novas)
    print(f"registrado: {novas[0]['id']}–{novas[-1]['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
