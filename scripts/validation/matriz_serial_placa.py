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
   `divisao_rotacao.json`; com `--lista`, os grupos e os ids do `referencias.csv`
   (na robustez ao ruído, 3 = branco e 4 = rosa; ver "Lista de segmentos" abaixo).
5. Reinício: uma linha de fora do modo (`hello`, `demo`, `ponta` ou um $SEG de origem
   R, C, D ou A) no meio da rodada é a placa que reiniciou: o PC pede o modo de novo,
   confere o modelo de novo e retoma do segmento pendente (sem gastar tentativa: o
   envio refeito tem o mesmo número de tentativa na captura).
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
placa × Python. No HUSTmotor, a matriz "verdadeira × placa" é a da LDA do Jung sem
retreino (`exp274` do Protocolo C), e o que se mede é de novo a concordância com o
Python. As duas matrizes ficam separadas, nunca as 335 juntas.

Referência do Python
--------------------
O escore em float64 de cada segmento é recalculado do mesmo int16 que vai no
quadro (`dsp.mfcc` e a forma dobrada do `lda_final.json`, como no `09`). No Jung,
ele tem de bater com o `escores_referencia.csv` (o CSV tem 5 casas) e a decisão,
com a do CSV; no HUST, com o `escore_sem_recal` do `decisoes_por_segmento.csv` da
recalibração (6 casas). Se não bater, o script aborta antes de mandar qualquer coisa.

Lista de segmentos (`--lista`)
------------------------------
Em vez do Jung e do HUSTmotor, manda os segmentos de uma lista exportada: uma pasta
com o `manifest.json` e o `referencias.csv`, e o `.bin` com os int16 na ordem do CSV
(o `bin` do manifesto; `--bin` aponta outro lugar). É o recorte da robustez ao ruído
(`run_varredura_snr.py --exportar-placa`, opção 2 da página de 05/11), o padrão de
`--lista` sem pasta. Antes de mandar qualquer quadro, o script confere:

- o sha256 do `.bin` inteiro contra o do manifesto, e o tamanho;
- o modelo e o JSON do manifesto contra o `lda_final.json`;
- o CSV: a ordem 0..n-1, n linhas como no manifesto, grupos do manifesto, (grupo, id)
  sem repetição e E > 0;
- o escore de cada segmento, recalculado do int16 do `.bin` (`escore_python`),
  contra o do CSV (9 casas), e a decisão.

Cada grupo vira um conjunto (branco e rosa), com matriz e linha de registry
próprias. Além da fração da tolerância relativa do A4, cada segmento sai com
|Δs|/E, com E o limite absoluto do erro do escore do CSV. O critério da rodada (fixado
antes dela, discussão da página "Robustez ao ruído também na placa, pela serial") é
o dos conjuntos da lista: todos os segmentos processados, as decisões iguais às do
Python e |Δs| < E em todos. A fração pela tolerância relativa entra só como
informação (perto do limiar, |s| pequeno, ela cresce sem que o erro do porte cresça).
A captura leva o sha256[:12] do manifesto no cabeçalho, e `--captura` só refaz a
análise com a mesma lista; sem o .bin (que não é versionado), ela usa o escore do
CSV e não registra. Numa rodada registrada, a captura é a
`capturas/serial-ruido-DD-MM.log` e as saídas vão para `reports/embarcado/serial_ruido_DD_MM/`,
separadas das do Jung e do HUST.

Uso (da raiz do repositório)
----------------------------
    python scripts/validation/matriz_serial_placa.py --simulada            # sem placa (eco simulado)
    python scripts/validation/matriz_serial_placa.py --porta COM5 --sem-registro   # ensaio com a placa
    python scripts/validation/matriz_serial_placa.py --porta COM5 --responsavel Amanda   # a rodada
    python scripts/validation/matriz_serial_placa.py --captura reports/embarcado/capturas/serial-07-11.log --sem-registro
    python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --sem-registro --limite 10   # ensaio da lista
    python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --responsavel Amanda          # a rodada da robustez
    python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --ids 3:48,3:3 --pausa 4      # LEDs (ensaio)

Teste dos LEDs (`--ids` e `--pausa`)
------------------------------------
`--ids G:ID,G:ID,...` manda só esses segmentos, na ordem dada (os grupos e ids da
regra 4; com `--lista`, os do `referencias.csv`); `--pausa S` espera S segundos depois
de cada resposta certa e mostra a decisão do Python, a da placa (lida do `$SEG`) e a
cor que o LED tem de ter, pela da placa (verde = normal, vermelho = falha); uma
divergência aparece na hora, marcada com `DIVERGE`, e não só na matriz. No modo de recepção, o LED da decisão fica aceso
até a próxima resposta (seção "Sinalização por LED" do README do firmware), então a
pausa é o tempo de olhar (ou filmar) o LED. Os dois são ensaio: não registram. A
matriz no fim confere, como sempre, a decisão da placa contra a do Python.

Com a placa: feche o picocom (só um programa abre a porta) e grave o firmware com a
recepção. A porta abre a 921.600 bit/s (o padrão do firmware desde 03/11);
`--baud 115200` para um firmware compilado a 115.200. O envio exige a linha rx com
voltas e estouros, que o firmware manda desde 04/11 (PR #12 do firmware); com um
firmware anterior, toda resposta é recusada (`rx_sem_voltas`). A análise de uma
captura antiga (`--captura`) continua lendo as rx sem os dois. O DTR e o RTS ficam
desligados para não reiniciar a placa em adaptadores que os ligam ao reset. Se a placa já estiver no
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

GRUPOS = {"jung": 1, "hust": 2, "branco": 3, "rosa": 4}
COR_LED = {"normal": "verde", "falha": "vermelho"}   # README do firmware, "Sinalização por LED"
NOMES_CONJUNTO = {"jung": "Jung", "hust": "HUSTmotor", "branco": "Ruído branco", "rosa": "Ruído rosa"}
CONJUNTOS_FIXOS = ("jung", "hust")         # os de --conjuntos; os outros vêm de --lista
CLASSES = ("normal", "falha")

CAMPOS_SEG = ("versao", "n", "origem", "clipe", "segmento", "decisao", "disparou",
              "escore_lda", "cod_novidade", "pont_novidade", "lim_novidade", "ciclos")
# a linha rx inteira, campo a campo, como o firmware escreve (recepcao_serial.c); voltas e
# estouros vêm juntos desde o PR #12 do firmware (04/11), e as capturas até 31/10 não têm os dois
RE_RX = re.compile(r"^rx grupo=(\d+) id=(\d+) bytes=(\d+) ms=(\d+) ignorados=(\d+) a_mais=(\d+) "
                   r"erros_uart=(\d+) ciclos=(\d+) escore_hex=0x([0-9A-F]{8})"
                   r"(?: voltas=(\d+) estouros=(\d+)(?: anel=estourado)?)?$")
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
LISTA_PADRAO = Path("reports/validation/varredura_snr/placa")   # recorte da robustez ao ruído
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


def inteiro_kv(kv: dict, chave: str, padrao=None):
    """O campo `chave` de uma linha de texto da placa; None se veio estragado (bytes perdidos na linha)."""
    if chave not in kv:
        return padrao
    return int(kv[chave]) if kv[chave].isdigit() else None


def ler_rx(rx: str, exigir_voltas: bool = False) -> dict | None:
    """
    A linha rx de um segmento, ou None se ela não está inteira. A rx não tem checksum:
    uma serial que perde bytes (o teste de 22/11, a 921.600 bit/s no Windows) pode
    juntar a rx com o $SEG seguinte ("estouros=$SEG,..."), cortar um campo
    ("erros_u0 ciclos=..."), deixar um campo vazio ("estouros=") ou tirar o fim da linha.
    Por isso a linha tem de vir inteira, com todos os campos na ordem do firmware e só
    dígitos nos números; senão é recusada (o segmento vai como "sem_rx" e é reenviado),
    nunca interpretada. `exigir_voltas`: o firmware manda voltas e estouros, então uma rx
    sem os dois perdeu o fim. O envio sempre exige (o firmware atual sempre manda os
    dois); a análise exige quando alguma rx da captura os tem, porque as capturas até
    31/10 são de um firmware que não os mandava.
    """
    m = RE_RX.match(rx or "")
    if not m or m.group(3) != str(TAM_QUADRO):
        return None
    if exigir_voltas and m.group(10) is None:
        return None
    return {"grupo": int(m.group(1)), "id": int(m.group(2)), "bytes": int(m.group(3)),
            "ms": int(m.group(4)), "ciclos": int(m.group(8)), "hexa": m.group(9),
            "voltas": int(m.group(10)) if m.group(10) is not None else None,
            "estouros": int(m.group(11)) if m.group(11) is not None else 0}


def validar_resposta(rx: str | None, seg: list[str],
                     exigir_voltas: bool = False) -> tuple[Resposta | None, str]:
    """
    Um $SEG de origem S (campos já com o checksum conferido) e a linha rx que veio antes
    dele. Devolve (Resposta, "") ou (None, motivo da rejeição). `exigir_voltas`: ver ler_rx.
    """
    if len(seg) != 1 + len(CAMPOS_SEG) or seg[0] != "SEG" or seg[1] != "1":
        return None, "seg_fora_do_formato"
    p = dict(zip(CAMPOS_SEG, seg[1:]))
    if p["origem"] != "S":
        return None, "origem"
    r = ler_rx(rx, exigir_voltas)
    if r is None:
        if exigir_voltas and ler_rx(rx) is not None:
            return None, "rx_sem_voltas"        # o fim da linha se perdeu (ou um firmware até 31/10)
        return None, "sem_rx"
    if (r["grupo"], r["id"]) != (int(p["clipe"]), int(p["segmento"])):
        return None, "rx_de_outro_segmento"
    if not p["ciclos"].isdigit() or r["ciclos"] != int(p["ciclos"]):
        return None, "rx_ciclos_nao_bate"       # um dígito perdido na rx, que não tem checksum
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
    E: float | None = None                       # limite absoluto do erro do escore (só na --lista)
    extra: dict = field(default_factory=dict)    # ruido, snr_db, semente, id_jung (só na --lista)

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


@dataclass
class Lista:
    segmentos: list[Segmento]
    manifesto: dict
    manifesto_sha: str     # sha256[:12] do manifest.json: vai no cabeçalho da captura
    referencias_sha: str   # sha256[:12] do referencias.csv
    bin_sha256: str        # o do .bin inteiro, conferido contra o do manifesto ("" sem o .bin)

    @property
    def bin_conferido(self) -> bool:
        return bool(self.bin_sha256)


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with Path(caminho).open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_lista(pasta: Path, modelo_id: str, json_sha12: str, pesos, bias, bin_path: Path | None = None,
                   escore=escore_python, exigir_bin: bool = True) -> Lista:
    """
    Os segmentos de uma lista exportada (manifest.json, referencias.csv e o .bin), conferidos.

    Com `exigir_bin=False` (só para refazer a análise de uma captura, `--captura`) e sem o
    .bin, o escore do Python é o do CSV (9 casas, conferido pelo exportador com o
    `escore_python`), os segmentos vêm sem o int16 e a lista não serve para registrar.
    """
    pasta = Path(pasta)
    man_path, ref_path = pasta / "manifest.json", pasta / "referencias.csv"
    for c in (man_path, ref_path):
        if not c.exists():
            raise Abortar(f"--lista: não achei {c}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    problemas = []
    if (str(man.get("modelo")), man.get("json_sha12")) != (modelo_id, json_sha12):
        problemas.append(f"a lista é do modelo {man.get('modelo')} (JSON {man.get('json_sha12')}); "
                         f"o Python, {modelo_id} (JSON {json_sha12})")
    if man.get("amostras_por_segmento") != N_AMOSTRAS or man.get("fs_hz") != 12800 \
            or man.get("dtype") != "int16 little-endian":
        problemas.append("o manifesto não descreve segmentos de 12.800 amostras int16 little-endian a 12,8 kHz")
    grupos_man = {int(g): str(r) for r, g in (man.get("grupos") or {}).items()}
    if not grupos_man or any(r in CONJUNTOS_FIXOS or GRUPOS.get(r) != g for g, r in grupos_man.items()):
        problemas.append(f"grupos do manifesto inválidos: {man.get('grupos')} (o script conhece "
                         f"{ {r: g for r, g in GRUPOS.items() if r not in CONJUNTOS_FIXOS} }; 1 e 2 são do "
                         "Jung e do HUST)")
    if problemas:
        raise Abortar("--lista: " + "; ".join(problemas))

    bin_path = Path(bin_path or man["bin"])
    n = int(man["n_segmentos"])
    pcm, sha = None, ""
    if bin_path.exists():
        tam = bin_path.stat().st_size
        if tam != n * N_AMOSTRAS * 2:
            raise Abortar(f"--lista: {bin_path} tem {tam} B; o manifesto pede {n} × {N_AMOSTRAS} amostras int16")
        sha = sha256_arquivo(bin_path)
        if sha != man.get("sha256"):
            raise Abortar(f"--lista: o sha256 de {bin_path} é {sha[:12]}…, e o do manifesto, "
                          f"{str(man.get('sha256'))[:12]}…: o .bin não é o desta lista")
        pcm = np.fromfile(bin_path, dtype="<i2").reshape(n, N_AMOSTRAS)
    elif exigir_bin:
        raise Abortar(f"--lista: não achei {bin_path} (não versionado; refaça com "
                      "scripts/validation/run_varredura_snr.py --exportar-placa)")

    with ref_path.open(encoding="utf-8", newline="") as fh:
        linhas = list(csv.DictReader(fh))
    if len(linhas) != n:
        raise Abortar(f"--lista: o referencias.csv tem {len(linhas)} linhas, e o manifesto, {n} segmentos")
    segs, vistos = [], set()
    for k, r in enumerate(linhas):
        g, i = int(r["grupo"]), int(r["id"])
        if int(r["ordem"]) != k:
            problemas.append(f"linha {k}: ordem {r['ordem']} (o .bin segue a ordem do CSV)")
        if g not in grupos_man or grupos_man[g] != r["ruido"]:
            problemas.append(f"{r['nome']}: grupo {g} com o ruído {r['ruido']} (manifesto: {man.get('grupos')})")
        if (g, i) in vistos:
            problemas.append(f"{r['nome']}: (grupo, id) = ({g}, {i}) repetido")
        vistos.add((g, i))
        lim = float(r["E"])
        if not lim > 0:
            problemas.append(f"{r['nome']}: E = {r['E']}")
        e_csv = float(r["escore"])
        if pcm is None:                         # --captura sem o .bin: o escore do CSV
            x, e = np.empty(0, dtype=np.int16), e_csv
        else:
            x = np.ascontiguousarray(pcm[k], dtype=np.int16)
            e = escore(x, pesos, bias)
        if abs(e - e_csv) > 1e-6 * max(1.0, abs(e_csv)) or r["decisao"] != decisao_de(e):
            problemas.append(f"{r['nome']}: escore {e:.9f} ({decisao_de(e)}) do int16 e o CSV "
                             f"{r['escore']} ({r['decisao']})")
        segs.append(Segmento(r["ruido"], g, i, r["nome"], r["classe"], x, e, decisao_de(e), E=lim,
                             extra={"ruido": r["ruido"], "snr_db": int(r["snr_db"]),
                                    "semente": int(r["semente"]), "id_jung": int(r["segmento"])}))
    if problemas:
        raise Abortar(f"--lista: {len(problemas)} problema(s) no referencias.csv:\n  "
                      + "\n  ".join(problemas[:10]))
    return Lista(segs, man, sha12(man_path), sha12(ref_path), sha)


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
            self.fh.flush()       # o painel (run_painel.py --seguir) lê a captura enquanto ela é gravada
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


def linha_pausa(k: int, total: int, s: Segmento, decisao_placa: str, pausa_s: float) -> str:
    """A linha do --pausa: as duas decisões e a cor do LED, que segue a da placa."""
    aviso = "" if decisao_placa == s.decisao_py else "  <- DIVERGE do Python"
    return (f"  {k}/{total} {s.nome}: Python {s.decisao_py}, placa {decisao_placa} -> LED "
            f"{COR_LED[decisao_placa]} ({pausa_s:g} s){aviso}")


def ler_ids(texto: str) -> list[tuple[int, int]]:
    """'3:48,3:3' -> [(3, 48), (3, 3)], na ordem dada (--ids)."""
    pares = []
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        g, sep, i = parte.partition(":")
        if not sep or not g.strip().isdigit() or not i.strip().isdigit():
            raise ValueError(f"--ids: '{parte}' não é grupo:id (ex.: 3:48)")
        pares.append((int(g), int(i)))
    if not pares:
        raise ValueError("--ids vazio")
    if len(set(pares)) != len(pares):
        raise ValueError("--ids com um segmento repetido")
    return pares


def escolher(segs: list[Segmento], pares: list[tuple[int, int]]) -> list[Segmento]:
    """Os segmentos de --ids, na ordem dada; recusa o que não está no plano."""
    por_chave = {s.chave: s for s in segs}
    faltam = [p for p in pares if p not in por_chave]
    if faltam:
        raise Abortar("--ids: " + ", ".join(f"{g}:{i}" for g, i in faltam) + " não estão nos segmentos "
                      "carregados (grupo e id da regra 4; com --lista, os do referencias.csv)")
    return [por_chave[p] for p in pares]


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
    (rx com estouros=1), "hex_errado" (rx com outro escore), "corta_fim_rx" (a rx
    chega sem " voltas=12 estouros=0", o fim que bytes perdidos podem levar),
    "reinicia" (a placa reinicia no meio do quadro), "erro_conta" ($SEG com disparou 4)
    ou "atrasa" (a resposta só sai quando o PC escreve de novo, depois do timeout dele:
    dá duplicata).
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

    def _escore32(self, grupo: int, id_seg: int, valor: float | None = None) -> float:
        """O escore em float32 mais os ulps; a linha ref e a serial usam a mesma conta, como na placa."""
        x = np.float32(self.escore(grupo, id_seg) if valor is None else valor)
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
            e32 = self._escore32(GRUPOS["jung"], i, e)
            self._linha(f"ref seg={i} clipe=? k=0 escore={e32:.3f} escore_hex=0x{hex_de_f32(e32)}")
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
        fim = "" if acao == "corta_fim_rx" else self._rx_fim(estouros)
        self._linha(f"rx grupo={grupo} id={id_seg} bytes={TAM_QUADRO} ms=307 ignorados=0 a_mais=0 "
                    f"erros_uart=0 ciclos=5843000 escore_hex=0x{hexa}{fim}")
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
                 tempos: Tempos = Tempos(), mostrar=print, pausa_s: float = 0.0, dormir=time.sleep):
        self.porta, self.cap = porta, captura
        self.pausa_s, self.dormir = pausa_s, dormir
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
        self.resposta = None       # a Resposta do 'ok' (a linha do --pausa)
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
                # sempre com voltas e estouros: o firmware atual sempre manda os dois, e uma
                # rx sem eles perdeu o fim da linha (a análise da captura também recusa)
                resp, motivo = validar_resposta(rx, campos, exigir_voltas=True)
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
        if achado[0] == "ok":
            self.resposta = achado[1]
            return "ok"
        return f"{achado[0]}:{achado[1]}"

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
                    if self.pausa_s > 0:            # --pausa: o LED da decisão fica aceso até a próxima
                        self.mostrar(linha_pausa(k, len(segs), s, self.resposta.decisao, self.pausa_s))
                        self.dormir(self.pausa_s)
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
    envios: Counter = field(default_factory=Counter)        # quadros mandados, inclusive os refeitos
    tentativas: dict = field(default_factory=dict)          # a maior tentativa de cada segmento (regra 2)
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
    # o firmware desta captura manda voltas e estouros na rx (todos desde 04/11): então
    # uma rx sem eles perdeu o fim da linha
    exigir_voltas = any(l.startswith("rx ") and " voltas=" in l for l in linhas)
    for num, l in enumerate(linhas, 1):
        if l.startswith("#pc "):
            texto = l[4:]
            kv = dict(RE_KV.findall(texto))
            if texto.startswith("rodada "):
                a.cabecalho.update(kv)
            elif texto.startswith("plano "):
                a.plano[int(kv["grupo"])] = de_faixas(kv["ids"])
            elif texto.startswith("envio "):
                chave = (int(kv["grupo"]), int(kv["id"]))
                a.envios[chave] += 1
                a.tentativas[chave] = max(a.tentativas.get(chave, 0), int(kv.get("tentativa", 1)))
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
            r = ler_rx(l, exigir_voltas)
            kv = dict(RE_KV.findall(l))
            a.estouros += (inteiro_kv(kv, "estouros", 0) or 0) if (r or l.startswith("rx erro")) else 0
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
                resp, motivo = validar_resposta(rx, campos, exigir_voltas)
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


def criterio_lista(ls: list[dict]) -> dict:
    """
    Os conjuntos da --lista: |Δs|/E por segmento e por SNR, e o critério da rodada (todos
    processados, decisões iguais às do Python e |Δs| < E em todos). A fração pela tolerância
    relativa do A4 continua nas linhas, só como informação.
    """
    ok = [x for x in ls if x["estado"] == "processado"]
    por_snr = []
    for snr in sorted({x["snr_db"] for x in ls}, reverse=True):
        lp = [x for x in ls if x["snr_db"] == snr]
        okp = [x for x in lp if x["estado"] == "processado"]
        por_snr.append({
            "snr_db": snr, "n": len(lp), "processados": len(okp),
            "sementes": len({x["semente"] for x in lp}),
            "concordancia": sum(x["decisao_placa"] == x["decisao_python"] for x in okp),
            "difE_max": max((x["dif_E"] for x in okp), default=None),
            "frac_max": max((x["frac_tol"] for x in okp), default=None),
            "fora_da_tolerancia": sum(x["frac_tol"] > 1.0 for x in okp),
            "menor_margem_E": min(abs(x["escore_python"]) / x["E"] for x in lp),
        })
    pior_E = max(ok, key=lambda x: x["dif_E"]) if ok else None
    fora_de_E = [x["segmento"] for x in ok if x["dif_E"] >= 1.0]
    discord = [x["segmento"] for x in ok if x["decisao_placa"] != x["decisao_python"]]
    nao = [x["segmento"] for x in ls if x["estado"] != "processado"]
    motivos = ([f"{len(nao)} não processado(s)"] if nao else []) \
        + ([f"{len(discord)} decisão(ões) diferente(s)"] if discord else []) \
        + ([f"{len(fora_de_E)} com |Δs| ≥ E"] if fora_de_E else [])
    return {"por_snr": por_snr, "pior_E": pior_E, "fora_de_E": fora_de_E,
            "difE_mediana": float(np.median([x["dif_E"] for x in ok])) if ok else None,
            "aprovado": not motivos, "motivos": motivos}


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
                 "tentativas": a.tentativas.get(s.chave, 0),
                 "duplicatas": len(a.duplicatas.get(s.chave, []))}
            if r is None:
                x.update(estado="nao_processado")
            else:
                tol = TOL_REL * abs(s.escore_py)
                dif = abs(r.escore - s.escore_py)
                x.update(estado="processado", escore_placa=r.escore, escore_hex=r.hexa, dif=dif, tol=tol,
                         frac_tol=dif / tol, decisao_placa=r.decisao, n=r.n, ms=r.ms, ciclos=r.ciclos)
                if s.E is not None:
                    x["dif_E"] = dif / s.E
            if s.E is not None:
                x.update(s.extra, E=s.E)
            linhas.append(x)
    conjuntos = {}
    for c in dict.fromkeys(x["conjunto"] for x in linhas):          # na ordem dos grupos
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
        if "E" in ls[0]:
            conjuntos[c].update(criterio_lista(ls))
    # os segmentos dos clipes da flash (linhas ref) estão no Jung: o escore pela serial
    # tem de ser bit a bit o da flash (o PR #11 já mostrou nos 10)
    clipes = [(i, a.refs[i], a.respostas[(GRUPOS["jung"], i)].hexa) for i in sorted(a.refs)
              if (GRUPOS["jung"], i) in a.respostas]
    # o que o PC deu como não processado tem de bater com o que a análise acha na captura
    sem_resposta = {(x["grupo"], x["id"]) for x in linhas if x["estado"] != "processado"}
    divergencias = (
        [f"{g}/{i}: o PC desistiu, mas a captura tem um $SEG válido (resposta atrasada)"
         for g, i in sorted(a.nao_processados_pc - sem_resposta)]
        + [f"{g}/{i}: sem resposta válida e sem a desistência do PC (captura interrompida?)"
           for g, i in sorted(sem_resposta - a.nao_processados_pc)])
    return {"linhas": linhas, "conjuntos": conjuntos, "faltam_na_referencia": faltam,
            "clipes": clipes, "clipes_iguais": sum(r == s for _, r, s in clipes),
            "divergencias_pc": divergencias}


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
    lista = any("E" in x for x in res["linhas"])
    if lista:                                   # só na --lista: o CSV do Jung e do HUST não muda
        campos[campos.index("frac_tol") + 1:campos.index("frac_tol") + 1] = ["E", "dif_E"]
        campos[campos.index("classe") + 1:campos.index("classe") + 1] = ["ruido", "snr_db", "semente", "id_jung"]
    with destino.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for x in res["linhas"]:
            y = dict(x, escore_python=f"{x['escore_python']:.6f}")
            if "E" in x:
                y["E"] = f"{x['E']:.6f}"
            if x["estado"] == "processado":
                y.update(escore_placa=f32_txt(x["escore_placa"]), dif=f"{x['dif']:.4g}",
                         tol=f"{x['tol']:.6f}", frac_tol=f"{x['frac_tol']:.4f}")
                if "dif_E" in x:
                    y["dif_E"] = f"{x['dif_E']:.3g}"
            w.writerow(y)


def _md_matriz(m: dict, linhas_rot: str, colunas_rot: str) -> list[str]:
    return [f"| {linhas_rot} \\ {colunas_rot} | normal | falha |", "|---|---:|---:|",
            *(f"| {x} | {m[(x, 'normal')]} | {m[(x, 'falha')]} |" for x in CLASSES)]


LEITURA = {
    "jung": "Os 295 segmentos estão no treino do modelo final: a matriz \"verdadeira × placa\" é acurácia "
            "de treino; o que se mede é a fidelidade do porte (concordância e fração da tolerância).",
    "hust": "A LDA do Jung aplicada ao HUSTmotor sem retreino (`exp274` do Protocolo C): a matriz "
            "\"verdadeira × placa\" repete a do Python, e o que se mede é a concordância. Os escores são "
            "menores que os do Jung, então a mesma diferença absoluta pesa mais na fração da tolerância.",
}
LEITURA_LISTA = ("Os segmentos do Jung com ruído somado no int16 a 51,2 kHz, antes da decimação (recorte da "
                 "varredura de SNR para a placa). O critério da rodada é o das decisões iguais às do Python e "
                 "|Δs| < E em todos os segmentos, com E o limite absoluto do erro do escore do porte; a "
                 "fração da tolerância relativa do A4 é só informação, porque perto do limiar (|s| pequeno) "
                 "ela cresce sem que o erro do porte cresça.")


def escrever_md(res: dict, a: Analise, meta: dict, destino: Path) -> None:
    t = [f"# Matriz do sistema embarcado pela serial ({meta['captura_nome']})", "",
         f"Modelo {meta['modelo']} (JSON {meta['json']}); captura sha {meta['captura_sha']}; "
         f"{meta.get('baud', '?')} bit/s. Os quadros vão já decimados a 12,8 kHz: a placa roda o MFCC e a "
         f"LDA. A decimação fica fora desta rodada; na placa, ela é testada pelo recorte de 1 s a 51,2 kHz "
         f"do segmento 29 (`DECIMACAO OK` e a linha `ponta` no boot desta captura), e no PC, nas 5 "
         f"gravações inteiras (`decimar_gravacao.c` do firmware). "
         f"Escore da placa pelo `escore_hex` da linha `rx` (float32 exato), "
         f"contra o float64 do Python; tolerância encadeada de 10⁻⁵ relativo (A4).", ""]
    for c, r in res["conjuntos"].items():
        nome = NOMES_CONJUNTO[c]
        t += [f"## {nome} ({r['processados']}/{r['n']} processados)", "", LEITURA.get(c, LEITURA_LISTA), ""]
        if "aprovado" in r:
            t.append("- **Critério da rodada: " + ("aprovado" if r["aprovado"] else "reprovado") + "**"
                     + (f" ({'; '.join(r['motivos'])})" if r["motivos"] else "")
                     + ": todos processados, decisões iguais às do Python e |Δs| < E em todos.")
            if r["pior_E"]:
                p = r["pior_E"]
                t.append(f"- |Δs|/E: pior caso {br(p['dif_E'], 4)} ({p['segmento']}, |s| = "
                         f"{br(abs(p['escore_python']), 3)}, E = {br(p['E'], 4)}), mediana "
                         f"{br(r['difE_mediana'], 4)}"
                         + (f"; **|Δs| ≥ E: {', '.join(r['fora_de_E'])}**" if r["fora_de_E"] else "") + ".")
        t.append(f"- Concordância placa × Python: **{r['concordancia']}/{r['processados']}**"
                 + (f"; discordantes: {', '.join(r['discordantes'])}" if r["discordantes"] else "") + ".")
        if r["pior"]:
            p = r["pior"]
            if "aprovado" in r:
                t.append(f"- Fração da tolerância relativa (só informação): pior caso {pct(p['frac_tol'])} "
                         f"({p['segmento']}, |s| = {br(abs(p['escore_python']), 3)}), mediana "
                         f"{pct(r['frac_mediana'])}; acima dela: {len(r['fora_da_tolerancia'])}.")
            else:
                t.append(f"- Fração da tolerância: pior caso {pct(p['frac_tol'])} ({p['segmento']}, dif "
                         f"{br(p['dif'], 6)} de {br(p['tol'], 6)}), mediana {pct(r['frac_mediana'])}"
                         + (f"; **fora da tolerância: {', '.join(r['fora_da_tolerancia'])}**"
                            if r["fora_da_tolerancia"] else "") + ".")
            ciclos = (f"mediana de {br(r['ciclos_mediana'], 0)} ciclos ({br(r['ciclos_mediana'] / 1e5, 1)} ms "
                      "a 100 MHz)" if r["ciclos_mediana"] else "sem medida (sem o DWT: emulador ou simulada)")
            t.append(f"- Recepção: mediana de {br(r['ms_mediana'], 0)} ms por quadro, da mágica ao último "
                     f"byte; processamento (MFCC e LDA, sem a decimação: o quadro já chega a 12,8 kHz), "
                     f"{ciclos}.")
        t.append(f"- Menor |escore| do Python: {br(r['menor_abs_escore'], 1)} (a tolerância é "
                 f"10⁻⁵ dele).")
        t.append(f"- Não processados: {len(r['nao_processados'])}"
                 + (f" ({', '.join(r['nao_processados'])})" if r["nao_processados"] else "")
                 + f"; tentativas a mais: {r['tentativas_extras']}; duplicatas: {r['duplicatas']}.")
        if "por_snr" in r:
            t += ["", "**Por SNR**", "",
                  # as barras de |Δs| e |s| escapadas: sem isso, viram separadores de coluna
                  "| SNR | segmentos | sementes | processados | decisões iguais | \\|Δs\\|/E máx. | "
                  "fração da tol. rel. máx. | acima da tol. rel. | menor \\|s\\|/E |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                  *(f"| {q['snr_db']} dB | {q['n']} | {q['sementes']} | {q['processados']} | "
                    f"{q['concordancia']}/{q['processados']} | "
                    + (f"{br(q['difE_max'], 4)} | {pct(q['frac_max'])} | " if q["difE_max"] is not None
                       else "— | — | ")
                    + f"{q['fora_da_tolerancia']} | {br(q['menor_margem_E'], 1)} |" for q in r["por_snr"])]
        t += ["", "**Placa**", "", *_md_matriz(r["m_placa"], "verdadeira", "placa"), "",
              "**Python**", "", *_md_matriz(r["m_python"], "verdadeira", "Python"), "",
              "**Concordância (linhas: Python; colunas: placa)**", "",
              *_md_matriz(r["m_concordancia"], "Python", "placa"), ""]
    erros = ", ".join(f"{k} {v}" for k, v in sorted(a.erros.items())) or "nenhum"
    rej = ", ".join(f"{k} {v}" for k, v in sorted(a.rejeitadas.items())) or "nenhuma"
    t += ["## Recepção", "",
          f"- Envios: {sum(a.envios.values())} (com os refeitos depois de um reinício, que não gastam "
          f"tentativa); $ERR da placa: {erros}; respostas rejeitadas pelo PC: {rej}; "
          f"timeouts: {a.falhas_pc.get('timeout', 0)}.",
          "- Não processados pelo PC × pela captura: "
          + ("batem." if not res["divergencias_pc"] else "**" + "; ".join(res["divergencias_pc"]) + "**."),
          f"- Estouros do anel: {a.estouros}; linhas com checksum errado: {a.checksum_errado}; "
          f"reinícios da placa: {a.reinicios}."]
    if "jung" in res["conjuntos"]:
        t.append(f"- Clipes da flash pela serial: {res['clipes_iguais']}/{len(res['clipes'])} com o escore bit "
                 "a bit igual ao da linha `ref` (a mesma conta da demonstração)"
                 + ("." if res["clipes_iguais"] == len(res["clipes"]) else
                    "; **diferem: " + ", ".join(str(i) for i, r_, s_ in res["clipes"] if r_ != s_) + "**."))
    if meta.get("lista"):
        lm = meta["lista"]
        t.append(f"- Lista: manifesto {lm['manifesto_sha']}, referencias.csv {lm['referencias_sha']}, "
                 + (f".bin sha256 {lm['bin_sha256'][:12]}… (conferido inteiro antes do primeiro quadro)."
                    if lm["bin_sha256"] else "análise refeita sem o .bin (escore do Python do referencias.csv)."))
    t.append("")
    destino.write_text("\n".join(t), encoding="utf-8", newline="\n")


def escrever_tex_lista(res: dict, destino: Path) -> None:
    """A tabela da --lista: uma linha por ruído e SNR."""
    t = [r"\begin{table}[H]", r"\centering",
         r"\caption{MFCC e LDA na placa com ruído (segmentos do Jung com ruído somado a 51,2~kHz, antes da "
         r"decimação, mandados pela serial a 12,8~kHz). Critério: decisões iguais às do Python e "
         r"$|\Delta s| < E$ em todos os segmentos, com $E$ o limite absoluto do erro do escore do porte; a "
         r"fração da tolerância relativa ($10^{-5}$ do escore) é só informação.}",
         r"\label{tab:serial_placa_ruido}", r"\small",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"\textbf{Ruído} & \textbf{SNR} & \textbf{Segmentos} & \textbf{Concordância} & "
         r"\textbf{$|\Delta s|/E$ máx.} & \textbf{\% tol. rel. máx.} & "
         r"\textbf{Acima da tol. rel.} \\", r"\midrule"]
    cj = res["conjuntos"]
    for c, r in cj.items():
        for j, q in enumerate(r["por_snr"]):
            difE = br(q["difE_max"], 4) if q["difE_max"] is not None else "---"
            frac = br(100 * q["frac_max"], 1) if q["frac_max"] is not None else "---"
            t.append(f"{NOMES_CONJUNTO[c].replace('Ruído ', '') if j == 0 else ''} & {q['snr_db']}~dB & "
                     f"{q['n']} & {q['concordancia']}/{q['processados']} & {difE} & {frac} & "
                     f"{q['fora_da_tolerancia']} \\\\")
        if c != list(cj)[-1]:
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    destino.write_text("\n".join(t), encoding="utf-8", newline="\n")


def escrever_tex(res: dict, destino: Path) -> None:
    cj = res["conjuntos"]
    if any("por_snr" in r for r in cj.values()):
        escrever_tex_lista(res, destino)
        return
    t = [r"\begin{table}[H]", r"\centering",
         r"\caption{MFCC e LDA na placa, com os segmentos já decimados a 12,8~kHz mandados pela serial "
         r"(a decimação fica fora desta rodada; na placa, ela é testada pelo recorte de 1~s a 51,2~kHz do "
         r"segmento 29, no boot da mesma captura, e no PC, nas 5 gravações inteiras): concordância com o "
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
          r"Python. No HUSTmotor, a LDA do Jung sem retreino (\texttt{exp274} do Protocolo C).}",
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


NOTAS_PADRAO = {
    "jung": "fidelidade do porte, não desempenho: os segmentos estão no treino do modelo final",
    "hust": "a LDA do Jung sem retreino (exp274 do Protocolo C); mede a concordância com o Python",
}
NOTA_LISTA = ("robustez ao ruído na placa (opção 2 da página de 05/11): critério decisões iguais e "
              "|Δs| < E em todos os segmentos; a fração da tolerância relativa é só informação")


def linhas_registry(res: dict, a: Analise, meta: dict, responsavel: str, notas: str) -> list[dict]:
    linhas, base = [], meta["proximo_exp"]
    for c, r in res["conjuntos"].items():
        p = r["pior"]
        linhas.append({
            "id": f"exp{base + len(linhas):03d}",
            "data": meta["data"],
            "etapa": "validacao_embarcada",
            "script": "validation/matriz_serial_placa.py",
            "git_commit": meta["commit"],
            "parametros": experimentos.kv({
                "conjunto": c, "grupo": GRUPOS[c], "captura": meta["captura_nome"],
                "captura_sha": meta["captura_sha"], "modelo": meta["modelo"], "json": meta["json"],
                "referencia_sha": meta["referencia_sha"] if c == "jung" else None,
                "divisao_hust": meta["divisao_sha"] if c == "hust" else None,
                **({"lista": meta["lista"]["manifesto_sha"], "referencias_sha": meta["lista"]["referencias_sha"],
                    "bin_sha256": meta["lista"]["bin_sha256"][:12],
                    "criterio": "decisoes_iguais_e_dif_menor_que_E"} if "aprovado" in r else {}),
                "baud": meta.get("baud"), "tol_rel": TOL_REL, "tentativas_max": TENTATIVAS}),
            "dataset": (f"jung_{r['n']}seg_treino_{meta['modelo']}" if c == "jung"
                        else f"hustmotor_{r['n']}seg_12800Hz" if c == "hust"
                        else f"jung_ruido_{c}_{r['n']}seg_12800Hz"),
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
                **({"aprovado": r["aprovado"], "fora_de_E": len(r["fora_de_E"]),
                    "dif_E_max": f"{r['pior_E']['dif_E']:.4g}" if r["pior_E"] else None,
                    "seg_dif_E_max": r["pior_E"]["segmento"] if r["pior_E"] else None}
                   if "aprovado" in r else {}),
                "tentativas_extras": r["tentativas_extras"], "duplicatas": r["duplicatas"],
                "ms_mediana": f"{r['ms_mediana']:.0f}" if p else None,
                "ciclos_mediana": f"{r['ciclos_mediana']:.0f}" if p else None,
                "clipes_bit_a_bit": f"{res['clipes_iguais']}/{len(res['clipes'])}" if c == "jung" else None,
                "erros_recepcao": sum(a.erros.values()), "estouros": a.estouros,
                "reinicios": a.reinicios}),
            "responsavel": responsavel,
            "notas": NOTAS_PADRAO.get(c, NOTA_LISTA) + (f"; {notas}" if notas else ""),   # --notas acrescenta
        })
    return linhas


def registrar_rodada(registry: Path, res: dict, a: Analise, meta: dict, responsavel: str,
                     notas: str) -> list[dict]:
    """
    As linhas da rodada no registry, uma por conjunto. Recusa uma captura que já tem linha
    (o mesmo captura_sha: refazer a matriz com --captura não é rodada nova) e uma captura
    com segmento sem resposta e sem a desistência do PC (interrompida no meio).
    """
    if Path(registry).exists():
        with Path(registry).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                kv = dict(par.split("=", 1) for par in (row.get("parametros") or "").split(";") if "=" in par)
                if kv.get("captura_sha") == meta["captura_sha"]:
                    raise Abortar(f"a captura {meta['captura_nome']} (sha {meta['captura_sha']}) já está no "
                                  f"registry, em {row['id']}")
    incompletos = [d for d in res["divergencias_pc"] if "sem a desistência" in d]
    if incompletos:
        raise Abortar(f"a captura está incompleta ({len(incompletos)} segmentos sem resposta e sem a "
                      "desistência do PC)")
    meta = dict(meta, proximo_exp=experimentos.next_exp_number(registry))
    novas = linhas_registry(res, a, meta, responsavel, notas)
    experimentos.append_registry(registry, novas)
    return novas


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
    ap.add_argument("--conjuntos", default=None, help="jung, hust ou jung,hust (padrão: os dois)")
    ap.add_argument("--lista", type=Path, nargs="?", const=LISTA_PADRAO,
                    help="manda os segmentos de uma lista exportada (pasta com o manifest.json e o "
                         f"referencias.csv; sem pasta, {LISTA_PADRAO.as_posix()}) no lugar do Jung e do HUST")
    ap.add_argument("--bin", type=Path, help="com --lista: o .bin em outro lugar que o do manifesto")
    ap.add_argument("--limite", type=int, help="só os N primeiros de cada conjunto (ensaio; não registra)")
    ap.add_argument("--ids", help="só estes segmentos, na ordem dada: grupo:id,grupo:id,... (ensaio; não registra)")
    ap.add_argument("--pausa", type=float, default=0.0,
                    help="segundos de espera depois de cada resposta certa, com a cor do LED (ensaio; não registra)")
    ap.add_argument("--out-dir", type=Path, help="pasta das saídas")
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true", help="ensaio: saídas fora do Git, registry intocado")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="", help="acrescentado às notas padrão de cada conjunto")
    args = ap.parse_args(argv)

    if args.lista and args.conjuntos:
        ap.error("--lista e --conjuntos não vão juntos: a lista substitui o Jung e o HUST")
    if args.bin and not args.lista:
        ap.error("--bin só vale com --lista")
    if args.pausa < 0:
        ap.error("--pausa não pode ser negativa")
    if (args.ids or args.pausa) and not args.porta and not args.simulada:
        ap.error("--ids e --pausa são do envio (--porta ou --simulada), não de --captura")
    if args.ids and args.limite:
        ap.error("--ids e --limite não vão juntos")
    try:
        pares_ids = ler_ids(args.ids) if args.ids else None
    except ValueError as e:
        ap.error(str(e))
    conjuntos = [c.strip() for c in (args.conjuntos or "jung,hust").split(",") if c.strip()]
    if not args.lista and (not conjuntos or any(c not in CONJUNTOS_FIXOS for c in conjuntos)):
        ap.error("--conjuntos: jung, hust ou jung,hust")
    if args.porta:                              # avisa antes de recalcular os escores
        try:
            import serial  # noqa: F401
        except ImportError:
            ap.error("--porta precisa do pyserial: pip install pyserial")
    registrar = not (args.sem_registro or args.simulada or args.limite or args.ids or args.pausa)
    if registrar and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada (ou use --sem-registro)")
    hoje = datetime.now().strftime("%d-%m")
    prefixo = "serial-ruido" if args.lista else "serial"      # a lista não mistura com a rodada do Jung e do HUST
    if args.out_dir:
        out_dir = args.out_dir
    elif args.simulada:
        out_dir = SAIDA_SIMULADA
    elif not registrar:
        out_dir = SAIDA_TESTE
    else:
        out_dir = Path("reports/embarcado") / f"{prefixo.replace('-', '_')}_{hoje.replace('-', '_')}"

    caminho_cap = None
    try:
        modelo_id, json_sha12, pesos, bias = info_modelo(MODELO)
        print("referência do Python: recalculando os escores do int16 dos quadros...")
        segs, lista = [], None
        if args.lista:
            lista = carregar_lista(args.lista, modelo_id, json_sha12, pesos, bias, args.bin,
                                   exigir_bin=not args.captura)
            if not lista.bin_conferido:
                if registrar:
                    raise Abortar("sem o .bin, a lista não é conferida e a rodada não é registrada: "
                                  "use --sem-registro para só refazer a análise")
                print("  sem o .bin: o escore do Python é o do referencias.csv (9 casas), sem recalcular")
            conjuntos = list(dict.fromkeys(s.conjunto for s in lista.segmentos))
            for c in conjuntos:                 # --limite: os N primeiros de cada grupo
                segs += [s for s in lista.segmentos if s.conjunto == c][: args.limite]
        else:
            if "jung" in conjuntos:
                segs += carregar_jung(pesos, bias)
            if "hust" in conjuntos:
                segs += carregar_hust(pesos, bias)
        esperado = Counter(s.grupo for s in (lista.segmentos if lista else segs))   # o plano inteiro
        if args.limite and not lista:
            segs = [s for c in conjuntos for s in [x for x in segs if x.conjunto == c][: args.limite]]
        if pares_ids:
            segs = escolher(segs, pares_ids)
            conjuntos = [c for c in conjuntos if any(s.conjunto == c for s in segs)]
        print(f"  {len(segs)} segmentos ({', '.join(conjuntos)}), conferidos com "
              + ((f"o referencias.csv e o .bin da lista ({args.lista})" if lista.bin_conferido
                  else f"o referencias.csv da lista ({args.lista})") if lista else "os CSV de referência"))

        if args.captura:
            caminho_cap = args.captura
            linhas = ler_linhas_captura(caminho_cap)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            caminho_cap = caminho_livre((out_dir if not registrar else CAPTURAS) / f"{prefixo}-{hoje}.log")
            caminho_cap.parent.mkdir(parents=True, exist_ok=True)
            cap = Captura(caminho_cap)
            cap.pc(f"rodada inicio data={date.today().isoformat()} porta={args.porta or 'simulada'} "
                   f"baud={args.baud} modelo={modelo_id} json={json_sha12} commit={experimentos.git_short_hash()}"
                   + (f" lista={lista.manifesto_sha}" if lista else ""))
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
                               tempos, pausa_s=args.pausa).enviar(segs)
            finally:
                cap.pc("rodada fim")
                cap.fechar()
                porta.close()
            print(f"envio: {envio}")
            linhas = ler_linhas_captura(caminho_cap)

        a = analisar(linhas, modelo_id, json_sha12)
        if a.cabecalho.get("lista") != (lista.manifesto_sha if lista else None):
            a.problemas.append(f"a captura é da lista {a.cabecalho.get('lista') or '(nenhuma)'}, e a análise, "
                               f"da {lista.manifesto_sha if lista else '(nenhuma: Jung e HUST)'}")
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
            "baud": a.cabecalho.get("baud", args.baud),
            # o commit e a data são os da rodada (cabeçalho da captura), não os da análise
            "commit": a.cabecalho.get("commit") or experimentos.git_short_hash(),
            "data": a.cabecalho.get("data") or date.today().isoformat()}
    if lista:
        meta["lista"] = {"manifesto_sha": lista.manifesto_sha, "referencias_sha": lista.referencias_sha,
                         "bin_sha256": lista.bin_sha256}
    out_dir.mkdir(parents=True, exist_ok=True)
    escrever_csv(res, out_dir / "segmentos.csv")
    escrever_md(res, a, meta, out_dir / "resumo.md")
    escrever_tex(res, out_dir / "tabelas.tex")
    for c, r in res["conjuntos"].items():
        p = r["pior"]
        print(f"{NOMES_CONJUNTO[c]}: {r['processados']}/{r['n']} processados; concordância "
              f"{r['concordancia']}/{r['processados']}"
              + (f"; pior caso {p['segmento']} com {pct(p['frac_tol'])} da tolerância" if p else ""))
        if "aprovado" in r:
            pe = r["pior_E"]
            print(f"  critério (decisões iguais e |Δs| < E): {'APROVADO' if r['aprovado'] else 'REPROVADO'}"
                  + (f" ({'; '.join(r['motivos'])})" if r["motivos"] else "")
                  + (f"; |Δs|/E máx. {pe['dif_E']:.4g} ({pe['segmento']})" if pe else ""))
    for d in res["divergencias_pc"]:
        print(f"ATENÇÃO: {d}")
    if res["clipes_iguais"] != len(res["clipes"]):
        print(f"ATENÇÃO: só {res['clipes_iguais']}/{len(res['clipes'])} clipes da flash com o escore pela serial "
              "bit a bit igual ao da linha ref (a placa calcula os dois com a mesma conta)")
    print(f"saídas: {out_dir}/ (captura {caminho_cap})")

    completo = all(len(a.plano.get(g, [])) == n for g, n in esperado.items())
    if not registrar:
        print("(ensaio: registry não alterado)")
        return 0
    if not completo:
        print("(a captura não tem o plano inteiro: registry não alterado)")
        return 0
    try:
        novas = registrar_rodada(args.registry, res, a, meta, args.responsavel, args.notas)
    except Abortar as e:
        print(f"registry não alterado: {e}")
        return 1
    print(f"registrado: {novas[0]['id']}–{novas[-1]['id']} (commit {meta['commit']}, rodada de {meta['data']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
