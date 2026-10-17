#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_export_clips.py — clipes de áudio para a flash da placa (Frente 3C)
======================================================================

Gera um cabeçalho C com trechos de áudio `int16` das gravações, para o firmware
(`lab3-firmware-stm32`) rodar a cadeia inteira na placa (MFCC, LDA e decisão) a
partir do som, e não mais das características prontas do `07`:

- **Clipes a 12,8 kHz** (o PCM decimado do `02`): por gravação, `--segundos`
  segmentos consecutivos da partição, a partir do segmento do meio (o mesmo das
  referências por estágio do `08`). Cada segmento traz o id, a classe e o
  escore e a decisão do `escores_referencia.csv`.
- **Recorte a 51,2 kHz** (o PCM original do `01`), para demonstrar a decimação
  na placa: um segmento de 1 s com as margens que o decimador causal do
  firmware precisa ler (`dsp.recorte_para_decimar`, fase 0 do
  `arm_fir_decimate_f32`: 75 amostras antes e 73 depois; as 37 primeiras saídas
  são descartadas). Vem com os 147 coeficientes do FIR (`dsp.taps_decimacao`).
  O segmento do recorte é o primeiro do clipe da `normal`, então a placa pode
  comparar o que decimou com o clipe a 12,8 kHz.

**Escore do recorte.** A entrada do MFCC é `int16`, então a placa requantiza a
saída do decimador antes do MFCC (`docs/contrato_numerico.md`, seção
"Decimação", item 4): `lrintf(y · 32767.0f)`, saturado em [−32768, 32767]
**antes** do cast (sem saturar, um pico acima de ±1 dá a volta no `int16`).

- `RECORTE_ESCORE_REF_INT16` é **a referência da placa**: o escore do segmento
  requantizado como o `02` (`round(y·32767)` em float64, saturado), que é o do
  `escores_referencia.csv`.
- `RECORTE_ESCORE_REF` é só diagnóstico do caminho em float: o MFCC da saída da
  decimação sem requantizar.

As duas diferem em ~7·10⁻⁶ relativo, e a tolerância encadeada do escore no A4 é
10⁻⁵: comparar com a referência errada quase esgota a margem.

**Na placa, a decimação é em float32.** Em float64 (Python), o recorte decimado
e requantizado é bit a bit o segmento do `02`; em float32 não precisa ser: a
saída difere da de float64 em ~0,003 LSB, e uma amostra que cai perto de um
empate (x,5) pode arredondar para o outro lado. `decimar_cmsis_f32` emula o
`arm_fir_decimate_f32` (acumulador float32, do tap mais antigo ao mais novo,
com e sem o FMA do Cortex-M4) e o script aborta se a emulação diferir do clipe
em mais de 1 LSB ou se o escore sair da tolerância encadeada do A4 (a
requantização também é emulada em float32, `lrintf(y * 32767.0f)`). Com o
recorte padrão, diferem 1 amostra sem FMA e 2 com FMA, e o `.h` registra isso:
na placa, compare o recorte com o clipe com folga de 1 LSB, e o escore com
tolerância.

**Uso na placa (flash e RAM).** O `.h` só *declara* os vetores (`extern const`);
eles são definidos no único `.c` que fizer `#define CLIPES_DEMO_DADOS` antes do
`#include`. Com `static const` no `.h`, cada `.c` que o incluísse levaria uma
cópia dos ~350 KB de áudio, e dois `.c` já passariam dos 512 KB da flash da
F411. O recorte não cabe inteiro em float na RAM (51.348 × 4 B = 200 KB, contra
128 KB): a placa decima em blocos, convertendo `int16` → float um bloco por vez
(blocos de 512 amostras dão 128 saídas, um passo de quadro). Como 51.348 =
100 × 512 + 148, depois dos blocos cheios vem uma última chamada com 148
amostras (`RECORTE_ULTIMO_BLOCO`, múltiplo do fator); um laço só com blocos
cheios perderia as 37 últimas saídas sem dar erro. O `.h` traz o laço.

**Orçamento** (C1): a flash da F411 tem 512 KB; o README do firmware reserva
64 KB para o resto do firmware e deixa ~434 KB para áudio, ~13 s a 12,8 kHz
com o recorte de 51,2 kHz. O script aborta se os clipes passarem de
`--orcamento-kb`. O padrão (2 s por gravação, 10 s, mais o recorte) ocupa
~350 KB (358.696 B).

**Conferências antes de gravar** (aborta se alguma falhar): PCM e partição
íntegros; o escore de cada segmento, recalculado do `int16` do clipe com o
`dsp.mfcc` e a LDA final, é o do `escores_referencia.csv`; o recorte, decimado
em float64 e requantizado, é **bit a bit** o segmento a 12,8 kHz do `02`, e,
decimado como na placa (float32), fica a no máximo 1 LSB dele.

**Fidelidade, não desempenho.** Todos os segmentos estão no treino do modelo
final: a demonstração mede se a placa reproduz o Python, não o desempenho do
classificador (que é o do Protocolo B).

**Commit antes de gerar.** Como no `07` e no `08`, o `.h` grava o commit; no
caminho versionado, o script aborta se ele mesmo, o `dsp.py`, o
`lda_final.json` ou o `escores_referencia.csv` tiverem mudanças não
commitadas. Não há linha no registry: o arquivo é uma fixture, sem métrica.

Saída: reports/c_reference/clipes_demo.h (--out para outro caminho)

Uso
---
    python scripts/pipeline/09_export_clips.py
    python scripts/pipeline/09_export_clips.py --segundos 1 --sem-recorte
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from scipy.signal import lfilter

import config
import dsp
import experimentos
import pcm_io
from validation import particao

_spec = importlib.util.spec_from_file_location(
    "export_lda_header", Path(__file__).resolve().with_name("06_export_lda_header.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

NOME_H = "clipes_demo.h"
OUT_VERSIONADO = Path("reports/c_reference") / NOME_H
FASE = 0                      # arm_fir_decimate_f32 (docs/contrato_numerico.md, "Decimação")
BLOCO = 512                   # amostras por chamada do decimador na placa: 128 saídas, um passo de quadro
ORCAMENTO_KB = 434            # README do firmware, "Orçamento da flash para o áudio"


@dataclass
class Clipe:
    rotulo: str
    segmentos: list            # particao.Segmento, consecutivos
    pcm: np.ndarray            # int16, len = n_segmentos · amostras_por_segmento
    escores: list[float]       # do escores_referencia.csv
    falha: list[bool]


@dataclass
class Recorte:
    segmento: object           # particao.Segmento (no domínio decimado)
    pcm: np.ndarray            # int16 a 51,2 kHz, com as margens
    inicio_original: int       # primeira amostra do recorte na gravação a 51,2 kHz
    antes: int
    depois: int
    descartar: int
    taps: np.ndarray           # float64, os do resample_clip
    escore_float: float
    escore_int16: float
    difere_na_placa: dict      # amostras que, decimadas em float32, diferem do 02 (sem e com FMA)


def selecionar(segmentos, segundos: int) -> dict[str, list]:
    """
    Por gravação, `segundos` segmentos consecutivos a partir do segmento do meio
    (o do `08`). Escolhe pela posição, sem olhar o sinal nem o escore.
    """
    if segundos < 1:
        raise ValueError("--segundos tem que ser pelo menos 1")
    escolha = {}
    for rotulo in dict.fromkeys(s.rotulo for s in segmentos):
        da_gravacao = [s for s in segmentos if s.rotulo == rotulo]
        meio = len(da_gravacao) // 2
        trecho = da_gravacao[meio:meio + segundos]
        if len(trecho) < segundos:
            raise ValueError(f"{rotulo} não tem {segundos} segmentos a partir do meio")
        for a, b in zip(trecho, trecho[1:]):
            if b.inicio != a.fim:
                raise ValueError(f"{rotulo}: segmentos {a.id} e {b.id} não são contíguos")
        escolha[rotulo] = trecho
    return escolha


def escore_lda(x: np.ndarray, pesos: np.ndarray, bias: float) -> float:
    """Escore da LDA final (forma dobrada, float64) para um segmento em float."""
    m = dsp.mfcc(x, config.FS_TRABALHO)
    return float(np.concatenate([m.mean(axis=0), m.std(axis=0)]) @ pesos + bias)


def decimar_recorte(pcm_recorte: np.ndarray, taps: np.ndarray, descartar: int,
                    fator: int = config.FATOR_DECIMACAO, fase: int = FASE) -> np.ndarray:
    """
    O que o decimador causal da placa faz com o recorte, em float64: estado
    inicial zerado, uma saída a cada `fator` amostras (fase `fase`), sem as
    `descartar` primeiras. Devolve float (sem requantizar).
    """
    x = pcm_recorte.astype(np.float64) / config.INT16_FULL
    return lfilter(taps, 1.0, x)[fase::fator][descartar:]


def decimar_cmsis_f32(pcm_recorte: np.ndarray, fir: np.ndarray, descartar: int,
                      fator: int = config.FATOR_DECIMACAO, fma: bool = True) -> np.ndarray:
    """
    Emula o `arm_fir_decimate_f32` (fase 0) em float32: `x = int16 / 32767.0f`,
    estado inicial zerado e, para cada saída, um acumulador float32 somado do tap
    mais antigo ao mais novo, como no laço da CMSIS. Com `fma`, cada passo
    arredonda uma vez só, como o `vfma.f32` que o gcc gera para o Cortex-M4 (a
    soma em float64 antes de arredondar para float32 é uma aproximação disso).
    `fir` está na ordem do tempo; a CMSIS o recebe invertido, o que, simétrico,
    dá no mesmo. Devolve float32, sem as `descartar` primeiras saídas.
    """
    fir = np.asarray(fir, dtype=np.float32)
    n = len(fir)
    x = pcm_recorte.astype(np.float32) / np.float32(config.INT16_FULL)
    xp = np.concatenate([np.zeros(n - 1, np.float32), x])
    inicio = np.arange(len(x) // fator) * fator           # a janela da saída j termina em x[fator·j]
    coef = fir[::-1]                                       # pCoeffs: coef[0] multiplica a mais antiga
    acc = np.zeros(len(inicio), np.float32)
    for k in range(n):
        if fma:
            acc = (acc.astype(np.float64) + float(coef[k]) * xp[inicio + k].astype(np.float64)
                   ).astype(np.float32)
        else:
            acc = acc + coef[k] * xp[inicio + k]
    return acc[descartar:]


def blocos(n: int, bloco: int = BLOCO, fator: int = config.FATOR_DECIMACAO) -> tuple[int, int]:
    """
    Quantos blocos cheios de `bloco` amostras e o tamanho da última chamada do
    decimador para `n` amostras. Cada chamada do `arm_fir_decimate_f32` precisa de um
    múltiplo do fator (e de no máximo o bloco do init).
    """
    if bloco % fator:
        raise ValueError(f"o bloco ({bloco}) tem que ser múltiplo do fator ({fator})")
    cheios, ultimo = divmod(n, bloco)
    if ultimo % fator:
        raise ValueError(f"a última chamada teria {ultimo} amostras, que não é múltiplo de {fator}")
    return cheios, ultimo


def requantizar_f32(y: np.ndarray) -> np.ndarray:
    """`lrintf(y * 32767.0f)` saturado em 16 bits, em float32, como a placa faria."""
    v = np.rint(np.asarray(y, dtype=np.float32) * np.float32(config.INT16_FULL))
    return np.clip(v, -32768, 32767).astype(np.int16)


def montar_recorte(seg, raw: np.ndarray, decimado: np.ndarray, pesos, bias,
                   escore_csv: float) -> Recorte:
    spec = dsp.design_decimation(config.FS_ORIGINAL, config.FS_TRABALHO)
    taps = dsp.taps_decimacao(spec, config.FS_ORIGINAL)
    n = seg.fim - seg.inicio
    a, b, descartar = dsp.recorte_para_decimar(seg.inicio, n, spec, FASE)
    if a < 0 or b > len(raw):
        raise SystemExit(f"Abortado: o recorte do segmento {seg.id} sai da gravação ({a}..{b})")
    pcm = raw[a:b]
    y = decimar_recorte(pcm, taps, descartar)
    if len(y) != n:
        raise SystemExit(f"Abortado: o recorte decimado tem {len(y)} amostras, não {n}")
    if not np.array_equal(pcm_io.para_pcm(y), decimado[seg.inicio:seg.fim]):
        raise SystemExit("Abortado: o recorte decimado e requantizado não é o segmento do 02")
    escore_int16 = escore_lda(decimado[seg.inicio:seg.fim].astype(np.float64) / config.INT16_FULL,
                              pesos, bias)
    if abs(escore_int16 - escore_csv) > 1e-8 * abs(escore_csv) + 1e-9:
        raise SystemExit(f"Abortado: escore int16 do recorte ({escore_int16}) difere do CSV ({escore_csv})")
    escore_float = escore_lda(y, pesos, bias)
    difere = conferir_como_na_placa(pcm, taps, descartar, decimado[seg.inicio:seg.fim],
                                    pesos, bias, escore_float, escore_csv)
    d = config.FATOR_DECIMACAO * seg.inicio - a
    return Recorte(segmento=seg, pcm=pcm, inicio_original=a, antes=d, depois=b - a - d - 4 * n,
                   descartar=descartar, taps=taps, escore_float=escore_float,
                   escore_int16=escore_csv, difere_na_placa=difere)


def conferir_como_na_placa(pcm, taps, descartar, segmento_02, pesos, bias,
                           escore_float: float, escore_int16: float) -> dict[str, int]:
    """
    Decima o recorte como a placa (float32, sem e com FMA) e confere: requantizado,
    fica a no máximo 1 LSB do segmento do `02`, e os dois escores ficam dentro da
    tolerância encadeada do escore no A4. Devolve quantas amostras diferem.
    """
    tol = next(c for e, _, _, c in dsp.ESTAGIOS_PORTE if e == "escore")
    difere = {}
    for fma in (False, True):
        nome = "com FMA" if fma else "sem FMA"
        y32 = decimar_cmsis_f32(pcm, taps.astype(np.float32), descartar, fma=fma)
        q = requantizar_f32(y32)
        dif = np.abs(q.astype(np.int32) - segmento_02.astype(np.int32))
        if dif.max() > 1:
            raise SystemExit(f"Abortado: decimado em float32 ({nome}), o recorte difere do segmento "
                             f"do 02 em até {dif.max()} LSB")
        for e, ref, o_que in ((escore_lda(y32.astype(np.float64), pesos, bias), escore_float, "float"),
                              (escore_lda(q.astype(np.float64) / config.INT16_FULL, pesos, bias),
                               escore_int16, "int16")):
            if abs(e - ref) > tol * abs(ref):
                raise SystemExit(f"Abortado: decimado em float32 ({nome}), o escore {o_que} do recorte "
                                 f"({e:.4f}) sai da tolerância {tol:.0e} de {ref:.4f}")
        difere[nome] = int(np.count_nonzero(dif))
    return difere


def _lista(valores, fmt, por_linha: int) -> list[str]:
    itens = [fmt(v) for v in valores]
    return ["    " + ", ".join(itens[k:k + por_linha]) + "," for k in range(0, len(itens), por_linha)]


def _f32(v: float) -> str:
    return H._f32(v)


def _c_ident(rotulo: str) -> str:
    return rotulo.replace(".", "_").replace("-", "_")


def gerar_header(clipes: list[Clipe], recorte: Recorte | None, origem: dict, p: dict) -> str:
    n_amostras = config.amostras_por_segmento(config.FS_TRABALHO)
    total_seg = sum(len(c.segmentos) for c in clipes)
    bytes_audio = sum(c.pcm.nbytes for c in clipes) + (recorte.pcm.nbytes if recorte else 0)
    linhas = [
        "/*",
        " * clipes_demo.h -- clipes de audio para a flash da placa (Frente 3C)",
        " *",
        " * GERADO por scripts/pipeline/09_export_clips.py; nao editar a mao.",
        f" * Modelo: {origem['json']} ({p.get('id', '?')}), SHA-256 do JSON {origem['sha256_json'][:12]}",
        f" * Escores: {origem['csv']} (SHA-256 {origem['sha256_csv'][:12]})",
        f" * Particao: {origem['splits']}; commit do gerador: {origem['commit']}",
        " *",
        " * TODOS OS SEGMENTOS ESTAO NO TREINO DO MODELO FINAL: a demonstracao mede a",
        " * fidelidade do porte (a placa reproduz o Python), nao o desempenho do",
        " * classificador, que e o do Protocolo B.",
        " *",
        " * USO: este .h so DECLARA os vetores. Em UM unico .c, faca",
        " *     #define CLIPES_DEMO_DADOS",
        " *     #include \"clipes_demo.h\"",
        " * para defini-los (na flash, const); nos outros .c, so o #include. Assim o",
        f" * audio ({bytes_audio} B) entra uma vez so: com static const no .h, cada .c que",
        " * o incluisse levaria uma copia, e dois .c ja passariam dos 512 KB da F411.",
        " *",
        f" * Clipes a {config.FS_TRABALHO} Hz: int16 do PCM decimado (02), segmentos consecutivos de",
        f" * {n_amostras} amostras; o segmento k de um clipe comeca em k * CLIPES_AMOSTRAS_POR_SEGMENTO.",
        " * Escala da entrada: x = int16 / 32767 (docs/contrato_numerico.md).",
    ]
    if recorte:
        cheios, ultimo = blocos(len(recorte.pcm))
        linhas += [
            " *",
            f" * Recorte a {config.FS_ORIGINAL} Hz: o segmento {recorte.segmento.id} ({recorte.segmento.rotulo}) com as",
            f" * margens do decimador causal de fase {FASE} (arm_fir_decimate_f32): {recorte.antes} amostras",
            f" * antes e {recorte.depois} depois; descartar as {recorte.descartar} primeiras saidas.",
            " * - REFERENCIA DA PLACA: RECORTE_ESCORE_REF_INT16. A entrada do MFCC e int16,",
            " *   entao a placa requantiza a saida do decimador antes do MFCC",
            " *   (docs/contrato_numerico.md, \"Decimacao\", item 4): lrintf(y * 32767.0f),",
            " *   SATURADO em [-32768, 32767] antes do cast para int16 (sem saturar, um",
            " *   pico acima de +-1 da a volta). RECORTE_ESCORE_REF_INT16 e o escore do",
            " *   segmento requantizado como o 02 (round em float64; o escore do CSV).",
            " * - RECORTE_ESCORE_REF: so diagnostico do caminho em float (MFCC da saida",
            " *   da decimacao sem requantizar). As duas diferem em ~7e-6 relativo, e a",
            " *   tolerancia do escore e 1e-5: a referencia errada quase esgota a margem.",
            " * - Em float64 (Python), o recorte decimado e requantizado da bit a bit o",
            f" *   primeiro segmento do clipe '{recorte.segmento.rotulo}'. Em float32 (placa) nao precisa:",
            " *   amostras perto de um empate (x,5) podem arredondar para o outro lado.",
            " *   Na emulacao da CMSIS em float32, amostras que diferem (de 1 LSB): "
            + ", ".join(f"{v} {k}" for k, v in recorte.difere_na_placa.items()) + ";",
            " *   compare com folga de 1 LSB e o escore com tolerancia, nao bit a bit.",
            f" * - RAM: o recorte inteiro em float ocuparia {recorte.pcm.size * 4} B (a F411 tem 128 KB).",
            " *   Decime em blocos, convertendo int16 -> float um bloco por vez. Com",
            f" *   RECORTE_BLOCO = {BLOCO} no init, o estado da CMSIS tem RECORTE_N_TAPS +",
            f" *   RECORTE_BLOCO - 1 floats ({(len(recorte.taps) + BLOCO - 1) * 4} B), e cada bloco da"
            f" {BLOCO // config.FATOR_DECIMACAO} saidas,",
            " *   um passo de quadro do MFCC.",
            f" * - RECORTE_N_AMOSTRAS = {cheios} x {BLOCO} + {ultimo}: depois dos blocos cheios vem uma",
            f" *   ultima chamada com RECORTE_ULTIMO_BLOCO = {ultimo} amostras (multiplo de",
            " *   RECORTE_FATOR; a chamada aceita um bloco menor que o do init). Um laco so",
            f" *   com blocos cheios perde as {ultimo // config.FATOR_DECIMACAO} ultimas saidas sem dar erro:",
            " *",
            " *     float x[RECORTE_BLOCO], y[RECORTE_BLOCO / RECORTE_FATOR];",
            " *     float estado[RECORTE_N_TAPS + RECORTE_BLOCO - 1];",
            " *     arm_fir_decimate_init_f32(&d, RECORTE_N_TAPS, RECORTE_FATOR,",
            " *                               recorte_fir, estado, RECORTE_BLOCO);",
            " *     for (int b = 0; b <= RECORTE_N_BLOCOS_CHEIOS; ++b) {",
            " *         const int n = b < RECORTE_N_BLOCOS_CHEIOS ? RECORTE_BLOCO",
            " *                                                   : RECORTE_ULTIMO_BLOCO;",
            " *         for (int i = 0; i < n; ++i)",
            f" *             x[i] = (float)recorte_{config.FS_ORIGINAL}[b * RECORTE_BLOCO + i] / 32767.0f;",
            " *         arm_fir_decimate_f32(&d, x, y, n);     // n / RECORTE_FATOR saidas",
            " *         // saidas de indice global < RECORTE_DESCARTAR (todas no 1o bloco):",
            " *         // descartar; as outras, requantizar (saturando) e seguir para o MFCC",
            " *     }",
            " *",
            f" *   No total, {len(recorte.pcm) // config.FATOR_DECIMACAO} saidas - RECORTE_DESCARTAR ="
            " CLIPES_AMOSTRAS_POR_SEGMENTO.",
        ]
    linhas += [
        " */",
        "",
        "#ifndef CLIPES_DEMO_H",
        "#define CLIPES_DEMO_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define CLIPES_FS_HZ {config.FS_TRABALHO}",
        f"#define CLIPES_AMOSTRAS_POR_SEGMENTO {n_amostras}",
        f"#define CLIPES_N_CLIPES {len(clipes)}",
        f"#define CLIPES_N_SEGMENTOS {total_seg}",
        f"#define CLIPES_MODELO_ID \"{p.get('id', '?')}\"",
        f"#define CLIPES_JSON_SHA256_12 \"{origem['sha256_json'][:12]}\"",
        f"#define CLIPES_CSV_SHA256_12 \"{origem['sha256_csv'][:12]}\"   /* escores_referencia.csv */",
        f"#define CLIPES_BYTES_AUDIO {bytes_audio}",
        "",
        "typedef struct {",
        "    const char *classe;        /* gravacao de origem */",
        "    const int16_t *pcm;",
        "    uint32_t n_amostras;",
        "    uint16_t primeiro_segmento; /* indice em clipes_segmentos[] */",
        "    uint16_t n_segmentos;",
        "} clipe_t;",
        "",
        "typedef struct {",
        "    uint16_t id;               /* segmento da particao (splits.json) */",
        "    uint8_t clipe;             /* indice em clipes[] */",
        "    uint8_t falha_ref;         /* decisao do CSV: 1 = falha, 0 = normal */",
        "    float escore_ref;          /* escores_referencia.csv */",
        "} clipe_segmento_t;",
        "",
    ]
    decl = [f"extern const int16_t clipe_{_c_ident(c.rotulo)}[{len(c.pcm)}];" for c in clipes]
    decl += ["extern const clipe_t clipes[CLIPES_N_CLIPES];",
             "extern const clipe_segmento_t clipes_segmentos[CLIPES_N_SEGMENTOS];"]
    linhas += decl + [""]
    if recorte:
        s = recorte.segmento
        linhas += [
            f"#define RECORTE_FS_HZ {config.FS_ORIGINAL}",
            f"#define RECORTE_N_AMOSTRAS {len(recorte.pcm)}",
            f"#define RECORTE_SEGMENTO {s.id}",
            f"#define RECORTE_CLIPE {[c.rotulo for c in clipes].index(s.rotulo)}  /* o segmento e o primeiro deste clipe */",
            f"#define RECORTE_INICIO_ORIGINAL {recorte.inicio_original}  /* amostra a {config.FS_ORIGINAL} Hz na gravacao */",
            f"#define RECORTE_MARGEM_ANTES {recorte.antes}",
            f"#define RECORTE_MARGEM_DEPOIS {recorte.depois}",
            f"#define RECORTE_FATOR {config.FATOR_DECIMACAO}",
            f"#define RECORTE_FASE {FASE}",
            f"#define RECORTE_DESCARTAR {recorte.descartar}",
            f"#define RECORTE_N_TAPS {len(recorte.taps)}",
            f"#define RECORTE_BLOCO {BLOCO}  /* amostras por chamada do decimador (blockSize do init) */",
            f"#define RECORTE_N_BLOCOS_CHEIOS {cheios}",
            f"#define RECORTE_ULTIMO_BLOCO {ultimo}  /* a ultima chamada: nao esquecer */",
            f"#define RECORTE_ESCORE_REF {_f32(recorte.escore_float)}  /* so diagnostico (saida em float) */",
            f"#define RECORTE_ESCORE_REF_INT16 {_f32(recorte.escore_int16)}  /* a referencia da placa */",
            "",
            "/* FIR da decimacao (dsp.taps_decimacao), em float32; simetrico bit a bit, entao",
            " * a ordem invertida que a CMSIS espera nao muda nada */",
            "extern const float recorte_fir[RECORTE_N_TAPS];",
            f"extern const int16_t recorte_{config.FS_ORIGINAL}[RECORTE_N_AMOSTRAS];",
            "",
        ]
    linhas += [
        "#endif /* CLIPES_DEMO_H */",
        "",
        "/* Definicoes: so no .c que define CLIPES_DEMO_DADOS (uma vez, mesmo incluido de novo) */",
        "#if defined(CLIPES_DEMO_DADOS) && !defined(CLIPES_DEMO_DADOS_DEFINIDOS)",
        "#define CLIPES_DEMO_DADOS_DEFINIDOS",
        "",
    ]
    for c in clipes:
        linhas += [f"/* {c.rotulo}: segmentos {', '.join(str(s.id) for s in c.segmentos)} */",
                   f"const int16_t clipe_{_c_ident(c.rotulo)}[{len(c.pcm)}] = {{",
                   *_lista(c.pcm, str, 16), "};", ""]
    linhas.append("const clipe_t clipes[CLIPES_N_CLIPES] = {")
    primeiro = 0
    for c in clipes:
        linhas.append(f"    {{ \"{c.rotulo}\", clipe_{_c_ident(c.rotulo)}, {len(c.pcm)}u, "
                      f"{primeiro}u, {len(c.segmentos)}u }},")
        primeiro += len(c.segmentos)
    linhas += ["};", "", "const clipe_segmento_t clipes_segmentos[CLIPES_N_SEGMENTOS] = {"]
    for i, c in enumerate(clipes):
        for s, e, f in zip(c.segmentos, c.escores, c.falha):
            linhas.append(f"    {{ {s.id}u, {i}u, {int(f)}u, {_f32(e)} }},  /* {c.rotulo} */")
    linhas += ["};", ""]
    if recorte:
        linhas += [
            "const float recorte_fir[RECORTE_N_TAPS] = {",
            *_lista(recorte.taps, _f32, 6),
            "};",
            "",
            f"const int16_t recorte_{config.FS_ORIGINAL}[RECORTE_N_AMOSTRAS] = {{",
            *_lista(recorte.pcm, str, 16),
            "};",
            "",
        ]
    linhas += ["#endif /* CLIPES_DEMO_DADOS */", ""]
    texto = "\n".join(linhas)
    if not texto.isascii():
        raise ValueError("o .h tem caracteres fora do ASCII")
    return texto


def ler_header(texto: str) -> dict:
    """
    Lê de volta o `.h` gerado (o teste do clipe, C3): os clipes `int16`, a tabela
    de segmentos, os #define e, se houver, o recorte e o FIR, sem depender de data/.
    """
    import re
    defs = {m.group(1): m.group(2).strip() for m in
            re.finditer(r"^#define (\w+) (.+?)(?:\s*/\*.*)?$", texto, re.M)}

    def bloco(nome: str) -> str:
        m = re.search(rf"\b{re.escape(nome)}\[[^\]]*\] = \{{(.*?)\n\}};", texto, re.S)
        if m is None:
            raise ValueError(f"{nome} não está no .h")
        return re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)

    def numeros(nome, conv):
        return [conv(v.strip().rstrip("fu")) for v in bloco(nome).split(",") if v.strip()]

    clipes = []
    for m in re.finditer(r'\{ "([^"]+)", (\w+), (\d+)u, (\d+)u, (\d+)u \}', bloco("clipes")):
        rotulo, ident, n, primeiro, n_seg = m.groups()
        pcm = np.array(numeros(ident, int), dtype=np.int16)
        if len(pcm) != int(n):
            raise ValueError(f"{ident}: {len(pcm)} amostras, a tabela diz {n}")
        clipes.append({"rotulo": rotulo, "pcm": pcm, "primeiro": int(primeiro), "n_seg": int(n_seg)})
    segmentos = [{"id": int(a), "clipe": int(b), "falha": bool(int(c)), "escore": float(d.rstrip("f"))}
                 for a, b, c, d in re.findall(r"\{ (\d+)u, (\d+)u, (\d)u, ([-+0-9.eE]+f) \}",
                                              bloco("clipes_segmentos"))]
    out = {"defs": defs, "clipes": clipes, "segmentos": segmentos}
    if "RECORTE_N_AMOSTRAS" in defs:
        out["recorte_fir"] = np.array(numeros("recorte_fir", float), dtype=np.float32)
        out["recorte"] = np.array(numeros(f"recorte_{config.FS_ORIGINAL}", int), dtype=np.int16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--escores", type=Path, default=Path("reports/modelo_final/escores_referencia.csv"))
    ap.add_argument("--out", type=Path, default=OUT_VERSIONADO)
    ap.add_argument("--segundos", type=int, default=2, help="segmentos (s) por gravação")
    ap.add_argument("--sem-recorte", action="store_true", help="não inclui o recorte a 51,2 kHz")
    ap.add_argument("--orcamento-kb", type=float, default=ORCAMENTO_KB)
    args = ap.parse_args()

    raiz_scripts = Path(__file__).resolve().parents[1]
    sujos = H.nao_commitados([Path(__file__), raiz_scripts / "dsp.py", args.modelo, args.escores])
    if sujos:
        lista = "\n  " + "\n  ".join(sujos)
        if args.out.resolve() == OUT_VERSIONADO.resolve():
            print("Abortado: commit antes de gerar; há mudanças não commitadas em:" + lista)
            return 1
        print("aviso: mudanças não commitadas (o commit no .h fica marcado com +mudancas):" + lista)

    clipes_dec = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(clipes_dec)
    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, clipes_dec, config.FS_TRABALHO)
                 + particao.verificar(particoes))
    raw = {}
    if not args.sem_recorte:
        clipes_raw = pcm_io.carregar_clipes(args.raw_dir, fs_esperado=config.FS_ORIGINAL)
        divergencias += pcm_io.verificar_integridade(clipes_raw)
        raw = {c.rotulo: c.pcm for c in clipes_raw}
    pcm_io.relatar_integridade(divergencias)
    if divergencias or problemas:
        print("Abortado: PCM ou partição inválidos:\n  " + "\n  ".join(problemas))
        return 1
    dec = {c.rotulo: c.pcm for c in clipes_dec}

    p = json.loads(args.modelo.read_text(encoding="utf-8"))
    pesos = np.asarray(p["dobrada"]["pesos"], dtype=float)
    bias = float(p["dobrada"]["bias"])
    csv_seg = {int(l["segmento"]): (float(l["escore"]), l["previsao"] == "falha")
               for l in csv.DictReader(args.escores.open(encoding="utf-8"))}

    clipes = []
    for rotulo, segs in selecionar(particao.segmentos_de(particoes), args.segundos).items():
        pcm = dec[rotulo][segs[0].inicio:segs[-1].fim]
        escores, falha = [], []
        for s in segs:
            e_csv, f_csv = csv_seg[s.id]
            e = escore_lda(dec[rotulo][s.inicio:s.fim].astype(np.float64) / config.INT16_FULL, pesos, bias)
            if abs(e - e_csv) > 1e-8 * abs(e_csv) + 1e-9:
                print(f"Abortado: escore do segmento {s.id} ({e:.6f}) difere do CSV ({e_csv})")
                return 1
            escores.append(e_csv)
            falha.append(f_csv)
        clipes.append(Clipe(rotulo, segs, pcm, escores, falha))

    recorte = None
    if not args.sem_recorte:
        seg = next(c for c in clipes if c.rotulo == "normal").segmentos[0]
        recorte = montar_recorte(seg, raw["normal"], dec["normal"], pesos, bias, csv_seg[seg.id][0])

    total = sum(c.pcm.nbytes for c in clipes) + (recorte.pcm.nbytes if recorte else 0)
    if total > args.orcamento_kb * 1024:
        print(f"Abortado: os clipes ocupam {total / 1024:.0f} KB, acima do orçamento de "
              f"{args.orcamento_kb:.0f} KB da flash (README do firmware)")
        return 1

    origem = {
        "json": args.modelo.as_posix(),
        "sha256_json": hashlib.sha256(args.modelo.read_bytes()).hexdigest(),
        "csv": args.escores.as_posix(),
        "sha256_csv": hashlib.sha256(args.escores.read_bytes()).hexdigest(),
        "splits": particao.hash_arquivo(args.splits),
        "commit": experimentos.git_short_hash() + ("+mudancas" if sujos else ""),
    }
    texto = gerar_header(clipes, recorte, origem, p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(texto, encoding="utf-8", newline="\n")
    for c in clipes:
        print(f"  {c.rotulo:<11} segmentos {[s.id for s in c.segmentos]}  escores "
              + ", ".join(f"{e:.1f}" for e in c.escores))
    if recorte:
        print(f"  recorte a {config.FS_ORIGINAL} Hz: segmento {recorte.segmento.id}, "
              f"{len(recorte.pcm)} amostras ({recorte.antes} antes, {recorte.depois} depois, "
              f"descartar {recorte.descartar}); escore float {recorte.escore_float:.4f}, "
              f"int16 {recorte.escore_int16:.4f}")
        print("  decimado em float32 como na placa, amostras a 1 LSB do 02: "
              + ", ".join(f"{v} {k}" for k, v in recorte.difere_na_placa.items()))
    print(f"cabeçalho: {args.out} ({total / 1024:.0f} KB de áudio na flash, "
          f"orçamento {args.orcamento_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
