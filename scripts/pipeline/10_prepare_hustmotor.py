#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_prepare_hustmotor.py — HUSTmotor no formato do pipeline (Protocolo C)
=========================================================================

Primeira parte do Protocolo C. Lê os .txt do HUSTmotor multimodal (Zhao et al.,
2025) das classes saudável (H) e falha de rolamento (BF), nas 4 velocidades (5,
10, 20 e 30 Hz), e grava o canal de som decimado para a taxa de trabalho, no
mesmo formato do Jung: .bin int16 (não versionados) e manifest.json (versionado).

Entra: data/raw/hustmotor/{H,BF}_{5,10,20,30}HZ.txt   (download manual, ver docs/hustmotor.md)
Sai:   data/processed/hustmotor/pcm_decimated/<fs>/   (.bin + manifest.json)

Cada .txt NÃO é um registro contínuo de 6,4 s
---------------------------------------------
O cabeçalho diz "Number of Blocks 5", e a coluna de tempo volta a zero a cada
32.768 amostras: são 5 aquisições de 1,28 s, sem continuidade entre elas. Por
isso cada bloco é decimado sozinho (o filtro não atravessa a emenda) e dá um
único segmento de 1 s, tirado do meio do bloco, longe dos transitórios do
filtro nas pontas. O .bin de cada arquivo tem os 5 segmentos em sequência,
alinhados a 1 s: 5 segmentos por classe e velocidade, 20 por classe.

Mesma cadeia do Jung
--------------------
- Quantização como no `01`: um fator único para todos os arquivos, 0,98 ×
  32767 / pico global, e o sinal em int16 à taxa original.
- Decimação como no `02`: `dsp.design_decimation` com a mesma regra (corte em
  0,45 × fs de saída, Kaiser, `config.FIR_ATTENUATION`) e `dsp.resample_clip`.
  O fator é 2 (25,6 → 12,8 kHz), e não 4: os coeficientes não são os mesmos do
  Jung, a regra de projeto é.

A unidade do HUST é volt na saída do microfone (sensibilidade não publicada), e
a do Jung é pascal: os níveis absolutos não são comparáveis. Isso pesa no C1
(transferência sem recalibrar) e é registrado no manifest.

Como o `02`, esta etapa só transforma: não escreve em `experiments/registry.csv`.

Uso
---
    python scripts/pipeline/10_prepare_hustmotor.py
    python scripts/pipeline/10_prepare_hustmotor.py --raw-dir outro/lugar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np

import config
import pcm_io
from dsp import design_decimation, resample_clip

FS_HUST = 25600
CLASSES_HUST = {"H": "normal", "BF": "falha"}      # as outras 4 classes do HUST ficam de fora
VELOCIDADES_HZ = (5, 10, 20, 30)                   # rotação nominal (tabela 2 da descrição)
CANAIS = ("X", "Y", "Z", "Sound")                  # colunas 2 a 5 (a 1 é o tempo)
CANAL_SOM = 3

# Rolamento 6203 (8 esferas, 6,747 mm, diâmetro primitivo 28,5 mm), contato 0°.
# A descrição do dataset não traz a geometria: o 6203 é o rolamento do motor
# Marathon D393 da bancada SpectraQuest, e as linhas do BF batem com ele (BPFO a
# até ~1 %, BPFI 2 a 3 % abaixo; docs/hustmotor.md). Inferido, não confirmado pelos autores.
N_ESFERAS, D_ESFERA, D_PRIMITIVO = 8, 6.747, 28.5
RAZAO_BPFO = N_ESFERAS / 2 * (1 - D_ESFERA / D_PRIMITIVO)     # 3,0531
RAZAO_BPFI = N_ESFERAS / 2 * (1 + D_ESFERA / D_PRIMITIVO)     # 4,9469

# Divisão do C2 (critério v4, seção 5: metade calibra, 1 de descarte, metade mede),
# aplicada bloco a bloco em cada arquivo da normal e fixada antes de rodar qualquer
# detector. É pela posição do bloco, sem olhar o conteúdo. Só a normal se divide,
# porque só ela calibra: a falha não calibra nada e mede com os 5 blocos. Juntar as
# 4 rotações na normal de calibração é proposta, que depende do Registro de Decisões.
BLOCOS_CALIBRACAO = (1, 2)
BLOCOS_DESCARTE = (3,)
BLOCOS_MEDIDA = (4, 5)
BLOCOS_MEDIDA_FALHA = (1, 2, 3, 4, 5)


def nome_arquivo(cls: str, v: int) -> str:
    return f"{cls}_{v}HZ.txt"


def rotulo(cls: str, v: int) -> str:
    return f"{cls.lower()}_{v}hz"


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def ler_txt(caminho: Path) -> np.ndarray:
    """
    Lê um .txt do HUSTmotor e devolve os blocos do canal de som, (n_blocos, n_por_bloco).

    Confere o cabeçalho (canais, linhas, blocos), o passo de tempo e que o tempo
    volta a zero exatamente na emenda de cada bloco.
    """
    linhas = caminho.read_text(encoding="latin-1").splitlines()
    cab, inicio = {}, None
    for i, l in enumerate(linhas):
        partes = l.strip().split("\t")
        if partes[0].startswith("Time (seconds)"):
            inicio = i + 1
            break
        if len(partes) >= 2:
            cab[partes[0].strip()] = [p.strip() for p in partes[1:]]
    if inicio is None:
        raise ValueError(f"{caminho.name}: sem a linha 'Time (seconds) and Data Channels'")
    if tuple(cab.get("Legend", [])) != CANAIS:
        raise ValueError(f"{caminho.name}: canais {cab.get('Legend')}, esperava {CANAIS}")
    n_total = int(cab["Total Data Rows"][0])
    n_blocos = int(cab["Number of Blocks"][0])
    if n_total % n_blocos:
        raise ValueError(f"{caminho.name}: {n_total} linhas não dividem em {n_blocos} blocos")

    dados = np.loadtxt(linhas[inicio:], dtype=np.float64)
    if dados.shape != (n_total, 1 + len(CANAIS)):
        raise ValueError(f"{caminho.name}: forma {dados.shape}, esperava ({n_total}, {1 + len(CANAIS)})")
    n_bloco = n_total // n_blocos
    t = dados[:, 0].reshape(n_blocos, n_bloco)
    if not np.all(t[:, 0] == 0.0):
        raise ValueError(f"{caminho.name}: o tempo não volta a zero em cada bloco")
    dt = np.diff(t, axis=1)
    if not np.allclose(dt, 1 / FS_HUST, atol=1e-7):
        raise ValueError(f"{caminho.name}: passo de tempo {dt.mean():.3e} s, esperava 1/{FS_HUST}")
    return dados[:, 1 + CANAL_SOM].reshape(n_blocos, n_bloco)


def cortar_segmentos(x: np.ndarray, n: int, n_segmentos: int) -> np.ndarray:
    """
    Os segmentos independentes de um .bin do HUST, (n_segmentos, n). Cada um vem de
    uma aquisição separada: corte só assim, em 1 s, sem sobreposição, e nunca filtre
    nem decime o .bin inteiro, porque o filtro cruzaria as emendas.
    """
    if x.size != n * n_segmentos:
        raise ValueError(f"{x.size} amostras não são {n_segmentos} segmentos de {n}")
    return x.reshape(n_segmentos, n)


def segmento_do_meio(y: np.ndarray, n: int) -> np.ndarray:
    """O trecho de `n` amostras do meio de `y` (longe dos transitórios do filtro nas pontas)."""
    if y.size < n:
        raise ValueError(f"bloco decimado com {y.size} amostras, menos que um segmento ({n})")
    i = (y.size - n) // 2
    return y[i:i + n]


def preparar(blocos: dict[str, np.ndarray], fs_out: float, escala: float):
    """Quantiza à taxa original, decima bloco a bloco e devolve {rótulo: sinal em float}."""
    spec = design_decimation(FS_HUST, fs_out)
    if not spec.integer_factor:
        raise ValueError(f"{FS_HUST} → {fs_out} Hz não é fator inteiro")
    n_seg = config.amostras_por_segmento(fs_out)
    # Depois do resample_clip, só o início tem transitório (o fim é truncado):
    # ceil(atraso / fator) saídas. Corta-se o mesmo nas duas pontas só para o trecho
    # do meio ficar centrado no bloco.
    atraso = (spec.numtaps - 1) // 2
    margem = -(-atraso // spec.down)
    saida = {}
    for rot, b in blocos.items():
        # como o 01, mas a escala vem só da normal: uma falha mais forte que ela
        # passaria do int16, e o astype daria a volta (um pico vira negativo, sem
        # aviso). Para aqui, em vez de saturar calado: pede uma decisão.
        n_sat = int(np.count_nonzero(np.abs(np.round(b * escala)) > config.INT16_FULL))
        if n_sat:
            raise ValueError(f"{rot}: {n_sat} amostras passam do int16 com a escala da normal")
        pcm = pcm_io.para_pcm(b * escala / config.INT16_FULL)
        segs = []
        for bloco in pcm:
            y = resample_clip(bloco.astype(np.float64) / config.INT16_FULL, FS_HUST, fs_out, spec)
            if y.size - 2 * margem < n_seg:
                raise ValueError(f"{rot}: bloco curto demais para 1 s sem os transitórios")
            segs.append(segmento_do_meio(y[margem:y.size - margem], n_seg))
        saida[rot] = np.concatenate(segs)
    return saida, spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/hustmotor"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/hustmotor/pcm_decimated"))
    ap.add_argument("--fs", type=float, default=float(config.FS_TRABALHO))
    args = ap.parse_args(argv)

    arquivos = {(c, v): args.raw_dir / nome_arquivo(c, v) for c in CLASSES_HUST for v in VELOCIDADES_HZ}
    faltam = [p.name for p in arquivos.values() if not p.is_file()]
    if faltam:
        print(f"faltam em {args.raw_dir}: {', '.join(faltam)} (download em docs/hustmotor.md)")
        return 1

    blocos, hashes = {}, {}
    for (c, v), p in arquivos.items():
        blocos[rotulo(c, v)] = ler_txt(p)
        hashes[rotulo(c, v)] = sha256(p)
        print(f"[load] {p.name}: {blocos[rotulo(c, v)].shape[0]} blocos de "
              f"{blocos[rotulo(c, v)].shape[1] / FS_HUST:.2f} s")

    # escala como no 01, mas só com a normal: os dados de falha não definem o ganho
    pico = max(float(np.abs(b).max()) for (c, v) in arquivos if CLASSES_HUST[c] == "normal"
               for b in [blocos[rotulo(c, v)]])
    escala = 0.98 * config.INT16_FULL / pico
    sinais, spec = preparar(blocos, args.fs, escala)
    print(f"[escala] pico da normal {pico:.6f} V → fator {escala:.1f} (como no 01)")
    print(f"[filtro] {FS_HUST / 1000:.1f} → {args.fs / 1000:.1f} kHz, fator /{spec.down}, "
          f"{spec.numtaps} taps, corte {spec.cutoff_hz:.0f} Hz, atenuação {spec.stopband_atten_db:.1f} dB")

    clipes = []
    for (c, v), p in arquivos.items():
        r = rotulo(c, v)
        clipes.append(pcm_io.Clip(rotulo=r, binario=CLASSES_HUST[c], fs=args.fs,
                                  x_float=sinais[r], origem=p.name))
    destino = args.out_dir / f"{int(args.fs)}"
    caminho = pcm_io.gravar_clipes(clipes, destino, args.fs, extras={
        "dataset": "HUSTmotor multimodal (Zhao, Shen, Zio e Ma, 2025, EAAI 162, 112559)",
        "licenca": "o repositório não declara licença; a descrição pede a citação do artigo acima",
        "canal": "Sound (coluna 5)",
        "unidade_original": "V na saída do microfone AWA14004 (sensibilidade não publicada)",
        "fs_original_hz": FS_HUST,
        "escala_int16": escala,
        "pico_normal_original": pico,
        "decimacao": asdict(spec),
        "segmentos": "5 blocos não contíguos de 1,28 s por arquivo; 1 segmento de 1 s do meio "
                     "de cada bloco, decimado sozinho; os segmentos do .bin são independentes",
        "corte": "cortar o .bin em segmentos de 1 s, sem sobreposição (cortar_segmentos); "
                 "não filtrar nem decimar o .bin inteiro, que cruzaria as emendas",
        "divisao_c2": {"blocos_calibracao": list(BLOCOS_CALIBRACAO),
                       "blocos_descarte": list(BLOCOS_DESCARTE),
                       "blocos_medida": list(BLOCOS_MEDIDA),
                       "blocos_medida_falha": list(BLOCOS_MEDIDA_FALHA),
                       "regra": "critério v4, seção 5, por posição do bloco, só na normal (a "
                                "falha não calibra e mede com os 5 blocos); fixada em 20/10, "
                                "depois de o BF ter sido visto para a geometria"},
        "rolamento": {"modelo": "6203 (inferido)", "n_esferas": N_ESFERAS, "d_esfera_mm": D_ESFERA,
                      "d_primitivo_mm": D_PRIMITIVO, "razao_bpfo": round(RAZAO_BPFO, 4),
                      "razao_bpfi": round(RAZAO_BPFI, 4)},
    })
    # campos por arquivo que o formato plano do gravar_clipes não tem
    manifest = json.loads(caminho.read_text(encoding="utf-8"))
    for (c, v), p in arquivos.items():
        manifest[rotulo(c, v)].update({"classe_hust": c, "velocidade_nominal_hz": v,
                                       "n_segmentos": len(blocos[rotulo(c, v)]),
                                       "sha256_origem": hashes[rotulo(c, v)]})
    caminho.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[grava] {len(clipes)} clipes em {destino}, "
          f"{sum(m['n_segmentos'] for k, m in manifest.items() if not k.startswith('_'))} segmentos de 1 s")
    print(f"[manifest] {caminho}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
