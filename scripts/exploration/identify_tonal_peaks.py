#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
identify_tonal_peaks.py — de onde vêm os tons que diferem entre as classes
=========================================================================

Tarefa do Notion: "Caracterizar a natureza da assinatura acústica no dataset
(tonal × impulsiva) e o caso bpfo_0.3mm", passo 3 do plano. Complementa o
`inspect_signature_spectra.py`, que diz ONDE as classes diferem; este diz DE QUÊ.

A pergunta
----------
O espectro das cinco gravações é dominado por picos discretos. Se os picos que
distinguem uma falha da normal forem harmônicos de BPFI/BPFO, a assinatura é do
rolamento. Se forem do eixo, do motor, da alimentação ou de uma portadora do
inversor, as gravações diferem também pela condição da bancada, e o
classificador pode estar aprendendo isso.

Cinemática da bancada (Jung et al., 2023, seção 3.1)
-----------------------------------------------------
- Eixo dos mancais: 3.010 rpm → 50,17 Hz. O rolamento com defeito está no mancal A,
  perto do microfone.
- Caixa multiplicadora de 2,07× entre motor e eixo: o motor gira a ~24,2 Hz.
- Motor de indução de 4 polos (2 pares), nominal 1.770 rpm a 60 Hz. A 24,2 Hz no
  eixo do motor, a alimentação precisa estar perto de 49 Hz: há um inversor, e
  não a rede de 60 Hz. f_e = 2·f_motor / (1 − s), com escorregamento s de alguns %.
  (Inferência pela rotação; o artigo não descreve o acionamento.)
- O artigo informa BPFI ≈ 272 Hz e BPFO ≈ 179 Hz a 50,17 Hz. Na primeira rodada
  (25/09) os dados mostraram outros valores: BPFI ≈ 268,3 Hz e BPFO ≈ 182,7–183,4 Hz
  (−1,4 % e +2 %). Por isso o script NÃO usa os valores do artigo para rotular:
  ele ESTIMA BPFI e BPFO a partir dos picos de cada gravação (ver item 4).

Nenhuma frequência é tomada como exata: f_eixo, f_motor e f_e também são
estimados em cada gravação, por busca de pente harmônico na PSD de alta
resolução. Se as gravações não rodaram na mesma velocidade, aparece aqui.

Armadilha do pente (corrigida em 25/09): todo harmônico do eixo também é
harmônico da METADE do eixo, então um pente em ~25 Hz "acerta" todos os picos do
eixo e ganha a busca pelo motor. A busca agora ignora os harmônicos que coincidem
com os do eixo e descarta candidatos colados em f_eixo/2 e em f_eixo. Quando o
pente não encontra harmônicos suficientes, a frequência sai como "não
identificada" (nan) em vez de um número qualquer.

O que é medido
--------------
1. Velocidades por gravação: f_eixo, f_motor, razão da caixa, f_e e escorregamento.
2. Picos: PSD de Welch com resolução de ~0,2 Hz; piso local pela mediana móvel;
   pico = proeminência ≥ `--proeminencia` dB acima do piso.
3. Picos que diferem: para cada falha, união dos picos dela com os da normal;
   Δ = nível na falha − nível na normal, na mesma frequência. Ficam os com
   |Δ| ≥ `--delta-min` dB.
4. BPFI e BPFO medidos: para cada gravação de falha, procura-se o f0 que melhor
   explica os 40 maiores picos que SOBEM em relação à normal como n·f0 ± k·f_eixo
   (n até 40, k de −2 a 2: harmônicos e bandas laterais a ± o eixo). Faz-se isso
   nas duas faixas (BPFI: 255–285 Hz; BPFO: 170–195 Hz) e compara-se com f0
   sorteados ao acaso, que dão a linha de base. A tabela cruzada é a evidência:
   gravação de pista interna deve encaixar só na série da BPFI, e de pista
   externa, só na da BPFO.
5. Rótulo de cada pico: harmônico n de qual família (eixo, motor, f_e, BPFI,
   BPFO), ou banda lateral "BPFI×n±k·eixo". Ambíguo quando cabe em mais de uma;
   "?" quando em nenhuma.
6. Espaçamento: entre os picos que diferem acima de 1 kHz, os espaçamentos mais
   frequentes. Bandas laterais a ±f_eixo sugerem engrenamento ou modulação pelo
   eixo; a ±2·f_e, portadora do inversor.

Uso
---
    python scripts/exploration/identify_tonal_peaks.py
    python scripts/exploration/identify_tonal_peaks.py --fmax 12800     # além do Nyquist de trabalho
    python scripts/exploration/identify_tonal_peaks.py --sem-registro
    python scripts/exploration/identify_tonal_peaks.py --sintetico      # auto-teste, sem data/

Saída em reports/signature/: picos_diferentes.csv, velocidades.csv, figuras.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.ndimage import median_filter

import config
import experimentos
import pcm_io

ALVO = "bpfo_0.3mm"
FALHAS = [c for c in config.CLASSES if config.BINARIO[c] == "falha"]

# Cinemática nominal (Jung et al., 2023). Só dão as faixas de busca e as razões.
F_EIXO_NOMINAL = 3010 / 60          # Hz
RAZAO_CAIXA_NOMINAL = 2.07
PARES_POLOS = 2
# BPFI/BPFO do artigo (272 e 179 Hz) NÃO são usadas: ver BPF_FAIXAS e ajustar_bpf.

NPERSEG = 1 << 18                   # 0,195 Hz a 51,2 kHz; ~22 médias em 60 s com 50 %
PISO_HZ = 25.0                      # janela da mediana móvel do piso, em Hz
N_PENTE = 10                        # harmônicos no pente da estimação de velocidade
BPF_N_MAX = 40                      # ordens de BPFI/BPFO consideradas no ajuste e nos rótulos
BPF_K_MAX = 2                       # bandas laterais a ±k·f_eixo
BPF_FAIXAS = {"BPFI": (255.0, 285.0), "BPFO": (170.0, 195.0)}   # Hz, busca do f0
BPF_TOP = 40                        # picos que sobem, usados no ajuste
BPF_TOL_HZ = 0.3                    # tolerância do ajuste
BPF_MIN_ACERTOS = 10                # abaixo disto a série não é considerada encontrada
N_BASE = 200                        # f0 sorteados para a linha de base
VISIVEL_DB = 6.0                    # harmônico "visível" no pente: excesso sobre o piso
MIN_VISIVEIS = 0.3                  # fração mínima de harmônicos visíveis para aceitar o pente
FAIXA_ESPACAMENTO = (5.0, 300.0)    # Hz


# --------------------------------------------------------------------------- #
# Espectro e picos
# --------------------------------------------------------------------------- #
def espectro(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PSD em dB, e o piso local (mediana móvel) em dB."""
    f, p = sg.welch(x, fs=fs, nperseg=min(NPERSEG, len(x)), window="hann")
    db = 10 * np.log10(p + 1e-30)
    n_piso = int(PISO_HZ / (f[1] - f[0])) | 1
    return f, db, median_filter(db, size=n_piso, mode="nearest")


def achar_picos(f, db, piso, proeminencia: float, fmin: float, fmax: float) -> np.ndarray:
    sel = np.where((f >= fmin) & (f <= fmax))[0]
    idx, _ = sg.find_peaks(db[sel] - piso[sel], height=proeminencia, distance=3)
    return sel[idx]


def freq_interpolada(f: np.ndarray, db: np.ndarray, i: int) -> float:
    """Frequência do pico com interpolação parabólica em dB (erro ≪ 1 bin)."""
    if i <= 0 or i + 1 >= f.size:
        return float(f[i])
    a, b, c = db[i - 1], db[i], db[i + 1]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den < 0 else 0.0
    return float(f[i] + np.clip(delta, -0.5, 0.5) * (f[1] - f[0]))


def nivel_em(db: np.ndarray, i: int, raio: int = 2) -> float:
    return float(db[max(0, i - raio):i + raio + 1].max())


# --------------------------------------------------------------------------- #
# Velocidades
# --------------------------------------------------------------------------- #
def pente(f, excesso, candidatos, n_harm, excluir=None, tol_excl=0.3) -> float:
    """
    Frequência do pente harmônico de maior soma de excesso sobre o piso.
    Harmônicos a menos de `tol_excl` de uma frequência em `excluir` não contam.
    """
    df = f[1] - f[0]
    melhor, f_melhor = -np.inf, float("nan")
    for f0 in candidatos:
        s = 0.0
        for k in range(1, n_harm + 1):
            fk = k * f0
            if excluir is not None and np.any(np.abs(fk - excluir) < tol_excl):
                continue
            i = int(round(fk / df))
            if i + 1 >= f.size:
                break
            s += max(0.0, excesso[i - 1:i + 2].max())
        if s > melhor:
            melhor, f_melhor = s, float(f0)
    return f_melhor


def refinar(f, db, exc, f0, n_harm, excluir=None, tol_excl=0.3) -> float:
    """
    Refina f0 pelo ajuste de mínimos quadrados f_k ≈ k·f0 sobre os harmônicos
    visíveis, cada um localizado com interpolação parabólica do pico em dB.
    O pente sozinho fica preso à grade de bins (~0,2 Hz); o erro, multiplicado
    pela ordem do harmônico, estragaria os rótulos de alta ordem.
    """
    df = f[1] - f[0]
    num = den = 0.0
    for k in range(1, n_harm + 1):
        fk = k * f0
        if excluir is not None and np.any(np.abs(fk - excluir) < tol_excl):
            continue
        i0 = int(round(fk / df))
        if i0 + 3 >= f.size:
            break
        i = i0 - 2 + int(np.argmax(db[i0 - 2:i0 + 3]))
        if exc[i] < 6.0:                      # harmônico não visível: não entra no ajuste
            continue
        a, b, c = db[i - 1], db[i], db[i + 1]
        den_p = a - 2 * b + c
        delta = 0.5 * (a - c) / den_p if den_p < 0 else 0.0
        fk_med = (i + float(np.clip(delta, -0.5, 0.5))) * df
        w = exc[i]
        num += w * k * fk_med
        den += w * k * k
    return num / den if den > 0 else f0


def fracao_visivel(f, exc, f0, n_harm, excluir=None, tol_excl=0.3) -> float:
    """Fração dos harmônicos (fora dos excluídos) com excesso ≥ VISIVEL_DB sobre o piso."""
    df = f[1] - f[0]
    vis = tot = 0
    for k in range(1, n_harm + 1):
        fk = k * f0
        if excluir is not None and np.any(np.abs(fk - excluir) < tol_excl):
            continue
        i = int(round(fk / df))
        if i + 1 >= f.size:
            break
        tot += 1
        vis += exc[i - 1:i + 2].max() >= VISIVEL_DB
    return vis / tot if tot else 0.0


def estimar_velocidades(f, db, piso) -> dict:
    exc = db - piso
    f_eixo = pente(f, exc, np.arange(49.0, 51.5, 0.005), N_PENTE)
    f_eixo = refinar(f, db, exc, f_eixo, N_PENTE)
    harm_eixo = f_eixo * np.arange(1, 4 * N_PENTE + 1)

    # motor: sem os harmônicos que coincidem com os do eixo e longe de f_eixo/2,
    # senão o pente em f_eixo/2 "acerta" todos os picos do eixo (armadilha de 25/09)
    cand = np.arange(f_eixo / 2.2, f_eixo / 1.95, 0.005)
    cand = cand[np.abs(cand - f_eixo / 2) > 0.5]
    f_motor = pente(f, exc, cand, N_PENTE, excluir=harm_eixo)
    f_motor = refinar(f, db, exc, f_motor, N_PENTE, excluir=harm_eixo)
    vis_motor = fracao_visivel(f, exc, f_motor, N_PENTE, excluir=harm_eixo)
    if vis_motor < MIN_VISIVEIS:
        f_motor = float("nan")

    f_e, vis_fe = float("nan"), 0.0
    if np.isfinite(f_motor):
        lo = PARES_POLOS * f_motor
        cand = np.arange(lo * 1.002, lo * 1.06, 0.005)
        cand = cand[np.abs(cand - f_eixo) > 0.5]
        f_e = pente(f, exc, cand, 6, excluir=harm_eixo)
        f_e = refinar(f, db, exc, f_e, 6, excluir=harm_eixo)
        vis_fe = fracao_visivel(f, exc, f_e, 6, excluir=harm_eixo)
        if vis_fe < MIN_VISIVEIS or abs(f_e - f_eixo) < 0.5:
            f_e = float("nan")
    s = 1 - PARES_POLOS * f_motor / f_e if np.isfinite(f_e) else float("nan")
    return {"f_eixo": f_eixo, "f_motor": f_motor, "razao_caixa": f_eixo / f_motor,
            "f_e": f_e, "escorregamento": s, "rpm_eixo": 60 * f_eixo,
            "visiveis_motor": vis_motor, "visiveis_f_e": vis_fe}


# --------------------------------------------------------------------------- #
# BPFI e BPFO medidos
# --------------------------------------------------------------------------- #
def cabe(fp: np.ndarray, f0: float, fr: float) -> np.ndarray:
    """Máscara: quais picos cabem em n·f0 ± k·fr."""
    ks = np.arange(-BPF_K_MAX, BPF_K_MAX + 1)
    r = fp[:, None] - ks[None, :] * fr
    n = np.clip(np.round(r / f0), 1, BPF_N_MAX)
    return np.abs(r - n * f0).min(axis=1) < BPF_TOL_HZ


def acertos(fp: np.ndarray, f0s: np.ndarray, fr: float) -> np.ndarray:
    """Para cada f0, quantos picos cabem em n·f0 ± k·fr (n ≤ BPF_N_MAX, |k| ≤ BPF_K_MAX)."""
    ks = np.arange(-BPF_K_MAX, BPF_K_MAX + 1)
    r = fp[None, :, None] - ks[None, None, :] * fr              # [1, picos, k]
    n = np.clip(np.round(r / f0s[:, None, None]), 1, BPF_N_MAX)  # [f0, picos, k]
    dist = np.abs(r - n * f0s[:, None, None]).min(axis=2)       # [f0, picos]
    return (dist < BPF_TOL_HZ).sum(axis=1)


def ajustar_bpf(linhas: list[dict], fr: float, faixa: tuple[float, float],
                rng: np.random.Generator) -> dict:
    """Melhor f0 na faixa, refinado por mínimos quadrados, com a linha de base."""
    sobem = sorted((r for r in linhas if r["delta_db"] > 0), key=lambda r: -r["delta_db"])
    fp = np.array([r["f_hz"] for r in sobem[:BPF_TOP]])
    if fp.size < BPF_MIN_ACERTOS:
        return {"f0": float("nan"), "acertos": 0, "de": int(fp.size), "base": float("nan")}
    grade = np.arange(faixa[0], faixa[1], 0.005)
    ac = acertos(fp, grade, fr)
    f0 = float(grade[int(np.argmax(ac))])
    # refino: f_p − k·fr = n·f0 → f0 = Σ n·(f_p − k·fr) / Σ n²
    num = den = 0.0
    for x in fp:
        best = min(((abs(x - k * fr - round((x - k * fr) / f0) * f0), k)
                    for k in range(-BPF_K_MAX, BPF_K_MAX + 1)))
        if best[0] < BPF_TOL_HZ:
            k = best[1]
            n = round((x - k * fr) / f0)
            if 1 <= n <= BPF_N_MAX:
                num += n * (x - k * fr)
                den += n * n
    if den:
        f0 = num / den
    base = float(acertos(fp, rng.uniform(150.0, 300.0, N_BASE), fr).mean())
    n_ac = int(acertos(fp, np.array([f0]), fr)[0])
    return {"f0": f0, "acertos": n_ac, "de": int(fp.size), "base": base,
            "picos": fp.tolist()}


def decidir_bpf(res: dict[str, dict], fr: float) -> None:
    """
    Aceita cada série só pelos picos que ela explica SOZINHA. Com bandas laterais
    a ±eixo, séries de f0 comensuráveis (por exemplo 3·f0 ≈ 2·BPFI) acabam
    "acertando" os mesmos picos; sem esta regra, uma gravação de pista interna
    poderia ganhar uma BPFO fantasma feita das linhas da BPFI.
    """
    nomes = list(res)
    for nome in nomes:
        r = res[nome]
        fp = np.array(r["picos"])
        if fp.size == 0:
            r.update(exclusivos=0, f0_medida=float("nan"))
            continue
        minha = cabe(fp, r["f0"], fr)
        outras = np.zeros_like(minha)
        for outro in nomes:
            if outro != nome and res[outro]["acertos"] >= r["acertos"] and np.array(res[outro]["picos"]).size:
                outras |= cabe(fp, res[outro]["f0"], fr)
        r["exclusivos"] = int((minha & ~outras).sum())
        r["f0_medida"] = r["f0"] if r["exclusivos"] >= BPF_MIN_ACERTOS else float("nan")
    for r in res.values():
        r.pop("picos", None)


def familias(v: dict, bpf: dict) -> dict[str, tuple[float, int, bool]]:
    """
    nome → (f0, n máximo, admite bandas laterais a ±k·eixo). Frequências não
    identificadas (nan) ficam de fora: não se rotula com o que não foi medido.
    """
    fams = {"eixo": (v["f_eixo"], 10_000, False),
            "motor": (v["f_motor"], 10_000, False),
            "f_e": (v["f_e"], 10_000, False),
            "BPFI": (bpf.get("BPFI", float("nan")), BPF_N_MAX, True),
            "BPFO": (bpf.get("BPFO", float("nan")), BPF_N_MAX, True)}
    return {k: t for k, t in fams.items() if np.isfinite(t[0])}


def rotular(fp: float, fams: dict, fr: float) -> list[str]:
    """Famílias em que o pico cabe, na forma 'eixo×12' ou 'BPFO×4+1·eixo'."""
    out = []
    for nome, (f0, n_max, laterais) in fams.items():
        ks = range(-BPF_K_MAX, BPF_K_MAX + 1) if laterais else (0,)
        for k in ks:
            n = int(round((fp - k * fr) / f0))
            if n < 1 or n > n_max:
                continue
            # erro da frequência do pico (interpolada, ~0,1 Hz) + erro de f0
            # multiplicado pela ordem. Em ordem alta a chance de coincidência
            # cresce: rótulo como "motor×161" é candidato, não prova.
            tol = 0.1 + 0.002 * n if not laterais else 0.15 + 0.005 * n
            if abs(fp - k * fr - n * f0) <= tol:
                out.append(f"{nome}×{n}" + (f"{k:+d}·eixo" if k else ""))
                break
    return out


def familia_de(rotulo: str) -> str:
    """'BPFO×4' → 'BPFO'; 'BPFI×3+1·eixo' → 'BPFI±eixo'."""
    nome = rotulo.split("×")[0]
    return nome + "±eixo" if "·eixo" in rotulo else nome


# --------------------------------------------------------------------------- #
# Comparação
# --------------------------------------------------------------------------- #
def picos_diferentes(f, esp_f, esp_n, picos_f, picos_n, delta_min, fams, fr) -> list[dict]:
    db_f, piso_f = esp_f
    db_n, piso_n = esp_n
    linhas, vistos = [], set()
    for origem, idxs in (("falha", picos_f), ("normal", picos_n)):
        for i in idxs:
            if any(abs(i - j) <= 2 for j in vistos):
                continue
            nf, nn = nivel_em(db_f, i), nivel_em(db_n, i)
            delta = nf - nn
            if abs(delta) < delta_min:
                continue
            vistos.add(i)
            fp = freq_interpolada(f, db_f if origem == "falha" else db_n, i)
            rot = rotular(fp, fams, fr)
            linhas.append({
                "f_hz": fp, "origem": origem, "delta_db": delta,
                "nivel_falha_db": nf, "nivel_normal_db": nn,
                "acima_piso_db": (nf - piso_f[i]) if origem == "falha" else (nn - piso_n[i]),
                "potencia_excesso": max(0.0, 10 ** (nf / 10) - 10 ** (nn / 10)),
                "potencia_deficit": max(0.0, 10 ** (nn / 10) - 10 ** (nf / 10)),
                "rotulos": ";".join(rot) if rot else "?",
                "familia": (familia_de(rot[0]) if len(rot) == 1 else
                            "ambiguo" if rot else "?"),
            })
    return sorted(linhas, key=lambda r: r["f_hz"])


def atribuir(linhas: list[dict], chave: str) -> dict[str, float]:
    tot = sum(r[chave] for r in linhas) or 1.0
    out: dict[str, float] = {}
    for r in linhas:
        out[r["familia"]] = out.get(r["familia"], 0.0) + r[chave] / tot
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def espacamentos(linhas: list[dict], f_acima: float = 1000.0, top: int = 5,
                 resol: float = 0.25) -> list[tuple[float, int]]:
    fs_ = np.array([r["f_hz"] for r in linhas if r["f_hz"] >= f_acima])
    if fs_.size < 3:
        return []
    d = np.abs(fs_[:, None] - fs_[None, :])[np.triu_indices(fs_.size, 1)]
    d = d[(d >= FAIXA_ESPACAMENTO[0]) & (d <= FAIXA_ESPACAMENTO[1])]
    if d.size == 0:
        return []
    bins = np.arange(FAIXA_ESPACAMENTO[0], FAIXA_ESPACAMENTO[1] + resol, resol)
    h, e = np.histogram(d, bins=bins)
    h = h + np.r_[0, h[:-1]] + np.r_[h[1:], 0]           # tolera ±1 bin
    ordem = np.argsort(h)[::-1]
    out, usados = [], []
    for k in ordem:
        c = float(e[k] + resol / 2)
        if h[k] < 3 or any(abs(c - u) < 1.0 for u in usados):
            continue
        out.append((c, int(h[k])))
        usados.append(c)
        if len(out) == top:
            break
    return out


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
CORES = {"eixo": "tab:blue", "motor": "tab:cyan", "f_e": "tab:purple", "BPFI": "tab:red",
         "BPFI±eixo": "salmon", "BPFO": "tab:orange", "BPFO±eixo": "gold",
         "ambiguo": "0.55", "?": "k"}


def fig_zoom(f, esp: dict, vel: dict, bpf: dict, caminho: Path, fmax: float = 1500.0) -> None:
    """PSD até `fmax`, com os harmônicos do eixo e da BPFI/BPFO MEDIDAS em cada gravação."""
    sel = f <= fmax
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(11, 2.4 * len(FALHAS)), sharex=True)
    for ax, falha in zip(axes, FALHAS):
        ax.plot(f[sel], esp["normal"][0][sel], color="k", lw=0.6, label="normal")
        ax.plot(f[sel], esp[falha][0][sel], color="tab:green" if falha == ALVO else "tab:red",
                lw=0.6, alpha=0.8, label=falha)
        marcas = [("eixo", vel[falha]["f_eixo"])] + [
            (nome, bpf[falha][nome]["f0_medida"]) for nome in BPF_FAIXAS
            if np.isfinite(bpf[falha][nome]["f0_medida"])]
        for nome, f0 in marcas:
            for k in range(1, int(fmax / f0) + 1):
                ax.axvline(k * f0, color=CORES[nome], lw=0.5, ls=":", alpha=0.8)
        ax.set_ylabel("dB")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frequência (Hz) — pontilhado: harmônicos do eixo (azul) e da "
                        "BPFO (laranja) / BPFI (vermelho) medidas na gravação")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_picos(res: dict, caminho: Path, fmax: float) -> None:
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(11, 2.3 * len(FALHAS)), sharex=True)
    for ax, falha in zip(axes, FALHAS):
        for r in res[falha]:
            ax.vlines(r["f_hz"], 0, r["delta_db"], color=CORES[r["familia"]], lw=1)
        for nome, cor in CORES.items():
            ax.plot([], [], color=cor, label=nome)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(0, fmax)
        ax.set_ylabel("Δ (dB)")
        ax.set_title(f"{falha} − normal: picos com |Δ| acima do limiar", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=7, loc="upper right")
    axes[-1].set_xlabel("frequência (Hz)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Dados sintéticos
# --------------------------------------------------------------------------- #
SINT_BPFI, SINT_BPFO = 5.345, 3.653     # × eixo, os valores medidos em 25/09


def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Resposta conhecida: eixo 50,17 Hz, motor 50,17/2,07, f_e com 1,5 % de
    escorregamento. BPFI e BPFO DIFERENTES das do artigo, como nos dados reais:
    5,345× e 3,653× o eixo. BPFI com bandas laterais a ±eixo. Uma "portadora" a
    4 kHz com bandas a ±2·f_e só na bpfo_0.3mm. O script tem que estimar
    velocidades, BPFI e BPFO, fechar a tabela cruzada na diagonal e deixar a
    portadora como "?".
    """
    rng = np.random.default_rng(seed)
    fs, dur = config.FS_ORIGINAL, 60.0
    t = np.arange(int(fs * dur)) / fs
    fe_, fm = 50.17, 50.17 / RAZAO_CAIXA_NOMINAL
    fel = PARES_POLOS * fm / (1 - 0.015)

    def tons(f0, n, a):
        return sum(a / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) for k in range(1, n + 1))

    out = []
    for c in config.CLASSES:
        x = 0.002 * rng.standard_normal(t.size) + tons(fe_, 12, 0.02) + tons(fm, 8, 0.01) \
            + tons(fel, 4, 0.01)
        g = 1.0 if c.endswith("1.0mm") else 0.3
        if c.startswith("bpfo"):
            x += g * tons(SINT_BPFO * fe_, 14, 0.02)
        if c.startswith("bpfi"):
            fb = SINT_BPFI * fe_
            for n in range(1, 13):
                for k, a in ((0, 0.02), (-1, 0.01), (1, 0.01)):
                    x += g * a / n * np.sin(2 * np.pi * (n * fb + k * fe_) * t + rng.uniform(0, 6.28))
        if c == ALVO:
            for m in range(-3, 4):
                x += 0.003 / (1 + abs(m)) * np.sin(2 * np.pi * (4000 + 2 * m * fel) * t)
        out.append(pcm_io.Clip(rotulo=c, binario=config.BINARIO[c], fs=fs, x_float=x))
    return out


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pcm-raw", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/signature"))
    ap.add_argument("--fmin", type=float, default=15.0)
    ap.add_argument("--fmax", type=float, default=config.FS_TRABALHO / 2,
                    help="até onde procurar picos (padrão: Nyquist da taxa de trabalho)")
    ap.add_argument("--proeminencia", type=float, default=10.0, help="dB acima do piso local")
    ap.add_argument("--delta-min", type=float, default=6.0, help="|Δ| mínimo contra a normal, dB")
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="Clara")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        clipes = sintetico()
        print(f"Modo sintético: esperado f_eixo 50,170 · f_motor {50.17 / RAZAO_CAIXA_NOMINAL:.3f} · "
              f"f_e {2 * 50.17 / RAZAO_CAIXA_NOMINAL / 0.985:.3f} Hz · "
              f"BPFI {SINT_BPFI * 50.17:.2f} · BPFO {SINT_BPFO * 50.17:.2f} Hz; "
              f"portadora em 4 kHz com espaçamento {2 * 2 * 50.17 / RAZAO_CAIXA_NOMINAL / 0.985:.2f} Hz.")
    else:
        clipes = pcm_io.carregar_clipes(args.pcm_raw, fs_esperado=config.FS_ORIGINAL)
        pcm_io.relatar_integridade(pcm_io.verificar_integridade(clipes))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    esp, vel, picos = {}, {}, {}
    for c in clipes:
        f, db, piso = espectro(c.x, c.fs)
        esp[c.rotulo] = (db, piso)
        vel[c.rotulo] = estimar_velocidades(f, db, piso)
        picos[c.rotulo] = achar_picos(f, db, piso, args.proeminencia, args.fmin, args.fmax)
    df = f[1] - f[0]

    def fmt(x, casas=3):
        return f"{x:.{casas}f}" if np.isfinite(x) else "n/id"

    print(f"\n1. Velocidades estimadas (resolução da PSD: {df:.3f} Hz; n/id = não identificada)")
    print(f"  {'classe':<12}{'f_eixo':>9}{'rpm':>8}{'f_motor':>9}{'caixa':>8}{'f_e':>9}{'escorr.':>9}"
          f"{'picos':>7}")
    for c in config.CLASSES:
        v = vel[c]
        esc = f"{v['escorregamento']:.1%}" if np.isfinite(v["escorregamento"]) else "n/id"
        print(f"  {c:<12}{v['f_eixo']:>9.3f}{v['rpm_eixo']:>8.0f}{fmt(v['f_motor']):>9}"
              f"{fmt(v['razao_caixa']):>8}{fmt(v['f_e']):>9}{esc:>9}{len(picos[c]):>7d}")
    spread = max(v["f_eixo"] for v in vel.values()) - min(v["f_eixo"] for v in vel.values())
    print(f"  variação de f_eixo entre gravações: {spread:.3f} Hz ({spread / F_EIXO_NOMINAL:.2%})")

    # BPFI/BPFO medidas: ajuste sobre os picos que sobem, antes de rotular
    rng = np.random.default_rng(config.SEMENTE)
    bpf: dict[str, dict] = {}
    for falha in FALHAS:
        brutas = picos_diferentes(f, esp[falha], esp["normal"], picos[falha], picos["normal"],
                                  args.delta_min, {}, vel[falha]["f_eixo"])
        bpf[falha] = {nome: ajustar_bpf(brutas, vel[falha]["f_eixo"], faixa, rng)
                      for nome, faixa in BPF_FAIXAS.items()}
        decidir_bpf(bpf[falha], vel[falha]["f_eixo"])

    print(f"\n2. BPFI e BPFO medidas: dos {BPF_TOP} maiores picos que sobem, quantos cabem em "
          f"n·f0 ± k·eixo (tolerância {BPF_TOL_HZ} Hz; linha de base = f0 ao acaso)")
    print(f"  {'':<12}{'---------- BPFI ----------':>28}{'---------- BPFO ----------':>30}")
    print(f"  {'classe':<12}{'f0 (Hz)':>9}{'acertos':>9}{'só dela':>8}{'base':>6}"
          f"{'f0 (Hz)':>11}{'acertos':>9}{'só dela':>8}{'base':>6}")
    for falha in FALHAS:
        bi, bo = bpf[falha]["BPFI"], bpf[falha]["BPFO"]
        print(f"  {falha:<12}{fmt(bi['f0_medida'], 2):>9}{bi['acertos']:>6}/{bi['de']:<2}"
              f"{bi['exclusivos']:>8}{bi['base']:>6.1f}"
              f"{fmt(bo['f0_medida'], 2):>11}{bo['acertos']:>6}/{bo['de']:<2}"
              f"{bo['exclusivos']:>8}{bo['base']:>6.1f}")
    print(f"  Uma série só é aceita (f0 ≠ n/id) com ≥ {BPF_MIN_ACERTOS} picos que só ela explica.")
    print("  Esperado se a assinatura for do rolamento: pista interna só na coluna BPFI,")
    print("  pista externa só na BPFO. Referência do artigo: BPFI 272, BPFO 179 Hz.")

    res, resumo = {}, {}
    print(f"\n3. Picos que diferem da normal (|Δ| ≥ {args.delta_min:g} dB, "
          f"{args.fmin:g}–{args.fmax:g} Hz)")
    for falha in FALHAS:
        # rótulos pela velocidade e pelas BPFI/BPFO medidas na própria gravação de falha
        medidas = {nome: bpf[falha][nome]["f0_medida"] for nome in BPF_FAIXAS}
        fams = familias(vel[falha], medidas)
        linhas = picos_diferentes(f, esp[falha], esp["normal"], picos[falha], picos["normal"],
                                  args.delta_min, fams, vel[falha]["f_eixo"])
        res[falha] = linhas
        exc, dfc = atribuir(linhas, "potencia_excesso"), atribuir(linhas, "potencia_deficit")
        esp_top = espacamentos(linhas)
        resumo[falha] = {"n_picos": len(linhas),
                         "n_excesso": sum(r["delta_db"] > 0 for r in linhas),
                         "n_deficit": sum(r["delta_db"] < 0 for r in linhas),
                         "excesso_por_familia": exc, "deficit_por_familia": dfc,
                         "espacamentos_hz": esp_top}
        marca = "   ← caso difícil" if falha == ALVO else ""
        print(f"\n  {falha}: {len(linhas)} picos ({resumo[falha]['n_excesso']} a mais, "
              f"{resumo[falha]['n_deficit']} a menos){marca}")
        print("    potência a mais, por família:  " +
              ", ".join(f"{k} {v:.0%}" for k, v in exc.items() if v >= 0.01))
        if resumo[falha]["n_deficit"]:
            print("    potência a menos, por família: " +
                  ", ".join(f"{k} {v:.0%}" for k, v in dfc.items() if v >= 0.01))
        top = sorted(linhas, key=lambda r: -abs(r["delta_db"]))[:8]
        for r in top:
            print(f"      {r['f_hz']:>9.2f} Hz  Δ {r['delta_db']:+6.1f} dB  ({r['origem']})  {r['rotulos']}")
        if esp_top:
            ref = {"eixo": vel[falha]["f_eixo"], "2·eixo": 2 * vel[falha]["f_eixo"],
                   "motor": vel[falha]["f_motor"], "f_e": vel[falha]["f_e"],
                   "2·f_e": 2 * vel[falha]["f_e"],
                   **{nome: bpf[falha][nome]["f0_medida"] for nome in BPF_FAIXAS}}
            ref = {k: v for k, v in ref.items() if np.isfinite(v)}
            txt = []
            for d, n in esp_top:
                perto = [k for k, v in ref.items() if abs(d - v) < 0.5]
                txt.append(f"{d:.2f} Hz (×{n}{', ≈ ' + perto[0] if perto else ''})")
            print("    espaçamentos mais frequentes acima de 1 kHz: " + "; ".join(txt))

    fig_zoom(f, esp, vel, bpf, args.out_dir / "fig_picos_zoom_baixa_freq.png")
    fig_picos(res, args.out_dir / "fig_picos_diferentes_por_familia.png", args.fmax)

    with (args.out_dir / "velocidades.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["classe", *vel["normal"].keys(), "bpfi_hz", "bpfo_hz"])
        w.writeheader()
        for c in config.CLASSES:
            extra = ({"bpfi_hz": fmt(bpf[c]["BPFI"]["f0_medida"]), "bpfo_hz": fmt(bpf[c]["BPFO"]["f0_medida"])}
                     if c in bpf else {"bpfi_hz": "", "bpfo_hz": ""})
            w.writerow({"classe": c, **{k: f"{v:.4f}" for k, v in vel[c].items()}, **extra})
    with (args.out_dir / "picos_diferentes.csv").open("w", encoding="utf-8", newline="") as fh:
        campos = ["classe", "f_hz", "origem", "delta_db", "nivel_falha_db", "nivel_normal_db",
                  "acima_piso_db", "familia", "rotulos"]
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for falha in FALHAS:
            for r in res[falha]:
                w.writerow({"classe": falha, **{k: (f"{v:.3f}" if isinstance(v, float) else v)
                                                for k, v in r.items()}})
    def sem_nan(o):
        if isinstance(o, dict):
            return {k: sem_nan(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [sem_nan(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    (args.out_dir / "picos_metrics.json").write_text(json.dumps(sem_nan(
        {"data": date.today().isoformat(), "resolucao_hz": df, "velocidades": vel,
         "bpf_medidas": bpf,
         "parametros": {"proeminencia_db": args.proeminencia, "delta_min_db": args.delta_min,
                        "fmin": args.fmin, "fmax": args.fmax},
         "por_falha": resumo}), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaída em {args.out_dir}/")

    if args.sem_registro:
        print("(sem registro)")
        return
    linha = {
        "id": f"exp{experimentos.next_exp_number(args.registry):03d}",
        "data": date.today().isoformat(),
        "etapa": "caracterizacao_assinatura",
        "script": "exploration/identify_tonal_peaks.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "fs_hz": config.FS_ORIGINAL, "nperseg": NPERSEG, "resolucao_hz": f"{df:.3f}",
            "proeminencia_db": args.proeminencia, "delta_min_db": args.delta_min,
            "fmin": args.fmin, "fmax": args.fmax}),
        "dataset": "jung2023_acustico_0Nm",
        "metricas": experimentos.kv({
            **{f"f_eixo_{c}": f"{vel[c]['f_eixo']:.3f}" for c in config.CLASSES},
            **{f"{nome.lower()}_{fa}": fmt(bpf[fa][nome]["f0_medida"], 2)
               for fa in FALHAS for nome in BPF_FAIXAS},
            **{f"acertos_{nome.lower()}_{fa}": bpf[fa][nome]["acertos"]
               for fa in FALHAS for nome in BPF_FAIXAS},
            **{f"npicos_{fa}": resumo[fa]["n_picos"] for fa in FALHAS},
            **{f"exc_sem_rotulo_{fa}": f"{resumo[fa]['excesso_por_familia'].get('?', 0.0):.3f}"
               for fa in FALHAS}}),
        "responsavel": args.responsavel,
        "notas": args.notas or "identificação de tons; BPFI/BPFO medidas por ajuste de série harmônica",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")


if __name__ == "__main__":
    main()