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
- BPFI ≈ 272 Hz e BPFO ≈ 179 Hz a 50,17 Hz, ou seja, 5,42× e 3,57× o eixo.

Nenhuma dessas frequências é tomada como exata: o script ESTIMA f_eixo, f_motor e
f_e em cada gravação, por busca de pente harmônico na PSD de alta resolução. Isso
já é um resultado: se as gravações não rodaram na mesma velocidade, aparece aqui.

O que é medido
--------------
1. Velocidades por gravação: f_eixo, f_motor, razão da caixa, f_e e escorregamento.
2. Picos: PSD de Welch com resolução de ~0,2 Hz; piso local pela mediana móvel;
   pico = proeminência ≥ `--proeminencia` dB acima do piso.
3. Picos que diferem: para cada falha, união dos picos dela com os da normal;
   Δ = nível na falha − nível na normal, na mesma frequência. Ficam os com
   |Δ| ≥ `--delta-min` dB.
4. Rótulo de cada pico: harmônico n de qual família (eixo, motor, f_e, BPFI,
   BPFO), com tolerância. Ambíguo quando cabe em mais de uma; "?" quando em
   nenhuma — que é o caso da portadora do inversor e do engrenamento, cujas
   frequências não são conhecidas.
5. Espaçamento: entre os picos que diferem acima de 1 kHz, os espaçamentos mais
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
RAZAO_BPFI = 272.0 / F_EIXO_NOMINAL  # ≈ 5,42 × eixo
RAZAO_BPFO = 179.0 / F_EIXO_NOMINAL  # ≈ 3,57 × eixo

NPERSEG = 1 << 18                   # 0,195 Hz a 51,2 kHz; ~22 médias em 60 s com 50 %
PISO_HZ = 25.0                      # janela da mediana móvel do piso, em Hz
N_PENTE = 10                        # harmônicos no pente da estimação de velocidade
BPF_N_MAX = 5                       # harmônicos de BPFI/BPFO considerados
BPF_TOL_REL = 0.01                  # escorregamento dos elementos: ~1–2 % (Randall & Antoni)
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
    """Frequência do pente harmônico de maior soma de excesso sobre o piso."""
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


def estimar_velocidades(f, db, piso) -> dict:
    exc = db - piso
    f_eixo = pente(f, exc, np.arange(49.0, 51.5, 0.005), N_PENTE)
    f_eixo = refinar(f, db, exc, f_eixo, N_PENTE)
    f_motor = pente(f, exc, np.arange(f_eixo / 2.2, f_eixo / 1.95, 0.005), N_PENTE)
    f_motor = refinar(f, db, exc, f_motor, N_PENTE, excluir=f_eixo * np.arange(1, 6))
    harm_eixo = f_eixo * np.arange(1, 8)
    lo = PARES_POLOS * f_motor
    f_e = pente(f, exc, np.arange(lo * 1.002, lo * 1.06, 0.005), 6, excluir=harm_eixo)
    f_e = refinar(f, db, exc, f_e, 6, excluir=harm_eixo)
    s = 1 - PARES_POLOS * f_motor / f_e
    return {"f_eixo": f_eixo, "f_motor": f_motor, "razao_caixa": f_eixo / f_motor,
            "f_e": f_e, "escorregamento": s, "rpm_eixo": 60 * f_eixo}


def familias(v: dict) -> dict[str, tuple[float, int, str]]:
    """nome → (f0, n máximo, tipo de tolerância)."""
    return {
        "eixo": (v["f_eixo"], 10_000, "abs"),
        "motor": (v["f_motor"], 10_000, "abs"),
        "f_e": (v["f_e"], 10_000, "abs"),
        "BPFI": (RAZAO_BPFI * v["f_eixo"], BPF_N_MAX, "rel"),
        "BPFO": (RAZAO_BPFO * v["f_eixo"], BPF_N_MAX, "rel"),
    }


def rotular(fp: float, fams: dict) -> list[str]:
    """Famílias em que o pico cabe como harmônico, na forma 'eixo×12'."""
    out = []
    for nome, (f0, n_max, tipo) in fams.items():
        n = int(round(fp / f0))
        if n < 1 or n > n_max:
            continue
        # erro da frequência do pico (interpolada, ~0,1 Hz) + erro de f0 (poucos
        # mHz depois do refino) multiplicado pela ordem. Em ordem alta a chance de
        # coincidência cresce: rótulo como "motor×161" é candidato, não prova.
        tol = 0.1 + 0.002 * n if tipo == "abs" else BPF_TOL_REL * fp
        if abs(fp - n * f0) <= tol:
            out.append(f"{nome}×{n}")
    return out


# --------------------------------------------------------------------------- #
# Comparação
# --------------------------------------------------------------------------- #
def picos_diferentes(f, esp_f, esp_n, picos_f, picos_n, delta_min, fams) -> list[dict]:
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
            rot = rotular(fp, fams)
            linhas.append({
                "f_hz": fp, "origem": origem, "delta_db": delta,
                "nivel_falha_db": nf, "nivel_normal_db": nn,
                "acima_piso_db": (nf - piso_f[i]) if origem == "falha" else (nn - piso_n[i]),
                "potencia_excesso": max(0.0, 10 ** (nf / 10) - 10 ** (nn / 10)),
                "potencia_deficit": max(0.0, 10 ** (nn / 10) - 10 ** (nf / 10)),
                "rotulos": ";".join(rot) if rot else "?",
                "familia": (rot[0].split("×")[0] if len(rot) == 1 else
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
         "BPFO": "tab:orange", "ambiguo": "0.55", "?": "k"}


def fig_zoom(f, esp: dict, vel: dict, caminho: Path, fmax: float = 700.0) -> None:
    sel = f <= fmax
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(11, 2.4 * len(FALHAS)), sharex=True)
    v = vel["normal"]
    for ax, falha in zip(axes, FALHAS):
        ax.plot(f[sel], esp["normal"][0][sel], color="k", lw=0.6, label="normal")
        ax.plot(f[sel], esp[falha][0][sel], color="tab:green" if falha == ALVO else "tab:red",
                lw=0.6, alpha=0.8, label=falha)
        for nome, f0 in (("eixo", v["f_eixo"]), ("BPFO", RAZAO_BPFO * v["f_eixo"]),
                         ("BPFI", RAZAO_BPFI * v["f_eixo"])):
            for k in range(1, int(fmax / f0) + 1):
                ax.axvline(k * f0, color=CORES[nome], lw=0.5, ls=":", alpha=0.8)
        ax.set_ylabel("dB")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("frequência (Hz) — pontilhado: harmônicos do eixo (azul), "
                        "BPFO (laranja), BPFI (vermelho)")
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
def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Resposta conhecida: eixo 50,17 Hz, motor 50,17/2,07, f_e com 1,5 % de
    escorregamento; BPFO nas classes bpfo_*; uma "portadora" a 4 kHz com bandas
    laterais a ±2·f_e só na bpfo_0.3mm. O script tem que estimar as velocidades,
    rotular BPFO e deixar a portadora como "?", com espaçamento ≈ 2·f_e.
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
        if c.startswith("bpfo"):
            x += tons(RAZAO_BPFO * fe_, 3, 0.02 if c.endswith("1.0mm") else 0.004)
        if c.startswith("bpfi"):
            x += tons(RAZAO_BPFI * fe_, 3, 0.02)
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
        print(f"Modo sintético: esperado f_eixo 50,17 · f_motor {50.17 / RAZAO_CAIXA_NOMINAL:.3f} · "
              f"f_e {2 * 50.17 / RAZAO_CAIXA_NOMINAL / 0.985:.3f} Hz; "
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

    print(f"\n1. Velocidades estimadas (resolução da PSD: {df:.3f} Hz)")
    print(f"  {'classe':<12}{'f_eixo':>9}{'rpm':>8}{'f_motor':>9}{'caixa':>8}{'f_e':>9}{'escorr.':>9}"
          f"{'picos':>7}")
    for c in config.CLASSES:
        v = vel[c]
        print(f"  {c:<12}{v['f_eixo']:>9.3f}{v['rpm_eixo']:>8.0f}{v['f_motor']:>9.3f}"
              f"{v['razao_caixa']:>8.3f}{v['f_e']:>9.3f}{v['escorregamento']:>9.1%}"
              f"{len(picos[c]):>7d}")
    spread = max(v["f_eixo"] for v in vel.values()) - min(v["f_eixo"] for v in vel.values())
    print(f"  variação de f_eixo entre gravações: {spread:.3f} Hz ({spread / F_EIXO_NOMINAL:.2%})")

    res, resumo = {}, {}
    fams_n = familias(vel["normal"])
    print(f"\n2. Picos que diferem da normal (|Δ| ≥ {args.delta_min:g} dB, "
          f"{args.fmin:g}–{args.fmax:g} Hz)")
    for falha in FALHAS:
        # rótulos pela velocidade da própria gravação de falha
        fams = familias(vel[falha])
        linhas = picos_diferentes(f, esp[falha], esp["normal"], picos[falha], picos["normal"],
                                  args.delta_min, fams)
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
            ref = {"eixo": vel[falha]["f_eixo"], "motor": vel[falha]["f_motor"],
                   "f_e": vel[falha]["f_e"], "2·f_e": 2 * vel[falha]["f_e"]}
            txt = []
            for d, n in esp_top:
                perto = [k for k, v in ref.items() if abs(d - v) < 0.5]
                txt.append(f"{d:.2f} Hz (×{n}{', ≈ ' + perto[0] if perto else ''})")
            print("    espaçamentos mais frequentes acima de 1 kHz: " + "; ".join(txt))

    fig_zoom(f, esp, vel, args.out_dir / "fig_picos_zoom_baixa_freq.png")
    fig_picos(res, args.out_dir / "fig_picos_diferentes_por_familia.png", args.fmax)

    with (args.out_dir / "velocidades.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["classe", *vel["normal"].keys()])
        w.writeheader()
        for c in config.CLASSES:
            w.writerow({"classe": c, **{k: f"{v:.4f}" for k, v in vel[c].items()}})
    with (args.out_dir / "picos_diferentes.csv").open("w", encoding="utf-8", newline="") as fh:
        campos = ["classe", "f_hz", "origem", "delta_db", "nivel_falha_db", "nivel_normal_db",
                  "acima_piso_db", "familia", "rotulos"]
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for falha in FALHAS:
            for r in res[falha]:
                w.writerow({"classe": falha, **{k: (f"{v:.3f}" if isinstance(v, float) else v)
                                                for k, v in r.items()}})
    (args.out_dir / "picos_metrics.json").write_text(json.dumps(
        {"data": date.today().isoformat(), "resolucao_hz": df, "velocidades": vel,
         "parametros": {"proeminencia_db": args.proeminencia, "delta_min_db": args.delta_min,
                        "fmin": args.fmin, "fmax": args.fmax},
         "por_falha": resumo}, indent=2, ensure_ascii=False), encoding="utf-8")
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
            **{f"npicos_{fa}": resumo[fa]["n_picos"] for fa in FALHAS},
            **{f"exc_sem_rotulo_{fa}": f"{resumo[fa]['excesso_por_familia'].get('?', 0.0):.3f}"
               for fa in FALHAS}}),
        "responsavel": args.responsavel,
        "notas": args.notas or "identificação de tons; velocidades estimadas por pente harmônico",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")


if __name__ == "__main__":
    main()