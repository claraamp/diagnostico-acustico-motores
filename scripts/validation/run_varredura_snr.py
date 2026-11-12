#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_varredura_snr.py — robustez ao ruído: varredura de SNR e análise de margem
==============================================================================

Card "Rodar os testes de robustez com ruído sintético (varredura de SNR)", com o
escopo da decisão "Robustez ao ruído também na placa, pela serial" (05/11).

Onde o ruído entra
------------------
No int16 a 51,2 kHz, antes da decimação, com a SNR definida nessa entrada
(decisão de 05/11): para cada segmento, o trecho da gravação original que o
decimador lê (`dsp.recorte_para_decimar`) recebe o ruído, volta para int16 com
saturação, é decimado pelo mesmo FIR do `02` e volta para int16 a 12,8 kHz, que
é o que a placa recebe pela serial. Sem ruído, esse caminho reproduz bit a bit o
PCM decimado do `02` (o script confere antes de rodar).

- A SNR é a do próprio segmento: a potência do ruído nas 51.200 amostras do
  segmento é a do sinal dividida por 10^(SNR/10), como no aumento de dados.
- Ruído branco e rosa (1/f, gerado no domínio da frequência). Em cada semente,
  o ruído de um segmento é o mesmo em todas as SNRs, só com outra escala, para
  a curva comparar SNR e não sorteio.
- O filtro de decimação tira o ruído acima de 5,76 kHz, então a SNR que a cadeia
  vê na banda é maior que a da entrada. O eixo é a SNR da entrada; a da banda
  sai ao lado (mediana dos segmentos).

O que mede
----------
1. Curva do Protocolo B: em cada fold, a LDA treinada nos segmentos limpos, sem
   e com aumento (4 variantes, deslocamento + estiramento + ruído, como no
   exp224), avaliada nos segmentos de teste com ruído. Acurácia balanceada,
   sensibilidade e especificidade médias, por SNR, tipo de ruído e semente.
   A semente fixa o ruído do teste e o sorteio do aumento: os dois modelos são
   comparados no mesmo ruído.
2. Passo 0 (análise de margem), com o modelo final (`lda_final.json`, exp239)
   nos 295 segmentos com ruído:
   - segmentos com |s| < E, com E o limite absoluto do erro do escore do porte
     (`limite_escore`);
   - amostras saturadas no int16, a 51,2 kHz e a 12,8 kHz;
   - a cadeia em float32 (a mesma emulação do exp253) contra a referência em
     float64: erro de cada estágio em fração da tolerância encadeada, e
     decisões diferentes. O escore é julgado por |Δs|/E, e não pela tolerância
     relativa dele, que vai a zero perto do limiar (`emular`).

Regras, fixadas antes dos números
---------------------------------
- Placa (página "Robustez ao ruído também na placa", 05/11): nenhum segmento
  com |s| < E em nenhum ponto e a emulação dentro da tolerância → a placa é só
  confirmação; senão, a rodada na placa é necessária nos pontos em que isso
  aconteceu (`decidir_placa`).
- Modelo final (card, 09/11): o modelo com aumento só substitui o exp239 se,
  em todos os pontos com SNR ≤ 10 dB, a acurácia balanceada e a especificidade
  médias forem maiores, com diferença maior que um desvio (o maior dos dois
  desvios entre sementes); senão, fica o modelo sem aumento (`decidir_aumento`).

Saídas
------
reports/validation/varredura_snr/ (com --sem-registro, teste_varredura_snr/):
  curva.csv       um ponto por modelo, ruído, SNR e semente
  margem.csv      o passo 0 por ruído, SNR e semente
  metrics.json    tudo, com as médias, as duas decisões e os parâmetros
  resumo.md       as tabelas e as duas decisões
  tab_varredura_snr.tex   a curva no formato do relatório
  curva_snr.png   acurácia balanceada e especificidade por SNR
e uma linha no registry por ponto da grade (ruído × SNR), mais a do sinal limpo.

Uso
---
    python scripts/validation/run_varredura_snr.py --responsavel <nome>
    python scripts/validation/run_varredura_snr.py --sem-registro             # teste
    python scripts/validation/run_varredura_snr.py --sem-registro --rapido    # 1 semente, 3 SNRs
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from scipy import signal as sg

import config
import dsp
import experimentos
import pcm_io
from augmentation import variantes
from exploration.inspect_tolerancias_estagio import CadeiaFloat32
from validation import metricas, particao
from validation.run_protocol import novo_modelo

SNRS_DB = (30, 25, 20, 15, 10, 5, 0)
RUIDOS = ("branco", "rosa")
SEMENTES = (config.SEMENTE, 1, 2)
COPIAS_AUMENTO = 4
SNR_MAX_REGRA_AUMENTO = 10          # dB: a regra do aumento olha os pontos com SNR ≤ 10 dB
MODELOS = ("sem_aumento", "com_aumento")

U_FLOAT32 = 2.0 ** -24              # unidade de arredondamento do float32
TOL_ENCADEADA = {e: (m, enc) for e, m, _, enc in dsp.ESTAGIOS_PORTE}
ESTAGIOS_EMULADOS = ("escala", "janela", "espectro", "mel", "logmel", "mfcc", "media", "desvio",
                     "escore")
METRICAS_B = ("acuracia_balanceada", "sensibilidade", "especificidade")
SAIDA = Path("reports/validation/varredura_snr")
SAIDA_TESTE = Path("reports/validation/teste_varredura_snr")


# --------------------------------------------------------------------------- #
# Ruído
# --------------------------------------------------------------------------- #
def ruido_base(tipo: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Ruído de variância 1: branco gaussiano, ou rosa (densidade ∝ 1/f, sem a componente DC)."""
    w = rng.standard_normal(n)
    if tipo == "branco":
        return w
    if tipo != "rosa":
        raise ValueError(f"ruído desconhecido: {tipo}")
    W = np.fft.rfft(w)
    f = np.arange(len(W), dtype=float)
    f[0] = np.inf                                   # zera a DC
    r = np.fft.irfft(W / np.sqrt(f), n)
    return r / r.std()


def somar_ruido(entrada: np.ndarray, base: np.ndarray, parte: slice, snr_db: float) -> np.ndarray:
    """
    `entrada + k·base`, com k tal que, em `parte` (o segmento), a potência do ruído
    seja a do sinal dividida por 10^(snr_db/10). Volta em float, antes do int16.
    """
    p_sinal = float(np.mean(entrada[parte] ** 2))
    p_base = float(np.mean(base[parte] ** 2))
    if p_sinal <= 0 or p_base <= 0:
        return entrada.astype(np.float64)
    k = math.sqrt(p_sinal / 10.0 ** (snr_db / 10.0) / p_base)
    return entrada + k * base


def saturar_int16(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Arredonda para int16 com saturação; devolve também quantas amostras saturaram."""
    r = np.round(x)
    sat = int(np.count_nonzero((r > 32767) | (r < -32768)))
    return np.clip(r, -32768, 32767).astype(np.int16), sat


class Decimador:
    """O trecho a 51,2 kHz que o decimador lê, para cada segmento, e o FIR do `02`."""

    def __init__(self, raw: dict, segmentos):
        self.spec = dsp.design_decimation(config.FS_ORIGINAL, config.FS_TRABALHO)
        self.taps = dsp.taps_decimacao(self.spec, config.FS_ORIGINAL)
        self.raw = raw
        self.n = config.AMOSTRAS_POR_SEGMENTO
        self.recortes = {s.id: dsp.recorte_para_decimar(s.inicio, self.n, self.spec) for s in segmentos}

    def entrada(self, s) -> tuple[np.ndarray, slice]:
        """O int16 a 51,2 kHz do recorte (zeros fora da gravação, como o `lfilter`) e onde fica o segmento."""
        a, b, descartar = self.recortes[s.id]
        x = self.raw[s.rotulo].pcm
        trecho = np.zeros(b - a, dtype=np.float64)
        ini, fim = max(a, 0), min(b, len(x))
        trecho[ini - a:fim - a] = x[ini:fim]
        m = self.spec.down
        p0 = m * s.inicio - a
        return trecho, slice(p0, p0 + m * self.n)

    def decimar(self, s, entrada: np.ndarray) -> np.ndarray:
        """Float em [-1, 1] a 12,8 kHz: as `n` amostras do segmento, como o `resample_clip`."""
        _, _, descartar = self.recortes[s.id]
        y = sg.lfilter(self.taps, 1.0, entrada / config.INT16_FULL)[:: self.spec.down]
        return y[descartar:descartar + self.n]


# --------------------------------------------------------------------------- #
# Limite absoluto do erro do escore (página da decisão de 05/11)
# --------------------------------------------------------------------------- #
def limite_escore(x: np.ndarray, pesos: np.ndarray, bias: float) -> np.ndarray:
    """
    E de cada segmento (linhas de `x`, as 26 características):
    E = Σ|w_i|·δ_i + E_LDA, com δ_i a tolerância encadeada da característica
    (média: absoluta; desvio: relativa ao próprio valor) e E_LDA a cota do
    arredondamento da soma em float32 na ordem do `lda_escore()`, bias primeiro e
    26 produtos: γ_27·(|b| + Σ|w_i·x_i|), γ_n = n·u/(1 − n·u), u = 2⁻²⁴.
    """
    x = np.atleast_2d(x)
    nc = config.MFCC_N_COEFS
    delta = np.empty_like(x)
    delta[:, :nc] = TOL_ENCADEADA["media"][1]
    delta[:, nc:] = TOL_ENCADEADA["desvio"][1] * np.abs(x[:, nc:])
    n = len(pesos) + 1
    gama = n * U_FLOAT32 / (1 - n * U_FLOAT32)
    e_lda = gama * (abs(bias) + np.abs(x * pesos).sum(axis=1))
    return (np.abs(pesos) * delta).sum(axis=1) + e_lda


# --------------------------------------------------------------------------- #
# Um ponto da grade: os 295 segmentos com ruído
# --------------------------------------------------------------------------- #
def caracteristicas(estagios: dict) -> np.ndarray:
    return np.concatenate([estagios["media"], estagios["desvio"]])


def emular(cadeia: CadeiaFloat32, ref: dict, E: float) -> dict:
    """
    Cadeia em float32 desde o int16; erro de cada estágio em fração da tolerância
    encadeada. O escore é julgado pelo limite absoluto E do segmento, |Δs|/E: a
    tolerância relativa dele (10⁻⁵ do próprio escore) vai a zero perto do limiar,
    que é onde o ruído põe os segmentos. A relativa sai à parte, em `escore_rel`.
    """
    enc = {"escala": cadeia.escala(ref["entrada"])}
    anterior = "escala"
    for e in ("janela", "espectro", "mel", "logmel", "mfcc"):
        enc[e] = getattr(cadeia, e)(enc[anterior])
        anterior = e
    enc["media"], enc["desvio"] = cadeia.resumo(enc["mfcc"])
    enc["escore"] = cadeia.escore(np.concatenate([enc["media"], enc["desvio"]]))
    frac = {e: dsp.erro_estagio(enc[e], ref[e], TOL_ENCADEADA[e][0]) / TOL_ENCADEADA[e][1]
            for e in ESTAGIOS_EMULADOS}
    s32 = float(enc["escore"][0])
    return frac | {"escore_rel": frac["escore"], "escore": abs(s32 - float(ref["escore"][0])) / E,
                   "escore32": s32}


def ponto(dec: Decimador, segmentos, ruido: str | None, snr_db: float | None, semente: int,
          pesos: np.ndarray, bias: float, cadeia: CadeiaFloat32 | None) -> dict:
    """Características e passo 0 dos segmentos num ponto (ruído None = limpo)."""
    X, sat51, sat12, snr_banda, linhas = [], 0, 0, [], []
    for s in segmentos:
        entrada, parte = dec.entrada(s)
        if ruido is None:
            ruidosa = entrada
        else:
            base = ruido_base(ruido, len(entrada), np.random.default_rng([semente, RUIDOS.index(ruido), s.id]))
            ruidosa_f = somar_ruido(entrada, base, parte, snr_db)
            sat51 += saturar_int16(ruidosa_f[parte])[1]
            ruidosa = saturar_int16(ruidosa_f)[0].astype(np.float64)
        y = dec.decimar(s, ruidosa)
        pcm, n_sat12 = saturar_int16(y * config.INT16_FULL)
        sat12 += n_sat12
        if ruido is not None:
            limpo = dec.decimar(s, entrada)
            p_r = float(np.mean((y - limpo) ** 2))
            snr_banda.append(10 * math.log10(float(np.mean(limpo ** 2)) / p_r) if p_r > 0 else math.inf)
        ref = dsp.estagios_mfcc(pcm)
        x = caracteristicas(ref)
        X.append(x)
        if cadeia is not None:
            ref["escore"] = np.array([x @ pesos + bias])
            linhas.append(emular(cadeia, ref, float(limite_escore(x, pesos, bias)[0])))
    X = np.vstack(X)
    escore = X @ pesos + bias
    E = limite_escore(X, pesos, bias)
    out = {"X": X, "escore": escore, "E": E, "sat_51k2": sat51, "sat_12k8": sat12,
           "snr_banda_mediana": float(np.median(snr_banda)) if snr_banda else None}
    if cadeia is not None:
        frac = {e: float(max(l[e] for l in linhas)) for e in ESTAGIOS_EMULADOS}
        out["emul_escore_rel_max"] = float(max(l["escore_rel"] for l in linhas))
        dec32 = np.array([l["escore32"] >= 0 for l in linhas])
        out |= {"emul_frac_tol": frac,
                "decisoes_diferentes": int(np.count_nonzero(dec32 != (escore >= 0)))}
    return out


def resumo_margem(p: dict) -> dict:
    razao = np.abs(p["escore"]) / p["E"]
    i = int(np.argmin(razao))
    out = {"perto_limiar": int(np.count_nonzero(razao < 1)), "margem_min_E": float(razao[i]),
           "segmento_margem_min": i, "sat_51k2": p["sat_51k2"], "sat_12k8": p["sat_12k8"],
           "snr_banda_mediana": p["snr_banda_mediana"]}
    if "emul_frac_tol" in p:
        out |= {"emul_frac_tol_max": max(p["emul_frac_tol"].values()),
                "emul_estagio_max": max(p["emul_frac_tol"], key=p["emul_frac_tol"].get),
                "emul_escore_rel_max": p["emul_escore_rel_max"],
                "decisoes_diferentes": p["decisoes_diferentes"]}
    return out


# --------------------------------------------------------------------------- #
# Protocolo B com o teste ruidoso
# --------------------------------------------------------------------------- #
def carregar_extrair_aumento():
    """O `extrair_aumento` do `04` (o nome do módulo começa com dígito)."""
    caminho = Path(__file__).resolve().parents[1] / "pipeline" / "04_extract_features.py"
    spec = importlib.util.spec_from_file_location("extracao_04", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extrair_aumento


def treinar_folds(folds, X: np.ndarray, y: np.ndarray, aumento: dict | None) -> list:
    """Um modelo por fold, treinado nos segmentos limpos (e nas variantes aceitas, com aumento)."""
    modelos = []
    for f in folds:
        X_fit, y_fit = X[f.treino], y[f.treino]
        if aumento is not None:
            aceitas = variantes.mascara_fold(f.treino, aumento["lidos"])
            X_fit = np.vstack([X_fit, aumento["X"][aceitas]])
            y_fit = np.concatenate([y_fit, y[aumento["origem"][aceitas]]])
        m = novo_modelo("lda", 2)
        m.fit(X_fit, y_fit)
        modelos.append(m)
    return modelos


def avaliar_b(folds, modelos, X_teste: np.ndarray, y: np.ndarray) -> dict:
    resultados = []
    for f, m in zip(folds, modelos):
        r = metricas.metricas_binarias(y[f.teste], m.predict(X_teste[f.teste]))
        resultados.append({"nome": f.nome, "info": f.info, **r})
    return metricas.resumo_protocolo_b(resultados)


# --------------------------------------------------------------------------- #
# Regras
# --------------------------------------------------------------------------- #
def media_dp(valores) -> tuple[float, float]:
    v = np.asarray(list(valores), dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def decidir_aumento(curva: list[dict]) -> dict:
    """
    Regra do card (09/11): o modelo com aumento só substitui o atual se, em todos os
    pontos com SNR ≤ 10 dB, a acurácia balanceada e a especificidade médias forem
    maiores, com diferença maior que o maior dos dois desvios entre sementes.
    """
    pontos, falhas = [], []
    for ruido in RUIDOS:
        for snr in (s for s in SNRS_DB if s <= SNR_MAX_REGRA_AUMENTO):
            sel = {m: [c for c in curva if c["modelo"] == m and c["ruido"] == ruido and c["snr_db"] == snr]
                   for m in MODELOS}
            if not all(sel.values()):
                continue
            for k in ("acuracia_balanceada", "especificidade"):
                (ms, ds), (ma, da) = (media_dp(c[k] for c in sel[m]) for m in MODELOS)
                ok = ma - ms > max(ds, da)
                pontos.append({"ruido": ruido, "snr_db": snr, "metrica": k, "sem": ms, "com": ma,
                               "desvio": max(ds, da), "passa": ok})
                if not ok:
                    falhas.append(f"{k} a {snr} dB ({ruido})")
    troca = bool(pontos) and not falhas
    return {"modelo_final": "com_aumento" if troca else "sem_aumento (exp239)",
            "pontos": pontos, "nao_passa": falhas}


def decidir_placa(margem: list[dict]) -> dict:
    """Regra da página de 05/11: |s| < E em algum ponto, ou emulação fora da tolerância → necessária."""
    pontos = [m for m in margem if m["ruido"] is not None and
              (m["perto_limiar"] > 0 or m["emul_frac_tol_max"] > 1 or m["decisoes_diferentes"] > 0)]
    return {"placa": "necessária" if pontos else "só confirmação (opcional)",
            "pontos": sorted({(m["ruido"], m["snr_db"]) for m in pontos})}


# --------------------------------------------------------------------------- #
# Saídas
# --------------------------------------------------------------------------- #
def _f(v, casas=3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def agregar(curva: list[dict], snrs) -> list[dict]:
    """Média ± desvio entre sementes, por modelo, ruído e SNR (None = limpo)."""
    out = []
    for m in MODELOS:
        for ruido in (None, *RUIDOS):
            for snr in ((None,) if ruido is None else snrs):
                sel = [c for c in curva if c["modelo"] == m and c["ruido"] == ruido and c["snr_db"] == snr]
                if not sel:
                    continue
                linha = {"modelo": m, "ruido": ruido, "snr_db": snr, "n_sementes": len(sel)}
                for k in METRICAS_B:
                    linha[k], linha[k + "_dp"] = media_dp(c[k] for c in sel)
                out.append(linha)
    return out


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    campos = list(dict.fromkeys(k for l in linhas for k in l))
    with caminho.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def resumo_md(agregado, margem_ag, dec_aumento, dec_placa, ids: str) -> str:
    def rotulo(r, s):
        return "limpo" if r is None else f"{r}, {s} dB"
    t = ["# Robustez ao ruído: varredura de SNR", "",
         f"Rodadas {ids}. Ruído somado no int16 a 51,2 kHz, antes da decimação; SNR da entrada, "
         f"por segmento. Sementes: {', '.join(map(str, SEMENTES))}.", "",
         "## Protocolo B com o teste ruidoso (média ± desvio entre sementes)", "",
         "| Ponto | Modelo | Acurácia balanceada | Sensibilidade | Especificidade |",
         "|---|---|---:|---:|---:|"]
    for a in agregado:
        t.append(f"| {rotulo(a['ruido'], a['snr_db'])} | {a['modelo'].replace('_', ' ')} | "
                 + " | ".join(f"{_f(a[k])} ± {_f(a[k + '_dp'])}" for k in METRICAS_B) + " |")
    t += ["", "## Passo 0: análise de margem com o modelo final (exp239), 295 segmentos", "",
          "| Ponto | SNR na banda (mediana) | Segmentos com \\|s\\| < E | Menor \\|s\\|/E | "
          "Saturadas 51,2 kHz | Saturadas 12,8 kHz | Emulação float32 (fração da tol.) | "
          "Escore pela tol. relativa | Decisões diferentes |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in margem_ag:
        banda = "—" if m["snr_banda_mediana"] is None else f"{_f(m['snr_banda_mediana'], 1)} dB"
        t.append(f"| {rotulo(m['ruido'], m['snr_db'])} | {banda} | {m['perto_limiar']} | "
                 f"{_f(m['margem_min_E'], 1)} | {m['sat_51k2']} | {m['sat_12k8']} | "
                 f"{_f(m['emul_frac_tol_max'], 4)} ({m['emul_estagio_max']}) | "
                 f"{_f(m['emul_escore_rel_max'], 2)} | {m['decisoes_diferentes']} |")
    t += ["", "Contagens somadas nas sementes; menor razão e maior fração, o pior caso entre elas. "
          "Na emulação, o escore entra por |Δs|/E; a coluna da tolerância relativa (10⁻⁵ do próprio "
          "escore) é só informativa, porque ela vai a zero perto do limiar.", "",
          "## Decisões pelas regras fixadas antes de rodar", "",
          f"- **Placa com ruído:** {dec_placa['placa']}"
          + (f", nos pontos {', '.join(f'{r} a {s} dB' for r, s in dec_placa['pontos'])}." if dec_placa["pontos"]
             else ": nenhum segmento com |s| < E e a emulação em float32 dentro da tolerância em todos os pontos."),
          f"- **Modelo final:** {dec_aumento['modelo_final']}"
          + (f"; não passa em: {'; '.join(dec_aumento['nao_passa'])}." if dec_aumento["nao_passa"] else "."),
          ""]
    return "\n".join(t)


def tabela_tex(agregado) -> str:
    linhas = []
    for ruido in (None, *RUIDOS):
        for snr in sorted({a["snr_db"] for a in agregado if a["ruido"] == ruido}, key=lambda v: -(v or 99)):
            sel = {a["modelo"]: a for a in agregado if a["ruido"] == ruido and a["snr_db"] == snr}
            nome = "limpo" if ruido is None else f"{ruido}, {snr}~dB"
            cel = [f"{_f(sel[m][k])} $\\pm$ {_f(sel[m][k + '_dp'])}"
                   for m in MODELOS for k in ("acuracia_balanceada", "especificidade")]
            linhas.append(f"{nome} & " + " & ".join(cel) + r" \\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering",
        r"\caption{Protocolo B com ruído somado ao teste no int16 a 51,2~kHz, antes da decimação "
        r"(SNR da entrada): acurácia balanceada e especificidade médias, média $\pm$ desvio entre "
        f"{len(SEMENTES)} sementes, com a LDA treinada sem e com aumento de dados.}}",
        r"\label{tab:varreduraSNR}", r"\footnotesize",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r" & \multicolumn{2}{c}{Sem aumento} & \multicolumn{2}{c}{Com aumento} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Ruído, SNR & Acur. bal. & Especif. & Acur. bal. & Especif. \\", r"\midrule",
        *linhas, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def figura(agregado, caminho: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, k, titulo in zip(eixos, ("acuracia_balanceada", "especificidade"),
                             ("Acurácia balanceada", "Especificidade")):
        for m, cor in zip(MODELOS, ("C0", "C1")):
            for ruido, estilo in zip(RUIDOS, ("-", "--")):
                sel = sorted((a for a in agregado if a["modelo"] == m and a["ruido"] == ruido),
                             key=lambda a: a["snr_db"])
                x = [a["snr_db"] for a in sel]
                ax.errorbar(x, [a[k] for a in sel], yerr=[a[k + "_dp"] for a in sel], color=cor,
                            ls=estilo, marker="o", ms=3, capsize=2,
                            label=f"{m.replace('_', ' ')}, {ruido}")
        ax.set_title(titulo)
        ax.set_xlabel("SNR na entrada (dB)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    eixos[0].set_ylim(0, 1.02)
    eixos[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sem-registro", action="store_true", help="teste: registry intocado")
    ap.add_argument("--rapido", action="store_true",
                    help="teste: 1 semente e 3 SNRs (30, 10 e 0 dB); não registra")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args(argv)
    registrar = not (args.sem_registro or args.rapido)
    if registrar and not args.responsavel:
        ap.error("--responsavel é obrigatório numa rodada registrada (ou use --sem-registro)")
    sementes = SEMENTES[:1] if args.rapido else SEMENTES
    snrs = (30, 10, 0) if args.rapido else SNRS_DB
    out_dir = args.out_dir or (SAIDA if registrar else SAIDA_TESTE)

    # ------------------------------------------------ dados e conferências
    raw = pcm_io.carregar_clipes(args.raw_dir, fs_esperado=config.FS_ORIGINAL)
    dec12 = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
    divergencias = pcm_io.verificar_integridade(raw) + pcm_io.verificar_integridade(dec12)
    particoes = particao.carregar(args.splits)
    problemas = (particao.conferir_compatibilidade(particoes, dec12, config.FS_TRABALHO)
                 + particao.verificar(particoes))
    if divergencias or problemas:
        print("Abortado: PCM ou partição inválidos:\n  " + "\n  ".join(divergencias + problemas))
        return 1
    segmentos = particao.segmentos_de(particoes)
    folds = particao.folds_de(particoes, "B")
    y = np.array([s.binario for s in segmentos])
    modelo = json.loads(args.modelo.read_text(encoding="utf-8"))
    pesos = np.asarray(modelo["dobrada"]["pesos"], dtype=float)
    bias = float(modelo["dobrada"]["bias"])
    dec = Decimador({c.rotulo: c for c in raw}, segmentos)
    por_rotulo12 = {c.rotulo: c for c in dec12}
    for s in segmentos:                  # sem ruído, o caminho reproduz o PCM do 02
        e, _ = dec.entrada(s)
        if not np.array_equal(saturar_int16(dec.decimar(s, e) * config.INT16_FULL)[0],
                              por_rotulo12[s.rotulo].pcm[s.inicio:s.fim]):
            print(f"Abortado: o recorte decimado do segmento {s.id} não reproduz o PCM do 02.")
            return 1
    print(f"{len(segmentos)} segmentos, {len(folds)} folds do B; recorte a 51,2 kHz confere com o 02")

    cadeia = CadeiaFloat32(config.FS_TRABALHO, pesos, bias)
    limpo = ponto(dec, segmentos, None, None, 0, pesos, bias, cadeia)
    print("limpo:", resumo_margem(limpo))

    # ------------------------------------------------ modelos por fold
    extrair_aumento = carregar_extrair_aumento()
    por_rotulo_dec = {c.rotulo: c for c in dec12}
    modelos = {("sem_aumento", None): treinar_folds(folds, limpo["X"], y, None)}
    info_aumento = {}
    for sem in sementes:
        print(f"aumento, semente {sem}: {COPIAS_AUMENTO} variantes por segmento...")
        dados = extrair_aumento(segmentos, por_rotulo_dec, COPIAS_AUMENTO, config.AUMENTO_TECNICAS,
                                config.AUMENTO_ESTIR_MODO, False, sem)
        aum = {"X": dados["X"], "origem": dados["origem"],
               "lidos": variantes.ids_lidos(segmentos, dados["origem"], dados["inicio_lido"], dados["fim_lido"])}
        modelos[("com_aumento", sem)] = treinar_folds(folds, limpo["X"], y, aum)
        info_aumento[sem] = int(len(dados["X"]))

    # ------------------------------------------------ grade
    curva, margem = [], []

    def avaliar(ruido, snr, sem, p, com_margem=True):
        for m in MODELOS:
            chave = (m, None) if m == "sem_aumento" else (m, sem)
            r = avaliar_b(folds, modelos[chave], p["X"], y)
            curva.append({"modelo": m, "ruido": ruido, "snr_db": snr, "semente": sem,
                          **{k: r[k + "_media"] for k in METRICAS_B},
                          "pior_falha": r["pior_falha"],
                          "pior_falha_acuracia_balanceada": r["pior_falha_acuracia_balanceada"]})
        if com_margem:
            margem.append({"ruido": ruido, "snr_db": snr, "semente": sem, **resumo_margem(p)})

    for sem in sementes:
        avaliar(None, None, sem, limpo, com_margem=sem == sementes[0])   # o limpo não depende da semente
        for ruido in RUIDOS:
            for snr in snrs:
                p = ponto(dec, segmentos, ruido, snr, sem, pesos, bias, cadeia)
                avaliar(ruido, snr, sem, p)
                c = {m: curva[-2 + i] for i, m in enumerate(MODELOS)}
                print(f"  semente {sem}, {ruido:<6} {snr:>2} dB: acc bal "
                      f"{c['sem_aumento']['acuracia_balanceada']:.3f} / {c['com_aumento']['acuracia_balanceada']:.3f}, "
                      f"especif. {c['sem_aumento']['especificidade']:.3f} / {c['com_aumento']['especificidade']:.3f}; "
                      f"|s|<E: {margem[-1]['perto_limiar']}, emulação {margem[-1]['emul_frac_tol_max']:.3f} da tol.")

    agregado = agregar(curva, snrs)
    margem_ag = []
    for ruido in (None, *RUIDOS):
        for snr in ((None,) if ruido is None else snrs):
            sel = [m for m in margem if m["ruido"] == ruido and m["snr_db"] == snr]
            pior = min(sel, key=lambda m: m["margem_min_E"])
            margem_ag.append({
                "ruido": ruido, "snr_db": snr,
                **{k: sum(m[k] for m in sel) for k in ("perto_limiar", "sat_51k2", "sat_12k8", "decisoes_diferentes")},
                "margem_min_E": pior["margem_min_E"], "segmento_margem_min": pior["segmento_margem_min"],
                "snr_banda_mediana": (None if ruido is None else float(np.median([m["snr_banda_mediana"] for m in sel]))),
                "emul_frac_tol_max": max(m["emul_frac_tol_max"] for m in sel),
                "emul_estagio_max": max(sel, key=lambda m: m["emul_frac_tol_max"])["emul_estagio_max"],
                "emul_escore_rel_max": max(m["emul_escore_rel_max"] for m in sel)})
    dec_aumento = decidir_aumento(curva)
    dec_placa = decidir_placa(margem_ag)
    print(f"\nplaca com ruído: {dec_placa['placa']} {dec_placa['pontos'] or ''}")
    print(f"modelo final: {dec_aumento['modelo_final']}")

    # ------------------------------------------------ gravação
    n0 = experimentos.next_exp_number(args.registry)
    pontos_reg = [(None, None)] + [(r, s) for r in RUIDOS for s in snrs]
    ids = {p: (f"exp{n0 + i:03d}" if registrar else "teste") for i, p in enumerate(pontos_reg)}
    faixa = f"exp{n0:03d}–exp{n0 + len(pontos_reg) - 1:03d}" if registrar else "de teste"
    commit = experimentos.git_short_hash()
    parametros = {
        "protocolo": "B", "modelo": "lda", "modelos": "+".join(MODELOS), "ponto_do_ruido": "int16_51200Hz",
        "snr": "da_entrada_por_segmento", "ruidos": "+".join(RUIDOS), "snrs_db": "-".join(map(str, snrs)),
        "sementes": "-".join(map(str, sementes)), "aumento": "+".join(config.AUMENTO_TECNICAS),
        "aumento_copias": COPIAS_AUMENTO, "aumento_snr_db": "-".join(map(str, config.AUMENTO_SNR_DB)),
        "modelo_margem": modelo.get("id"), "splits": particao.hash_arquivo(args.splits),
        "fs_hz": config.FS_TRABALHO, "regra_aumento_snr_max_db": SNR_MAX_REGRA_AUMENTO,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    gravar_csv(out_dir / "curva.csv", curva)
    gravar_csv(out_dir / "margem.csv", margem)
    (out_dir / "metrics.json").write_text(json.dumps({
        "ids": {("limpo" if r is None else f"{r}_{s}db"): i for (r, s), i in ids.items()},
        "git_commit": commit, "parametros": parametros, "variantes_aumento": info_aumento,
        "curva": agregado, "margem": margem_ag, "decisao_aumento": dec_aumento, "decisao_placa": dec_placa,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "resumo.md").write_text(resumo_md(agregado, margem_ag, dec_aumento, dec_placa, faixa),
                                       encoding="utf-8", newline="\n")
    (out_dir / "tab_varredura_snr.tex").write_text(tabela_tex(agregado), encoding="utf-8", newline="\n")
    figura(agregado, out_dir / "curva_snr.png")
    print(f"saídas: {out_dir}/")
    if not registrar:
        print("(teste: registry não alterado)")
        return 0

    linhas = []
    for (ruido, snr), exp_id in ids.items():
        met = {}
        for a in (a for a in agregado if a["ruido"] == ruido and a["snr_db"] == snr):
            sigla = "sem" if a["modelo"] == "sem_aumento" else "com"
            for k, abrev in zip(METRICAS_B, ("acc_bal", "sens", "espec")):
                met[f"{abrev}_{sigla}_media"] = f"{a[k]:.4f}"
                met[f"{abrev}_{sigla}_dp"] = f"{a[k + '_dp']:.4f}"
        m = next(m for m in margem_ag if m["ruido"] == ruido and m["snr_db"] == snr)
        met |= {"perto_limiar": m["perto_limiar"], "margem_min_E": f"{m['margem_min_E']:.0f}",
                "sat_51k2": m["sat_51k2"], "sat_12k8": m["sat_12k8"],
                "snr_banda_mediana_db": None if m["snr_banda_mediana"] is None else f"{m['snr_banda_mediana']:.1f}",
                "emul_frac_tol_max": f"{m['emul_frac_tol_max']:.4f}", "decisoes_diferentes": m["decisoes_diferentes"]}
        linhas.append({
            "id": exp_id, "data": date.today().isoformat(), "etapa": "robustez_ruido",
            "script": "validation/run_varredura_snr.py", "git_commit": commit,
            "parametros": experimentos.kv({**parametros, "ruido": ruido or "nenhum", "snr_db": snr}),
            "dataset": f"jung2023_acustico_0Nm_{config.FS_ORIGINAL}Hz_decimado_{config.FS_TRABALHO}Hz",
            "metricas": experimentos.kv(met), "responsavel": args.responsavel,
            "notas": args.notas or ("referência sem ruído da varredura" if ruido is None else
                                    f"varredura de SNR, ponto {ruido} a {snr} dB; {faixa}"),
        })
    experimentos.append_registry(args.registry, linhas)
    print(f"registrado: {faixa} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
