#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_lda_harmonicos.py — quanto da decisão da LDA vem dos harmônicos do eixo
===============================================================================

Controle da tarefa Classificador 2/6, levantado pela caracterização da
assinatura (exp023–exp026): a gravação normal tem harmônicos do eixo mais
fortes que as de falha, e com uma única gravação normal o classificador pode
ter aprendido "harmônicos do eixo fortes = normal" em vez da falha. A curva de
aprendizado não separa as duas coisas (o B já fica perto do valor final com 2 s
por classe), então este script olha direto para os pesos do modelo.

Método (Protocolo B, treino inteiro, features de referência do 04):
1. **Bandas dos harmônicos.** Em cada gravação, a potência nos múltiplos de
   f_eixo (PSD de 0,2 Hz no sinal de 12,8 kHz, máximo a ±0,5 Hz de cada n·f_eixo)
   é somada em cada filtro de Mel do `dsp`. Em cada fold, as bandas em que a
   normal tem ≥ LIMIAR_DB a mais que a média das falhas de treino são as
   "bandas dos harmônicos".
2. **Pesos da LDA por banda.** A média dos MFCC de um segmento é D·(média do
   log-Mel), com D as 13 primeiras linhas da DCT-II ortonormal. Então a parte da
   LDA que lê a média dos MFCC equivale a pesos v = Dᵀ·w nas 20 bandas de Mel
   (w já dividido pela escala do StandardScaler).
3. **Contribuição.** A separação entre as médias de treino das duas classes,
   w·(x̄_normal − x̄_falha), é decomposta em v_b·Δlog-Mel_b por banda, mais a
   parte do desvio dos MFCC, que não tem mapa por banda e é reportada à parte.
   A pergunta é que fração da separação vem das bandas dos harmônicos,
   comparada com a fração de bandas que elas são.

Não é prova de atalho: uma banda dos harmônicos também pode conter energia de
falha. É uma medida de quanto o modelo se apoia nessas bandas.

Auto-teste: `--sintetico` monta gravações em que a normal só difere das falhas
por harmônicos do eixo mais fortes acima de 3 kHz. O script tem que achar essas
bandas e atribuir a elas quase toda a separação. Implica `--sem-registro` e
grava em `reports/validation/controle_harmonicos/sintetico/` (fora do Git).

Uso
---
    python scripts/exploration/inspect_lda_harmonicos.py --sintetico
    python scripts/exploration/inspect_lda_harmonicos.py --responsavel <nome>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.fft import dct

import config
import experimentos
import pcm_io
from exploration.inspect_signature_mel import (MEDIDAS_JSON, NPERSEG_FINO, TOL_LINHA_HZ,
                                               carregar_features, carregar_medidas,
                                               centros_mel, features_como_o_04,
                                               filtros_na_grade, logmel_por_segmento)
from validation import particao
from validation.run_protocol import FEATURES_ID, novo_modelo

LIMIAR_DB = 3.0
SINT_F_EIXO = 50.20
N_MELS = config.MFCC_N_MELS
N_COEFS = config.MFCC_N_COEFS


# --------------------------------------------------------------------------- #
# 1. Bandas dos harmônicos
# --------------------------------------------------------------------------- #
def potencia_harmonicos_por_banda(clipes: dict[str, pcm_io.Clip], f_eixo: float) -> dict[str, np.ndarray]:
    """Potência somada nos múltiplos de f_eixo, ponderada por cada filtro de Mel."""
    fs = config.FS_TRABALHO
    fn = f_eixo * np.arange(1, int((fs / 2) // f_eixo) + 1)
    fn = fn[fn < fs / 2 - TOL_LINHA_HZ]
    W = filtros_na_grade(fn, fs, N_MELS)                       # (bandas, harmônicos)
    saida = {}
    for rotulo, c in clipes.items():
        f, P = sg.welch(c.x, fs, nperseg=NPERSEG_FINO)
        pn = np.array([P[(f >= h - TOL_LINHA_HZ) & (f <= h + TOL_LINHA_HZ)].max() for h in fn])
        saida[rotulo] = W @ pn
    return saida


def excesso_normal_db(H: dict[str, np.ndarray], falhas_treino: list[str]) -> np.ndarray:
    """
    Normal − média das falhas de treino, em dB. A média é feita em dB (e não em
    potência) porque a LDA compara médias de log-Mel: em potência, a média
    ficaria dominada pelas falhas mais fortes.
    """
    db = lambda h: 10 * np.log10(h + 1e-30)
    return db(H["normal"]) - np.mean([db(H[f]) for f in falhas_treino], axis=0)


# --------------------------------------------------------------------------- #
# 2 e 3. Pesos da LDA por banda e contribuição
# --------------------------------------------------------------------------- #
def matriz_dct() -> np.ndarray:
    """D (N_COEFS × N_MELS): média dos MFCC = D · média do log-Mel, como no dsp.mfcc."""
    return dct(np.eye(N_MELS), type=2, axis=0, norm="ortho")[:N_COEFS]


def decompor_fold(X: np.ndarray, L: np.ndarray, y: np.ndarray, treino: list[int],
                  bandas_harm: np.ndarray) -> dict:
    mdl = novo_modelo("lda", 2)
    mdl.fit(X[treino], y[treino])
    scaler, lda = mdl[0], mdl[-1]
    sinal = 1.0 if lda.classes_[1] == "normal" else -1.0       # separação > 0 → normal
    w = sinal * lda.coef_[0] / scaler.scale_                   # pesos nas features brutas
    yt, Xt, Lt = y[treino], X[treino], L[treino]
    dX = Xt[yt == "normal"].mean(axis=0) - Xt[yt == "falha"].mean(axis=0)
    dL = Lt[yt == "normal"].mean(axis=0) - Lt[yt == "falha"].mean(axis=0)

    v = matriz_dct().T @ w[:N_COEFS]                           # pesos por banda de Mel
    contrib = v * dL
    parte_media = float(w[:N_COEFS] @ dX[:N_COEFS])
    parte_desvio = float(w[N_COEFS:] @ dX[N_COEFS:])
    if not np.isclose(contrib.sum(), parte_media, rtol=1e-6, atol=1e-9):
        raise SystemExit("a decomposição por banda não fecha com a parte da média dos MFCC: "
                         f"{contrib.sum():.6g} × {parte_media:.6g}")
    total = parte_media + parte_desvio
    return {
        "separacao_total": total,
        "fracao_media_mfcc": parte_media / total,
        "fracao_desvio_mfcc": parte_desvio / total,
        "fracao_bandas_harmonicos": float(contrib[bandas_harm].sum() / total),
        "fracao_de_bandas": float(bandas_harm.mean()),
        "n_bandas_harmonicos": int(bandas_harm.sum()),
        "contrib_por_banda": (contrib / total).tolist(),
        "peso_por_banda": v.tolist(),
    }


# --------------------------------------------------------------------------- #
# Auto-teste
# --------------------------------------------------------------------------- #
def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Todas as gravações têm o mesmo ruído colorido e a mesma série de harmônicos
    do eixo, exceto acima de 3 kHz, onde os harmônicos da normal são 20 dB mais
    fortes. Resposta esperada: as bandas acima de ~3 kHz como bandas dos
    harmônicos, e quase toda a separação atribuída a elas.
    """
    rng = np.random.default_rng(seed)
    fs = config.FS_TRABALHO
    n = int(59.9986 * fs)
    t = np.arange(n) / fs
    out = []
    for c in config.CLASSES:
        x = 0.01 * sg.lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
        for k in range(1, int((fs / 2) // SINT_F_EIXO)):
            f = k * SINT_F_EIXO
            a = 0.002 * (10.0 if (c == "normal" and f > 3000) else 1.0)
            x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
        out.append(pcm_io.Clip(rotulo=c, binario=config.BINARIO[c], fs=fs, x_float=x))
    return out


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
def figura(centros: np.ndarray, contrib_media: np.ndarray, frac_harm: np.ndarray, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    cores = ["#c0392b" if h >= 0.5 else "#7f8c8d" for h in frac_harm]
    ax.bar(np.arange(len(centros)), contrib_media, color=cores)
    ax.set_xticks(np.arange(len(centros)))
    ax.set_xticklabels([f"{c:.0f}" for c in centros], rotation=60, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("centro da banda de Mel (Hz)")
    ax.set_ylabel("fração da separação normal × falha")
    ax.set_title("Contribuição de cada banda de Mel para a LDA (Protocolo B, média dos folds)\n"
                 "vermelho: banda dos harmônicos do eixo em ≥ 50 % dos folds")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dec", type=Path, default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--features", type=Path, default=Path("data/processed/features/mfcc_features.npz"))
    ap.add_argument("--medidas", type=Path, default=MEDIDAS_JSON)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/validation/controle_harmonicos"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        lista = sintetico()
        segmentos = particao.segmentar(lista)
        folds = particao.folds_protocolo_b(segmentos)
        clipes = {c.rotulo: c for c in lista}
        X = features_como_o_04(clipes, segmentos)
        f_eixo, manif, hash_splits = SINT_F_EIXO, {"git_commit": "sintetico"}, "sintetico"
        print("Modo sintético: esperado bandas dos harmônicos acima de ~3 kHz e quase toda a "
              "separação atribuída a elas.")
    else:
        lista = pcm_io.carregar_clipes(args.pcm_dec, fs_esperado=config.FS_TRABALHO)
        divergencias = pcm_io.verificar_integridade(lista)
        pcm_io.relatar_integridade(divergencias)
        if divergencias:
            raise SystemExit("Abortado: os PCM não batem com o manifest versionado.")
        particoes = particao.carregar(args.splits)
        erros = particao.conferir_compatibilidade(particoes, lista) + particao.verificar(particoes)
        if erros:
            raise SystemExit("splits.json inválido para estes dados:\n  " + "\n  ".join(erros))
        segmentos = particao.segmentos_de(particoes)
        folds = particao.folds_de(particoes, "B")
        clipes = {c.rotulo: c for c in lista}
        X, manif = carregar_features(args.features, args.splits, len(segmentos))
        _, f_eixo = carregar_medidas(args.medidas)
        hash_splits = particao.hash_arquivo(args.splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    y = np.array([s.binario for s in segmentos])
    Lpor = logmel_por_segmento(clipes, segmentos, N_MELS)
    # logmel_por_segmento agrupa por rótulo, na ordem dos segmentos: remonta na ordem global
    cont = {k: 0 for k in Lpor}
    L = np.empty((len(segmentos), N_MELS))
    for i, s in enumerate(segmentos):
        L[i] = Lpor[s.rotulo][cont[s.rotulo]]
        cont[s.rotulo] += 1

    H = potencia_harmonicos_por_banda(clipes, f_eixo)
    centros = centros_mel(config.FS_TRABALHO, N_MELS)
    resultados = []
    for f in folds:
        falha_fora = f.info["falha_de_fora"]
        falhas_treino = [c for c in config.CLASSES if c not in ("normal", falha_fora)]
        exc = excesso_normal_db(H, falhas_treino)
        bandas = exc >= LIMIAR_DB
        r = decompor_fold(X, L, y, f.treino, bandas)
        resultados.append({"fold": f.nome, "falha_de_fora": falha_fora,
                           "excesso_normal_db": exc.tolist(), **r})

    frac = np.array([r["fracao_bandas_harmonicos"] for r in resultados])
    base = np.array([r["fracao_de_bandas"] for r in resultados])
    desv = np.array([r["fracao_desvio_mfcc"] for r in resultados])
    contrib = np.array([r["contrib_por_banda"] for r in resultados])
    marcada = np.array([np.array(r["excesso_normal_db"]) >= LIMIAR_DB for r in resultados]).mean(axis=0)
    exc_medio = np.array([r["excesso_normal_db"] for r in resultados]).mean(axis=0)

    print(f"\nf_eixo = {f_eixo:.3f} Hz; limiar = {LIMIAR_DB:g} dB; {len(folds)} folds do Protocolo B")
    print(f"\n  {'centro (Hz)':>11} {'excesso normal (dB)':>20} {'banda harm.':>12} {'contribuição':>13}")
    for b in range(N_MELS):
        print(f"  {centros[b]:>11.0f} {exc_medio[b]:>+20.1f} {marcada[b]:>11.0%} {contrib[:, b].mean():>+13.3f}")
    print("\n  por falha deixada de fora (média dos 6 folds):")
    print(f"  {'falha':<12} {'bandas harm.':>13} {'fração das bandas':>18} {'fração da separação':>20} {'desvio MFCC':>12}")
    por_falha = {}
    for falha in sorted({r["falha_de_fora"] for r in resultados}):
        rs = [r for r in resultados if r["falha_de_fora"] == falha]
        por_falha[falha] = {k: float(np.mean([r[k] for r in rs])) for k in
                            ("n_bandas_harmonicos", "fracao_de_bandas", "fracao_bandas_harmonicos",
                             "fracao_desvio_mfcc")}
        p = por_falha[falha]
        print(f"  {falha:<12} {p['n_bandas_harmonicos']:>13.1f} {p['fracao_de_bandas']:>18.2f} "
              f"{p['fracao_bandas_harmonicos']:>20.2f} {p['fracao_desvio_mfcc']:>12.2f}")
    print(f"\n  todos os folds: fração da separação nas bandas dos harmônicos mediana "
          f"{np.median(frac):.2f} (mín. {frac.min():.2f}, máx. {frac.max():.2f}), "
          f"contra {np.median(base):.2f} das bandas; desvio dos MFCC {np.median(desv):.2f}")

    resumo = {
        "f_eixo_hz": f_eixo, "limiar_db": LIMIAR_DB, "n_folds": len(folds),
        "fracao_separacao_harm_mediana": float(np.median(frac)),
        "fracao_separacao_harm_min": float(frac.min()),
        "fracao_separacao_harm_max": float(frac.max()),
        "fracao_de_bandas_mediana": float(np.median(base)),
        "fracao_desvio_mfcc_mediana": float(np.median(desv)),
        "por_falha": por_falha,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(
        {"resumo": resumo, "folds": resultados, "centros_hz": centros.tolist(),
         "features_commit": manif.get("git_commit"), "splits": hash_splits},
        ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "bandas.csv").open("w", encoding="utf-8", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["banda", "centro_hz", "excesso_normal_db_medio", "fracao_folds_banda_harm",
                       "contribuicao_media"])
        for b in range(N_MELS):
            wcsv.writerow([b, f"{centros[b]:.1f}", f"{exc_medio[b]:.2f}", f"{marcada[b]:.3f}",
                           f"{contrib[:, b].mean():.4f}"])
    figura(centros, contrib.mean(axis=0), marcada, args.out_dir / "fig_contribuicao_bandas.png")
    print(f"\nsaídas: {args.out_dir}/")

    if args.sem_registro:
        print("(registry não alterado)")
        return 0
    exp_id = f"exp{experimentos.next_exp_number(args.registry):03d}"
    experimentos.append_registry(args.registry, [{
        "id": exp_id,
        "data": date.today().isoformat(),
        "etapa": "controle_classificador",
        "script": "exploration/inspect_lda_harmonicos.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "protocolo": "B", "modelo": "lda", "features": FEATURES_ID, "splits": hash_splits,
            "features_commit": manif.get("git_commit"), "fs_hz": config.FS_TRABALHO,
            "n_mels": N_MELS, "n_coefs": N_COEFS, "f_eixo_hz": f"{f_eixo:.3f}",
            "limiar_db": LIMIAR_DB, "nperseg": NPERSEG_FINO, "tol_hz": TOL_LINHA_HZ}),
        "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
        "metricas": experimentos.kv({
            "frac_sep_harm_mediana": f"{np.median(frac):.3f}",
            "frac_sep_harm_min": f"{frac.min():.3f}", "frac_sep_harm_max": f"{frac.max():.3f}",
            "frac_bandas_mediana": f"{np.median(base):.3f}",
            "frac_desvio_mfcc_mediana": f"{np.median(desv):.3f}",
            **{f"frac_sep_harm_{k}": f"{v['fracao_bandas_harmonicos']:.3f}" for k, v in por_falha.items()}}),
        "responsavel": args.responsavel,
        "notas": args.notas or "controle dos harmônicos do eixo (pesos da LDA por banda de Mel)",
    }])
    print(f"registrado: {exp_id} em {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
