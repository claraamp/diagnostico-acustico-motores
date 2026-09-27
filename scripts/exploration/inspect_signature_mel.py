#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_signature_mel.py — o que o MFCC oficial enxerga da assinatura de falha
============================================================================

Tarefa do Notion: "Caracterizar a natureza da assinatura acústica no dataset
(tonal × impulsiva) e o caso bpfo_0.3mm". É a parte que esperava o MFCC oficial
(`dsp.log_mel` / `dsp.mfcc` e `pipeline/04_extract_features.py`).

O que já se sabe (25–26/09)
---------------------------
- A `bpfo_0.3mm` tem a série de BPFO (~182,7 Hz) no espectro do áudio: linhas
  estreitas, +29 a +43 dB sobre a normal (`identify_tonal_peaks.py`), confirmadas
  pela vibração (`confirm_bpf_envelope.py`).
- Mas não tem a elevação larga do espectro que as outras três falhas têm
  (`inspect_signature_spectra.py`): Δ de potência total de −0,3 dB.
- No Protocolo B, 0 % dos segmentos dela são detectados (exp009, exp013, exp014).

Hipótese a testar: o classificador aprende "elevação larga"; as linhas da
`bpfo_0.3mm` existem, mas somar energia dentro de um filtro de Mel as dilui.

O que é medido
--------------
1. **Diluição das linhas.** Para cada falha, PSD de alta resolução (0,2 Hz) a
   12,8 kHz, a taxa que o classificador usa. Em cada filtro de Mel:
   - Δ da banda: a energia da banda, ponderada pelo filtro triangular, da falha
     contra a normal. É o que chega ao MFCC, em média no tempo.
   - Δ da linha: a maior diferença numa linha de falha dentro da banda
     (harmônicos n·f0 e, na pista interna, bandas laterais a ± eixo), com f0 =
     medido no áudio.
   - Diluição = Δ da linha − Δ da banda.
   Diluição grande com Δ da banda pequeno = a informação existe e o Mel a apaga.
2. **O que cada filtro recebe, por segmento.** `dsp.log_mel` (a função oficial)
   sobre os segmentos do splits.json, com 20 filtros (o do config), 40 e 64.
   Por banda: Δ em dB contra a normal e d de Cohen. Cosseno entre o perfil da
   `bpfo_0.3mm` e o das outras falhas: perto de 1, "mesma direção"; perto de 0
   ou negativo, "direção diferente".
3. **O eixo do classificador.** A mesma LDA do `run_protocol.py`, com as features
   oficiais do `04_extract_features.py` (o .npz), treinada com a normal e as três
   outras falhas. Posição relativa da `bpfo_0.3mm` no eixo:
   (média_bpfo03 − média_normal) / (média_outras − média_normal) —
   0 = em cima da normal, 1 = em cima das outras falhas.
   Diagnóstico do exp013/exp014, não protocolo de validação: não entra na meta.

Uso
---
    python scripts/pipeline/04_extract_features.py          # se o .npz não existir
    python scripts/exploration/inspect_signature_mel.py --sem-registro
    python scripts/exploration/inspect_signature_mel.py --sintetico    # auto-teste, sem data/

Saída em reports/signature/: mel_metrics.json, mel_bandas.csv, diluicao_linhas.csv
e três figuras.
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

import config
import dsp
import experimentos
import pcm_io
from validation import particao
from validation.run_protocol import novo_modelo, FEATURES_ID

ALVO = "bpfo_0.3mm"
FALHAS = [c for c in config.CLASSES if config.BINARIO[c] == "falha"]
N_MELS_TESTE = (config.MFCC_N_MELS, 40, 64)
DB_POR_NEPER = 10.0 / np.log(10.0)          # log natural de potência → dB

# Frequências das linhas de falha: as MEDIDAS no áudio, lidas em main() do
# picos_metrics.json do identify_tonal_peaks.py (não ficam escritas aqui, para
# não divergirem dele). Pista interna leva bandas laterais a ± eixo (k = −2…2);
# pista externa, só harmônicos.
F_EIXO: float = float("nan")
LINHAS_F0: dict[str, float] = {}
MEDIDAS_JSON = Path("reports/signature/picos_metrics.json")

# Valores do SINAL SINTÉTICO do auto-teste (definem o sinal; não são medidas).
SINT_F_EIXO = 50.20
SINT_LINHAS_F0 = {"bpfi_0.3mm": 268.33, "bpfi_1.0mm": 268.25,
                  "bpfo_0.3mm": 182.65, "bpfo_1.0mm": 183.39}
NPERSEG_FINO = 1 << 16                      # 0,195 Hz a 12,8 kHz
TOL_LINHA_HZ = 0.5


# --------------------------------------------------------------------------- #
# Bandas de Mel sobre uma grade fina
# --------------------------------------------------------------------------- #
def bordas_mel(fs: float, n_mels: int, fmin: float = config.MFCC_FMIN) -> np.ndarray:
    """As n_mels + 2 bordas em Hz do banco de `dsp.mel_filterbank`."""
    return dsp.mel_to_hz(np.linspace(dsp.hz_to_mel(fmin), dsp.hz_to_mel(fs / 2), n_mels + 2))


def filtros_na_grade(f: np.ndarray, fs: float, n_mels: int) -> np.ndarray:
    """
    Os mesmos filtros triangulares de `dsp.mel_filterbank`, mas avaliados numa
    grade de frequência fina (a PSD de 0,2 Hz), em vez da FFT de 512 pontos.
    Mesmas bordas; só a amostragem do triângulo muda.
    """
    b = bordas_mel(fs, n_mels)
    W = np.zeros((n_mels, f.size))
    for m in range(n_mels):
        l, c, r = b[m], b[m + 1], b[m + 2]
        sub = (f >= l) & (f < c)
        W[m, sub] = (f[sub] - l) / (c - l)
        des = (f >= c) & (f < r)
        W[m, des] = (r - f[des]) / (r - c)
    return W


def centros_mel(fs: float, n_mels: int) -> np.ndarray:
    return bordas_mel(fs, n_mels)[1:-1]


# --------------------------------------------------------------------------- #
# 1. Diluição das linhas
# --------------------------------------------------------------------------- #
def frequencias_das_linhas(classe: str, fmax: float) -> np.ndarray:
    f0 = LINHAS_F0[classe]
    ks = range(-2, 3) if classe.startswith("bpfi") else (0,)
    fl = [n * f0 + k * F_EIXO for n in range(1, int(fmax / f0) + 2) for k in ks]
    return np.array(sorted(x for x in fl if 0 < x < fmax))


def diluicao(clipes: dict[str, pcm_io.Clip], n_mels: int) -> dict[str, list[dict]]:
    fs = config.FS_TRABALHO
    psds = {c: sg.welch(clipes[c].x, fs=fs, nperseg=NPERSEG_FINO, window="hann")
            for c in config.CLASSES}
    f = psds["normal"][0]
    W = filtros_na_grade(f, fs, n_mels)
    Pn = psds["normal"][1]
    En = W @ Pn
    cf = centros_mel(fs, n_mels)
    bordas = bordas_mel(fs, n_mels)
    saida = {}
    for falha in FALHAS:
        Pf = psds[falha][1]
        Ef = W @ Pf
        linhas = frequencias_das_linhas(falha, fs / 2)
        # nível de cada linha: máximo em ±TOL_LINHA_HZ
        info = []
        mascara_linha = np.zeros_like(f, dtype=bool)
        for fl in linhas:
            sel = np.abs(f - fl) <= TOL_LINHA_HZ
            if not sel.any():
                continue
            mascara_linha |= sel
            i = np.where(sel)[0][np.argmax(Pf[sel])]
            info.append((fl, 10 * np.log10(Pf[i] / (Pn[i] + 1e-30) + 1e-30)))
        linhas_arr = np.array(info) if info else np.zeros((0, 2))
        por_banda = []
        for m in range(n_mels):
            dentro = (linhas_arr[:, 0] >= bordas[m]) & (linhas_arr[:, 0] < bordas[m + 2]) \
                if linhas_arr.size else np.zeros(0, bool)
            d_banda = 10 * np.log10(Ef[m] / (En[m] + 1e-30) + 1e-30)
            d_linha = float(linhas_arr[dentro, 1].max()) if dentro.any() else float("nan")
            frac = float((W[m] * Pf * mascara_linha).sum() / (Ef[m] + 1e-30))
            por_banda.append({"banda": m, "centro_hz": float(cf[m]), "delta_banda_db": float(d_banda),
                              "delta_linha_max_db": d_linha,
                              "diluicao_db": d_linha - d_banda if np.isfinite(d_linha) else float("nan"),
                              "fracao_energia_linhas": frac, "n_linhas": int(dentro.sum())})
        saida[falha] = por_banda
    return saida


def resumo_diluicao(bandas: list[dict]) -> dict:
    com = [b for b in bandas if np.isfinite(b["delta_linha_max_db"])]
    if not com:
        return {"n_bandas_com_linha": 0}
    return {"n_bandas_com_linha": len(com),
            "delta_linha_mediana_db": float(np.median([b["delta_linha_max_db"] for b in com])),
            "delta_banda_mediana_db": float(np.median([b["delta_banda_db"] for b in com])),
            "diluicao_mediana_db": float(np.median([b["diluicao_db"] for b in com])),
            "fracao_linhas_mediana": float(np.median([b["fracao_energia_linhas"] for b in com]))}


# --------------------------------------------------------------------------- #
# 2. O que cada filtro recebe, por segmento
# --------------------------------------------------------------------------- #
def logmel_por_segmento(clipes: dict[str, pcm_io.Clip], segmentos, n_mels: int) -> dict[str, np.ndarray]:
    saida: dict[str, list] = {}
    for s in segmentos:
        c = clipes[s.rotulo]
        L = dsp.log_mel(c.x[s.inicio:s.fim], c.fs, n_mels=n_mels)
        saida.setdefault(s.rotulo, []).append(L.mean(axis=0))
    return {k: np.vstack(v) for k, v in saida.items()}


def efeito_mel(L: dict[str, np.ndarray]) -> dict[str, dict]:
    n = L["normal"]
    out = {}
    for falha in FALHAS:
        a = L[falha]
        delta = (a.mean(axis=0) - n.mean(axis=0)) * DB_POR_NEPER
        sp = np.sqrt((a.var(axis=0, ddof=1) + n.var(axis=0, ddof=1)) / 2) + 1e-9
        out[falha] = {"delta_db": delta, "d": (a.mean(axis=0) - n.mean(axis=0)) / sp}
    return out


def cosseno(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def similaridades(ef: dict[str, dict]) -> dict:
    outras = [f for f in FALHAS if f != ALVO]
    alvo = ef[ALVO]["delta_db"]
    res = {f: cosseno(alvo, ef[f]["delta_db"]) for f in outras}
    res["media_outras"] = cosseno(alvo, np.mean([ef[f]["delta_db"] for f in outras], axis=0))
    return res


# --------------------------------------------------------------------------- #
# 3. O eixo do classificador (features oficiais)
# --------------------------------------------------------------------------- #
def carregar_medidas(caminho: Path) -> tuple[dict[str, float], float]:
    """BPFI/BPFO medidas e f_eixo, do picos_metrics.json. Para se faltar algo."""
    if not caminho.exists():
        raise SystemExit(f"{caminho} não encontrado: rode o identify_tonal_peaks.py antes.")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    medidas = {}
    for c in FALHAS:
        fam = "BPFO" if "bpfo" in c else "BPFI"
        f0 = d["bpf_medidas"][c][fam]["f0_medida"]
        if f0 is None:
            raise SystemExit(f"{caminho}: {fam} de {c} não foi medida (n/id).")
        medidas[c] = float(f0)
    return medidas, float(np.mean([v["f_eixo"] for v in d["velocidades"].values()]))


def carregar_features(caminho: Path, splits: Path, n_seg: int) -> tuple[np.ndarray, dict]:
    """
    Features do 04, com as mesmas conferências do run_protocol: mesma partição,
    mesma taxa e mesmos parâmetros de MFCC do config, e sem normalização por
    segmento (as rodadas de referência, exp013/exp015, são sem --norm-clipe).
    """
    manifesto_path = caminho.with_name("manifest_features.json")
    for arq in (caminho, manifesto_path):
        if not arq.exists():
            raise SystemExit(f"{arq} não encontrado: rode o 04_extract_features.py.")
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    esperado = {
        "splits_hash": particao.hash_arquivo(splits),
        "fs_hz": config.FS_TRABALHO,
        "mfcc_janela_ms": config.MFCC_WINDOW_MS,
        "mfcc_hop_ms": config.MFCC_HOP_MS,
        "mfcc_n_mels": config.MFCC_N_MELS,
        "mfcc_n_coefs": config.MFCC_N_COEFS,
        "norm_clipe": False,
    }
    divergentes = [f"{k}: features={manifesto.get(k, False if k == 'norm_clipe' else None)}, esperado={v}"
                   for k, v in esperado.items()
                   if manifesto.get(k, False if k == "norm_clipe" else None) != v]
    if divergentes:
        raise SystemExit("features extraídas com outra configuração; rode o "
                         "04_extract_features.py sem flags:\n  " + "\n  ".join(divergentes))
    X = np.load(caminho)["X"]
    if len(X) != n_seg:
        raise SystemExit(f"o .npz tem {len(X)} linhas e a partição tem {n_seg} segmentos")
    return X, manifesto


def eixo_lda_sem_alvo(X: np.ndarray, segmentos) -> dict:
    rot = np.array([s.rotulo for s in segmentos])
    y = np.array([s.binario for s in segmentos])
    treino = rot != ALVO
    mdl = novo_modelo("lda", 2)                 # o mesmo modelo do run_protocol.py
    mdl.fit(X[treino], y[treino])
    sinal = 1.0 if mdl[-1].classes_[1] == "falha" else -1.0     # score > 0 → falha
    score = sinal * mdl.decision_function(X)
    medias = {c: float(score[rot == c].mean()) for c in config.CLASSES}
    m_out = float(np.mean([medias[f] for f in FALHAS if f != ALVO]))
    return {"score": score, "rotulos": rot, "medias": medias,
            "posicao_relativa_alvo": (medias[ALVO] - medias["normal"]) / (m_out - medias["normal"] + 1e-12),
            "fracao_alvo_como_falha": float(np.mean(score[rot == ALVO] > 0))}


def features_como_o_04(clipes: dict[str, pcm_io.Clip], segmentos) -> np.ndarray:
    """Só para o modo sintético, que não tem .npz: a mesma conta do 04 (média e desvio do MFCC)."""
    linhas = []
    for s in segmentos:
        m = dsp.mfcc(clipes[s.rotulo].x[s.inicio:s.fim], config.FS_TRABALHO)
        linhas.append(np.concatenate([m.mean(axis=0), m.std(axis=0)]))
    return np.vstack(linhas)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_diluicao(dil: dict[str, list[dict]], caminho: Path) -> None:
    fig, axes = plt.subplots(len(FALHAS), 1, figsize=(10, 2.4 * len(FALHAS)), sharex=True)
    for ax, falha in zip(axes, FALHAS):
        b = dil[falha]
        cf = np.array([x["centro_hz"] for x in b])
        ax.step(cf, [x["delta_banda_db"] for x in b], where="mid", color="k", lw=1.2,
                label="Δ da banda (o que o Mel recebe)")
        dl = np.array([x["delta_linha_max_db"] for x in b])
        ok = np.isfinite(dl)
        ax.plot(cf[ok], dl[ok], "o", color="tab:orange" if "bpfo" in falha else "tab:red", ms=4,
                label="Δ da linha de falha mais forte na banda")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_ylabel("Δ (dB)")
        ax.set_title(falha, fontsize=9, loc="left")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("centro do filtro de Mel (Hz), 20 filtros")
    fig.suptitle("Diluição: a linha de falha contra a banda de Mel que a contém", fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_mel(efeitos: dict[int, dict], caminho: Path) -> None:
    n20 = config.MFCC_N_MELS
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7))
    cf = centros_mel(config.FS_TRABALHO, n20)
    w = 0.2
    for i, falha in enumerate(FALHAS):
        a1.bar(np.arange(n20) + (i - 1.5) * w, efeitos[n20][falha]["delta_db"], w, label=falha)
    a1.set_xticks(np.arange(n20))
    a1.set_xticklabels([f"{c:.0f}" for c in cf], rotation=60, fontsize=7)
    a1.axhline(0, color="k", lw=0.6)
    a1.set_ylabel("Δ log-Mel (dB)")
    a1.set_xlabel("centro da banda de Mel (Hz)")
    a1.set_title(f"O que cada um dos {n20} filtros recebe a mais (ou a menos) que na normal",
                 fontsize=10, loc="left")
    a1.legend(fontsize=8, ncol=4)
    a1.grid(alpha=0.3, axis="y")
    for n_mels, ef in efeitos.items():
        a2.plot(centros_mel(config.FS_TRABALHO, n_mels), ef[ALVO]["delta_db"], marker="o", ms=3,
                lw=1, label=f"{n_mels} filtros")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_xscale("log")
    a2.set_xlabel("frequência (Hz)")
    a2.set_ylabel("Δ log-Mel (dB)")
    a2.set_title(f"{ALVO} − normal com resoluções de Mel diferentes", fontsize=10, loc="left")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def fig_lda(lda: dict, caminho: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(lda["score"].min(), lda["score"].max(), 60)
    for c in config.CLASSES:
        ax.hist(lda["score"][lda["rotulos"] == c], bins=bins, alpha=0.55, label=c,
                histtype="stepfilled", color="k" if c == "normal" else None)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("score da LDA (> 0 → falha)")
    ax.set_ylabel("segmentos")
    ax.set_title(f"Eixo aprendido SEM {ALVO}: posição relativa {lda['posicao_relativa_alvo']:.2f} "
                 "(0 = normal, 1 = outras falhas)", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sintético
# --------------------------------------------------------------------------- #
def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Resposta conhecida, direto a 12,8 kHz: as falhas "fáceis" levantam o ruído de
    banda larga (+10 dB) e têm as suas linhas; a `bpfo_0.3mm` tem SÓ as linhas de
    BPFO, fortes sobre o piso, sem elevação larga. O script tem que mostrar
    diluição grande nela e posição perto de 0 no eixo da LDA.
    """
    rng = np.random.default_rng(seed)
    fs = config.FS_TRABALHO
    n = int(59.9986 * fs)
    t = np.arange(n) / fs
    out = []
    for c in config.CLASSES:
        ganho = 3.0 if c in ("bpfi_0.3mm", "bpfi_1.0mm", "bpfo_1.0mm") else 1.0
        x = 0.01 * ganho * sg.lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
        for k in range(1, 20):
            x += 0.005 / k * np.sin(2 * np.pi * k * F_EIXO * t + rng.uniform(0, 6.28))
        if c in LINHAS_F0:
            for fl in frequencias_das_linhas(c, 3000.0):
                x += 0.003 * np.sin(2 * np.pi * fl * t + rng.uniform(0, 6.28))
        out.append(pcm_io.Clip(rotulo=c, binario=config.BINARIO[c], fs=fs, x_float=x))
    return out


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--pcm-dec", type=Path, default=Path(f"data/processed/pcm_decimated/{config.FS_TRABALHO}"))
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits/splits.json"))
    ap.add_argument("--features", type=Path, default=Path("data/processed/features/mfcc_features.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/signature"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--medidas", type=Path, default=MEDIDAS_JSON,
                    help="picos_metrics.json do identify_tonal_peaks.py")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    global F_EIXO, LINHAS_F0
    if args.sintetico:
        LINHAS_F0, F_EIXO = dict(SINT_LINHAS_F0), SINT_F_EIXO
    else:
        LINHAS_F0, F_EIXO = carregar_medidas(args.medidas)

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        lista = sintetico()
        segmentos = particao.segmentar(lista)
        clipes = {c.rotulo: c for c in lista}
        X = features_como_o_04(clipes, segmentos)
        manif = {"git_commit": "sintetico", "splits_hash": "sintetico", "norm_clipe": False}
        print("Modo sintético: esperado diluição grande na bpfo_0.3mm (Δ da linha ≫ Δ da "
              "banda) e posição perto de 0 no eixo da LDA.")
    else:
        lista = pcm_io.carregar_clipes(args.pcm_dec, fs_esperado=config.FS_TRABALHO)
        pcm_io.relatar_integridade(pcm_io.verificar_integridade(lista))
        particoes = particao.carregar(args.splits)
        erros = particao.conferir_compatibilidade(particoes, lista)
        if erros:
            raise SystemExit("splits.json não corresponde aos PCM:\n  " + "\n  ".join(erros))
        segmentos = particao.segmentos_de(particoes)
        clipes = {c.rotulo: c for c in lista}
        X, manif = carregar_features(args.features, args.splits, len(segmentos))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. diluição -------------------------------------------------------- #
    n20 = config.MFCC_N_MELS
    dil = diluicao(clipes, n20)
    resumo_dil = {f: resumo_diluicao(dil[f]) for f in FALHAS}
    print(f"\n1. Diluição das linhas de falha nos {n20} filtros de Mel (PSD de 0,2 Hz a "
          f"{config.FS_TRABALHO} Hz; mediana sobre as bandas que contêm linhas)")
    print(f"  {'classe':<12}{'bandas':>7}{'Δ linha':>10}{'Δ banda':>10}{'diluição':>10}{'energia nas linhas':>20}")
    for f in FALHAS:
        r = resumo_dil[f]
        if not r["n_bandas_com_linha"]:
            print(f"  {f:<12}  sem linhas na faixa")
            continue
        print(f"  {f:<12}{r['n_bandas_com_linha']:>7}{r['delta_linha_mediana_db']:>+9.1f} "
              f"{r['delta_banda_mediana_db']:>+9.1f} {r['diluicao_mediana_db']:>9.1f} "
              f"{r['fracao_linhas_mediana']:>19.1%}")
    print(f"\n  {ALVO}, banda a banda (só as que contêm linhas de BPFO):")
    print(f"    {'centro (Hz)':>12}{'linhas':>8}{'Δ linha':>10}{'Δ banda':>10}{'energia nas linhas':>20}")
    for b in dil[ALVO]:
        if b["n_linhas"]:
            print(f"    {b['centro_hz']:>12.0f}{b['n_linhas']:>8}{b['delta_linha_max_db']:>+10.1f}"
                  f"{b['delta_banda_db']:>+10.1f}{b['fracao_energia_linhas']:>20.1%}")

    # ---- 2. log-Mel por segmento -------------------------------------------- #
    print(f"\n2. dsp.log_mel sobre os {len(segmentos)} segmentos da partição")
    efeitos, sims = {}, {}
    for n_mels in N_MELS_TESTE:
        L = logmel_por_segmento(clipes, segmentos, n_mels)
        efeitos[n_mels] = efeito_mel(L)
        sims[n_mels] = similaridades(efeitos[n_mels])
    print(f"  {'filtros':<9}{'máx |Δ| ' + ALVO:>22}{'cos c/ média outras':>22}")
    for n_mels in N_MELS_TESTE:
        print(f"  {n_mels:<9}{np.max(np.abs(efeitos[n_mels][ALVO]['delta_db'])):>19.2f} dB"
              f"{sims[n_mels]['media_outras']:>22.2f}")
    print(f"\n  Perfil com {n20} filtros (Δ dB por banda; centro em Hz):")
    print("    " + "  ".join(f"{c:>6.0f}" for c in centros_mel(config.FS_TRABALHO, n20)))
    for falha in FALHAS:
        print("    " + "  ".join(f"{v:>+6.1f}" for v in efeitos[n20][falha]["delta_db"]) + f"   {falha}")

    # ---- 3. eixo da LDA ------------------------------------------------------ #
    lda = eixo_lda_sem_alvo(X, segmentos)
    print(f"\n3. Eixo da LDA sem {ALVO} (features do 04, commit {manif.get('git_commit')}, "
          f"norm_clipe={manif.get('norm_clipe')})")
    for c in config.CLASSES:
        print(f"    {c:<12} score médio {lda['medias'][c]:+9.2f}")
    print(f"  posição relativa de {ALVO}: {lda['posicao_relativa_alvo']:+.2f} (0 = normal, 1 = outras falhas)")
    print(f"  segmentos de {ALVO} do lado da falha: {lda['fracao_alvo_como_falha']:.1%}")

    # ---- saída -------------------------------------------------------------- #
    fig_diluicao(dil, args.out_dir / "fig_diluicao_linhas_mel.png")
    fig_mel(efeitos, args.out_dir / "fig_mel_diferenca_por_banda.png")
    fig_lda(lda, args.out_dir / "fig_lda_eixo_sem_bpfo03.png")
    with (args.out_dir / "diluicao_linhas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["classe", *dil[FALHAS[0]][0].keys()])
        w.writeheader()
        for f in FALHAS:
            for b in dil[f]:
                w.writerow({"classe": f, **b})
    with (args.out_dir / "mel_bandas.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_mels", "banda", "centro_hz", "classe", "delta_db", "d_cohen"])
        for n_mels, ef in efeitos.items():
            for i, cc in enumerate(centros_mel(config.FS_TRABALHO, n_mels)):
                for falha in FALHAS:
                    w.writerow([n_mels, i, f"{cc:.1f}", falha,
                                f"{ef[falha]['delta_db'][i]:.3f}", f"{ef[falha]['d'][i]:.3f}"])
    (args.out_dir / "mel_metrics.json").write_text(json.dumps({
        "data": date.today().isoformat(), "features": manif, "linhas_f0_hz": LINHAS_F0,
        "diluicao": resumo_dil,
        "mel": {str(n): {"cosseno_alvo": sims[n],
                         "max_abs_delta_db_alvo": float(np.max(np.abs(efeitos[n][ALVO]["delta_db"])))}
                for n in N_MELS_TESTE},
        "lda_sem_alvo": {k: lda[k] for k in ("medias", "posicao_relativa_alvo", "fracao_alvo_como_falha")},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaída em {args.out_dir}/")

    if args.sem_registro:
        print("(sem registro)")
        return
    ra = resumo_dil[ALVO]
    linha = {
        "id": f"exp{experimentos.next_exp_number(args.registry):03d}",
        "data": date.today().isoformat(),
        "etapa": "caracterizacao_assinatura",
        "script": "exploration/inspect_signature_mel.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({
            "fs_hz": config.FS_TRABALHO, "n_mels": "/".join(map(str, N_MELS_TESTE)),
            "nperseg_fino": NPERSEG_FINO, "splits": manif.get("splits_hash"),
            "features": FEATURES_ID, "features_commit": manif.get("git_commit"),
            "norm_clipe": manif.get("norm_clipe")}),
        "dataset": "jung2023_acustico_0Nm_12800Hz",
        "metricas": experimentos.kv({
            f"diluicao_mediana_db_{ALVO}": f"{ra.get('diluicao_mediana_db', float('nan')):.1f}",
            f"delta_banda_mediana_db_{ALVO}": f"{ra.get('delta_banda_mediana_db', float('nan')):.1f}",
            **{f"cos{n}_{ALVO}": f"{sims[n]['media_outras']:.3f}" for n in N_MELS_TESTE},
            f"lda_posicao_{ALVO}": f"{lda['posicao_relativa_alvo']:.3f}",
            f"lda_fracao_falha_{ALVO}": f"{lda['fracao_alvo_como_falha']:.3f}"}),
        "responsavel": args.responsavel,
        "notas": args.notas or "diagnóstico do exp013/exp014; não é protocolo de validação",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")


if __name__ == "__main__":
    main()