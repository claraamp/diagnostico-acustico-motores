#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
confirm_bpf_envelope.py — as frequências de falha medidas no áudio aparecem na vibração?
=====================================================================================

Tarefa do Notion: "Caracterizar a natureza da assinatura acústica no dataset
(tonal × impulsiva) e o caso bpfo_0.3mm". Confirmação independente do achado de
25/09.

O achado a confirmar
--------------------
No ÁUDIO, os picos das gravações de falha formam séries com fundamental de
~268,3 Hz (pista interna) e ~182,7–183,4 Hz (pista externa)
(`identify_tonal_peaks.py`, versão 2). As frequências CINEMÁTICAS do artigo,
calculadas pela geometria do NSK 6205 com ângulo de contato θ = 0° e sem
escorregamento, são 272,07 e 179,43 Hz a 50,17 Hz (Jung et al., 2023, Tabela 1).

Antes de pôr isso no relatório, é preciso uma segunda medição, com outro sensor.
O dataset tem VIBRAÇÃO a 0 Nm, com os mesmos defeitos: quatro acelerômetros,
direções x e y dos mancais A e B (Jung et al., 2023, seção 2; o microfone fica
perto do mancal A). O artigo não diz em qual mancal está o rolamento com defeito
nos arquivos de 0 Nm. Se o espectro de envelope da vibração mostrar a BPFO em
~182,7 Hz e a BPFI em ~268,3 Hz, e não em 179,5 e 272,3 Hz, o achado passa de
inferência a constatação com dois sensores.

Método (Randall & Antoni, 2011, seções 4.2 e 5)
-----------------------------------------------
Para cada sensor e cada gravação:

1. Banda de demodulação escolhida pela CURTOSE ESPECTRAL (SK), sem olhar para
   BPFI/BPFO: STFT com janelas de 64 a 512 amostras, SK(f) = <|X|⁴>/<|X|²>² − 2,
   e escolhe-se a janela e a frequência de SK máxima. É o princípio do kurtograma:
   a banda mais impulsiva. Largura mínima de 2 kHz, para o envelope enxergar até
   ~1 kHz de modulação.
2. Espectro de envelope (`dsp.envelope_spectrum`): passa-faixa, Hilbert, FFT
   média sobre segmentos de 4 s (resolução 0,25 Hz).
3. Nas faixas de busca (BPFO: 170–195 Hz; BPFI: 255–285 Hz), o f0 cujo pente de
   harmônicos (n = 1…4) tem mais energia acima do piso: f0 do envelope.
4. SNR em duas frequências, pelo MESMO critério: a cinemática do artigo e a
   medida no áudio. Pico em ±0,5 Hz; piso = mediana entre 1,5 e 10 Hz de
   distância. Critério diferente do `dsp.peak_snr` da decimação, que procurava
   em ±1,5 Hz do alvo nominal e media o piso até 25 Hz — largo o bastante para
   engolir a linha verdadeira, a 3–4 Hz do alvo.

A gravação normal entra como controle: não deve ter pico em nenhuma das duas.

Critério de confirmação (por sensor e gravação de falha)
--------------------------------------------------------
- f0 do envelope só é aceito com SNR no próprio f0 ≥ SNR_MIN_DB; abaixo disso
  sai n/id, como no identify_tonal_peaks.py.
- "sim": f0 aceito e a até TOL_CONFIRMA_HZ (2 × a resolução) da medida no áudio.
- "parcial": há energia na medida do áudio (SNR ≥ SNR_MIN_DB), mas o f0 do
  envelope não cai nela, ou não foi aceito.
- "não": nem uma coisa nem outra.

As frequências medidas no áudio NÃO estão escritas aqui: são lidas do
`picos_metrics.json` do identify_tonal_peaks.py (a rodada registrada dele). Se
o arquivo faltar, o script para. Só o modo --sintetico usa valores próprios,
que definem o sinal de teste.

Uso
---
    python scripts/exploration/confirm_bpf_envelope.py --sem-registro
    python scripts/exploration/confirm_bpf_envelope.py --canais 0 1 --sem-registro   # colunas da vibração
    python scripts/exploration/confirm_bpf_envelope.py --sintetico                  # auto-teste, sem data/

Precisa dos .mat de VIBRAÇÃO a 0 Nm em data/raw/vibracao/ (mesmos nomes dos de
áudio: 0Nm_Normal.mat, 0Nm_BPFI_03.mat, ...), e dos PCM de áudio de sempre.
Saída em reports/signature/: envelope_bpf.csv, envelope_metrics.json e a figura.
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
import scipy.io as sio
from scipy import signal as sg

import config
import dsp
import experimentos
import pcm_io

FALHAS = [c for c in config.CLASSES if config.BINARIO[c] == "falha"]

ARQUIVOS = {                         # mesmos nomes do áudio, outra pasta
    "normal": "0Nm_Normal.mat",
    "bpfi_0.3mm": "0Nm_BPFI_03.mat",
    "bpfi_1.0mm": "0Nm_BPFI_10.mat",
    "bpfo_0.3mm": "0Nm_BPFO_03.mat",
    "bpfo_1.0mm": "0Nm_BPFO_10.mat",
}

# Geometria do NSK 6205 (Jung et al., 2023, Tabela 1 e texto): N, d, D, θ.
N_ESFERAS, D_ESFERA_MM, D_PRIMITIVO_MM, THETA_DEG = 9, 7.90, 38.5, 0.0

# Valores do SINAL SINTÉTICO do auto-teste. Definem o sinal de teste; não são
# medidas e não entram na análise dos dados reais.
SINT_F_EIXO = 50.20
SINT_MEDIDAS = {"bpfi_0.3mm": 268.33, "bpfi_1.0mm": 268.25,
                "bpfo_0.3mm": 182.65, "bpfo_1.0mm": 183.39}

# Medidas usadas na análise: preenchidas em main(), a partir do
# picos_metrics.json (dados reais) ou de SINT_* (--sintetico).
F_EIXO: float = float("nan")
MEDIDAS_AUDIO: dict[str, float] = {}
MEDIDAS_JSON = Path("reports/signature/picos_metrics.json")

# Nomes dos canais de vibração. Jung et al. (2023, seção 2) listam as colunas na
# ordem x e y do mancal A, x e y do mancal B; o .mat traz Point1–Point4, que se
# supõe estarem na mesma ordem.
NOMES_VIB = {0: "vib_A_x", 1: "vib_A_y", 2: "vib_B_x", 3: "vib_B_y"}

SNR_MIN_DB = 6.0                    # f0 do envelope abaixo disto sai n/id
TOL_CONFIRMA_HZ = 0.5               # 2 × a resolução de 0,25 Hz

FAIXAS = {"BPFO": (170.0, 195.0), "BPFI": (255.0, 285.0)}
N_HARM = 4
JANELAS_SK = (64, 128, 256, 512)
F_MIN_BANDA = 500.0                 # Hz — abaixo disto a SK pega os tons do eixo
LARGURA_MIN = 2000.0                # Hz
SEG_ENVELOPE_S = 4.0                # resolução de 0,25 Hz


def cinematicas(f_eixo: float | None = None) -> dict[str, float]:
    f_eixo = F_EIXO if f_eixo is None else f_eixo
    r = D_ESFERA_MM / D_PRIMITIVO_MM * np.cos(np.radians(THETA_DEG))
    return {"BPFO": N_ESFERAS * f_eixo / 2 * (1 - r),
            "BPFI": N_ESFERAS * f_eixo / 2 * (1 + r)}


def theta_efetivo(bpfo: float, f_eixo: float | None = None) -> float:
    """Ângulo de contato que reproduziria a BPFO medida (mantidos N, d, D)."""
    f_eixo = F_EIXO if f_eixo is None else f_eixo
    r_eff = 1 - 2 * bpfo / (N_ESFERAS * f_eixo)
    c = r_eff / (D_ESFERA_MM / D_PRIMITIVO_MM)
    return float(np.degrees(np.arccos(c))) if -1 <= c <= 1 else float("nan")


# --------------------------------------------------------------------------- #
# Leitura da vibração
# --------------------------------------------------------------------------- #
def _textos(v, saida: list[str], nivel: int = 0) -> None:
    """Coleta todos os campos de texto de uma struct (para achar nomes de canal)."""
    if nivel > 8:
        return
    if isinstance(v, np.ndarray) and v.dtype.names:
        for el in v.ravel()[:8]:
            for campo in v.dtype.names:
                _textos(el[campo], saida, nivel + 1)
    elif isinstance(v, np.ndarray) and v.dtype == object:
        for item in v.ravel()[:16]:
            _textos(item, saida, nivel + 1)
    elif isinstance(v, np.ndarray) and v.dtype.kind in ("U", "S"):
        saida.append("".join(str(c) for c in v.ravel()))


def _desembrulhar(a):
    """Tira as camadas (1, 1) e de objeto que o loadmat põe em volta de cada campo."""
    while isinstance(a, np.ndarray) and a.size == 1 and (a.dtype == object or a.dtype.names):
        a = a.ravel()[0]
    return np.asarray(a) if not (isinstance(a, np.void) or hasattr(a, "dtype") and a.dtype.names) else a


def carregar_vibracao(caminho: Path) -> tuple[float, np.ndarray, list[str]]:
    """
    Devolve (fs, valores[n, canais], textos encontrados). Tenta, na ordem:
    a estrutura `Signal` dos .mat de áudio (x_values.increment dá a taxa) e,
    se não houver, a maior matriz numérica do arquivo. Uma primeira coluna
    monotônica crescente é tomada como carimbo de tempo: dá a taxa e sai.
    Se a estrutura for outra, rode o inspect_vibration_mat.py e confira a
    estrutura do arquivo.
    """
    try:
        data = sio.loadmat(caminho, struct_as_record=True, squeeze_me=False)
    except NotImplementedError as e:
        raise SystemExit(f"{caminho}: MAT v7.3 — rode inspect_vibration_mat.py e confira a estrutura") from e

    fs, vals, textos = None, None, []
    if "Signal" in data:
        # um elemento com todos os canais, ou um elemento por canal: aceita os dois
        colunas = []
        elementos = []
        for el in data["Signal"].ravel():      # struct array ou cell de structs
            while isinstance(el, np.ndarray) and el.dtype == object and el.size == 1:
                el = el.ravel()[0]
            elementos.extend(el.ravel() if isinstance(el, np.ndarray) and el.dtype.names
                             and el.size > 1 else [el.ravel()[0] if isinstance(el, np.ndarray) else el])
        for sig in elementos:
            if fs is None and "x_values" in sig.dtype.names:
                x = _desembrulhar(sig["x_values"])
                fs = 1.0 / float(_desembrulhar(x["increment"]).ravel()[0])
            v = np.asarray(_desembrulhar(_desembrulhar(sig["y_values"])["values"]), dtype=np.float64)
            v = v[:, None] if v.ndim == 1 else (v.T if v.shape[0] < v.shape[1] else v)
            colunas.append(v)
            if "function_record" in sig.dtype.names:
                _textos(sig["function_record"], textos)
        n_min = min(c.shape[0] for c in colunas)
        vals = np.hstack([c[:n_min] for c in colunas])
    else:
        candidatos = [np.asarray(v) for k, v in data.items() if not k.startswith("__")
                      and isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.ndim == 2]
        if not candidatos:
            raise SystemExit(f"{caminho}: não achei matriz numérica — rode inspect_vibration_mat.py")
        vals = max(candidatos, key=lambda a: a.size).astype(np.float64)

    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] < vals.shape[1]:
        vals = vals.T
    col0 = vals[: min(1000, len(vals)), 0]
    if vals.shape[1] >= 2 and np.all(np.diff(col0) > 0):       # carimbo de tempo
        dt = float(np.median(np.diff(vals[:, 0])))
        fs = fs or 1.0 / dt
        vals = vals[:, 1:]
    if fs is None:
        raise SystemExit(f"{caminho}: não consegui achar a taxa — rode inspect_vibration_mat.py")
    return fs, vals, textos


def carregar_medidas(caminho: Path) -> tuple[dict[str, float], float]:
    """
    BPFI/BPFO medidas no áudio e f_eixo, do picos_metrics.json do
    identify_tonal_peaks.py. Para se faltar o arquivo ou alguma medida.
    """
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
    f_eixo = float(np.mean([v["f_eixo"] for v in d["velocidades"].values()]))
    return medidas, f_eixo


def classificar(f0: float | None, snr_medida: float, medida: float) -> str:
    """Critério de confirmação da docstring: sim / parcial / não."""
    if f0 is not None and abs(f0 - medida) <= TOL_CONFIRMA_HZ:
        return "sim"
    if np.isfinite(snr_medida) and snr_medida >= SNR_MIN_DB:
        return "parcial"
    return "não"


# --------------------------------------------------------------------------- #
# Análise
# --------------------------------------------------------------------------- #
def escolher_banda(x: np.ndarray, fs: float) -> dict:
    """Banda de SK máxima (kurtograma simplificado), sem olhar para BPFI/BPFO."""
    melhor = {"sk": -np.inf}
    x = x[: int(min(len(x), 30 * fs))]              # 30 s bastam para a estatística
    for nw in JANELAS_SK:
        f, _, Z = sg.stft(x, fs=fs, window="hann", nperseg=nw, noverlap=3 * nw // 4)
        p2 = np.abs(Z) ** 2
        sk = (p2 ** 2).mean(axis=1) / (p2.mean(axis=1) ** 2 + 1e-30) - 2
        sel = (f >= F_MIN_BANDA) & (f <= 0.45 * fs)
        i = np.argmax(np.where(sel, sk, -np.inf))
        if sk[i] > melhor["sk"]:
            melhor = {"sk": float(sk[i]), "fc": float(f[i]), "janela": nw}
    larg = max(2 * fs / melhor["janela"], LARGURA_MIN)
    lo = max(F_MIN_BANDA, melhor["fc"] - larg / 2)
    hi = min(0.45 * fs, lo + larg)
    melhor["banda"] = (lo, hi)
    return melhor


def piso_local(f: np.ndarray, m: np.ndarray, alvo: float) -> float:
    d = np.abs(f - alvo)
    sel = (d >= 1.5) & (d <= 10.0)
    return float(np.median(m[sel])) if sel.any() else float(np.median(m))


def snr_em(f: np.ndarray, m: np.ndarray, alvo: float) -> float:
    win = np.abs(f - alvo) <= 0.5
    if not win.any():
        return float("nan")
    return float(20 * np.log10(m[win].max() / (piso_local(f, m, alvo) + 1e-30) + 1e-30))


def f0_do_envelope(f: np.ndarray, m: np.ndarray, faixa: tuple[float, float]) -> dict:
    """f0 da faixa cujo pente n·f0 (n = 1…N_HARM) tem mais energia acima do piso."""
    grade = np.arange(faixa[0], faixa[1], 0.01)
    df = f[1] - f[0]
    rel = m / (sg.medfilt(m, int(10 / df) | 1) + 1e-30)          # em unidades do piso local
    score = np.zeros_like(grade)
    for n in range(1, N_HARM + 1):
        score += np.interp(n * grade, f, rel)
    i = int(np.argmax(score))
    f0 = float(grade[i])
    # refino: interpolação parabólica do pico da fundamental no espectro
    j = int(round(f0 / df))
    j = j - 2 + int(np.argmax(m[j - 2:j + 3]))
    a, b, c = m[j - 1], m[j], m[j + 1]
    den = a - 2 * b + c
    f0_ref = (j + (0.5 * (a - c) / den if den < 0 else 0.0)) * df
    return {"f0": float(f0_ref), "snr_f0_db": snr_em(f, m, f0_ref)}


def analisar(x: np.ndarray, fs: float) -> dict:
    banda = escolher_banda(x, fs)
    f, m = dsp.envelope_spectrum(x, fs, banda["banda"], seg_s=SEG_ENVELOPE_S)
    return {"banda": banda, "f": f, "m": m,
            "f0": {nome: f0_do_envelope(f, m, fx) for nome, fx in FAIXAS.items()}}


# --------------------------------------------------------------------------- #
# Sintético
# --------------------------------------------------------------------------- #
def sintetico(fs: float, seed: int) -> dict[str, np.ndarray]:
    """
    Impactos na BPF "real" (as medidas no áudio), com 1 % de jitter, excitando
    uma ressonância de 3 kHz; normal só com ruído e tons do eixo. O script tem
    que achar as frequências medidas, com SNR alto nelas e baixo nas cinemáticas.
    """
    rng = np.random.default_rng(seed)
    n = int(60 * fs)
    t = np.arange(n) / fs
    tt = np.arange(int(0.004 * fs)) / fs
    ir = np.exp(-800 * tt) * np.sin(2 * np.pi * 3000 * tt)
    out = {}
    for c in config.CLASSES:
        x = 0.05 * rng.standard_normal(n)
        for k in range(1, 6):
            x += 0.2 / k * np.sin(2 * np.pi * k * SINT_F_EIXO * t)
        if c in SINT_MEDIDAS:
            f0 = SINT_MEDIDAS[c]
            g = 1.0 if c.endswith("1.0mm") else 0.4
            tk = np.cumsum(1 / f0 * (1 + 0.01 * rng.standard_normal(int(60 * f0) + 10)))
            idx = (tk[tk < 60 - 0.01] * fs).astype(int)
            imp = np.zeros(n)
            amp = g * (1 + 0.5 * np.cos(2 * np.pi * SINT_F_EIXO * tk[: idx.size])) if "bpfi" in c else g
            imp[idx] = amp
            x += sg.lfilter(ir, [1.0], imp) * 3
        out[c] = x
    return out


# --------------------------------------------------------------------------- #
# Figura
# --------------------------------------------------------------------------- #
def figura(res: dict, sensores: list[str], caminho: Path, fmax: float = 600.0) -> None:
    cin = cinematicas()
    fig, axes = plt.subplots(len(config.CLASSES), len(sensores),
                             figsize=(5.2 * len(sensores), 2.1 * len(config.CLASSES)),
                             sharex=True, squeeze=False)
    for j, sensor in enumerate(sensores):
        for i, c in enumerate(config.CLASSES):
            ax = axes[i, j]
            r = res[sensor][c]
            sel = r["f"] <= fmax
            ax.plot(r["f"][sel], r["m"][sel], color="k", lw=0.6)
            fam = "BPFO" if "bpfo" in c else "BPFI" if "bpfi" in c else None
            for nome, cor in (("BPFO", "tab:orange"), ("BPFI", "tab:red")):
                if fam not in (None, nome):
                    continue
                for n in range(1, int(fmax / cin[nome]) + 1):
                    ax.axvline(n * cin[nome], color="0.5", ls="--", lw=0.7)
                if c in MEDIDAS_AUDIO and fam == nome:
                    for n in range(1, int(fmax / MEDIDAS_AUDIO[c]) + 1):
                        ax.axvline(n * MEDIDAS_AUDIO[c], color=cor, lw=0.9, alpha=0.8)
            ax.set_title(f"{sensor} · {c}", fontsize=8, loc="left")
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("frequência (Hz)")
    fig.suptitle("Espectro de envelope — tracejado cinza: cinemática do artigo (θ = 0°); "
                 "cor: medida no áudio", fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--vib-dir", type=Path, default=Path("data/raw/vibracao"))
    ap.add_argument("--canais", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="colunas da vibração a analisar (0–1: mancal A; 2–3: mancal B)")
    ap.add_argument("--medidas", type=Path, default=MEDIDAS_JSON,
                    help="picos_metrics.json do identify_tonal_peaks.py")
    ap.add_argument("--pcm-raw", type=Path, default=Path("data/processed/pcm_raw"))
    ap.add_argument("--sem-audio", action="store_true", help="só vibração")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/signature"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    global F_EIXO, MEDIDAS_AUDIO
    if args.sintetico:
        MEDIDAS_AUDIO, F_EIXO = dict(SINT_MEDIDAS), SINT_F_EIXO
    else:
        MEDIDAS_AUDIO, F_EIXO = carregar_medidas(args.medidas)

    sinais: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        for nome, fs, seed in (("vib_sint", 25600.0, 1), ("audio_sint", 51200.0, 2)):
            sinais[nome] = {c: (fs, x) for c, x in sintetico(fs, seed).items()}
        print("Modo sintético: esperado f0 do envelope ≈ medidas no áudio "
              f"({MEDIDAS_AUDIO}), SNR alto nelas e baixo nas cinemáticas.")
    else:
        for c, arq in ARQUIVOS.items():
            fs, vals, textos = carregar_vibracao(args.vib_dir / arq)
            if c == "normal":
                print(f"Vibração: {vals.shape[1]} canais a {fs:.0f} Hz, {vals.shape[0] / fs:.1f} s "
                      f"(normal). Textos no arquivo: {textos[:12]}")
                print(f"  Usando as colunas {args.canais} "
                      f"({', '.join(NOMES_VIB.get(k, f'vib_col{k}') for k in args.canais)}).")
            for k in args.canais:
                x = vals[: int(60 * fs), k].copy()                      # cópia: libera a matriz inteira (normal tem 300 s)
                sinais.setdefault(NOMES_VIB.get(k, f"vib_col{k}"), {})[c] = (fs, x - x.mean())
        if not args.sem_audio:
            for cl in pcm_io.carregar_clipes(args.pcm_raw, fs_esperado=config.FS_ORIGINAL):
                sinais.setdefault("audio", {})[cl.rotulo] = (cl.fs, cl.x)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cin = cinematicas()
    print(f"\nCinemáticas (θ = 0°, f_eixo {F_EIXO:.2f} Hz): BPFO {cin['BPFO']:.2f} · BPFI {cin['BPFI']:.2f} Hz")
    print(f"Medidas no áudio: {MEDIDAS_AUDIO}"
          + ("" if args.sintetico else f"  (de {args.medidas})"))
    print(f"f0 do envelope aceito com SNR ≥ {SNR_MIN_DB:.0f} dB no próprio f0; "
          f"confirma = f0 a até {TOL_CONFIRMA_HZ} Hz da medida\n")

    res: dict[str, dict] = {}
    linhas = []
    for sensor, por_classe in sinais.items():
        res[sensor] = {}
        print(f"── {sensor}")
        print(f"  {'classe':<12}{'banda (Hz)':>16}{'SK':>7}{'família':>9}{'f0 env.':>10}"
              f"{'SNR@f0':>8}{'SNR@cinem.':>12}{'SNR@áudio':>11}{'confirma':>10}")
        for c in config.CLASSES:
            fs, x = por_classe[c]
            r = analisar(x, fs)
            res[sensor][c] = r
            b = r["banda"]
            familias = (["BPFO", "BPFI"] if c == "normal"
                        else ["BPFO"] if "bpfo" in c else ["BPFI"])
            for fam in familias:
                # na normal (controle), a média das medidas daquela família
                alvo_audio = MEDIDAS_AUDIO.get(c, float(np.mean(
                    [v for k, v in MEDIDAS_AUDIO.items() if fam.lower() in k])))
                s_cin = snr_em(r["f"], r["m"], cin[fam])
                s_aud = snr_em(r["f"], r["m"], alvo_audio) if np.isfinite(alvo_audio) else float("nan")
                s_f0 = r["f0"][fam]["snr_f0_db"]
                f0 = r["f0"][fam]["f0"] if s_f0 >= SNR_MIN_DB else None
                conf = classificar(f0, s_aud, alvo_audio) if c != "normal" else "—"
                f0_txt = f"{f0:.2f}" if f0 is not None else "n/id"
                print(f"  {c:<12}{b['banda'][0]:>8.0f}–{b['banda'][1]:<7.0f}{b['sk']:>7.2f}{fam:>9}"
                      f"{f0_txt:>10}{s_f0:>8.1f}{s_cin:>12.1f}{s_aud:>11.1f}{conf:>10}")
                linhas.append({"sensor": sensor, "classe": c, "familia": fam,
                               "banda_lo_hz": round(b["banda"][0], 1), "banda_hi_hz": round(b["banda"][1], 1),
                               "sk_max": round(b["sk"], 3), "janela_sk": b["janela"],
                               "f0_envelope_hz": round(f0, 3) if f0 is not None else "n/id",
                               "snr_f0_db": round(s_f0, 2),
                               "f_cinematica_hz": round(cin[fam], 3), "snr_cinematica_db": round(s_cin, 2),
                               "f_audio_hz": alvo_audio, "snr_audio_db": round(s_aud, 2),
                               "confirma": conf,
                               "theta_efetivo_deg": (round(theta_efetivo(f0), 1)
                                                     if fam == "BPFO" and c != "normal" and f0 is not None
                                                     else "")})
        print()

    print("Leitura: confirma se, na vibração, o f0 do envelope de cada falha cai a poucos")
    print("décimos de Hz da medida no áudio e o SNR na medida supera com folga o SNR na")
    print("cinemática. Na normal, nenhum dos dois deve ter pico.")
    bpfos = [l for l in linhas if l["familia"] == "BPFO" and l["classe"] != "normal"
             and l["sensor"].startswith("vib") and l["theta_efetivo_deg"] != ""]
    if bpfos:
        print("θ efetivo que reproduziria a BPFO da vibração: " +
              ", ".join(f"{l['sensor']}/{l['classe']} {l['theta_efetivo_deg']}°" for l in bpfos))
        por_classe = {}
        for l in bpfos:
            por_classe.setdefault(l["classe"], []).append(float(l["theta_efetivo_deg"]))
        if len(por_classe) > 1:
            medias = {c: np.mean(v) for c, v in por_classe.items()}
            print("  θ médio por gravação: " + ", ".join(f"{c} {v:.1f}°" for c, v in medias.items())
                  + ". Um único ângulo de contato só explicaria as duas se esses valores coincidissem.")

    figura(res, list(sinais), args.out_dir / "fig_envelope_bpf.png")
    with (args.out_dir / "envelope_bpf.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    (args.out_dir / "envelope_metrics.json").write_text(json.dumps(
        {"data": date.today().isoformat(), "cinematicas": cin, "medidas_audio": MEDIDAS_AUDIO,
         "criterio": {"snr_min_db": SNR_MIN_DB, "tol_confirma_hz": TOL_CONFIRMA_HZ},
         "geometria": {"N": N_ESFERAS, "d_mm": D_ESFERA_MM, "D_mm": D_PRIMITIVO_MM,
                       "theta_deg": THETA_DEG, "f_eixo_hz": F_EIXO},
         "linhas": linhas}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaída em {args.out_dir}/")

    if args.sem_registro:
        print("(sem registro)")
        return
    met = {}
    for l in linhas:
        if l["classe"] == "normal":
            continue
        chave = f"{l['sensor']}_{l['classe']}"
        met[f"f0_{chave}"] = l["f0_envelope_hz"]
        met[f"snr_{chave}"] = l["snr_audio_db"]
        met[f"conf_{chave}"] = l["confirma"]
    linha = {
        "id": f"exp{experimentos.next_exp_number(args.registry):03d}",
        "data": date.today().isoformat(),
        "etapa": "caracterizacao_assinatura",
        "script": "exploration/confirm_bpf_envelope.py",
        "git_commit": experimentos.git_short_hash(),
        "parametros": experimentos.kv({"canais_vib": "/".join(map(str, args.canais)),
                                       "janelas_sk": "/".join(map(str, JANELAS_SK)),
                                       "largura_min_hz": LARGURA_MIN, "seg_envelope_s": SEG_ENVELOPE_S,
                                       "f_eixo_hz": round(F_EIXO, 3), "snr_min_db": SNR_MIN_DB,
                                       "tol_confirma_hz": TOL_CONFIRMA_HZ}),
        "dataset": "jung2023_vibracao_0Nm+acustico_0Nm",
        "metricas": experimentos.kv(met),
        "responsavel": args.responsavel,
        "notas": args.notas or "confirmação de BPFI/BPFO por envelope em vibração e áudio",
    }
    experimentos.append_registry(args.registry, [linha])
    print(f"registrado como {linha['id']}")


if __name__ == "__main__":
    main()