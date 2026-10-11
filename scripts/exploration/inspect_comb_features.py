#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_comb_features.py — features de linhas estreitas (pente harmônico de BPFO/BPFI)
======================================================================================

Card "Avaliar features de linhas estreitas (pente harmônico de BPFI/BPFO) junto ao
MFCC" (Frente 2, decisão em 13/10). O MFCC dilui as linhas finas da BPFO (+30 dB na
linha, +1 dB na banda de Mel); o pente mede a série harmônica diretamente. A
pergunta do card é se somar o pente ao MFCC, com a mesma LDA, melhora o detector, e
qual resolução de espectro o firmware precisaria para isso.

Candidatos de resolução (os do card, fixados antes de rodar)
-----------------------------------------------------------
A  espectro médio dos 98 quadros que a cadeia do MFCC já calcula (320 amostras,
   Hann simétrica, FFT de 512, potência); resolução efetiva fs/320 = 40 Hz
B  FFT dedicada de 4.096 pontos: 3 blocos de 0,32 s, Hann, potência média;
   resolução fs/4.096 = 3,125 Hz
C  FFT do segundo inteiro (12.800 pontos); 1 Hz; só referência do teto, em Python

Parâmetros do pente
-------------------
Vêm da física do rolamento, da rotação e da resolução de cada candidato, não do
espectro da bpfo_0.3mm:

- razões cinemáticas da geometria (NSK 6205: N = 9, d = 7,90 mm, D = 38,5 mm,
  θ = 0°): BPFO = 3,5766 × f_eixo e BPFI = 5,4234 × f_eixo;
- f₀ buscado em ±3 % em torno da razão cinemática (as medidas diferem 1,4–1,8 %,
  ~4 Hz, o erro que o card pede para cobrir);
- dentes até 5,6 kHz (abaixo do corte de 5.760 Hz do anti-aliasing);
- dente descartado a menos de 2 resoluções efetivas de uma harmônica do eixo,
  que é mais forte na gravação normal (atualização de 05/10 do card);
- saliência do dente = pico (±1 bin) / mediana do fundo entre 3 e 10 resoluções;
- dente "presente" com saliência ≥ 6 dB.

O candidato A é inviável por construção: com 40 Hz de resolução, a máscara (80 Hz)
é maior que metade do espaçamento das harmônicas do eixo (~25 Hz), e nenhum dente
sobra. É o risco que a atualização de 05/10 aponta (as linhas de BPFO caem na
vizinhança das harmônicas do eixo); o script registra a inviabilidade e não calcula
features para ele.

O f₀ escolhido só posiciona o pente e NÃO vira feature (seria a identidade da
gravação). Features por segmento (4): pontuação da BPFO, pontuação da BPFI, fração
de dentes da BPFO ≥ 6 dB e fração de dentes da BPFI ≥ 6 dB.

Rotação do eixo
---------------
O card pede a rotação estimada no próprio segmento: modo "estimado", pelo pente
harmônico do próprio eixo, em duas etapas (`estimar_eixo`). A decisão usa só esse
modo. O modo "fixo" (50,17 Hz, a rotação publicada por Jung et al., 2023, valor
externo) roda como sensibilidade: mostra quanto do resultado depende da precisão
da estimativa.

Desvio do plano: uma primeira versão estimava o eixo pelo pico entre 45 e 55 Hz.
Na primeira rodada real, num espectro de 0,64 s, o pico caía em ~45 Hz e o f₀ da
BPFO esperado ficava fora da própria faixa de busca. A conferência foi contra a
rotação conhecida (50,2 Hz em todas as gravações), não contra o resultado da falha,
mas os números daquela rodada foram vistos. Ela foi descartada (medição inválida).

Decisão (critério do card)
--------------------------
Mesma validação e mesma regra do estudo do classificador (`validation.selecao`):
dentro do treino de cada fold externo do B, os folds internos (cada falha de treino
× cada bloco normal de treino); acurácia balanceada interna média de "mfcc" e
"mfcc+pente", agregada por falha externa (escolha aninhada). O `escolher()` fica com
o mais barato dentro da margem de 0,02, então o pente só entra se ganhar por mais
que a margem; e só se não perder especificidade interna. O pente entra na cadeia se
for escolhido para as quatro falhas externas; dividido ou inconclusivo, fica fora
(regra de desempate do card). O teste do B externo não entra na decisão.

Reportado seja qual for (card): o Protocolo B externo de "mfcc" e "mfcc+pente"
(LDA de priors uniformes), com a bpfo_0.3mm.

Achado fora do escopo do card: variante "c"
-------------------------------------------
"c" = LDA do MFCC OU pontuação do pente acima de média + 4 desvios dos segmentos
normais do treino. Não é "o mesmo modelo com mais entradas": acrescenta um detector
treinado só com a normal, a ideia do card de Mahalanobis (detecção de novidade). Sai
no relatório como achado, com 20 permutações por bloco, e não entra na decisão.

Custo no STM32F411 (estimativa, a medir com o DWT na placa)
-----------------------------------------------------------
`custo_stm32()`: RAM dos buffers da `arm_rfft_fast_f32` (entrada e saída, N floats
cada) mais o acumulador do espectro médio (N/2 + 1 floats), e operações por
segmento (FFT ≈ 2,5·N·log2 N flops por bloco, mediana do fundo por bin e busca do
f₀). A `arm_rfft_fast_f32` vai até 4.096 pontos: o C (12.800) não tem FFT pronta e
é só referência.

Uso
---
    python scripts/exploration/inspect_comb_features.py --sintetico      # auto-teste
    python scripts/exploration/inspect_comb_features.py --responsavel <nome>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from scipy import signal as sg

import config
import dsp
import experimentos
import pcm_io
from validation import metricas, particao
from validation.run_protocol import novo_modelo, permutar_treino_por_bloco
from validation.selecao import escolher, folds_internos

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
N_ESFERAS, D_ESFERA, D_PRIMITIVO = 9, 7.90, 38.5
RAZAO_BPFO = N_ESFERAS / 2 * (1 - D_ESFERA / D_PRIMITIVO)     # 3,5766
RAZAO_BPFI = N_ESFERAS / 2 * (1 + D_ESFERA / D_PRIMITIVO)     # 5,4234
BUSCA_REL = 0.03
FMAX_DENTES = 5600.0
MASCARA_RES = 2.0            # máscara do eixo, em resoluções efetivas
FUNDO_RES = (3.0, 10.0)      # janela do fundo, em resoluções efetivas
LIMIAR_DENTE_DB = 6.0
EIXO_HZ = (45.0, 55.0)
EIXO_HARM_GROSSO = 3         # estimativa grossa do eixo (não se alinha com BPFO/4)
EIXO_HARMONICAS = 20         # refino do eixo (até ~1 kHz)
EIXO_REFINO_HZ = 1.0         # janela do refino em torno da estimativa grossa
EIXO_ARTIGO_HZ = 50.17       # Jung et al. (2023), Tabela 1: só no modo "fixo"
MODOS_EIXO = ("estimado", "fixo")
MODO_DECISAO = "estimado"
MARGEM = 0.02                # a do escolher() no estudo do classificador
K_DESVIOS_C = 4.0
N_PERMUTACOES = 20
RFFT_MAX_CMSIS = 4096


@dataclass(frozen=True)
class Candidato:
    nome: str
    n_bloco: int         # amostras de cada bloco (= janela de Hann)
    n_fft: int
    passo: int
    n_blocos: int
    descricao: str

    @property
    def resolucao(self) -> float:
        """Resolução efetiva, pela duração do bloco."""
        return config.FS_TRABALHO / self.n_bloco

    @property
    def bin_hz(self) -> float:
        return config.FS_TRABALHO / self.n_fft

    @property
    def mascara_hz(self) -> float:
        return MASCARA_RES * self.resolucao

    @property
    def fundo_hz(self) -> tuple[float, float]:
        return FUNDO_RES[0] * self.resolucao, FUNDO_RES[1] * self.resolucao


_N_QUADRO = int(round(config.MFCC_WINDOW_MS / 1000 * config.FS_TRABALHO))
_N_PASSO = int(round(config.MFCC_HOP_MS / 1000 * config.FS_TRABALHO))
CANDIDATOS = (
    Candidato("A_espectro_medio_quadros", _N_QUADRO, 1 << (_N_QUADRO - 1).bit_length(), _N_PASSO,
              1 + (config.AMOSTRAS_POR_SEGMENTO - _N_QUADRO) // _N_PASSO,
              "média do espectro de potência dos quadros do MFCC"),
    Candidato("B_fft4096_3blocos", 4096, 4096, 4096, 3, "FFT de 4.096 pontos, média de 3 blocos de 0,32 s"),
    Candidato("C_fft_1s", config.AMOSTRAS_POR_SEGMENTO, config.AMOSTRAS_POR_SEGMENTO,
              config.AMOSTRAS_POR_SEGMENTO, 1, "FFT do segundo inteiro (referência do teto)"),
)
FAMILIAS = (("bpfo", RAZAO_BPFO), ("bpfi", RAZAO_BPFI))
NOMES_FEATURES = ("pontuacao_bpfo", "pontuacao_bpfi", "fracao_bpfo", "fracao_bpfi")


def inviavel(c: Candidato) -> str | None:
    """Motivo se a máscara do eixo não deixa nenhum dente; None se o candidato é viável."""
    meia_distancia = EIXO_HZ[0] / 2
    if c.mascara_hz >= meia_distancia:
        return (f"máscara de {c.mascara_hz:.1f} Hz ({MASCARA_RES:g} × {c.resolucao:.1f} Hz de resolução) "
                f"≥ metade do espaçamento das harmônicas do eixo (≤ {meia_distancia:.1f} Hz): "
                "todo dente cai na vizinhança de uma harmônica do eixo")
    return None


# ---------------------------------------------------------------------------
# Espectro e pente
# ---------------------------------------------------------------------------
def espectro(x: np.ndarray, c: Candidato) -> tuple[np.ndarray, np.ndarray]:
    """Espectro de potência médio dos blocos do segmento, como seria calculado no chip."""
    x = np.asarray(x, dtype=float)
    janela = np.hanning(c.n_bloco)
    inicios = np.arange(c.n_blocos) * c.passo
    blocos = x[inicios[:, None] + np.arange(c.n_bloco)[None, :]] * janela
    p = (np.abs(np.fft.rfft(blocos, n=c.n_fft, axis=1)) ** 2).mean(axis=0)
    return np.fft.rfftfreq(c.n_fft, 1 / config.FS_TRABALHO), p


def pico_e_fundo(f: np.ndarray, p: np.ndarray, fundo_hz: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Para cada bin: máximo em ±1 bin e mediana do fundo entre ±fundo_hz."""
    res = f[1] - f[0]
    pico = np.maximum.reduce([np.roll(p, -1), p, np.roll(p, 1)])
    lo, hi = max(int(np.ceil(fundo_hz[0] / res)), 2), int(np.floor(fundo_hz[1] / res))
    offs = np.r_[-np.arange(hi, lo - 1, -1), np.arange(lo, hi + 1)]
    idx = np.clip(np.arange(len(p))[:, None] + offs[None, :], 0, len(p) - 1)
    return pico, np.median(p[idx], axis=1)


def _pente_eixo(f, pico, fundo, lo: float, hi: float, n_harm: int) -> float:
    """Candidato em [lo, hi] cujos n_harm primeiros múltiplos têm a maior saliência média."""
    res = f[1] - f[0]
    passo = res / (2 * n_harm)
    cand = np.arange(lo, hi + passo, passo)
    k = np.arange(1, n_harm + 1)
    b = np.clip(np.round(cand[:, None] * k[None, :] / res).astype(int), 1, len(pico) - 2)
    sal = 10 * np.log10((pico[b] + 1e-30) / (fundo[b] + 1e-30))
    return float(cand[int(np.argmax(sal.mean(axis=1)))])


def estimar_eixo(f: np.ndarray, pico: np.ndarray, fundo: np.ndarray) -> float:
    """
    f_eixo pelo pente harmônico do próprio eixo, em duas etapas:
    1. grosso: 3 múltiplos em 45–55 Hz. Com poucos múltiplos, uma série forte de
       falha não se alinha (com 20, o pente do eixo cai em BPFO/4 ≈ 45,7 Hz nas
       gravações de BPFO, porque acerta uma linha da BPFO a cada 4 dentes);
    2. fino: 20 múltiplos, só a ±1 Hz da estimativa grossa, para a máscara das
       harmônicas altas do eixo ficar no lugar.
    """
    grosso = _pente_eixo(f, pico, fundo, EIXO_HZ[0], EIXO_HZ[1], EIXO_HARM_GROSSO)
    return _pente_eixo(f, pico, fundo, grosso - EIXO_REFINO_HZ, grosso + EIXO_REFINO_HZ, EIXO_HARMONICAS)


def k_max(f0_cin: float) -> int:
    return int(FMAX_DENTES // (f0_cin * (1 + BUSCA_REL)))


def pente(f: np.ndarray, pico: np.ndarray, fundo: np.ndarray, f_eixo: float,
          razao: float, c: Candidato) -> dict:
    """
    Busca o f₀ (±3 % da razão cinemática × f_eixo) que maximiza a saliência média
    dos dentes válidos. Devolve a pontuação, a fração de dentes ≥ 6 dB e o f₀ (este
    só para diagnóstico: não é feature).
    """
    res = f[1] - f[0]
    f0_cin = razao * f_eixo
    km = k_max(f0_cin)
    passo = res / (2 * max(km, 1))                       # o último dente anda ≤ meio bin
    f0s = np.arange(f0_cin * (1 - BUSCA_REL), f0_cin * (1 + BUSCA_REL) + passo, passo)
    k = np.arange(1, km + 1)
    dentes = f0s[:, None] * k[None, :]                   # (candidatos, dentes)
    harm = np.round(dentes / f_eixo) * f_eixo
    valido = (np.abs(dentes - harm) >= c.mascara_hz) & (dentes <= FMAX_DENTES)
    n_validos = valido.sum(axis=1)
    if not n_validos.any():
        raise ValueError(f"nenhum dente válido abaixo de {FMAX_DENTES} Hz para f₀ ≈ {f0_cin:.1f} Hz")
    b = np.clip(np.round(dentes / res).astype(int), 1, len(pico) - 2)
    sal = 10 * np.log10((pico[b] + 1e-30) / (fundo[b] + 1e-30))
    sal = np.where(valido, sal, np.nan)
    pont = np.where(n_validos > 0, np.nansum(sal, axis=1) / np.maximum(n_validos, 1), -np.inf)
    i = int(np.argmax(pont))
    frac = float(np.nansum(sal[i] >= LIMIAR_DENTE_DB) / max(n_validos[i], 1))
    return {"pontuacao": float(pont[i]), "fracao": frac, "f0": float(f0s[i]),
            "n_dentes": int(n_validos[i]), "n_candidatos": len(f0s)}


def features_segmento(x: np.ndarray, c: Candidato, modo_eixo: str = "estimado") -> tuple[np.ndarray, dict]:
    if modo_eixo not in MODOS_EIXO:
        raise ValueError(f"modo_eixo={modo_eixo}; use um de {MODOS_EIXO}")
    f, p = espectro(x, c)
    pico, fundo = pico_e_fundo(f, p, c.fundo_hz)
    f_eixo = estimar_eixo(f, pico, fundo) if modo_eixo == "estimado" else EIXO_ARTIGO_HZ
    r = {nome: pente(f, pico, fundo, f_eixo, razao, c) for nome, razao in FAMILIAS}
    v = np.array([r["bpfo"]["pontuacao"], r["bpfi"]["pontuacao"],
                  r["bpfo"]["fracao"], r["bpfi"]["fracao"]])
    return v, {"f_eixo": f_eixo, **{f"f0_{n}": r[n]["f0"] for n in r}}


def extrair(clipes: dict, segmentos, c: Candidato, modo_eixo: str = "estimado") -> tuple[np.ndarray, list[dict]]:
    X, diag = [], []
    for s in segmentos:
        v, d = features_segmento(clipes[s.rotulo].x[s.inicio:s.fim], c, modo_eixo)
        X.append(v)
        diag.append(d)
    return np.vstack(X), diag


def custo_stm32(c: Candidato) -> dict:
    """Estimativa de RAM e operações a mais por segmento no STM32F411 (a medir com o DWT)."""
    n_bins = c.n_fft // 2 + 1
    if c.nome.startswith("A_"):
        # a FFT já existe na cadeia do MFCC: só o acumulador do espectro médio
        ram = 4 * n_bins
        flops_fft = 0.0
    else:
        ram = 4 * (2 * c.n_fft + n_bins)     # entrada e saída da rfft + acumulador
        flops_fft = c.n_blocos * 2.5 * c.n_fft * np.log2(c.n_fft)
    lo, hi = c.fundo_hz
    janela_fundo = 2 * max(int(hi / c.bin_hz) - int(lo / c.bin_hz) + 1, 1)
    ops_fundo = n_bins * janela_fundo * np.log2(max(janela_fundo, 2))   # mediana por bin
    ops_busca = 0.0
    for _, razao in FAMILIAS:
        f0 = razao * EIXO_ARTIGO_HZ
        km = k_max(f0)
        n_cand = 2 * BUSCA_REL * f0 / (c.bin_hz / (2 * km))
        ops_busca += n_cand * km
    return {"ram_kb": ram / 1024, "flops_fft": float(flops_fft), "ops_fundo": float(ops_fundo),
            "ops_busca": float(ops_busca), "ops_total": float(flops_fft + ops_fundo + ops_busca),
            "fft_na_cmsis": c.n_fft <= RFFT_MAX_CMSIS and (c.n_fft & (c.n_fft - 1)) == 0}


# ---------------------------------------------------------------------------
# Decisão: validação interna (critério do card)
# ---------------------------------------------------------------------------
def _lda(X_tr, y_tr, X_te):
    mdl = novo_modelo("lda", 2)
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)


def validacao_interna(folds, segmentos, y, conjuntos: dict, descarte: int) -> list[dict]:
    """Uma linha por (fold externo, fold interno, conjunto). O teste externo não é lido."""
    linhas = []
    for fe in folds:
        for tr, te, falha_int in folds_internos(segmentos, fe.treino, descarte):
            for nome, X in conjuntos.items():
                m = metricas.metricas_binarias(y[te], _lda(X[tr], y[tr], X[te]))
                linhas.append({"externo": fe.nome.rsplit("_bloco", 1)[0], "falha_interna": falha_int,
                               "conjunto": nome, **{k: m[k] for k in
                                                    ("sensibilidade", "especificidade", "acuracia_balanceada")}})
    return linhas


def decidir(linhas: list[dict], margem: float = MARGEM) -> dict:
    """
    Escolha aninhada entre "mfcc" e "mfcc+pente": por falha externa, `escolher()` (o
    mais barato dentro da margem) e, se ficar o pente, ele não pode perder
    especificidade interna. Entra na cadeia só se for escolhido para todas.
    """
    conjuntos = sorted({l["conjunto"] for l in linhas})
    custos = {"mfcc": {"macs": 26, "parametros": 27},
              "mfcc+pente": {"macs": 26 + len(NOMES_FEATURES), "parametros": 27 + len(NOMES_FEATURES)}}
    por_externo = {}
    for e in sorted({l["externo"] for l in linhas}):
        grupo = {c: [l for l in linhas if l["externo"] == e and l["conjunto"] == c] for c in conjuntos}
        medias = {c: float(np.mean([l["acuracia_balanceada"] for l in g])) for c, g in grupo.items()}
        espec = {c: float(np.mean([l["especificidade"] for l in g])) for c, g in grupo.items()}
        pior = {}
        for c, g in grupo.items():
            por_f = {}
            for l in g:
                por_f.setdefault(l["falha_interna"], []).append(l["acuracia_balanceada"])
            pior[c] = float(min(np.mean(v) for v in por_f.values()))
        escolhido, _ = escolher(medias, custos, margem, None, pior)
        perde_espec = escolhido == "mfcc+pente" and espec["mfcc+pente"] < espec["mfcc"] - 1e-12
        por_externo[e] = {"medias": medias, "especificidade": espec, "pior_falha": pior,
                          "escolhido_escolher": escolhido, "perde_especificidade": perde_espec,
                          "escolhido": "mfcc" if perde_espec else escolhido}
    escolhas = [d["escolhido"] for d in por_externo.values()]
    entra = all(e == "mfcc+pente" for e in escolhas)
    situacao = ("entra" if entra else
                "fica fora (dividido entre as falhas externas)" if "mfcc+pente" in escolhas else
                "fica fora (não ganha por mais que a margem em nenhuma falha externa)")
    return {"por_externo": por_externo, "entra": entra, "situacao": situacao, "margem": margem}


# ---------------------------------------------------------------------------
# Protocolo B externo (reportado, não decide)
# ---------------------------------------------------------------------------
def limiar_normal(P: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    n = P[y_tr == "normal"]
    return n.mean(axis=0) + K_DESVIOS_C * n.std(axis=0, ddof=1)


def rodar_b(folds, segmentos, y, X_mfcc, X_pente, variante: str,
            rng: np.random.Generator | None = None) -> dict:
    """Protocolo B de uma variante; com `rng`, rótulos de treino permutados por bloco."""
    resultados = []
    for f in folds:
        y_tr = permutar_treino_por_bloco(segmentos, f.treino, y, rng) if rng is not None else y[f.treino]
        X = {"mfcc": X_mfcc, "mfcc+pente": np.hstack([X_mfcc, X_pente]) if X_pente is not None else None,
             "c": X_mfcc}[variante]
        pred = _lda(X[f.treino], y_tr, X[f.teste])
        if variante == "c":
            # (c) usa a LDA do MFCC e soma, por fora, o detector do pente treinado só com a normal
            lim = limiar_normal(X_pente[f.treino][:, :2], y_tr)
            acima = (X_pente[f.teste][:, :2] > lim).any(axis=1)
            pred = np.where(acima, "falha", pred)
        m = metricas.metricas_binarias(y[f.teste], pred)
        resultados.append({"nome": f.nome, "info": f.info, **m})
    return metricas.resumo_protocolo_b(resultados)


def permutacao_c(folds, segmentos, y, X_mfcc, X_pente) -> list[float]:
    rng = np.random.default_rng(config.SEMENTE)
    return [rodar_b(folds, segmentos, y, X_mfcc, X_pente, "c", rng)["acuracia_balanceada_media"]
            for _ in range(N_PERMUTACOES)]


# ---------------------------------------------------------------------------
# Auto-teste sintético
# ---------------------------------------------------------------------------
SINT_EIXO = 50.20
SINT_SERIES = {"bpfi_0.3mm": ("bpfi", 268.30, 0.004), "bpfi_1.0mm": ("bpfi", 268.25, 0.010),
               "bpfo_0.3mm": ("bpfo", 182.65, 0.002), "bpfo_1.0mm": ("bpfo", 183.39, 0.010)}


def sintetico(seed: int = 0) -> list[pcm_io.Clip]:
    """
    Ruído colorido + harmônicos do eixo em todas as gravações (mais fortes na normal).
    Cada falha soma a série da própria pista nas frequências MEDIDAS (fora da razão
    cinemática, como nos dados); a bpfo_0.3mm tem a série mais fraca e nenhuma
    elevação larga, e as de 1,0 mm sobem o ruído de banda larga. Esperado: pente da
    própria família alto, o da outra família baixo e a normal baixa. Valida o cálculo
    (o f₀ é recuperado), não a conclusão: aqui o MFCC já separa as falhas sozinho.
    """
    rng = np.random.default_rng(seed)
    fs = config.FS_TRABALHO
    n = int(59.9986 * fs)
    t = np.arange(n) / fs
    out = []
    for c in config.CLASSES:
        ganho = 3.0 if c.endswith("1.0mm") else 1.0
        x = 0.01 * ganho * sg.lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
        a_eixo = 0.004 if c == "normal" else 0.002
        for k in range(1, 40):
            x += a_eixo / k * np.sin(2 * np.pi * k * SINT_EIXO * t + rng.uniform(0, 6.28))
        if c in SINT_SERIES:
            _, f0, a = SINT_SERIES[c]
            for k in range(1, int(FMAX_DENTES // f0) + 1):
                x += a * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28))
        out.append(pcm_io.Clip(rotulo=c, binario=config.BINARIO[c], fs=fs, x_float=x))
    return out


# ---------------------------------------------------------------------------
def _linha_b(nome: str, r: dict, perm: float | None = None) -> str:
    pf = r["por_falha"]
    return (f"  {nome:<12}{r['acuracia_balanceada_media']:>9.3f}{r['sensibilidade_media']:>7.3f}"
            f"{r['especificidade_media']:>8.3f}" + "".join(f"{pf[f]['acuracia_balanceada']:>10.3f}" for f in pf)
            + (f"{perm:>8.3f}" if perm is not None else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--features", type=Path,
                    default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/exploration/pente"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        lista = sintetico()
        particoes = particao.gerar_particoes(lista)
        segmentos = particao.segmentos_de(particoes)
        clipes = {c.rotulo: c for c in lista}
        X_mfcc = np.vstack([np.concatenate([m.mean(0), m.std(0)]) for m in
                            (dsp.mfcc(clipes[s.rotulo].x[s.inicio:s.fim], config.FS_TRABALHO)
                             for s in segmentos)])
        hash_splits, feat_commit = "sintetico", "sintetico"
    else:
        if not args.sem_registro and not args.responsavel:
            ap.error("--responsavel é obrigatório numa rodada registrada")
        lista = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
        if pcm_io.verificar_integridade(lista):
            raise SystemExit("Abortado: os PCM não batem com o manifest versionado.")
        particoes = particao.carregar(args.splits)
        erros = particao.conferir_compatibilidade(particoes, lista, config.FS_TRABALHO) + particao.verificar(particoes)
        if erros:
            raise SystemExit("splits.json inválido:\n  " + "\n  ".join(erros))
        segmentos = particao.segmentos_de(particoes)
        clipes = {c.rotulo: c for c in lista}
        hash_splits = particao.hash_arquivo(args.splits)
        manif = json.loads(args.features.with_name("manifest_features.json").read_text(encoding="utf-8"))
        if manif.get("splits_hash") != hash_splits or manif.get("norm_clipe"):
            raise SystemExit("Abortado: rode o 04 sem opções (features de referência desta partição).")
        X_mfcc = np.load(args.features)["X"]
        feat_commit = manif.get("git_commit")

    folds = particao.folds_de(particoes, "B")
    descarte = int(particoes["parametros"]["segmentos_descarte"])
    y = np.array([s.binario for s in segmentos])
    rot = np.array([s.rotulo for s in segmentos])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"razões cinemáticas: BPFO {RAZAO_BPFO:.4f} × eixo, BPFI {RAZAO_BPFI:.4f} × eixo; "
          f"busca ±{BUSCA_REL:.0%}; decisão pelo modo do eixo '{MODO_DECISAO}', margem {MARGEM}")
    b_mfcc = rodar_b(folds, segmentos, y, X_mfcc, None, "mfcc")
    cab = (f"  {'conjunto':<12}{'acc.bal.':>9}{'sens.':>7}{'espec.':>8}"
           f"{'bpfi_0.3':>10}{'bpfi_1.0':>10}{'bpfo_0.3':>10}{'bpfo_1.0':>10}{'perm.':>8}")
    resultado = {}
    for c in CANDIDATOS:
        custo = custo_stm32(c)
        motivo = inviavel(c)
        print(f"\n== {c.nome}: {c.descricao}; resolução {c.resolucao:.3f} Hz, máscara do eixo "
              f"{c.mascara_hz:.2f} Hz; custo estimado {custo['ram_kb']:.1f} KB, "
              f"{custo['ops_total'] / 1e6:.2f} Mops por segmento")
        resultado[c.nome] = {"candidato": {**c.__dict__, "resolucao_hz": c.resolucao,
                                           "mascara_hz": c.mascara_hz, "fundo_hz": c.fundo_hz},
                             "custo_stm32": custo, "inviavel": motivo, "modos": {}}
        if motivo:
            print(f"  inviável: {motivo}")
            continue
        for modo in MODOS_EIXO:
            X_pente, diag = extrair(clipes, segmentos, c, modo)
            dist = {k: {n: [float(X_pente[rot == k, j].mean()), float(X_pente[rot == k, j].std())]
                        for j, n in enumerate(NOMES_FEATURES)} for k in config.CLASSES}
            f0 = {k: {campo: float(np.median([d[campo] for d, r in zip(diag, rot) if r == k]))
                      for campo in ("f_eixo", "f0_bpfo", "f0_bpfi")} for k in config.CLASSES}
            print(f"\n  eixo {modo}" + (" (decide)" if modo == MODO_DECISAO else " (sensibilidade)"))
            print(f"  {'classe':<12}{'pont. BPFO':>12}{'pont. BPFI':>12}{'eixo':>8}{'f0 BPFO':>10}{'f0 BPFI':>10}")
            for k in config.CLASSES:
                d = dist[k]
                print(f"  {k:<12}{d['pontuacao_bpfo'][0]:>7.1f}±{d['pontuacao_bpfo'][1]:<4.1f}"
                      f"{d['pontuacao_bpfi'][0]:>7.1f}±{d['pontuacao_bpfi'][1]:<4.1f}"
                      f"{f0[k]['f_eixo']:>8.2f}{f0[k]['f0_bpfo']:>10.2f}{f0[k]['f0_bpfi']:>10.2f}")

            internas = validacao_interna(folds, segmentos, y,
                                         {"mfcc": X_mfcc, "mfcc+pente": np.hstack([X_mfcc, X_pente])},
                                         descarte)
            dec = decidir(internas)
            print("  validação interna (acc. bal. média | especificidade) por falha externa:")
            for e, d in dec["por_externo"].items():
                print(f"    {e:<12} mfcc {d['medias']['mfcc']:.3f} | {d['especificidade']['mfcc']:.3f}   "
                      f"mfcc+pente {d['medias']['mfcc+pente']:.3f} | {d['especificidade']['mfcc+pente']:.3f}"
                      f"   → {d['escolhido']}" + (" (perde especificidade)" if d["perde_especificidade"] else ""))
            print(f"  decisão: {dec['situacao']}")

            b_pente = rodar_b(folds, segmentos, y, X_mfcc, X_pente, "mfcc+pente")
            b_c = rodar_b(folds, segmentos, y, X_mfcc, X_pente, "c")
            perm_c = permutacao_c(folds, segmentos, y, X_mfcc, X_pente)
            print("  Protocolo B externo (reportado, não decide):")
            print(cab)
            print(_linha_b("mfcc", b_mfcc))
            print(_linha_b("mfcc+pente", b_pente))
            print(_linha_b("c (achado)", b_c, float(np.mean(perm_c))))
            resultado[c.nome]["modos"][modo] = {
                "decide": modo == MODO_DECISAO,
                "distribuicao_por_classe": dist, "f0_mediano_por_classe": f0,
                "decisao": dec,
                "protocolo_b_externo": {"mfcc": b_mfcc, "mfcc+pente": b_pente},
                "achado_c": {"resumo": b_c, "permutacao_media": float(np.mean(perm_c)),
                             "permutacao_max": float(np.max(perm_c)), "k_desvios": K_DESVIOS_C},
            }

    parametros = {"razao_bpfo": RAZAO_BPFO, "razao_bpfi": RAZAO_BPFI, "busca_rel": BUSCA_REL,
                  "fmax_dentes": FMAX_DENTES, "mascara_res": MASCARA_RES, "fundo_res": FUNDO_RES,
                  "limiar_dente_db": LIMIAR_DENTE_DB, "eixo_hz": EIXO_HZ, "modo_decisao": MODO_DECISAO,
                  "margem": MARGEM, "k_desvios_c": K_DESVIOS_C, "n_permutacoes": N_PERMUTACOES,
                  "splits": hash_splits, "features_commit": feat_commit}
    (args.out_dir / "metrics.json").write_text(json.dumps(
        {"parametros": parametros, "candidatos": resultado}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\nsaídas: {args.out_dir}/")
    if args.sem_registro:
        print("(registry não alterado)")
        return 0

    linhas = []
    base = experimentos.next_exp_number(args.registry)
    for nome, r in resultado.items():
        cu = r["custo_stm32"]
        comum = {"candidato": nome, "resolucao_hz": f"{r['candidato']['resolucao_hz']:.3f}",
                 "mascara_hz": f"{r['candidato']['mascara_hz']:.2f}", "busca_rel": BUSCA_REL,
                 "limiar_dente_db": LIMIAR_DENTE_DB, "splits": hash_splits, "features_commit": feat_commit}
        modos = r["modos"].items() if r["modos"] else [(None, None)]
        for modo, m in modos:
            met = {"ram_kb": f"{cu['ram_kb']:.1f}", "mops": f"{cu['ops_total'] / 1e6:.2f}"}
            if m is None:
                met["viavel"] = "nao"
                notas = f"candidato inviável: {r['inviavel']}"
            else:
                dec, bp, c_ = m["decisao"], m["protocolo_b_externo"]["mfcc+pente"], m["achado_c"]
                met.update({
                    **{f"interna_{e}_{k}": f"{d['medias'][k]:.4f}" for e, d in dec["por_externo"].items()
                       for k in ("mfcc", "mfcc+pente")},
                    "entra": "sim" if dec["entra"] else "nao",
                    "b_acc_bal_media": f"{bp['acuracia_balanceada_media']:.4f}",
                    "b_espec_media": f"{bp['especificidade_media']:.4f}",
                    "b_acc_bal_bpfo_0.3mm": f"{bp['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.4f}",
                    "c_acc_bal_bpfo_0.3mm": f"{c_['resumo']['por_falha']['bpfo_0.3mm']['acuracia_balanceada']:.4f}",
                    "c_espec_media": f"{c_['resumo']['especificidade_media']:.4f}",
                    "c_perm_media": f"{c_['permutacao_media']:.4f}",
                })
                notas = (f"pente harmônico, eixo {modo} ({'decide' if m['decide'] else 'sensibilidade'}): "
                         f"{dec['situacao']}; c = LDA OU pente acima da normal, achado fora do escopo do card")
            linhas.append({
                "id": f"exp{base + len(linhas):03d}",
                "data": date.today().isoformat(),
                "etapa": "caracterizacao_pente",
                "script": "exploration/inspect_comb_features.py",
                "git_commit": experimentos.git_short_hash(),
                "parametros": experimentos.kv({**comum, "modo_eixo": modo, "margem": MARGEM if modo else None}),
                "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
                "metricas": experimentos.kv(met),
                "responsavel": args.responsavel,
                "notas": notas,
            })
    experimentos.append_registry(args.registry, linhas)
    print(f"registrado: {linhas[0]['id']}–{linhas[-1]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
