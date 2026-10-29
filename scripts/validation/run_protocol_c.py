#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_protocol_c.py — Protocolo C no HUSTmotor: LDA sem retreino e especificidade dos detectores
==============================================================================================

Card "HUSTmotor — LDA sem retreino e especificidade dos detectores de novidade", com o
critério do docs/criterio_deteccao_novidade.md (versão 4, seções 2 a 5) e os dados do
10_prepare_hustmotor.py (docs/hustmotor.md).

Leituras
--------
LDA sem retreino
    A LDA final (reports/modelo_final/lda_final.json) em todos os 40 segmentos do HUST.

C1, transferência (só reporte: limite de transferência)
    Cada detector ajustado e com o limiar tirado no Jung inteiro, aplicado ao HUST sem
    recalibrar. Mede o alarme em todos os segmentos normais do HUST.
    D2  Mahalanobis nos 59 segmentos normais do Jung; limiar = maior pontuação fora do bloco.
    D1  pente; limiar = maior pontuação dos 59 normais do Jung, com a geometria e a rotação
        do Jung. No HUST, o pente usa a geometria do 6203 e a rotação de cada arquivo.
    D4  LDA treinada nos 295 segmentos do Jung mais as cópias sintéticas dos 59 normais.

C2, comissionamento (é o que decide)
    A normal aprendida no próprio HUST, com a divisão do manifest (`divisao_c2`): os blocos
    de calibração da normal ajustam o detector e dão o limiar; os blocos de medida da
    normal dão o alarme falso; a falha mede a detecção com os blocos dela.
    D2  normal junta (as quatro rotações): por rotação seriam 2 segmentos para 26
        dimensões, e o ajuste fora do bloco teria 1. Limiar = maior pontuação fora do
        bloco, cada segmento como o seu próprio bloco (as aquisições são separadas).
    D1  só a 20 e a 30 Hz (a 5 e a 10 Hz a máscara do eixo cobre o espectro); normal junta
        das duas rotações; limiar = maior pontuação da calibração, por série.
    D4  não tem limiar para recalibrar: fica só no C1.
O D3 saiu no Protocolo B (não supera o D2) e não roda aqui.

Limitações do C2 (gravadas no metrics.json)
    - O limiar é o máximo de poucas pontuações (8 no D2; 4 por série no D1). Um
      segmento normal novo passa do máximo de n com probabilidade ~1/(n+1): ~11 % no
      D2 e ~20 % por série no D1 (~36 % com BPFO OU BPFI), só pelo tamanho da
      calibração. A seção 2 do critério conta com ~47 pontuações (~2 %); aqui o limiar
      não tem esse significado.
    - No D2, cada ajuste fora do bloco tem 7 segmentos para 26 dimensões; o encolhimento
      do Ledoit–Wolf fica gravado no metrics.json (no HUST, 0,37 no ajuste final e 0,40
      em média fora do bloco).
    - Calibração e medida saem do mesmo arquivo de cada rotação (mesma gravação,
      montagem, microfone e sala): as aquisições são separadas, mas não independentes,
      e o alarme falso medido fica otimista.

Cada detector é reportado sozinho e OU a LDA final (o combinado, como ficaria no
firmware). A rotação nominal decide; a estimada no segmento (pente do eixo entre 0,85 e
1,02 vez a nominal) é só reporte, no D1.

Estatística
-----------
Alarme falso: k alarmes em n segmentos normais de medida, com o limite superior
unilateral de 95 % (Clopper–Pearson). Com 8 segmentos e nenhum alarme, ele é 31 %:
o C2 consegue reprovar um detector, mas não aprová-lo. Detecção: fração dos
segmentos de falha, com o intervalo bilateral de 95 %.

Decisão (`decidir`, contra X = 5 %): aprova se o limite superior unilateral de 95 % ficar
em até X; reprova se o limite inferior unilateral de 95 % passar de X; senão, inconclusivo.
Os dois limites unilaterais de 95 % equivalem a um intervalo bilateral de 90 %. Pelo
critério, decide o combinado (o detector OU a LDA); o detector sozinho é reportado à parte.

Decisões do Registro (22/10): normal do C2 junta (as quatro rotações), descarte do bloco 3
e a regra da `decidir()` acima, que completa o "alarme falso ≤ X" do critério v4.

Recalibração da LDA pela normal da máquina nova (--recalibracao)
----------------------------------------------------------------
Página do Registro de Decisões "Recalibrar a LDA pela normal da máquina nova" (29/10,
Definida). A LDA final do Jung é aplicada SEM MUDAR OS PESOS, depois de as características
do HUST serem levadas ao referencial da normal do Jung, só com gravações saudáveis: a
média de cada rotação vem das aquisições de calibração da normal (as mesmas do C2), e a
referência são a média e o desvio dos 59 segmentos normais do Jung (os do treino do
modelo final). A falha nunca entra na calibração. Base: a normalização da média
cepstral. Um microfone, uma sala ou um ganho diferentes são um filtro fixo, que desloca
as 13 médias do MFCC e não muda os 13 desvios.

Variantes (fixadas antes de rodar):
    medias13  PRINCIPAL, decide: x′ = x − μ_rotação + μ_Jung só nas 13 médias.
    sem_c0    CONTROLE, decide junto: a principal com o c0 neutralizado (a média e o
              desvio do c0 recebem o valor da normal do Jung em todos os segmentos).
    medias26  reporte: a mesma conta nas 26 características.
    desvio    reporte: x′ = μ_Jung + σ_Jung · (x − μ_rotação) / σ, com σ o desvio comum
              às rotações (resíduos em torno da média de cada uma) e piso de 0,5 · σ_Jung.

Critério (`classificar_recalibracao`): funciona se a principal e o controle detectarem
cada um pelo menos 12 dos 15 segmentos com falha a 10, 20 e 30 Hz (0,80) e o alarme
falso da principal não reprovar; "detecção pelo nível" se só a principal passar; não
funciona nos outros casos. A 5 Hz fica fora da conta e é reportada à parte. A leitura
combinada (D2 do C2 OU a LDA recalibrada) é calculada com as mesmas regras; o papel dela
na decisão do D2 é o da página da adoção do D2. A conferência no Protocolo B recalibra
cada fold com a normal do próprio treino (a transformação vira a identidade): mostra
que o código está certo, não que o método funciona.

Uso
---
    python scripts/validation/run_protocol_c.py --sintetico          # auto-teste, sem registro
    python scripts/validation/run_protocol_c.py --responsavel <nome>
    python scripts/validation/run_protocol_c.py --recalibracao --sintetico
    python scripts/validation/run_protocol_c.py --recalibracao --responsavel <nome>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from scipy import stats

import config
import dsp
import experimentos
import pcm_io
from models import lda as modelo_lda
from validation import metricas, particao
from validation.run_protocol import novo_modelo

_RAIZ = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "inspect_deteccao_novidade", _RAIZ / "scripts" / "exploration" / "inspect_deteccao_novidade.py")
N = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = N
_spec.loader.exec_module(N)
P = N.P                                    # inspect_comb_features (espectro e pente)

CAND = N.CAND_B                            # o espectro do D1 (FFT de 4.096 pontos, 3 blocos)
EIXO_JUNG_HZ = N.EIXO_NOMINAL_HZ           # 50,17 Hz
EIXO_ESTIMADO_REL = (0.85, 1.02)           # faixa do pente do eixo, em vezes a rotação nominal
CONFIANCA = 0.95
X_ALARME = 0.05                            # alarme falso máximo por segmento (critério, seção 4)


# ---------------------------------------------------------------------------
# Dados do HUST
# ---------------------------------------------------------------------------
@dataclass
class SegHust:
    rotulo: str        # h_20hz, bf_30hz, ...
    classe: str        # H ou BF
    binario: str       # normal ou falha
    rotacao: int       # rotação nominal, Hz
    bloco: int         # aquisição de origem, 1 a 5
    x: np.ndarray      # 1 s a 12,8 kHz, em float


def carregar_hust(pcm_dir: Path) -> tuple[list[SegHust], dict]:
    """Os segmentos do HUST, cortados pelo pcm_io.cortar_segmentos, e o `_conjunto` do manifest."""
    conjunto = json.loads((pcm_dir / "manifest.json").read_text(encoding="utf-8"))["_conjunto"]
    clipes = pcm_io.carregar_clipes(pcm_dir, fs_esperado=config.FS_TRABALHO)
    if pcm_io.verificar_integridade(clipes):
        raise SystemExit("Abortado: os .bin do HUST não batem com o manifest versionado.")
    n = config.AMOSTRAS_POR_SEGMENTO
    segs = []
    for c in clipes:
        m = c.meta
        for k, x in enumerate(pcm_io.cortar_segmentos(c.x, n, int(m["n_segmentos"])), start=1):
            segs.append(SegHust(c.rotulo, m["classe_hust"], c.binario, int(m["velocidade_nominal_hz"]), k, x))
    return segs, conjunto


def hust_sintetico(razoes: tuple[float, float], seed: int = 0) -> list[SegHust]:
    """HUST com resposta conhecida: harmônicas do eixo e ruído; a falha soma as duas séries."""
    rng = np.random.default_rng(seed)
    n = config.AMOSTRAS_POR_SEGMENTO
    t = np.arange(n) / config.FS_TRABALHO
    segs = []
    for v in (5, 10, 20, 30):
        for classe, binario in (("H", "normal"), ("BF", "falha")):
            for k in range(1, 6):
                x = 0.01 * rng.standard_normal(n)
                x += sum(0.02 / h * np.sin(2 * np.pi * h * v * t + rng.uniform(0, 6.28))
                         for h in range(1, 21))
                if classe == "BF":
                    for r in razoes:
                        f0 = r * v
                        x += sum(0.006 / np.sqrt(j) * np.sin(2 * np.pi * j * f0 * t + rng.uniform(0, 6.28))
                                 for j in range(1, int(5000 // f0) + 1))
                segs.append(SegHust(f"{classe.lower()}_{v}hz", classe, binario, v, k, x))
    return segs


def mfcc26(x: np.ndarray) -> np.ndarray:
    m = dsp.mfcc(x, config.FS_TRABALHO)
    return np.concatenate([m.mean(axis=0), m.std(axis=0)])


# ---------------------------------------------------------------------------
# Pente no HUST
# ---------------------------------------------------------------------------
def pente_viavel(rotacao: float, c=CAND) -> bool:
    """A máscara do eixo (±2 resoluções) deixa algum dente? (docs/hustmotor.md)"""
    return c.mascara_hz < rotacao / 2


def estimar_eixo(f, pico, fundo, nominal: float) -> float:
    """A rotação pelo pente do eixo, como no estudo do pente, mas em torno da nominal do arquivo."""
    lo, hi = EIXO_ESTIMADO_REL[0] * nominal, EIXO_ESTIMADO_REL[1] * nominal
    grosso = P._pente_eixo(f, pico, fundo, lo, hi, P.EIXO_HARM_GROSSO)
    return P._pente_eixo(f, pico, fundo, grosso - P.EIXO_REFINO_HZ, grosso + P.EIXO_REFINO_HZ,
                         P.EIXO_HARMONICAS)


def pontuacoes_pente(segs: list[SegHust], razoes: tuple[float, float], modo: str) -> np.ndarray:
    """(n, 2): a pontuação do pente de BPFO e de BPFI; NaN onde o pente é inviável."""
    out = np.full((len(segs), 2), np.nan)
    for i, s in enumerate(segs):
        if not pente_viavel(s.rotacao):
            continue
        f, p = P.espectro(s.x, CAND)
        pico, fundo = P.pico_e_fundo(f, p, CAND.fundo_hz)
        eixo = s.rotacao if modo == "nominal" else estimar_eixo(f, pico, fundo, s.rotacao)
        out[i] = [P.pente(f, pico, fundo, eixo, r, CAND)["pontuacao"] for r in razoes]
    return out


# ---------------------------------------------------------------------------
# Estatística e resumo
# ---------------------------------------------------------------------------
def limite_superior(k: int, n: int, conf: float = CONFIANCA) -> float:
    """Limite superior unilateral (Clopper–Pearson) da taxa, com k eventos em n."""
    if n == 0:
        return float("nan")
    return 1.0 if k >= n else float(stats.beta.ppf(conf, k + 1, n - k))


def limite_inferior(k: int, n: int, conf: float = CONFIANCA) -> float:
    """Limite inferior unilateral (Clopper–Pearson) da taxa, com k eventos em n."""
    if n == 0:
        return float("nan")
    return 0.0 if k == 0 else float(stats.beta.ppf(1 - conf, k, n - k + 1))


def decidir(k: int, n: int, x: float = X_ALARME, conf: float = CONFIANCA) -> str:
    """
    Aprova se o limite superior unilateral ≤ x; reprova se o limite inferior unilateral > x;
    senão, inconclusivo. Os dois unilaterais de 95 % equivalem a um bilateral de 90 %.
    """
    if n == 0:
        return "sem dados"
    if limite_superior(k, n, conf) <= x:
        return "aprova"
    if limite_inferior(k, n, conf) > x:
        return "reprova"
    return "inconclusivo"


def intervalo(k: int, n: int, conf: float = CONFIANCA) -> tuple[float, float]:
    """Intervalo bilateral (Clopper–Pearson) da taxa, com k eventos em n."""
    if n == 0:
        return float("nan"), float("nan")
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def contar(d: np.ndarray, mascara: np.ndarray) -> dict:
    k, n = int(d[mascara].sum()), int(mascara.sum())
    return {"k": k, "n": n, "taxa": k / n if n else float("nan")}


def resumir(segs: list[SegHust], dispara: np.ndarray, medida_normal: np.ndarray,
            medida_falha: np.ndarray) -> dict:
    """Alarme falso (com o limite superior) e detecção (com o intervalo), no total e por rotação."""
    rot = np.array([s.rotacao for s in segs])
    alarme, det = contar(dispara, medida_normal), contar(dispara, medida_falha)
    alarme["limite_superior_95"] = limite_superior(alarme["k"], alarme["n"])
    alarme["limite_inferior_95"] = limite_inferior(alarme["k"], alarme["n"])
    alarme["decisao"] = decidir(alarme["k"], alarme["n"])
    det["intervalo_95"] = intervalo(det["k"], det["n"])
    por_rot = {}
    for v in sorted(set(rot[medida_normal | medida_falha].tolist())):
        a = contar(dispara, medida_normal & (rot == v))
        a["limite_superior_95"] = limite_superior(a["k"], a["n"])
        por_rot[int(v)] = {"alarme": a, "deteccao": contar(dispara, medida_falha & (rot == v))}
    return {"alarme": alarme, "deteccao": det, "por_rotacao": por_rot}


def leitura(segs, dispara, lda_falha, medida_normal, medida_falha) -> dict:
    """O detector sozinho e OU a LDA final, nos mesmos segmentos (NaN do pente = não avaliado)."""
    validos = ~np.isnan(dispara.astype(float)) if dispara.dtype != bool else np.ones(len(segs), bool)
    d = np.where(validos, dispara, False).astype(bool)
    mn, mf = medida_normal & validos, medida_falha & validos
    return {"sozinho": resumir(segs, d, mn, mf), "ou_lda": resumir(segs, d | lda_falha, mn, mf)}


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------
def limiar_pente(pont: np.ndarray) -> np.ndarray:
    """Limiar de cada série: a maior pontuação da normal de calibração."""
    return np.nanmax(pont, axis=0)


def dispara_pente(pont: np.ndarray, lim: np.ndarray) -> np.ndarray:
    """O pente dispara se qualquer série passar do limiar; NaN onde não foi avaliado."""
    out = (pont > lim).any(axis=1).astype(float)
    out[np.isnan(pont).any(axis=1)] = np.nan
    return out


def mahalanobis_c2(X: np.ndarray, calib: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """
    D2 no C2: ajuste na calibração, limiar fora do bloco (cada segmento o seu bloco).
    Devolve também o encolhimento do Ledoit–Wolf, no ajuste final e nos ajustes fora do bloco.
    """
    idx = np.flatnonzero(calib)
    lim = N.limiar_fora_do_bloco(X, idx, np.arange(len(idx)))
    final = N.Mahalanobis().fit(X[idx])
    fora = [N.Mahalanobis().fit(X[np.delete(idx, i)]).lw.shrinkage_ for i in range(len(idx))]
    encolhimento = {"final": float(final.lw.shrinkage_), "fora_do_bloco_media": float(np.mean(fora)),
                    "n_ajuste_fora_do_bloco": len(idx) - 1, "dimensoes": X.shape[1]}
    return final.pontuar(X) > lim, lim, encolhimento


def alarme_esperado_pelo_maximo(n_calib: int, n_series: int = 1) -> float:
    """Chance de um segmento normal novo passar do máximo de n_calib (~1/(n+1) por série)."""
    p = 1 / (n_calib + 1)
    return 1 - (1 - p) ** n_series


# ---------------------------------------------------------------------------
# Recalibração da LDA pela normal da máquina nova (--recalibracao)
# ---------------------------------------------------------------------------
N_COEFS = config.MFCC_N_COEFS                   # colunas 0..12 = médias; 13..25 = desvios
COLUNAS_C0 = (0, N_COEFS)                       # a média e o desvio do c0
VARIANTES_RECAL = {                             # variante → papel (Registro, 29/10)
    "medias13": "decide",
    "sem_c0": "decide_junto",
    "medias26": "reporte",
    "desvio": "reporte",
}
PISO_DESVIO = 0.5                               # na variante desvio: σ ≥ 0,5 · σ_Jung
ROTACOES_CRITERIO = (10, 20, 30)                # a 5 Hz fica fora da conta, reportada à parte
FRACAO_DETECCAO = 0.80                          # 12 de 15: o 0,80 da adoção de um detector


def normal_referencia(X_jung: np.ndarray, y_jung: np.ndarray) -> dict:
    """Média e desvio (ddof = 1) das 26 características nos segmentos normais do Jung."""
    Xn = np.asarray(X_jung, dtype=float)[np.asarray(y_jung) == "normal"]
    if len(Xn) < 2:
        raise ValueError("a referência precisa de pelo menos 2 segmentos normais do Jung")
    return {"media": Xn.mean(axis=0), "desvio": Xn.std(axis=0, ddof=1), "n": int(len(Xn))}


def recalibrar(X: np.ndarray, rot: np.ndarray, calib: np.ndarray, ref: dict,
               variante: str) -> tuple[np.ndarray, dict]:
    """
    As características levadas ao referencial da normal do Jung, rotação a rotação, só com
    os segmentos de calibração (normais). Devolve X′ e o que a calibração estimou.

    medias13  x′ = x − μ_rotação + μ_Jung nas 13 médias; os 13 desvios ficam como medidos.
    sem_c0    medias13 e, depois, a média e o desvio do c0 iguais aos da normal do Jung.
    medias26  a mesma conta da medias13 nas 26 características.
    desvio    x′ = μ_Jung + σ_Jung · (x − μ_rotação) / σ, nas 26, com σ o desvio comum às
              rotações (resíduos em torno da média de cada uma, n − rotações graus de
              liberdade) e o piso de PISO_DESVIO · σ_Jung.
    """
    if variante not in VARIANTES_RECAL:
        raise ValueError(f"variante desconhecida: {variante}")
    X = np.asarray(X, dtype=float)
    rot, calib = np.asarray(rot), np.asarray(calib, dtype=bool)
    mu_ref, sd_ref = np.asarray(ref["media"], dtype=float), np.asarray(ref["desvio"], dtype=float)
    cols = slice(0, N_COEFS) if variante in ("medias13", "sem_c0") else slice(None)
    Xr, medias, residuos = X.copy(), {}, []
    for v in sorted(set(rot.tolist())):
        m, c = rot == v, calib & (rot == v)
        if not c.any():
            raise ValueError(f"rotação {v} sem segmento de calibração")
        medias[v] = X[c].mean(axis=0)
        residuos.append(X[c] - medias[v])
        if variante != "desvio":
            Xr[m, cols] = X[m, cols] - medias[v][cols] + mu_ref[cols]
    info = {"n_calibracao": int(calib.sum()),
            "media_por_rotacao": {int(v): mu.tolist() for v, mu in medias.items()}}
    if variante == "desvio":
        R = np.vstack(residuos)
        gl = len(R) - len(medias)
        if gl < 1:
            raise ValueError("variante desvio sem graus de liberdade (1 segmento de calibração por rotação)")
        sd_cal = np.sqrt((R ** 2).sum(axis=0) / gl)
        piso = PISO_DESVIO * sd_ref
        sd = np.maximum(sd_cal, piso)
        for v, mu_v in medias.items():
            m = rot == v
            Xr[m] = mu_ref + sd_ref * (X[m] - mu_v) / sd
        info.update({"graus_de_liberdade": int(gl), "desvio_calibracao": sd_cal.tolist(),
                     "caracteristicas_no_piso": int((sd_cal < piso).sum())})
    if variante == "sem_c0":
        for j in COLUNAS_C0:
            Xr[:, j] = mu_ref[j]
    return Xr, info


def bias_medias13(p_lda: dict, ref: dict, media_rotacao: np.ndarray) -> float:
    """
    No firmware, a variante principal entra só no bias: escore(x′) = pesos · x + bias′, com
    bias′ = bias + Σ_{i < 13} pesos_i (μ_Jung,i − μ_rotação,i). Os pesos não mudam.
    """
    d = p_lda["dobrada"]
    w = np.asarray(d["pesos"], dtype=float)
    delta = np.asarray(ref["media"], dtype=float)[:N_COEFS] - np.asarray(media_rotacao, dtype=float)[:N_COEFS]
    return float(d["bias"] + w[:N_COEFS] @ delta)


def avaliar_recalibracao(segs: list[SegHust], dispara: np.ndarray, medida_normal: np.ndarray,
                         medida_falha: np.ndarray) -> dict:
    """O resumir() do C2 mais a detecção do critério (10, 20 e 30 Hz) e a de fora dele (5 Hz)."""
    r = resumir(segs, dispara, medida_normal, medida_falha)
    rot = np.array([s.rotacao for s in segs])
    no_criterio = np.isin(rot, ROTACOES_CRITERIO)
    det = contar(dispara, medida_falha & no_criterio)
    det["minimo"] = int(np.ceil(FRACAO_DETECCAO * det["n"] - 1e-9))
    det["passa"] = bool(det["n"] > 0 and det["k"] >= det["minimo"])
    det["intervalo_95"] = intervalo(det["k"], det["n"])
    r["deteccao_criterio"] = det
    r["deteccao_fora_do_criterio"] = contar(dispara, medida_falha & ~no_criterio)
    return r


def classificar_recalibracao(principal: dict, controle: dict) -> str:
    """
    Funciona: a principal e o controle sem o c0 passam na detecção (10, 20 e 30 Hz) e o
    alarme falso da principal não reprova. Detecção pelo nível: só a principal passa.
    Não funciona: a principal não passa na detecção, ou o alarme falso reprova.
    """
    if principal["alarme"]["decisao"] == "reprova" or not principal["deteccao_criterio"]["passa"]:
        return "não funciona"
    if not controle["deteccao_criterio"]["passa"]:
        return "detecção pelo nível"
    return "funciona"


def conferencia_protocolo_b(X_jung: np.ndarray, y_jung: np.ndarray, folds) -> dict:
    """
    Cada fold do B recalibrado (variante principal) com a normal do próprio treino, que é
    também a referência: a transformação vira a identidade. Confere que o código não
    muda nada onde não há o que corrigir, e não que o método funciona.
    """
    X_jung, y_jung = np.asarray(X_jung, dtype=float), np.asarray(y_jung)
    sem, com, diferentes, maior = [], [], 0, 0.0
    for f in folds:
        tr, te = np.asarray(f.treino), np.asarray(f.teste)
        mdl = novo_modelo("lda", 2).fit(X_jung[tr], y_jung[tr])
        calib = np.zeros(len(X_jung), dtype=bool)
        calib[tr[y_jung[tr] == "normal"]] = True
        Xr, _ = recalibrar(X_jung, np.zeros(len(X_jung), dtype=int), calib,
                           normal_referencia(X_jung[tr], y_jung[tr]), "medias13")
        maior = max(maior, float(np.abs(Xr[te] - X_jung[te]).max()))
        p0, p1 = mdl.predict(X_jung[te]), mdl.predict(Xr[te])
        diferentes += int((p0 != p1).sum())
        for lista, p in ((sem, p0), (com, p1)):
            lista.append({"nome": f.nome, "info": f.info, **metricas.metricas_binarias(y_jung[te], p)})
    return {"n_folds": len(folds),
            "acc_bal_media_sem": metricas.resumo_protocolo_b(sem)["acuracia_balanceada_media"],
            "acc_bal_media_com": metricas.resumo_protocolo_b(com)["acuracia_balanceada_media"],
            "decisoes_diferentes": diferentes, "maior_mudanca_nas_caracteristicas": maior}


def _fmt_recal(r: dict) -> str:
    a, d, f = r["alarme"], r["deteccao_criterio"], r["deteccao_fora_do_criterio"]
    return (f"alarme {a['k']}/{a['n']} ({a['decisao']}; sup95 {a['limite_superior_95']:.0%})   "
            f"detecção 10–30 Hz {d['k']}/{d['n']} ({'passa' if d['passa'] else 'não passa'}, "
            f"mínimo {d['minimo']})   5 Hz {f['k']}/{f['n']}")


def rodar_recalibracao(args, segs: list[SegHust], X: np.ndarray, calib: np.ndarray,
                       med_normal: np.ndarray, med_falha: np.ndarray, X_jung: np.ndarray,
                       y_jung: np.ndarray, p_lda: dict, folds_b, divisao: dict, modelo_id: str,
                       hash_splits: str, feat_commit: str) -> int:
    """As leituras da recalibração, a combinada com o D2, a conferência no B e o registro."""
    rot = np.array([s.rotacao for s in segs])
    ref = normal_referencia(X_jung, y_jung)
    res, dispara = {}, {}
    for v, papel in VARIANTES_RECAL.items():
        Xr, info = recalibrar(X, rot, calib, ref, v)
        dispara[v] = modelo_lda.prever(p_lda, Xr) == "falha"
        res[f"C2_LDA_recal_{v}"] = {
            "leitura": "C2", "detector": "LDA_recal", "variante": v, "papel": papel, "calibracao": info,
            "resultado": {"sozinho": avaliar_recalibracao(segs, dispara[v], med_normal, med_falha)}}
    principal = res["C2_LDA_recal_medias13"]
    principal["bias_por_rotacao"] = {
        v: bias_medias13(p_lda, ref, np.asarray(mu)) for v, mu in principal["calibracao"]["media_por_rotacao"].items()}
    resultado = classificar_recalibracao(principal["resultado"]["sozinho"],
                                         res["C2_LDA_recal_sem_c0"]["resultado"]["sozinho"])
    d2_c2, lim_d2, _ = mahalanobis_c2(X, calib)
    res["C2_D2_ou_LDA_recal"] = {
        "leitura": "C2", "detector": "D2_OU_LDA_recal", "variante": "medias13",
        "papel": "combinado (papel na decisão: página da adoção do D2)", "limiar_d2": lim_d2,
        "resultado": {"combinado": avaliar_recalibracao(segs, d2_c2 | dispara["medias13"], med_normal, med_falha)}}
    comparacao = {"C2_LDA_sem_recalibrar": avaliar_recalibracao(
        segs, modelo_lda.prever(p_lda, X) == "falha", med_normal, med_falha)}
    conf_b = conferencia_protocolo_b(X_jung, y_jung, folds_b)

    print("\nRecalibração da LDA pela normal da máquina nova (C2; a falha não entra na calibração)")
    for nome, r in res.items():
        for modo, rr in r["resultado"].items():
            print(f"  {nome:<24} [{r['papel'].split(' ')[0]}] {_fmt_recal(rr)}")
            print(f"  {'':<24} por rotação (alarme e detecção): {_por_rot(rr)}")
    print(f"  {'LDA sem recalibrar':<24} [comparação] {_fmt_recal(comparacao['C2_LDA_sem_recalibrar'])}")
    print(f"\nresultado (principal e controle sem o c0): {resultado.upper()}")
    print(f"conferência no Protocolo B: acc. bal. {conf_b['acc_bal_media_sem']:.4f} sem e "
          f"{conf_b['acc_bal_media_com']:.4f} com a recalibração; {conf_b['decisoes_diferentes']} decisões "
          f"diferentes; maior mudança nas características {conf_b['maior_mudanca_nas_caracteristicas']:.1e}")

    destino = args.out_dir / "recalibracao"
    destino.mkdir(parents=True, exist_ok=True)
    normal = {"origem": "segmentos normais do Jung (os do treino do modelo final)", "n": ref["n"],
              "modelo": modelo_id, "splits": hash_splits, "features_commit": feat_commit, "ddof_desvio": 1,
              "media": ref["media"].tolist(), "desvio": ref["desvio"].tolist()}
    (destino / "normal_jung.json").write_text(json.dumps(normal, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    parametros = {"modelo": modelo_id, "splits": hash_splits, "features_commit": feat_commit,
                  "divisao_c2": divisao, "variantes": VARIANTES_RECAL, "piso_desvio": PISO_DESVIO,
                  "rotacoes_criterio": ROTACOES_CRITERIO, "fracao_deteccao": FRACAO_DETECCAO,
                  "confianca": CONFIANCA, "x_alarme": X_ALARME,
                  "registro": "Recalibrar a LDA pela normal da máquina nova (29/10, Definida)",
                  "criterio": "funciona: principal e controle sem o c0 com ≥ 12/15 a 10–30 Hz e alarme da "
                              "principal sem reprovar; detecção pelo nível: só a principal passa; 5 Hz à parte"}
    (destino / "metrics.json").write_text(json.dumps(
        {"parametros": parametros, "normal_referencia": normal, "resultado": resultado,
         "conferencia_protocolo_b": conf_b, "comparacao": comparacao, "leituras": res},
        ensure_ascii=False, indent=2,
        default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n", encoding="utf-8", newline="\n")
    print(f"\nsaídas: {destino}/")
    if args.sem_registro:
        print("(registry não alterado)")
        return 0

    blocos = lambda chave: ",".join(str(b) for b in divisao[chave])
    divisao_txt = (f"calib={blocos('blocos_calibracao')};descarte={blocos('blocos_descarte')};"
                   f"medida={blocos('blocos_medida')};falha={blocos('blocos_medida_falha')}")
    linhas, base = [], experimentos.next_exp_number(args.registry)
    for nome, r in res.items():
        met = {}
        for modo, rr in r["resultado"].items():
            a, d, f = rr["alarme"], rr["deteccao_criterio"], rr["deteccao_fora_do_criterio"]
            met.update({f"alarme_{modo}": f"{a['k']}/{a['n']}", f"alarme_sup95_{modo}": f"{a['limite_superior_95']:.4f}",
                        f"alarme_inf95_{modo}": f"{a['limite_inferior_95']:.4f}", f"decisao_{modo}": a["decisao"],
                        f"deteccao_{modo}": f"{rr['deteccao']['k']}/{rr['deteccao']['n']}",
                        f"deteccao_10a30hz_{modo}": f"{d['k']}/{d['n']}",
                        f"passa_10a30hz_{modo}": "sim" if d["passa"] else "nao",
                        f"deteccao_5hz_{modo}": f"{f['k']}/{f['n']}"})
        if nome == "C2_LDA_recal_medias13":
            met.update({"resultado": resultado, "conf_B_acc_bal_sem": f"{conf_b['acc_bal_media_sem']:.4f}",
                        "conf_B_acc_bal_com": f"{conf_b['acc_bal_media_com']:.4f}",
                        "conf_B_decisoes_diferentes": conf_b["decisoes_diferentes"]})
        linhas.append({
            "id": f"exp{base + len(linhas):03d}",
            "data": date.today().isoformat(),
            "etapa": "protocolo_c",
            "script": "validation/run_protocol_c.py",
            "git_commit": experimentos.git_short_hash(),
            "parametros": experimentos.kv({
                "protocolo": "C", "leitura": "C2", "detector": r["detector"], "variante": r["variante"],
                "papel": r["papel"].split(" ")[0], "referencia": f"normal_jung_{ref['n']}", "normal_c2": "por_rotacao",
                "piso_desvio": PISO_DESVIO if r["variante"] == "desvio" else None,
                "criterio": "12/15_a_10-30Hz", "modelo": modelo_id, "divisao_c2": divisao_txt,
                "splits": hash_splits, "features_commit": feat_commit, "fs_hz": config.FS_TRABALHO}),
            "dataset": f"hustmotor_som_H_BF_{config.FS_TRABALHO}Hz",
            "metricas": experimentos.kv(met),
            "responsavel": args.responsavel,
            "notas": ("Protocolo C, recalibração da LDA pela normal da máquina nova (Registro, 29/10): "
                      + {"decide": "variante principal, decide", "decide_junto": "controle sem o c0, decide junto",
                         "reporte": "reporte", "combinado": "D2 OU LDA recalibrada, papel na página da adoção do D2"}
                      [r["papel"].split(" ")[0]]
                      + (f"; {args.notas}" if args.notas else "")),
        })
    experimentos.append_registry(args.registry, linhas)
    print(f"registrado: {linhas[0]['id']}–{linhas[-1]['id']}")
    return 0


# ---------------------------------------------------------------------------
def _fmt(r: dict) -> str:
    a, d = r["alarme"], r["deteccao"]
    return (f"alarme {a['k']:>2}/{a['n']:<2} (≤ {a['limite_superior_95']:.0%}; {a['decisao']:<12})   "
            f"detecção {d['k']:>2}/{d['n']:<2} [{d['intervalo_95'][0]:.2f}, {d['intervalo_95'][1]:.2f}]")


def _por_rot(r: dict) -> str:
    return "  ".join(f"{v} Hz: {x['alarme']['k']}/{x['alarme']['n']} e {x['deteccao']['k']}/{x['deteccao']['n']}"
                     for v, x in r["por_rotacao"].items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hust-dir", type=Path, default=Path("data/processed/hustmotor/pcm_decimated/12800"))
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--features", type=Path, default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--modelo", type=Path, default=Path("reports/modelo_final/lda_final.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/validation/protocolo_c"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    ap.add_argument("--recalibracao", action="store_true",
                    help="só a recalibração da LDA pela normal da máquina nova (Registro, 29/10)")
    args = ap.parse_args()

    # ---- Jung (C1, LDA) -------------------------------------------------------------
    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        lista = P.sintetico()
        particoes = particao.gerar_particoes(lista)
        segmentos = particao.segmentos_de(particoes)
        clipes = {c.rotulo: c for c in lista}
        X_jung = np.vstack([mfcc26(clipes[s.rotulo].x[s.inicio:s.fim]) for s in segmentos])
        y_jung = np.array([s.binario for s in segmentos])
        p_lda = modelo_lda.parametros(modelo_lda.treinar(X_jung, y_jung), config.MFCC_N_COEFS)
        razoes = (3.0531, 4.9469)
        divisao = {"blocos_calibracao": [1, 2], "blocos_descarte": [3], "blocos_medida": [4, 5],
                   "blocos_medida_falha": [1, 2, 3, 4, 5]}
        segs = hust_sintetico(razoes)
        hash_splits, feat_commit, modelo_id = "sintetico", "sintetico", "sintetico"
    else:
        if not args.sem_registro and not args.responsavel:
            ap.error("--responsavel é obrigatório numa rodada registrada")
        lista = pcm_io.carregar_clipes(args.pcm_dir, fs_esperado=config.FS_TRABALHO)
        if pcm_io.verificar_integridade(lista):
            raise SystemExit("Abortado: os PCM do Jung não batem com o manifest versionado.")
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
        X_jung, feat_commit = np.load(args.features)["X"], manif.get("git_commit")
        y_jung = np.array([s.binario for s in segmentos])
        p_lda = json.loads(args.modelo.read_text(encoding="utf-8"))
        modelo_id = p_lda["id"]
        segs, conjunto = carregar_hust(args.hust_dir)
        rol = conjunto["rolamento"]
        razoes = (float(rol["razao_bpfo"]), float(rol["razao_bpfi"]))
        divisao = conjunto["divisao_c2"]

    # ---- HUST: características e máscaras --------------------------------------------
    X = np.vstack([mfcc26(s.x) for s in segs])
    classe = np.array([s.classe for s in segs])
    bloco = np.array([s.bloco for s in segs])
    normal, falha = classe == "H", classe == "BF"
    calib = normal & np.isin(bloco, divisao["blocos_calibracao"])
    med_normal = normal & np.isin(bloco, divisao["blocos_medida"])
    med_falha = falha & np.isin(bloco, divisao["blocos_medida_falha"])
    lda_falha = modelo_lda.prever(p_lda, X) == "falha"
    print(f"HUST: {len(segs)} segmentos; C2: {calib.sum()} de calibração, {med_normal.sum()} normais de "
          f"medida, {med_falha.sum()} de falha. Razões BPFO {razoes[0]:.4f}, BPFI {razoes[1]:.4f}.")

    if args.recalibracao:
        return rodar_recalibracao(args, segs, X, calib, med_normal, med_falha, X_jung, y_jung, p_lda,
                                  particao.folds_de(particoes, "B"), divisao, modelo_id, hash_splits,
                                  feat_commit)

    res = {}
    # LDA sem retreino: todos os segmentos
    res["LDA_sem_retreino"] = {"leitura": "LDA", "detector": "LDA", "rotacao": None,
                               "resultado": {"sozinho": resumir(segs, lda_falha, normal, falha)}}

    # C1: limiares do Jung
    normais_jung = np.flatnonzero(y_jung == "normal")
    bloco_jung = np.array([s.bloco for s in segmentos])
    lim_d2_jung = N.limiar_fora_do_bloco(X_jung, normais_jung, bloco_jung[normais_jung])
    d2_c1 = N.Mahalanobis().fit(X_jung[normais_jung]).pontuar(X) > lim_d2_jung
    res["C1_D2"] = {"leitura": "C1", "detector": "D2", "rotacao": None, "limiar": lim_d2_jung,
                    "resultado": leitura(segs, d2_c1, lda_falha, normal, falha)}

    segs_normais_jung = [segmentos[i] for i in normais_jung]
    cache_jung = N.preparar_espectros(clipes, segs_normais_jung, CAND)
    pont_jung = N.features_pente(cache_jung, np.full(len(cache_jung), EIXO_JUNG_HZ), CAND)[:, :2]
    lim_d1_jung = pont_jung.max(axis=0)
    pont = {m: pontuacoes_pente(segs, razoes, m) for m in ("nominal", "estimado")}
    for m in ("nominal", "estimado"):
        res[f"C1_D1_{m}"] = {"leitura": "C1", "detector": "D1", "rotacao": m, "limiar": lim_d1_jung.tolist(),
                             "nota": "limiar do Jung tirado com a rotação nominal do Jung (50,17 Hz) nos dois modos; "
                                     "a rotação estimada vale só no HUST" if m == "estimado" else None,
                             "resultado": leitura(segs, dispara_pente(pont[m], lim_d1_jung), lda_falha, normal, falha)}

    sint = N.features_sinteticas(clipes, segmentos, config.SEMENTE)
    ids_n = [segmentos[i].id for i in normais_jung]
    d4 = novo_modelo("lda", 2).fit(np.vstack([X_jung, np.array([sint[i] for i in ids_n])]),
                                   np.concatenate([y_jung, np.full(len(ids_n), "falha")]))
    d4_c1 = d4.predict(X) == "falha"
    res["C1_D4"] = {"leitura": "C1", "detector": "D4", "rotacao": None,
                    "resultado": {"sozinho": resumir(segs, d4_c1, normal, falha)}}

    # C2: normal do próprio HUST
    d2_c2, lim_d2, encolhimento = mahalanobis_c2(X, calib)
    res["C2_D2"] = {"leitura": "C2", "detector": "D2", "rotacao": None, "limiar": lim_d2,
                    "encolhimento_ledoit_wolf": encolhimento,
                    "alarme_esperado_pela_calibracao": alarme_esperado_pelo_maximo(int(calib.sum())),
                    "resultado": leitura(segs, d2_c2, lda_falha, med_normal, med_falha)}
    for m in ("nominal", "estimado"):
        lim = limiar_pente(pont[m][calib])
        n_cal_d1 = int((calib & ~np.isnan(pont[m]).any(axis=1)).sum())
        res[f"C2_D1_{m}"] = {"leitura": "C2", "detector": "D1", "rotacao": m, "limiar": lim.tolist(),
                             "alarme_esperado_pela_calibracao": alarme_esperado_pelo_maximo(n_cal_d1, 2),
                             "resultado": leitura(segs, dispara_pente(pont[m], lim), lda_falha, med_normal, med_falha)}

    # ---- saída ----------------------------------------------------------------------
    print("\nalarme = segmentos normais de medida que dispararam (≤ limite superior de 95 %);"
          " detecção = segmentos de falha [intervalo de 95 %]")
    for nome, r in res.items():
        for modo, rr in r["resultado"].items():
            print(f"  {nome:<18} {modo:<8} {_fmt(rr)}")
            print(f"  {'':<18} {'':<8} por rotação (alarme e detecção): {_por_rot(rr)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parametros = {"modelo": modelo_id, "splits": hash_splits, "features_commit": feat_commit,
                  "razoes_hust": razoes, "divisao_c2": divisao, "normal_c2": "junta",
                  "espectro_d1": CAND.nome, "eixo_jung_hz": EIXO_JUNG_HZ, "eixo_estimado_rel": EIXO_ESTIMADO_REL,
                  "confianca": CONFIANCA, "x_alarme": X_ALARME, "d4_semente": config.SEMENTE,
                  "decisao": "pelo combinado (OU a LDA); o detector sozinho é só reporte",
                  "limitacoes_c2": [
                      "limiar = máximo de poucas pontuações: alarme esperado ~1/(n+1) só pelo tamanho da "
                      "calibração (ver alarme_esperado_pela_calibracao); a seção 2 conta com ~47 pontuações",
                      "D2: 7 segmentos para 26 dimensões em cada ajuste fora do bloco; Ledoit–Wolf com "
                      "encolhimento em encolhimento_ledoit_wolf",
                      "calibração e medida saem do mesmo arquivo de cada rotação: aquisições separadas, mas não "
                      "independentes; o alarme falso medido fica otimista"]}
    (args.out_dir / "metrics.json").write_text(json.dumps(
        {"parametros": parametros, "leituras": res}, ensure_ascii=False, indent=2,
        default=lambda o: list(o) if isinstance(o, tuple) else float(o)) + "\n", encoding="utf-8", newline="\n")
    print(f"\nsaídas: {args.out_dir}/")
    if args.sem_registro:
        print("(registry não alterado)")
        return 0

    blocos = lambda chave: ",".join(str(b) for b in divisao[chave])
    divisao_txt = (f"calib={blocos('blocos_calibracao')};descarte={blocos('blocos_descarte')};"
                   f"medida={blocos('blocos_medida')};falha={blocos('blocos_medida_falha')}")
    linhas, base = [], experimentos.next_exp_number(args.registry)
    for nome, r in res.items():
        met = {}
        for modo, rr in r["resultado"].items():
            a, d = rr["alarme"], rr["deteccao"]
            met.update({f"alarme_{modo}": f"{a['k']}/{a['n']}", f"alarme_sup95_{modo}": f"{a['limite_superior_95']:.4f}",
                        f"decisao_{modo}": a["decisao"], f"deteccao_{modo}": f"{d['k']}/{d['n']}"})
        linhas.append({
            "id": f"exp{base + len(linhas):03d}",
            "data": date.today().isoformat(),
            "etapa": "protocolo_c",
            "script": "validation/run_protocol_c.py",
            "git_commit": experimentos.git_short_hash(),
            "parametros": experimentos.kv({
                "protocolo": "C", "leitura": r["leitura"], "detector": r["detector"], "rotacao": r["rotacao"],
                "normal_c2": "junta" if r["leitura"] == "C2" else None, "modelo": modelo_id,
                "divisao_c2": divisao_txt if r["leitura"] == "C2" else None,
                "espectro": CAND.nome if r["detector"] == "D1" else None,
                "splits": hash_splits, "features_commit": feat_commit, "fs_hz": config.FS_TRABALHO}),
            "dataset": f"hustmotor_som_H_BF_{config.FS_TRABALHO}Hz",
            "metricas": experimentos.kv(met),
            "responsavel": args.responsavel,
            "notas": (f"Protocolo C, {r['leitura']}: alarme falso com limite superior de 95 % (Clopper–Pearson)"
                      + ("; C2 só consegue reprovar, e decide o combinado (OU a LDA)" if r["leitura"] == "C2" else "")
                      + (f"; {args.notas}" if args.notas else "")),
        })
    experimentos.append_registry(args.registry, linhas)
    print(f"registrado: {linhas[0]['id']}–{linhas[-1]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
