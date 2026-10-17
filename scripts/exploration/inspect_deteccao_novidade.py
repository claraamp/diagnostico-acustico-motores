#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_deteccao_novidade.py — detecção de novidade para a bpfo_0.3mm (sensibilidade no B)
=========================================================================================

Segue o `docs/criterio_deteccao_novidade.md` (versão 2): roda, no Protocolo B, os
quatro candidatos do critério e diz quais ficam elegíveis. A especificidade em
outra máquina (Protocolo C, HUSTmotor) fica para depois; aqui nada é adotado.

Candidatos
----------
D1   pente (pontuação de BPFO e de BPFI) com limiar da normal, OU a LDA atual
D2   Mahalanobis no MFCC (26), OU a LDA atual
D3   Mahalanobis no MFCC + pente (30), OU a LDA atual
D4   LDA retreinada com falha sintética de linhas estreitas (troca os pesos)

O pente roda em dois espectros: B (FFT de 4.096 pontos a 12,8 kHz, a do estudo do
pente) e D1b (decima para 6,4 kHz e FFT de 2.048 pontos: a mesma resolução com
metade da RAM, dentes até 2,8 kHz). Os dois com a rotação nominal (50,17 Hz, a
que decide) e com a estimada no segmento (só reporte).

Limiar (D1, D2, D3)
-------------------
Em cada fold, os segmentos normais de treino ficam em 5 blocos. Para cada bloco, o
detector é ajustado nos outros 4 e pontua o deixado de fora; o limiar é a maior
dessas pontuações, e o detector final é ajustado nos 5. No D1 não há ajuste (a
pontuação do pente é direta): o limiar é a maior pontuação da normal de treino,
um por série. Mahalanobis: padronização e covariância de Ledoit–Wolf ajustadas só
na normal.

D4: injeção da série sintética
------------------------------
Na forma de onda (`int16 / 32767`), antes da cadeia, só em segmentos normais de
treino de cada fold: família BPFO ou BPFI (meio a meio), f₀ em ±3 % da razão
cinemática × 50,17 Hz, de 10 a 25 harmônicos até 5,6 kHz, fase aleatória, e cada
dente entre 10 e 40 dB (uniforme em dB) acima do fundo local do próprio segmento
(mediana do espectro de 1 s, Hann, entre 3 e 10 Hz do dente). A cópia sintética
de um segmento é a mesma em todos os folds (semente por segmento), e só entra no
treino de um fold se o segmento de origem for normal de treino nele.

Controles
---------
- D4: permutação por bloco dos rótulos de treino (20 sorteios); as cópias
  sintéticas continuam "falha". Passa com acurácia balanceada média ≤ 0,55.
- D1 e D3: pente fixo (sem a busca de ±3 %) em frequências sorteadas (20 sorteios),
  razões entre 2 e 7 ×
  rotação, fora de ±6 % de BPFO, de BPFI e das frações p/q delas (p, q ≤ 3, que
  alinhariam o pente com parte dos harmônicos da falha) e a mais de 0,15 de um inteiro.
  Controle relativo (critério v3): passa se a fração de segmentos da bpfo_0.3mm
  detectados pelo detector sozinho (sem a LDA) com o pente de rolamento superar
  em pelo menos 0,5 a média dos sorteios. Um limite absoluto não serve: com linhas
  fortes, até um pente fixo em frequência sorteada esbarra em algumas delas por
  acaso (o auto-teste sintético reprovava um detector correto).
- D3: só segue se a bpfo_0.3mm passar a do D2 por mais de 0,02; senão, o pente não
  acrescenta nada ao MFCC e é descartado pelo custo. O controle é o mesmo do D1.
- D2: sem controle no B (mede "diferente da normal" por construção); fica para o
  Protocolo C.

Elegível: especificidade do combinado ≥ 0,95, outras três falhas ≥ 0,95,
controle aprovado e, no D3, superar o D2. A rotação estimada é só reporte. Candidato à adoção (falta o C2): elegível e bpfo_0.3mm ≥ 0,80.

Uso
---
    python scripts/exploration/inspect_deteccao_novidade.py --sintetico      # auto-teste
    python scripts/exploration/inspect_deteccao_novidade.py --responsavel <nome>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/

import numpy as np
from sklearn.covariance import LedoitWolf

import config
import dsp
import experimentos
import pcm_io
from validation import metricas, particao
from validation.run_protocol import novo_modelo, permutar_treino_por_bloco

_spec = importlib.util.spec_from_file_location(
    "inspect_comb_features", Path(__file__).resolve().with_name("inspect_comb_features.py"))
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P          # o @dataclass do módulo procura o próprio módulo aqui
_spec.loader.exec_module(P)

# ---------------------------------------------------------------------------
# Parâmetros (docs/criterio_deteccao_novidade.md, versão 2)
# ---------------------------------------------------------------------------
EIXO_NOMINAL_HZ = P.EIXO_ARTIGO_HZ           # 50,17 Hz, Jung et al. (2023)
MODOS_EIXO = ("nominal", "estimado")         # o primeiro decide
CAND_B = P.CANDIDATOS[1]
CAND_D1B = P.Candidato("D1b_fft2048_6400hz_3blocos", 2048, 2048, 2048, 3,
                       "decima para 6,4 kHz e FFT de 2.048 pontos, média de 3 blocos",
                       decimar=2, fmax=2800.0)
ESPECTROS = (CAND_B, CAND_D1B)

D4_HARMONICOS = (10, 25)
D4_DB = (10.0, 40.0)
D4_FUNDO_HZ = (3.0, 10.0)
D4_FMAX = 5600.0
D4_SEMENTES = 5

N_CONTROLES = 20
CTRL_RAZOES = (2.0, 7.0)
CTRL_LONGE_BPF = 0.06
CTRL_LONGE_INTEIRO = 0.15
FRACOES_BPF = tuple(sorted({p / q for p in (1, 2, 3) for q in (1, 2, 3)}))
CTRL_MARGEM = 0.50              # detecção real − detecção com o pente sorteado
D3_MARGEM_SOBRE_D2 = 0.02
PERM_MAX = 0.55

ESPEC_MIN = 0.95
OUTRAS_MIN = 0.95
BPFO03_MIN = 0.80
FALHA_ALVO = "bpfo_0.3mm"


# ---------------------------------------------------------------------------
# Pente: espectro uma vez por segmento, pontuação por razão
# ---------------------------------------------------------------------------
def preparar_espectros(clipes: dict, segmentos, c) -> list[tuple]:
    """(f, pico, fundo) de cada segmento, para o candidato de espectro `c`."""
    out = []
    for s in segmentos:
        f, p = P.espectro(clipes[s.rotulo].x[s.inicio:s.fim], c)
        pico, fundo = P.pico_e_fundo(f, p, c.fundo_hz)
        out.append((f, pico, fundo))
    return out


def eixos(cache: list[tuple], modo: str) -> np.ndarray:
    if modo == "nominal":
        return np.full(len(cache), EIXO_NOMINAL_HZ)
    if modo == "estimado":
        return np.array([P.estimar_eixo(f, pico, fundo) for f, pico, fundo in cache])
    raise ValueError(f"modo do eixo desconhecido: {modo}")


def features_pente(cache: list[tuple], f_eixo: np.ndarray, c, razoes=(P.RAZAO_BPFO, P.RAZAO_BPFI)) -> np.ndarray:
    """As 4 características do pente (pontuação e fração de cada série) por segmento."""
    X = np.empty((len(cache), 4))
    for i, (f, pico, fundo) in enumerate(cache):
        r = [P.pente(f, pico, fundo, f_eixo[i], razao, c) for razao in razoes]
        X[i] = [r[0]["pontuacao"], r[1]["pontuacao"], r[0]["fracao"], r[1]["fracao"]]
    return X


def pente_fixo(f, pico, fundo, f_eixo: float, razao: float, c) -> tuple[float, float]:
    """
    Pontuação e fração do pente num f₀ fixo (razão × rotação), sem a busca de ±3 %.
    É o pente do controle: com a busca, um pente em qualquer frequência acaba
    alinhando parte dos dentes com as linhas fortes de uma falha, e passa a medir
    "há linhas estreitas", e não "são linhas de rolamento".
    """
    res = f[1] - f[0]
    f0 = razao * f_eixo
    dentes = f0 * np.arange(1, P.k_max(f0, c.fmax) + 1)
    harm = np.round(dentes / f_eixo) * f_eixo
    dentes = dentes[(np.abs(dentes - harm) >= c.mascara_hz) & (dentes <= c.fmax)]
    if len(dentes) == 0:
        raise ValueError(f"nenhum dente válido para f₀ = {f0:.1f} Hz")
    b = np.clip(np.round(dentes / res).astype(int), 1, len(pico) - 2)
    sal = 10 * np.log10((pico[b] + 1e-30) / (fundo[b] + 1e-30))
    return float(sal.mean()), float(np.mean(sal >= P.LIMIAR_DENTE_DB))


def features_pente_fixo(cache: list[tuple], f_eixo: np.ndarray, c, razoes) -> np.ndarray:
    """As 4 características do pente fixo (controle), no mesmo formato de `features_pente`."""
    X = np.empty((len(cache), 4))
    for i, (f, pico, fundo) in enumerate(cache):
        (p1, q1), (p2, q2) = (pente_fixo(f, pico, fundo, f_eixo[i], r, c) for r in razoes)
        X[i] = [p1, p2, q1, q2]
    return X


def razoes_controle(n: int, seed: int = config.SEMENTE) -> list[tuple[float, float]]:
    """Pares de razões (× rotação) que não são de rolamento nem múltiplos do eixo."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        par = []
        while len(par) < 2:
            r = rng.uniform(*CTRL_RAZOES)
            # longe de BPFO e BPFI e das frações simples delas (p/q, p e q até 3): um pente
            # em 2/3 da BPFO, por exemplo, acerta um dente em cada três harmônicos da falha
            longe_bpf = all(abs(r / (b * fr) - 1) > CTRL_LONGE_BPF
                            for b in (P.RAZAO_BPFO, P.RAZAO_BPFI) for fr in FRACOES_BPF)
            if longe_bpf and abs(r - round(r)) > CTRL_LONGE_INTEIRO:
                par.append(float(r))
        out.append(tuple(par))
    return out


# ---------------------------------------------------------------------------
# D4: série sintética de linhas estreitas na forma de onda
# ---------------------------------------------------------------------------
def injetar_serie(x: np.ndarray, rng: np.random.Generator, fs: float = config.FS_TRABALHO,
                  eixo: float = EIXO_NOMINAL_HZ) -> tuple[np.ndarray, dict]:
    """Soma a `x` uma série harmônica sintética (ver o docstring do módulo)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    familia, razao = (("bpfo", P.RAZAO_BPFO), ("bpfi", P.RAZAO_BPFI))[int(rng.integers(2))]
    f0 = razao * eixo * (1 + rng.uniform(-P.BUSCA_REL, P.BUSCA_REL))
    n_harm = min(int(rng.integers(D4_HARMONICOS[0], D4_HARMONICOS[1] + 1)), int(D4_FMAX // f0))
    potencia = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    res = fs / n
    lo, hi = int(np.ceil(D4_FUNDO_HZ[0] / res)), int(np.floor(D4_FUNDO_HZ[1] / res))
    t = np.arange(n) / fs
    y = x.copy()
    db = rng.uniform(*D4_DB, size=n_harm)
    fases = rng.uniform(0, 2 * np.pi, size=n_harm)
    for k in range(1, n_harm + 1):
        b = int(round(k * f0 / res))
        vizinhos = np.r_[b - hi:b - lo + 1, b + lo:b + hi + 1]
        fundo = float(np.median(potencia[np.clip(vizinhos, 0, len(potencia) - 1)]))
        # pico de uma senoide de amplitude a com janela de Hann: |X| = a·N/4
        a = 4.0 / n * np.sqrt(fundo * 10 ** (db[k - 1] / 10))
        y += a * np.sin(2 * np.pi * k * f0 * t + fases[k - 1])
    return y, {"familia": familia, "f0": float(f0), "n_harmonicos": n_harm}


def features_sinteticas(clipes: dict, segmentos, semente: int) -> dict[int, np.ndarray]:
    """Por segmento normal: as 26 características da cópia com a série sintética."""
    out = {}
    for s in segmentos:
        if s.binario != "normal":
            continue
        rng = np.random.default_rng([semente, s.id])
        y, _ = injetar_serie(clipes[s.rotulo].x[s.inicio:s.fim], rng)
        m = dsp.mfcc(y, config.FS_TRABALHO)
        out[s.id] = np.concatenate([m.mean(axis=0), m.std(axis=0)])
    return out


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------
class Mahalanobis:
    """Distância de Mahalanobis (ao quadrado) à normal, com padronização e Ledoit–Wolf."""

    def fit(self, X: np.ndarray) -> "Mahalanobis":
        self.media = X.mean(axis=0)
        self.escala = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        self.lw = LedoitWolf().fit((X - self.media) / self.escala)
        return self

    def pontuar(self, X: np.ndarray) -> np.ndarray:
        return self.lw.mahalanobis((X - self.media) / self.escala)


def limiar_fora_do_bloco(X: np.ndarray, normais: np.ndarray, blocos: np.ndarray) -> float:
    """Maior pontuação de Mahalanobis da normal, cada bloco pontuado por um ajuste sem ele."""
    pont = []
    for b in np.unique(blocos):
        ajuste, fora = normais[blocos != b], normais[blocos == b]
        pont.append(Mahalanobis().fit(X[ajuste]).pontuar(X[fora]))
    return float(np.max(np.concatenate(pont)))


def _lda(X_tr, y_tr, X_te):
    mdl = novo_modelo("lda", 2)
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)


def rodar_b(folds, segmentos, y, X_mfcc, detector: str, X_pente=None, sinteticas=None,
            rng: np.random.Generator | None = None) -> dict:
    """
    Protocolo B de um candidato. Devolve o resumo do combinado e, à parte, a fração
    de alarmes do detector sozinho na normal de teste e a detecção da bpfo_0.3mm.
    Com `rng` (só no D4), os rótulos de treino são permutados por bloco.
    """
    rot = np.array([s.rotulo for s in segmentos])
    bloco = np.array([s.bloco for s in segmentos])
    resultados, alarmes_normal, det_alvo = [], [], []
    for f in folds:
        tr, te = np.asarray(f.treino), np.asarray(f.teste)
        y_tr = permutar_treino_por_bloco(segmentos, f.treino, y, rng) if rng is not None else y[tr]
        normais = tr[y[tr] == "normal"]          # normal de treino (rótulos verdadeiros)
        if detector == "D4":
            X_aug = np.vstack([X_mfcc[tr], np.array([sinteticas[i] for i in normais])])
            y_aug = np.concatenate([y_tr, np.full(len(normais), "falha")])
            pred = _lda(X_aug, y_aug, X_mfcc[te])
            dispara = np.zeros(len(te), dtype=bool)
        else:
            pred = _lda(X_mfcc[tr], y_tr, X_mfcc[te])
            if detector == "D1":
                lim = X_pente[normais][:, :2].max(axis=0)
                dispara = (X_pente[te][:, :2] > lim).any(axis=1)
            else:
                X = X_mfcc if detector == "D2" else np.hstack([X_mfcc, X_pente])
                lim = limiar_fora_do_bloco(X, normais, bloco[normais])
                dispara = Mahalanobis().fit(X[normais]).pontuar(X[te]) > lim
            pred = np.where(dispara, "falha", pred)
        m = metricas.metricas_binarias(y[te], pred)
        resultados.append({"nome": f.nome, "info": f.info, **m})
        alarmes_normal.append(dispara[y[te] == "normal"])
        det_alvo.append(dispara[rot[te] == FALHA_ALVO])
    resumo = metricas.resumo_protocolo_b(resultados)
    fracao = lambda partes: float(np.mean(np.concatenate(partes))) if sum(map(len, partes)) else float("nan")
    return {"resumo": resumo, "alarme_detector_normal": fracao(alarmes_normal),
            "deteccao_detector_alvo": fracao(det_alvo)}


def avaliar(r: dict, controle: dict | None, decide: bool = True, d2_alvo: float | None = None) -> dict:
    """
    Elegibilidade e situação pela regra de escolha do critério (seção 6). `decide`
    falso (rotação estimada) vira só reporte; `d2_alvo`, no D3, é a bpfo_0.3mm do D2.
    """
    pf = r["resumo"]["por_falha"]
    outras = min(pf[f]["acuracia_balanceada"] for f in pf if f != FALHA_ALVO)
    alvo = pf[FALHA_ALVO]["acuracia_balanceada"]
    crit = {
        "especificidade >= 0,95": r["resumo"]["especificidade_media"] >= ESPEC_MIN,
        "outras falhas >= 0,95": outras >= OUTRAS_MIN,
        "controle": None if controle is None else bool(controle["passa"]),
        "supera o D2": None if d2_alvo is None else bool(alvo > d2_alvo + D3_MARGEM_SOBRE_D2),
    }
    elegivel = (crit["especificidade >= 0,95"] and crit["outras falhas >= 0,95"]
                and crit["controle"] is not False and crit["supera o D2"] is not False)
    if not decide:
        situacao = "só reporte (rotação estimada)"
    elif crit["supera o D2"] is False:
        situacao = "descartado: não supera o D2"
    else:
        situacao = ("candidato à adoção (falta o Protocolo C)" if elegivel and alvo >= BPFO03_MIN else
                    "elegível, mas não detecta a bpfo_0.3mm" if elegivel else "não elegível")
    return {**crit, "outras_falhas_min": outras, "bpfo_0.3mm": alvo, "elegivel": elegivel and decide,
            "situacao": situacao}


# ---------------------------------------------------------------------------
def _linha(nome: str, r: dict, aval: dict, ctrl: str) -> str:
    rs, pf = r["resumo"], r["resumo"]["por_falha"]
    return (f"  {nome:<26}{rs['acuracia_balanceada_media']:>7.3f}{rs['especificidade_media']:>8.3f}"
            + "".join(f"{pf[f]['acuracia_balanceada']:>9.3f}" for f in pf)
            + f"{r['alarme_detector_normal']:>8.3f}{ctrl:>10}  {aval['situacao']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcm-dir", type=Path, default=Path(config.dir_pcm_decimado(config.FS_TRABALHO)))
    ap.add_argument("--splits", type=Path, default=Path(config.arquivo_splits(config.FS_TRABALHO)))
    ap.add_argument("--features", type=Path,
                    default=Path(config.dir_features(config.FS_TRABALHO)) / "mfcc_features.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/exploration/novidade"))
    ap.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    ap.add_argument("--sem-registro", action="store_true")
    ap.add_argument("--sintetico", action="store_true", help="auto-teste (implica --sem-registro)")
    ap.add_argument("--responsavel", default="")
    ap.add_argument("--notas", default="")
    args = ap.parse_args()

    if args.sintetico:
        args.sem_registro = True
        args.out_dir = args.out_dir / "sintetico"
        lista = P.sintetico()
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
    y = np.array([s.binario for s in segmentos])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(segmentos)} segmentos, {len(folds)} folds do B; rotação nominal {EIXO_NOMINAL_HZ} Hz")

    cab = (f"  {'candidato':<26}{'acc.bal':>7}{'espec.':>8}"
           + "".join(f"{f[:9]:>9}" for f in ("bpfi_0.3", "bpfi_1.0", "bpfo_0.3", "bpfo_1.0"))
           + f"{'alarme':>8}{'controle':>10}  situação")
    linhas_saida, resultado = [], {}

    def guardar(nome, r, ctrl, extra, decide=True, d2_alvo=None):
        aval = avaliar(r, ctrl, decide, d2_alvo)
        ctrl_txt = "n/a" if ctrl is None else f"{ctrl['valor']:.3f}" + ("✓" if ctrl["passa"] else "✗")
        linhas_saida.append(_linha(nome, r, aval, ctrl_txt))
        resultado[nome] = {**extra, **r, "controle": ctrl, "avaliacao": aval}

    # D2: Mahalanobis no MFCC
    r2 = rodar_b(folds, segmentos, y, X_mfcc, "D2")
    d2_alvo = r2["resumo"]["por_falha"][FALHA_ALVO]["acuracia_balanceada"]
    guardar("D2_mahalanobis_mfcc", r2, None, {"detector": "D2", "espectro": None, "rotacao": None})

    # D1 e D3, nos dois espectros e nos dois modos de rotação
    controles_razoes = razoes_controle(N_CONTROLES)
    for c in ESPECTROS:
        cache = preparar_espectros(clipes, segmentos, c)
        for modo in MODOS_EIXO:
            f_eixo = eixos(cache, modo)
            X_pente = features_pente(cache, f_eixo, c)
            decide = modo == "nominal"
            for det in ("D1", "D3"):
                r = rodar_b(folds, segmentos, y, X_mfcc, det, X_pente)
                ctrl = None
                if decide:
                    vals = [rodar_b(folds, segmentos, y, X_mfcc, det,
                                    features_pente_fixo(cache, f_eixo, c, par))["deteccao_detector_alvo"]
                            for par in controles_razoes]
                    real = r["deteccao_detector_alvo"]
                    ctrl = {"tipo": "pente fixo em frequências sorteadas (relativo)", "valor": float(np.mean(vals)),
                            "max": float(np.max(vals)), "deteccao_real": real, "margem": CTRL_MARGEM,
                            "passa": real - float(np.mean(vals)) >= CTRL_MARGEM, "razoes": controles_razoes}
                nome = f"{det}_{'pente' if det == 'D1' else 'mahalanobis_mfcc_pente'}_{c.nome.split('_')[0]}_{modo}"
                guardar(nome, r, ctrl, {"detector": det, "espectro": c.nome, "rotacao": modo,
                                        "custo_stm32": P.custo_stm32(c)},
                        decide=decide, d2_alvo=d2_alvo if det == "D3" else None)

    # D4: LDA com falha sintética (semente principal + dispersão)
    sint = {k: features_sinteticas(clipes, segmentos, config.SEMENTE + k) for k in range(D4_SEMENTES)}
    r4 = rodar_b(folds, segmentos, y, X_mfcc, "D4", sinteticas=sint[0])
    rng = np.random.default_rng(config.SEMENTE)
    perm = [rodar_b(folds, segmentos, y, X_mfcc, "D4", sinteticas=sint[0], rng=rng)["resumo"]["acuracia_balanceada_media"]
            for _ in range(N_CONTROLES)]
    disp = [rodar_b(folds, segmentos, y, X_mfcc, "D4", sinteticas=sint[k])["resumo"]["por_falha"][FALHA_ALVO]["acuracia_balanceada"]
            for k in range(D4_SEMENTES)]
    guardar("D4_lda_falha_sintetica", r4,
            {"tipo": "permutação por bloco", "valor": float(np.mean(perm)), "max": float(np.max(perm)),
             "limite": PERM_MAX, "passa": float(np.mean(perm)) <= PERM_MAX},
            {"detector": "D4", "espectro": None, "rotacao": "nominal (só no treino)",
             "bpfo_0.3mm_por_semente": disp})

    print("\nProtocolo B, cada candidato OU a LDA (D4: a LDA retreinada); alarme = detector sozinho na normal")
    print(cab)
    print("\n".join(linhas_saida))
    print(f"\nD4, bpfo_0.3mm por semente: " + ", ".join(f"{v:.3f}" for v in resultado["D4_lda_falha_sintetica"]["bpfo_0.3mm_por_semente"]))

    parametros = {"eixo_nominal_hz": EIXO_NOMINAL_HZ, "espectros": [c.nome for c in ESPECTROS],
                  "d4_harmonicos": D4_HARMONICOS, "d4_db": D4_DB, "d4_fundo_hz": D4_FUNDO_HZ,
                  "d4_sementes": D4_SEMENTES, "n_controles": N_CONTROLES, "ctrl_razoes": CTRL_RAZOES,
                  "ctrl_longe_bpf": CTRL_LONGE_BPF, "ctrl_longe_inteiro": CTRL_LONGE_INTEIRO,
                  "ctrl_margem": CTRL_MARGEM, "d3_margem_sobre_d2": D3_MARGEM_SOBRE_D2, "perm_max": PERM_MAX, "espec_min": ESPEC_MIN,
                  "outras_min": OUTRAS_MIN, "bpfo03_min": BPFO03_MIN, "splits": hash_splits,
                  "features_commit": feat_commit}
    (args.out_dir / "metrics.json").write_text(json.dumps(
        {"parametros": parametros, "candidatos": resultado}, ensure_ascii=False, indent=2,
        default=lambda o: list(o) if isinstance(o, tuple) else str(o)) + "\n", encoding="utf-8", newline="\n")
    print(f"\nsaídas: {args.out_dir}/")
    if args.sem_registro:
        print("(registry não alterado)")
        return 0

    linhas = []
    base = experimentos.next_exp_number(args.registry)
    for nome, r in resultado.items():
        rs, av, ct = r["resumo"], r["avaliacao"], r["controle"]
        linhas.append({
            "id": f"exp{base + len(linhas):03d}",
            "data": date.today().isoformat(),
            "etapa": "deteccao_novidade",
            "script": "exploration/inspect_deteccao_novidade.py",
            "git_commit": experimentos.git_short_hash(),
            "parametros": experimentos.kv({
                "protocolo": "B", "tarefa": "binario", "candidato": nome, "detector": r["detector"],
                "espectro": r["espectro"], "rotacao": r["rotacao"], "combinacao": "OU_LDA" if r["detector"] != "D4" else "retreino_LDA",
                "limiar": "max_normal_fora_do_bloco" if r["detector"] in ("D2", "D3") else
                          ("max_normal_treino" if r["detector"] == "D1" else None),
                "splits": hash_splits, "fs_hz": config.FS_TRABALHO, "segmento_s": config.SEGMENTO_S,
                "mfcc_janela_ms": config.MFCC_WINDOW_MS, "mfcc_hop_ms": config.MFCC_HOP_MS,
                "mfcc_n_mels": config.MFCC_N_MELS, "mfcc_n_coefs": config.MFCC_N_COEFS,
                "features_commit": feat_commit, "aumento": "nenhum" if r["detector"] != "D4" else "falha_sintetica"}),
            "dataset": f"jung2023_acustico_0Nm_{config.FS_TRABALHO}Hz",
            "metricas": experimentos.kv({
                "acc_bal_media": f"{rs['acuracia_balanceada_media']:.4f}",
                "espec_media": f"{rs['especificidade_media']:.4f}",
                **{f"acc_bal_{f}": f"{t['acuracia_balanceada']:.4f}" for f, t in rs["por_falha"].items()},
                "alarme_detector_normal": f"{r['alarme_detector_normal']:.4f}",
                "controle": None if ct is None else f"{ct['valor']:.4f}",
                "controle_passa": None if ct is None else ("sim" if ct["passa"] else "nao"),
                "elegivel": "sim" if av["elegivel"] else "nao"}),
            "responsavel": args.responsavel,
            "notas": (f"detecção de novidade, critério v2: {av['situacao']}; sensibilidade no B, "
                      "especificidade só no Protocolo C" + (f"; {args.notas}" if args.notas else "")),
        })
    experimentos.append_registry(args.registry, linhas)
    print(f"registrado: {linhas[0]['id']}–{linhas[-1]['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
