"""
lda.py — LDA final do detector binário e seus parâmetros num formato neutro.

O `05_train_classifier.py` usa este módulo para treinar o modelo que vai para o
firmware. O modelo é o mesmo que o Protocolo B validou (`run_protocol.novo_modelo`:
padronização + LDA com priors uniformes), treinado uma única vez com todos os
segmentos. Ele não mede desempenho: o desempenho é o do Protocolo B (exp015).

Os parâmetros saem em JSON, sem depender do sklearn, em duas formas equivalentes:

- **padronizada**: z = (x − média) / escala, e escore = coef · z + intercepto;
- **dobrada**: escore = pesos · x + bias, com pesos = coef / escala e
  bias = intercepto − Σ coef · média / escala. É a forma que o firmware usa:
  26 multiplicações-acumulações por segmento, sem guardar média e escala.

Convenção do escore: **escore ≥ 0 → falha**. O sklearn ordena as classes em ordem
alfabética (`falha`, `normal`) e o seu `decision_function` é positivo para a
segunda; aqui o sinal é invertido para que o escore positivo seja a falha, a mesma
convenção do eixo da LDA no relatório. No empate exato, o sklearn binário escolhe
`classes_[0]` (`decision_function > 0` é a única condição para `classes_[1]`), que é
`falha`; o JSON registra essa classe em `escore_zero`, e `prever` a segue, para que o
firmware decida exatamente como o modelo validado. (Até a versão 1 do formato, o
docstring dizia que o empate era normal, o que não batia com o sklearn.)

Este módulo não é executável: é importado.
"""

from __future__ import annotations

import numpy as np

from validation.run_protocol import novo_modelo

CLASSE_POSITIVA = "falha"
CLASSE_NEGATIVA = "normal"
VERSAO_FORMATO = 2


def nomes_features(n_coefs: int) -> list[str]:
    """Ordem das colunas de X, a mesma do `04` (`resumo_mfcc`): médias, depois desvios."""
    return [f"media_c{i}" for i in range(n_coefs)] + [f"desvio_c{i}" for i in range(n_coefs)]


def treinar(X: np.ndarray, y: np.ndarray):
    """Padronização + LDA do protocolo, ajustados em todos os segmentos."""
    classes = set(np.unique(y).tolist())
    if classes != {CLASSE_POSITIVA, CLASSE_NEGATIVA}:
        raise ValueError(f"o modelo final é binário ({CLASSE_NEGATIVA}/{CLASSE_POSITIVA}); "
                         f"recebi as classes {sorted(classes)}")
    modelo = novo_modelo("lda", 2)
    modelo.fit(X, y)
    return modelo


def parametros(modelo, n_coefs: int) -> dict:
    """Parâmetros do pipeline ajustado, nas duas formas, com escore ≥ 0 → falha."""
    scaler, lda = modelo[0], modelo[-1]
    classes = list(lda.classes_)
    sinal = 1.0 if classes[1] == CLASSE_POSITIVA else -1.0
    # empate: o sklearn só escolhe classes_[1] com decision_function > 0
    escore_zero = str(classes[0])
    coef = sinal * lda.coef_.ravel()
    intercepto = float(sinal * lda.intercept_[0])
    media, escala = scaler.mean_, scaler.scale_
    pesos = coef / escala
    bias = intercepto - float(np.sum(coef * media / escala))
    nomes = nomes_features(n_coefs)
    if len(nomes) != len(coef):
        raise ValueError(f"esperava {len(nomes)} features ({n_coefs} coeficientes), "
                         f"o modelo tem {len(coef)}")
    return {
        "versao_formato": VERSAO_FORMATO,
        "modelo": "lda",
        "escore": ("escore >= 0 → falha; escore < 0 → normal" if escore_zero == CLASSE_POSITIVA
                   else "escore > 0 → falha; escore <= 0 → normal"),
        "classe_positiva": CLASSE_POSITIVA,
        "escore_zero": escore_zero,
        "priors": [float(p) for p in lda.priors_],
        "features": nomes,
        "padronizada": {
            "media": media.tolist(),
            "escala": escala.tolist(),
            "coef": coef.tolist(),
            "intercepto": intercepto,
        },
        "dobrada": {"pesos": pesos.tolist(), "bias": bias},
    }


def escore_dobrado(p: dict, X: np.ndarray) -> np.ndarray:
    d = p["dobrada"]
    return np.asarray(X, dtype=float) @ np.asarray(d["pesos"]) + d["bias"]


def escore_padronizado(p: dict, X: np.ndarray) -> np.ndarray:
    d = p["padronizada"]
    z = (np.asarray(X, dtype=float) - np.asarray(d["media"])) / np.asarray(d["escala"])
    return z @ np.asarray(d["coef"]) + d["intercepto"]


def prever(p: dict, X: np.ndarray) -> np.ndarray:
    """Previsão só com o JSON, como o firmware fará (forma dobrada), com o empate do sklearn."""
    s = escore_dobrado(p, X)
    falha = s >= 0 if p["escore_zero"] == CLASSE_POSITIVA else s > 0
    return np.where(falha, CLASSE_POSITIVA, CLASSE_NEGATIVA)


def conferir(modelo, p: dict, X: np.ndarray) -> dict:
    """
    Confere que o JSON reproduz o sklearn: as duas formas dão o mesmo escore que o
    `decision_function` (com o sinal da convenção) e a mesma previsão em todos os
    segmentos. Não é medida de desempenho: os segmentos são os do próprio treino.
    """
    lda = modelo[-1]
    sinal = 1.0 if list(lda.classes_)[1] == CLASSE_POSITIVA else -1.0
    ref = sinal * modelo.decision_function(X)
    s_pad, s_dob = escore_padronizado(p, X), escore_dobrado(p, X)
    escala = max(1.0, float(np.max(np.abs(ref))))
    return {
        "max_dif_escore_padronizado": float(np.max(np.abs(s_pad - ref))),
        "max_dif_escore_dobrado": float(np.max(np.abs(s_dob - ref))),
        "max_dif_relativa": float(max(np.max(np.abs(s_pad - ref)), np.max(np.abs(s_dob - ref))) / escala),
        "previsoes_iguais": bool(np.array_equal(prever(p, X), modelo.predict(X))),
    }