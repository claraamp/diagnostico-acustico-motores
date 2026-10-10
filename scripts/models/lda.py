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

Contrato com o firmware: além dos parâmetros, o JSON descreve a cadeia que produz
as 26 features (`cadeia_entrada`), nos pontos em que uma implementação em C costuma
divergir sem dar erro: escala da entrada, tipo de janela, FFT e espectro, banco de
Mel (fórmula e bins), log, DCT e desvio-padrão, e de onde vem o sinal a 12,8 kHz
(decimação). Os parâmetros configuráveis (taxas, janela, passo, bandas,
coeficientes, escala, piso do Mel, projeto do FIR) saem do `config` e do `dsp`.
Dois valores são literais que espelham o código do `dsp`, sem constante própria:
o `1e-10` do log e a DCT-II ortonormal. O
teste `test_descricao_da_cadeia_reproduz_o_dsp` refaz o MFCC só a partir da
descrição e compara com o `dsp.mfcc`, então uma mudança no `dsp` sem atualizar
esses literais faz o teste falhar. O `reports/c_reference/reference_data.h` continua sendo a
referência numérica bloco a bloco. (Até a versão 2 do formato, o JSON não trazia
essa descrição; até a 3, não trazia a origem do sinal.)

Este módulo não é executável: é importado.
"""

from __future__ import annotations

import numpy as np

import config
import dsp
from validation.run_protocol import novo_modelo

CLASSE_POSITIVA = "falha"
CLASSE_NEGATIVA = "normal"
VERSAO_FORMATO = 4


def cadeia_entrada(fs: int = config.FS_TRABALHO) -> dict:
    """
    Como as 26 features são calculadas a partir de um segmento de 1 s, com os números
    tirados do config e do dsp (as mesmas fórmulas do `dsp.log_mel`).
    """
    n_quadro = int(round(config.MFCC_WINDOW_MS / 1000 * fs))
    n_passo = int(round(config.MFCC_HOP_MS / 1000 * fs))
    n_fft = 1 << (n_quadro - 1).bit_length()
    n_amostras = config.amostras_por_segmento(fs)
    fmin = config.MFCC_FMIN   # o padrão do dsp.mel_filterbank, o que o dsp.mfcc usa
    mels = np.linspace(dsp.hz_to_mel(fmin), dsp.hz_to_mel(fs / 2), config.MFCC_N_MELS + 2)
    bins = np.clip(np.floor((n_fft + 1) * dsp.mel_to_hz(mels) / fs).astype(int), 0, n_fft // 2)
    fir = dsp.design_decimation(config.FS_ORIGINAL, fs)
    return {
        "origem_do_sinal": {
            "fs_original_hz": config.FS_ORIGINAL,
            "fator_decimacao": fir.down,
            "fir": {"tipo": "Kaiser (scipy.signal.firwin)", "numtaps": fir.numtaps,
                    "corte_hz": fir.cutoff_hz, "transicao_hz": fir.transition_hz,
                    "atenuacao_alvo_db": config.FIR_ATTENUATION,
                    "atenuacao_medida_db": round(fir.stopband_atten_db, 2)},
            "descricao": f"PCM a {config.FS_ORIGINAL} Hz / {config.INT16_FULL:g}, FIR passa-baixa causal "
                         f"(lfilter) de {fir.numtaps} taps, descarte das primeiras {(fir.numtaps - 1) // 2} "
                         f"amostras (atraso de grupo), uma amostra a cada {fir.down} a partir da primeira, "
                         "requantizado para int16 (round(y·32767)); é o 02_decimate_pcm.py "
                         "(dsp.design_decimation e dsp.resample_clip). Se o firmware ler o PCM já decimado, "
                         "esta etapa já está feita. Se decimar no STM32 (arm_fir_decimate_f32), usar os mesmos "
                         "taps e conferir a fase: qual das 4 amostras é mantida e o descarte do atraso podem "
                         "diferir do Python e deslocar os segmentos. Se amostrar em outra taxa ou usar outro "
                         "filtro, a entrada do MFCC muda",
            "referencia": "reports/decimation/ (estudo da taxa e do filtro) e "
                          "data/processed/pcm_decimated/12800/manifest.json",
        },
        "fs_hz": fs,
        "segmento_s": config.SEGMENTO_S,
        "amostras_por_segmento": n_amostras,
        "escala_entrada": {
            "divisor": config.INT16_FULL,
            "descricao": f"x = pcm_int16 / {config.INT16_FULL:g} (config.INT16_FULL); a conversão "
                         "Q15 da CMSIS-DSP divide por 32768, e a diferença desloca o log-Mel",
        },
        "quadros": {
            "amostras_por_quadro": n_quadro, "passo": n_passo,
            "n_quadros": 1 + (n_amostras - n_quadro) // n_passo,
            "descricao": "quadros só dentro do segmento, sem preenchimento nem pré-ênfase",
        },
        "janela": {"tipo": "hann", "simetrica": True,
                   "descricao": f"np.hanning({n_quadro}): w[n] = 0,5 − 0,5·cos(2πn/(N−1)), "
                                "simétrica, não a periódica"},
        "fft": {"n_fft": n_fft,
                "descricao": f"quadro de {n_quadro} amostras completado com zeros até {n_fft}; "
                             f"espectro de potência |X[k]|², k = 0…{n_fft // 2}, sem normalizar por N"},
        "mel": {
            "n_filtros": config.MFCC_N_MELS, "fmin_hz": fmin, "fmax_hz": fs / 2,
            "formula": "HTK: mel = 2595·log10(1 + f/700)",
            "bins": bins.tolist(),
            "bins_coincidentes": bool(np.any(np.diff(bins) == 0)),
            "descricao": f"{config.MFCC_N_MELS + 2} pontos igualmente espaçados em mel; "
                         "bin = floor((n_fft + 1)·f/fs), limitado a n_fft/2; o filtro m é um triângulo "
                         "de pico 1 (sem normalizar a área): sobe de bins[m−1] a bins[m] e desce até "
                         "bins[m+1], com o bin da borda direita fora. Com bins coincidentes, o dsp "
                         "desloca o do meio de 1 (não ocorre nesta configuração se bins_coincidentes "
                         "for false)",
        },
        "log": {"funcao": "ln", "epsilon": 1e-10, "descricao": "log natural de (energia Mel + 1e-10)"},
        "dct": {"tipo": "II", "norma": "ortonormal", "coeficientes": config.MFCC_N_COEFS,
                "descricao": f"scipy.fft.dct(type=2, norm='ortho'), mantidos c0…c{config.MFCC_N_COEFS - 1}, "
                             "sem lifter"},
        "resumo_por_segmento": "média e desvio-padrão populacional (ddof=0, dividir por N) de cada "
                               "coeficiente ao longo dos quadros do segmento",
    }


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