# Protocolo B em duas taxas de amostragem

Gerado por `scripts/validation/run_tabela_taxas.py` a partir dos `metrics.json`. Modelo `lda`, MFCC de 13 coeficientes (média e desvio por segmento; janela de 25 ms, passo de 10 ms, 20 bandas Mel até o Nyquist de cada taxa), sem aumento. As partições das taxas descrevem os mesmos segmentos de 1 s e os mesmos folds; só o número de amostras muda.

## Médias e pior caso

| taxa | rodada | Nyquist | corte do FIR | sensib. | especif. | acc. bal. | pior falha (acc. bal.) | pior fold |
|---|---|---:|---|---:|---:|---:|---|---:|
| 12.8 kHz | exp234 | 6400 Hz | 5760 Hz (147 taps) | 0.750 | 1.000 | **0.875** | 0.500 (bpfo_0.3mm) | 0.500 |
| 25.6 kHz | exp235 | 12800 Hz | 11520 Hz (75 taps) | 0.741 | 1.000 | **0.870** | 0.500 (bpfo_0.3mm) | 0.500 |
| diferença (25.6 − 12.8 kHz) | | | | -0.009 | 0 | **-0.005** | 0 | 0 |

## Acurácia balanceada por falha deixada de fora

| taxa | **bpfi_0.3mm** | bpfi_1.0mm | **bpfo_0.3mm** | bpfo_1.0mm |
|---|---:|---:|---:|---:|
| 12.8 kHz | 1.000 | 1.000 | 0.500 | 1.000 |
| 25.6 kHz | 0.982 | 1.000 | 0.500 | 1.000 |
| diferença | -0.018 | 0 | 0 | 0 |

## Sensibilidade por falha deixada de fora

| taxa | **bpfi_0.3mm** | bpfi_1.0mm | **bpfo_0.3mm** | bpfo_1.0mm |
|---|---:|---:|---:|---:|
| 12.8 kHz | 1.000 | 1.000 | 0.000 | 1.000 |
| 25.6 kHz | 0.963 | 1.000 | 0.000 | 1.000 |
| diferença | -0.037 | 0 | 0 | 0 |

Leitura: acurácia balanceada = (sensibilidade + especificidade) / 2. Um 0,500 numa falha é sensibilidade 0 com especificidade 1 — a falha não é detectada. As falhas de 0,3 mm (em negrito) são as que o estudo de decimação apontava como dependentes da banda alta: se a informação delas estivesse acima do Nyquist da taxa de trabalho, é nelas que a taxa maior faria diferença.
