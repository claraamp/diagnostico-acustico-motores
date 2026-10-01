# Protocolo B com e sem aumento de dados

Gerado por `scripts/validation/run_tabela_aumento.py` a partir dos `metrics.json`. Partição `772c8e1eb1`, modelo `lda`, MFCC de 13 coeficientes (média e desvio por segmento). O aumento entra só no treino de cada fold; o teste é sempre o segmento original.

## Médias e pior caso

| aumento | rodada | sensib. | especif. | acc. bal. | pior falha (acc. bal.) | pior fold | fator de expansão (nominal / efetivo) | variantes aceitas por fold |
|---|---|---:|---:|---:|---|---:|---|---|
| sem aumento | exp015 | 0.750 | 1.000 | **0.875** | 0.500 (bpfo_0.3mm) | 0.500 | 1 | — |
| três técnicas (10 sementes) | exp224–exp233 | 0.750 ± 0.000 | 1.000 ± 0.000 | **0.875 ± 0.000** | 0.500 (bpfo_0.3mm) | 0.500 | 5 / 4.98 | 889–903 |
| só deslocamento | exp019 | 0.750 | 1.000 | **0.875** | 0.500 (bpfo_0.3mm) | 0.500 | 5 / 4.99 | 889–903 |
| só estiramento | exp020 | 0.750 | 1.000 | **0.875** | 0.500 (bpfo_0.3mm) | 0.500 | 5 / 4.97 | 888–900 |
| só ruído | exp021 | 0.750 | 1.000 | **0.875** | 0.500 (bpfo_0.3mm) | 0.500 | 5 / 5.00 | 896–904 |
| três técnicas, estiramento *velocidade* | exp022 | 0.576 | 1.000 | **0.788** | 0.500 (bpfo_0.3mm) | 0.500 | 5 / 4.98 | 889–903 |

## Acurácia balanceada por falha deixada de fora

| aumento | bpfi_0.3mm | bpfi_1.0mm | bpfo_0.3mm | bpfo_1.0mm |
|---|---:|---:|---:|---:|
| sem aumento | 1.000 | 1.000 | 0.500 | 1.000 |
| três técnicas (10 sementes) | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 |
| só deslocamento | 1.000 | 1.000 | 0.500 | 1.000 |
| só estiramento | 1.000 | 1.000 | 0.500 | 1.000 |
| só ruído | 1.000 | 1.000 | 0.500 | 1.000 |
| três técnicas, estiramento *velocidade* | 0.651 | 1.000 | 0.500 | 1.000 |

Leitura: acurácia balanceada = (sensibilidade + especificidade) / 2. Um 0,500 numa falha é sensibilidade 0 com especificidade 1 — a falha não é detectada —, e não detecção parcial.

Fator de expansão: nominal = 1 + cópias por segmento (4 cópias); efetivo = média, entre os folds, de (segmentos de treino + variantes aceitas) / segmentos de treino. A diferença são as variantes recusadas por encostarem no teste ou na faixa de descarte.

Com mais de uma semente, a linha traz média ± desvio entre as sementes do aumento, e o pior caso é o pior entre elas.
