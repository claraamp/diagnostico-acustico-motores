# Controles obrigatórios do classificador

Gerado por `scripts/validation/run_tabela_controles.py` a partir dos `metrics.json`. MFCC oficial, partição `772c8e1eb1`, LDA, binário, sem aumento de dados. A: acurácia balanceada média ± desvio entre os folds. B: acurácia balanceada média (sensibilidade e especificidade médias; acurácia balanceada da `bpfo_0.3mm`).

## Ablação do ganho

| protocolo | referência | sem c0 | normalização RMS por segmento |
|---|---|---|---|
| A | 1.000 ± 0.000 (exp027) | 1.000 ± 0.000 (exp029) | 1.000 ± 0.000 (exp219) |
| B | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) (exp015) | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) (exp030) | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) (exp014) |

## Permutação por bloco (Protocolo B)

100 sementes (exp031–exp138), contra a referência 0.875 (exp015).

| média ± desvio | mediana | mín. | máx. | sementes ≥ referência | p empírico | média × 0,5 (teste t) |
|---|---|---|---|---|---|---|
| 0.450 ± 0.060 | 0.455 | 0.329 | 0.571 | 0 de 100 | 0.0099 | t = -8.29, p = 5.6e-13 |

p empírico = (1 + nº de sementes com acurácia ≥ a da referência) / (n + 1): a probabilidade de um modelo sem relação rótulo–sinal chegar ao resultado da referência. Com n sementes, o menor valor possível é 1 / (n + 1).

A média fica abaixo de 0,5 de forma significativa. A permutação por bloco mantém quantos blocos há de cada rótulo, e há uma única gravação normal: a maior parte dos blocos dela recebe o rótulo "falha" no treino, e parte dos blocos de falha recebe "normal". O modelo aprende uma regra invertida em relação ao teste, que usa os rótulos verdadeiros. Vazamento empurraria o resultado para cima, não para baixo.

## Curva de aprendizado

Segmentos de treino repartidos entre as gravações de cada classe; 10 sementes por ponto (exp139–exp218). Média ± desvio entre as sementes.

| s por classe | A | B (acc. bal.) | B, sensib. | B, especif. | B, bpfo_0.3mm (acc. bal.) |
|---|---|---|---|---|---|
| 2 | 0.840 ± 0.050 | 0.757 ± 0.036 | 0.568 ± 0.078 | 0.946 ± 0.021 | 0.682 ± 0.077 |
| 5 | 0.977 ± 0.017 | 0.836 ± 0.014 | 0.676 ± 0.026 | 0.996 ± 0.006 | 0.584 ± 0.042 |
| 10 | 0.997 ± 0.003 | 0.800 ± 0.031 | 0.603 ± 0.062 | 0.997 ± 0.003 | 0.570 ± 0.072 |
| 30 | 1.000 ± 0.000 | 0.867 ± 0.014 | 0.734 ± 0.028 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| todo o treino | 1.000 | 0.875 | 0.750 | 1.000 | 0.500 |
