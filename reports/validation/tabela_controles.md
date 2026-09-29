# Controles obrigatórios do classificador

Gerado por `scripts/validation/run_tabela_controles.py` a partir dos `metrics.json`. MFCC oficial, partição `772c8e1eb1`, LDA, binário, sem aumento de dados. B: acurácia balanceada média (sensibilidade e especificidade médias; acurácia balanceada da `bpfo_0.3mm`).

## Ablação do ganho (sem o c0)

| protocolo | com c0 | sem c0 |
|---|---|---|
| A | 1.000 ± 0.000 (exp027) | 1.000 ± 0.000 (exp029) |
| B | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) (exp015) | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) (exp030) |

## Permutação por bloco (Protocolo B)

20 sementes (exp031, exp032, exp033, exp034, exp035, exp036, exp037, exp038, exp039, exp040, exp041, exp042, exp043, exp044, exp045, exp046, exp047, exp048, exp049, exp050).

| média ± desvio | mediana | mín. | máx. | sementes acima de 0,5 |
|---|---|---|---|---|
| 0.448 ± 0.068 | 0.456 | 0.332 | 0.556 | 4 de 20 |

| semente | acc. bal. média | pior fold |
|---|---|---|
| 1 | 0.332 | 0.000 |
| 2 | 0.418 | 0.000 |
| 3 | 0.556 | 0.000 |
| 4 | 0.436 | 0.000 |
| 5 | 0.414 | 0.000 |
| 6 | 0.511 | 0.017 |
| 7 | 0.474 | 0.000 |
| 8 | 0.347 | 0.000 |
| 9 | 0.482 | 0.000 |
| 10 | 0.485 | 0.000 |
| 11 | 0.427 | 0.000 |
| 12 | 0.497 | 0.000 |
| 13 | 0.497 | 0.017 |
| 14 | 0.338 | 0.050 |
| 15 | 0.347 | 0.000 |
| 16 | 0.413 | 0.000 |
| 17 | 0.522 | 0.000 |
| 18 | 0.528 | 0.042 |
| 19 | 0.494 | 0.000 |
| 20 | 0.438 | 0.000 |

## Curva de aprendizado

| segundos de treino por classe | A | B |
|---|---|---|
| 2 | 0.823 ± 0.090 | 0.865 (sens. 0.768, espec. 0.962; bpfo_0.3mm 0.788) |
| 5 | 0.962 ± 0.044 | 0.824 (sens. 0.653, espec. 0.996; bpfo_0.3mm 0.525) |
| 10 | 0.965 ± 0.053 | 0.732 (sens. 0.464, espec. 1.000; bpfo_0.3mm 0.589) |
| 30 | 1.000 ± 0.000 | 0.835 (sens. 0.669, espec. 1.000; bpfo_0.3mm 0.500) |
| todo o treino | 1.000 ± 0.000 | 0.875 (sens. 0.750, espec. 1.000; bpfo_0.3mm 0.500) |

Rodadas da curva: A exp051, exp052, exp053, exp054; B exp055, exp056, exp057, exp058.
