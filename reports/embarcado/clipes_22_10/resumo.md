# Matriz do sistema embarcado: 10 segmentos dos clipes (primeira-inferencia-22-10.log)

Modelo exp239 (JSON ee0c3e088366), referência `escores_referencia.csv` (sha 2507e68b5489), captura sha bf40bca9640e. Uma passagem do boot e 2 voltas do laço.

Os 10 segmentos estão no treino do modelo final: a matriz mede a fidelidade do porte, e não o desempenho (que é o do Protocolo B).

## Por segmento (verificação do boot)

| Seg. | Clipe | k | Classe | Escore Python | Escore placa | Fração da tol. | Decisão Python | Decisão placa | Passagens iguais |
|---:|---|---:|---|---:|---:|---:|---|---|---:|
| 29 | normal | 0 | normal | -1.331,76767 | -1.331,768 | 0,9 % | normal | normal | 3/3 |
| 30 | normal | 1 | normal | -1.392,13252 | -1.392,132 | 1,8 % | normal | normal | 3/3 |
| 88 | bpfi_0.3mm | 0 | falha | 1.361,83595 | 1.361,836 | 2,7 % | falha | falha | 3/3 |
| 89 | bpfi_0.3mm | 1 | falha | 1.477,39064 | 1.477,391 | 0,8 % | falha | falha | 3/3 |
| 147 | bpfi_1.0mm | 0 | falha | 1.469,79876 | 1.469,799 | 0,8 % | falha | falha | 3/3 |
| 148 | bpfi_1.0mm | 1 | falha | 1.425,16537 | 1.425,165 | 6,0 % | falha | falha | 3/3 |
| 206 | bpfo_0.3mm | 0 | falha | 1.492,37453 | 1.492,374 | 0,8 % | falha | falha | 3/3 |
| 207 | bpfo_0.3mm | 1 | falha | 1.518,17746 | 1.518,177 | 2,4 % | falha | falha | 3/3 |
| 265 | bpfo_1.0mm | 0 | falha | 1.499,83921 | 1.499,839 | 0,0 % | falha | falha | 3/3 |
| 266 | bpfo_1.0mm | 1 | falha | 1.499,84737 | 1.499,847 | 0,0 % | falha | falha | 3/3 |

## Matrizes (linhas: classe verdadeira; colunas: decisão)

**Placa**

| verdadeira \ placa | normal | falha |
|---|---:|---:|
| normal | 2 | 0 |
| falha | 0 | 8 |

**Python**

| verdadeira \ Python | normal | falha |
|---|---:|---:|
| normal | 2 | 0 |
| falha | 0 | 8 |

**Concordância (linhas: Python; colunas: placa)**

| Python \ placa | normal | falha |
|---|---:|---:|
| normal | 2 | 0 |
| falha | 0 | 8 |

## Conferências

- Concordância placa × Python: 10/10.
- Pior caso: seg 148 (bpfi_1.0mm), 6,0 % da tolerância encadeada (dif 0,000854 de 0,014252, medidos na placa). No pior arredondamento do texto, contra o float64 do CSV: 7,3 %.
- Escore e decisão iguais ao boot nas 2 voltas do laço, nos 10 segmentos.
- 20 pacotes $SEG de origem C, cada um igual à linha demo correspondente; contador sem salto, nenhum checksum errado.
- Referência: o sha do CSV é o CLIPES_CSV_SHA256_12 do clipes_demo.h, e o ref= impresso pela placa é o escore do CSV em todos os segmentos.
- Ponta a ponta (origem D, fora da matriz): 3 inferências, todas OK e iguais ao boot. É o mesmo segundo do seg 29 por outro caminho; contá-lo duplicaria o segmento.
- Firmware: `cadeia resumo ok=10/10 decisoes_iguais=10/10 iguais_blocos=10/10 dif_max=0.000854 (seg 148) dif/tol_max=6.0% ciclos min=5854229 med=5855300 max=5857373 (med 58.55 ms, max 58.57 ms)`
