# Robustez ao ruído: varredura de SNR

Rodadas exp305–exp319. Ruído somado no int16 a 51,2 kHz, antes da decimação; SNR da entrada, por segmento. 10 sementes: 20260925, 1, 2, 3, 4, 5, 6, 7, 8, 9 (as do aumento nos exp224–exp233).

## Protocolo B com o teste ruidoso (média ± desvio entre sementes)

| Ponto | Modelo | Acurácia balanceada | Sensibilidade | Especificidade |
|---|---|---:|---:|---:|
| limpo | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 30 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 25 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 20 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 15 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 10 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 5 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 0 dB | sem aumento | 0,629 ± 0,005 | 0,750 ± 0,000 | 0,507 ± 0,010 |
| rosa, 30 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 25 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 20 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 15 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 10 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 5 dB | sem aumento | 0,753 ± 0,005 | 0,750 ± 0,000 | 0,756 ± 0,010 |
| rosa, 0 dB | sem aumento | 0,505 ± 0,004 | 0,750 ± 0,000 | 0,260 ± 0,007 |
| limpo | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 30 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 25 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 20 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 15 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 10 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 5 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 0 dB | com aumento | 0,805 ± 0,028 | 0,750 ± 0,000 | 0,860 ± 0,057 |
| rosa, 30 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 25 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 20 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 15 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 10 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 5 dB | com aumento | 0,851 ± 0,018 | 0,750 ± 0,000 | 0,951 ± 0,036 |
| rosa, 0 dB | com aumento | 0,526 ± 0,020 | 0,750 ± 0,000 | 0,302 ± 0,040 |

O eixo é a SNR da entrada, a 51,2 kHz. O filtro de decimação tira o ruído acima de 5,76 kHz, então a SNR na banda que a cadeia vê é maior: 6,3 dB a mais no branco e 0,4 dB a mais no rosa. O mesmo número de dB não é o mesmo nível de ruído para a cadeia nos dois ruídos.

## Passo 0: análise de margem com o modelo final (exp239), 295 segmentos

| Ponto | SNR na banda (mediana) | Segmentos com \|s\| < E | Menor \|s\|/E | Saturadas por semente, 51,2 kHz | Saturadas por semente, 12,8 kHz | Emulação float32, tol. do A4 (fração; estágio) | Escore por \|Δs\|/E | Decisões diferentes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| limpo | — | 0 | 3087,5 | 0 | 0 | 0,0994 (escore) | 0,0035 | 0 |
| branco, 30 dB | 36,3 dB | 0 | 3061,7 | 0 | 0 | 0,1244 (escore) | 0,0044 | 0 |
| branco, 25 dB | 31,3 dB | 0 | 3026,3 | 0 | 0 | 0,1577 (mel) | 0,0043 | 0 |
| branco, 20 dB | 26,3 dB | 0 | 2933,8 | 2 | 0 | 0,1247 (escore) | 0,0043 | 0 |
| branco, 15 dB | 21,3 dB | 0 | 2693,8 | 22 | 0 | 0,1062 (mel) | 0,0033 | 0 |
| branco, 10 dB | 16,3 dB | 0 | 2123,0 | 392 | 1 | 0,1230 (escore) | 0,0032 | 0 |
| branco, 5 dB | 11,3 dB | 0 | 959,2 | 6002 | 12 | 0,2431 (escore) | 0,0031 | 0 |
| branco, 0 dB | 6,3 dB | 0 | 1,3 | 73233 | 104 | 64,9563 (escore) | 0,0026 | 0 |
| rosa, 30 dB | 30,4 dB | 0 | 3000,3 | 0 | 0 | 0,1311 (mel) | 0,0041 | 0 |
| rosa, 25 dB | 25,4 dB | 0 | 2914,5 | 0 | 0 | 0,1200 (mel) | 0,0038 | 0 |
| rosa, 20 dB | 20,4 dB | 0 | 2739,0 | 1 | 0 | 0,1235 (mel) | 0,0038 | 0 |
| rosa, 15 dB | 15,4 dB | 0 | 2300,2 | 21 | 3 | 0,1160 (mel) | 0,0031 | 0 |
| rosa, 10 dB | 10,4 dB | 0 | 1343,4 | 382 | 45 | 0,1583 (escore) | 0,0031 | 0 |
| rosa, 5 dB | 5,4 dB | 0 | 4,8 | 5844 | 576 | 7,2791 (escore) | 0,0031 | 0 |
| rosa, 0 dB | 0,4 dB | 0 | 54,3 | 72391 | 5903 | 0,8019 (escore) | 0,0025 | 0 |

Segmentos perto do limiar e decisões diferentes, somados nas sementes; menor razão e maior fração, o pior caso entre elas. Na emulação, a tolerância é a encadeada do A4; no escore, a relativa (10⁻⁵ do próprio escore), que vai a zero perto do limiar. A coluna |Δs|/E mede o mesmo erro do escore contra o limite absoluto E. Fora da tolerância relativa: branco a 0 dB (64,96×, escore; |Δs|/E = 0,0026) e rosa a 5 dB (7,28×, escore; |Δs|/E = 0,0031).

## Decisões pelas regras fixadas antes de rodar

- **Placa com ruído (regra de 05/11):** necessária em branco a 0 dB e rosa a 5 dB.
- **Proposta para o grupo (não decide):** com o escore julgado por |Δs|/E, a placa seria só confirmação (opcional).
- **Modelo final (regra fixada em 12/11):** fica o exp239, sem aumento: a regra pede todos os pontos, e os dois modelos empatam em branco a 10 dB, branco a 5 dB e rosa a 10 dB (dₖ = 0 em todas as sementes: os dois ficam no teto, como sem ruído). Nos pontos em que passa, o modelo com aumento é melhor; ganho de acurácia balanceada (média ± desvio de dₖ): branco a 0 dB, +0,176 ± 0,030, rosa a 5 dB, +0,098 ± 0,018 e rosa a 0 dB, +0,021 ± 0,021.
