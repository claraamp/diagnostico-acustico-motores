# Robustez ao ruído: varredura de SNR

Rodadas exp305–exp319. Ruído somado no int16 a 51,2 kHz, antes da decimação; SNR da entrada, por segmento. Sementes: 20260925, 1, 2.

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
| branco, 0 dB | sem aumento | 0,627 ± 0,003 | 0,750 ± 0,000 | 0,505 ± 0,005 |
| rosa, 30 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 25 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 20 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 15 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 10 dB | sem aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 5 dB | sem aumento | 0,749 ± 0,002 | 0,750 ± 0,000 | 0,749 ± 0,005 |
| rosa, 0 dB | sem aumento | 0,504 ± 0,001 | 0,750 ± 0,000 | 0,258 ± 0,003 |
| limpo | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 30 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 25 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 20 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 15 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 10 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 5 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| branco, 0 dB | com aumento | 0,805 ± 0,037 | 0,750 ± 0,000 | 0,860 ± 0,075 |
| rosa, 30 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 25 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 20 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 15 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 10 dB | com aumento | 0,875 ± 0,000 | 0,750 ± 0,000 | 1,000 ± 0,000 |
| rosa, 5 dB | com aumento | 0,851 ± 0,015 | 0,750 ± 0,000 | 0,952 ± 0,030 |
| rosa, 0 dB | com aumento | 0,527 ± 0,020 | 0,750 ± 0,000 | 0,303 ± 0,041 |

## Passo 0: análise de margem com o modelo final (exp239), 295 segmentos

| Ponto | SNR na banda (mediana) | Segmentos com \|s\| < E | Menor \|s\|/E | Saturadas 51,2 kHz | Saturadas 12,8 kHz | Emulação float32 (fração da tol.) | Escore pela tol. relativa | Decisões diferentes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| limpo | — | 0 | 3087,5 | 0 | 0 | 0,0991 (mel) | 0,10 | 0 |
| branco, 30 dB | 36,3 dB | 0 | 3067,3 | 0 | 0 | 0,1082 (mel) | 0,12 | 0 |
| branco, 25 dB | 31,3 dB | 0 | 3037,4 | 1 | 0 | 0,1001 (mel) | 0,11 | 0 |
| branco, 20 dB | 26,3 dB | 0 | 2954,0 | 6 | 0 | 0,1234 (mel) | 0,12 | 0 |
| branco, 15 dB | 21,3 dB | 0 | 2728,6 | 66 | 0 | 0,1062 (mel) | 0,10 | 0 |
| branco, 10 dB | 16,3 dB | 0 | 2195,1 | 1190 | 7 | 0,1171 (mel) | 0,12 | 0 |
| branco, 5 dB | 11,3 dB | 0 | 1071,5 | 18113 | 38 | 0,0949 (mel) | 0,24 | 0 |
| branco, 0 dB | 6,3 dB | 0 | 1,6 | 219895 | 303 | 0,1128 (mel) | 64,96 | 0 |
| rosa, 30 dB | 30,4 dB | 0 | 3043,5 | 0 | 0 | 0,1311 (mel) | 0,10 | 0 |
| rosa, 25 dB | 25,4 dB | 0 | 2992,5 | 0 | 0 | 0,1150 (mel) | 0,11 | 0 |
| rosa, 20 dB | 20,4 dB | 0 | 2865,0 | 4 | 1 | 0,1224 (mel) | 0,11 | 0 |
| rosa, 15 dB | 15,4 dB | 0 | 2517,1 | 76 | 10 | 0,1056 (mel) | 0,09 | 0 |
| rosa, 10 dB | 10,4 dB | 0 | 1580,8 | 1104 | 129 | 0,1091 (mel) | 0,12 | 0 |
| rosa, 5 dB | 5,4 dB | 0 | 13,8 | 17232 | 1696 | 0,1079 (mel) | 7,28 | 0 |
| rosa, 0 dB | 0,4 dB | 0 | 632,6 | 217031 | 17832 | 0,1020 (mel) | 0,13 | 0 |

Contagens somadas nas sementes; menor razão e maior fração, o pior caso entre elas. Na emulação, o escore entra por |Δs|/E; a coluna da tolerância relativa (10⁻⁵ do próprio escore) é só informativa, porque ela vai a zero perto do limiar.

## Decisões pelas regras fixadas antes de rodar

- **Placa com ruído:** só confirmação (opcional): nenhum segmento com |s| < E e a emulação em float32 dentro da tolerância em todos os pontos.
- **Modelo final:** sem_aumento (exp239); não passa em: acuracia_balanceada a 10 dB (branco); especificidade a 10 dB (branco); acuracia_balanceada a 5 dB (branco); especificidade a 5 dB (branco); acuracia_balanceada a 10 dB (rosa); especificidade a 10 dB (rosa).
