# Matriz do sistema embarcado pela serial (serial-ruido-15-11-2.log)

Modelo exp239 (JSON ee0c3e088366); captura sha a35becc3b3dd; 921600 bit/s. Os quadros vão já decimados a 12,8 kHz: a placa roda o MFCC e a LDA. A decimação fica fora desta rodada; na placa, ela é testada pelo recorte de 1 s a 51,2 kHz do segmento 29 (`DECIMACAO OK` e a linha `ponta` no boot desta captura), e no PC, nas 5 gravações inteiras (`decimar_gravacao.c` do firmware). Escore da placa pelo `escore_hex` da linha `rx` (float32 exato), contra o float64 do Python; tolerância encadeada de 10⁻⁵ relativo (A4).

## Ruído branco (894/894 processados)

Os segmentos do Jung com ruído somado no int16 a 51,2 kHz, antes da decimação (recorte da varredura de SNR para a placa). O critério da rodada é o das decisões iguais às do Python e |Δs| < E em todos os segmentos, com E o limite absoluto do erro do escore do porte; a fração da tolerância relativa do A4 é só informação, porque perto do limiar (|s| pequeno) ela cresce sem que o erro do porte cresça.

- **Critério da rodada: aprovado**: todos processados, decisões iguais às do Python e |Δs| < E em todos.
- |Δs|/E: pior caso 0,0027 (normal_seg0017_branco_10db, |s| = 1.222,200, E = 0,4178), mediana 0,0005.
- Concordância placa × Python: **894/894**.
- Fração da tolerância relativa (só informação): pior caso 5.433,0 % (normal_seg0048_branco_0db, |s| = 0,676), mediana 1,6 %; acima dela: 16.
- Recepção: mediana de 306 ms por quadro, da mágica ao último byte; processamento (MFCC e LDA, sem a decimação: o quadro já chega a 12,8 kHz), mediana de 5.846.173 ciclos (58,5 ms a 100 MHz).
- Menor |escore| do Python: 0,5 (a tolerância é 10⁻⁵ dele).
- Não processados: 0; tentativas a mais: 0; duplicatas: 0.

**Por SNR**

| SNR | segmentos | sementes | processados | decisões iguais | \|Δs\|/E máx. | fração da tol. rel. máx. | acima da tol. rel. | menor \|s\|/E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 dB | 295 | 1 | 295 | 295/295 | 0,0027 | 9,3 % | 0 | 2.242,7 |
| 5 dB | 295 | 1 | 295 | 295/295 | 0,0025 | 19,6 % | 0 | 1.071,5 |
| 0 dB | 304 | 10 | 304 | 304/304 | 0,0018 | 5.433,0 % | 16 | 1,3 |

**Placa**

| verdadeira \ placa | normal | falha |
|---|---:|---:|
| normal | 138 | 48 |
| falha | 0 | 708 |

**Python**

| verdadeira \ Python | normal | falha |
|---|---:|---:|
| normal | 138 | 48 |
| falha | 0 | 708 |

**Concordância (linhas: Python; colunas: placa)**

| Python \ placa | normal | falha |
|---|---:|---:|
| normal | 138 | 0 |
| falha | 0 | 756 |

## Ruído rosa (894/894 processados)

Os segmentos do Jung com ruído somado no int16 a 51,2 kHz, antes da decimação (recorte da varredura de SNR para a placa). O critério da rodada é o das decisões iguais às do Python e |Δs| < E em todos os segmentos, com E o limite absoluto do erro do escore do porte; a fração da tolerância relativa do A4 é só informação, porque perto do limiar (|s| pequeno) ela cresce sem que o erro do porte cresça.

- **Critério da rodada: aprovado**: todos processados, decisões iguais às do Python e |Δs| < E em todos.
- |Δs|/E: pior caso 0,0020 (bpfi_1.0mm_seg0139_rosa_5db, |s| = 1.616,264, E = 0,4071), mediana 0,0005.
- Concordância placa × Python: **894/894**.
- Fração da tolerância relativa (só informação): pior caso 493,0 % (normal_seg0016_rosa_5db_s8, |s| = 1,980), mediana 1,4 %; acima dela: 8.
- Recepção: mediana de 306 ms por quadro, da mágica ao último byte; processamento (MFCC e LDA, sem a decimação: o quadro já chega a 12,8 kHz), mediana de 5.847.378 ciclos (58,5 ms a 100 MHz).
- Menor |escore| do Python: 2,0 (a tolerância é 10⁻⁵ dele).
- Não processados: 0; tentativas a mais: 0; duplicatas: 0.

**Por SNR**

| SNR | segmentos | sementes | processados | decisões iguais | \|Δs\|/E máx. | fração da tol. rel. máx. | acima da tol. rel. | menor \|s\|/E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 dB | 295 | 1 | 295 | 295/295 | 0,0020 | 11,9 % | 0 | 1.580,8 |
| 5 dB | 304 | 10 | 304 | 304/304 | 0,0020 | 493,0 % | 8 | 4,8 |
| 0 dB | 295 | 1 | 295 | 295/295 | 0,0019 | 9,5 % | 0 | 812,4 |

**Placa**

| verdadeira \ placa | normal | falha |
|---|---:|---:|
| normal | 123 | 63 |
| falha | 0 | 708 |

**Python**

| verdadeira \ Python | normal | falha |
|---|---:|---:|
| normal | 123 | 63 |
| falha | 0 | 708 |

**Concordância (linhas: Python; colunas: placa)**

| Python \ placa | normal | falha |
|---|---:|---:|
| normal | 123 | 0 |
| falha | 0 | 771 |

## Recepção

- Envios: 1788 (com os refeitos depois de um reinício, que não gastam tentativa); $ERR da placa: nenhum; respostas rejeitadas pelo PC: nenhuma; timeouts: 0.
- Não processados pelo PC × pela captura: batem.
- Estouros do anel: 0; linhas com checksum errado: 0; reinícios da placa: 0.
- Lista: manifesto 04e990853353, referencias.csv f2c71c8f1d19, .bin sha256 11f26bb8bfe4… (conferido inteiro antes do primeiro quadro).
