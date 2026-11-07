# Matriz do sistema embarcado pela serial (serial-07-11.log)

Modelo exp239 (JSON ee0c3e088366); captura sha 6a8dbee33272; 921600 bit/s. Escore da placa pelo `escore_hex` da linha `rx` (float32 exato), contra o float64 do Python; tolerância encadeada de 10⁻⁵ relativo (A4).

## Jung (295/295 processados)

Os 295 segmentos estão no treino do modelo final: a matriz "verdadeira × placa" é acurácia de treino; o que se mede é a fidelidade do porte (concordância e fração da tolerância).

- Concordância placa × Python: **295/295**.
- Fração da tolerância: pior caso 9,8 % (bpfi_1.0mm_seg0127, dif 0,001462 de 0,014950), mediana 2,2 %.
- Recepção: mediana de 306 ms por quadro, da mágica ao último byte; processamento, mediana de 5.845.552 ciclos (58,5 ms a 100 MHz).
- Menor |escore| do Python: 1.297,2 (a tolerância é 10⁻⁵ dele).
- Não processados: 0; tentativas a mais: 0; duplicatas: 0.

**Placa**

| verdadeira \ placa | normal | falha |
|---|---:|---:|
| normal | 59 | 0 |
| falha | 0 | 236 |

**Python**

| verdadeira \ Python | normal | falha |
|---|---:|---:|
| normal | 59 | 0 |
| falha | 0 | 236 |

**Concordância (linhas: Python; colunas: placa)**

| Python \ placa | normal | falha |
|---|---:|---:|
| normal | 59 | 0 |
| falha | 0 | 236 |

## HUSTmotor (40/40 processados)

A LDA do Jung aplicada ao HUSTmotor sem retreino (`exp274` do Protocolo C): a matriz "verdadeira × placa" repete a do Python, e o que se mede é a concordância. Os escores são menores que os do Jung, então a mesma diferença absoluta pesa mais na fração da tolerância.

- Concordância placa × Python: **40/40**.
- Fração da tolerância: pior caso 7,6 % (bf_20hz_a4, dif 0,000409 de 0,005392), mediana 1,3 %.
- Recepção: mediana de 306 ms por quadro, da mágica ao último byte; processamento, mediana de 5.845.877 ciclos (58,5 ms a 100 MHz).
- Menor |escore| do Python: 483,4 (a tolerância é 10⁻⁵ dele).
- Não processados: 0; tentativas a mais: 0; duplicatas: 0.

**Placa**

| verdadeira \ placa | normal | falha |
|---|---:|---:|
| normal | 0 | 20 |
| falha | 0 | 20 |

**Python**

| verdadeira \ Python | normal | falha |
|---|---:|---:|
| normal | 0 | 20 |
| falha | 0 | 20 |

**Concordância (linhas: Python; colunas: placa)**

| Python \ placa | normal | falha |
|---|---:|---:|
| normal | 0 | 0 |
| falha | 0 | 40 |

## Recepção

- Envios: 335 (com os refeitos depois de um reinício, que não gastam tentativa); $ERR da placa: nenhum; respostas rejeitadas pelo PC: nenhuma; timeouts: 0.
- Não processados pelo PC × pela captura: batem.
- Estouros do anel: 0; linhas com checksum errado: 0; reinícios da placa: 0.
- Clipes da flash pela serial: 10/10 com o escore bit a bit igual ao da linha `ref` (a mesma conta da demonstração).
