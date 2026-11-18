# Tempo de execução e ocupação de memória do firmware final

Captura `tempo-memoria-18-11.log` (sha 605280f58dfc), firmware `025449f` com `-DMEDIDA=ON`
para o tempo e a pilha medida; memória e pilha estática do build padrão (Debug, `-DMEDIDA=OFF`).
Relógio de 100 MHz; baud real 925.925 bit/s (1080 ciclos por byte de 10 bits). Rodada de 2026-11-18.

Escores contra a rodada `serial-07-11.log`: **335 iguais bit a bit**, 0 diferentes, 0 sem par. O build de medida não mexe na cadeia.

## Memória (build padrão)

| Parte | Flash 22/10 | Flash final | RAM 22/10 | RAM final |
|---|---:|---:|---:|---:|
| clipes a 12,8 kHz | 256.215 | 256.215 | 0 | 0 |
| recorte a 51,2 kHz | 103.284 | 103.284 | 0 | 0 |
| cadeia (MFCC) | 1.432 | 1.432 | 10.788 | 10.788 |
| cadeia completa na placa (cadeia_clipes.c) | 3.569 | 3.569 | 3.446 | 3.446 |
| decimacao (decimacao.c; a FIR esta na CMSIS-DSP) | 6.108 | 5.988 | 7.125 | 7.125 |
| recepcao pela serial (recepcao.c e recepcao_serial.c) | — | 3.055 | — | 3.245 |
| CRC-32 comum (crc32.c) | — | 200 | — | 0 |
| marco e LDA | 37.352 | 37.804 | 0 | 0 |
| pacote da serial | 1.398 | 1.598 | 0 | 0 |
| CMSIS-DSP | 13.028 | 13.028 | 0 | 0 |
| HAL e CubeMX | 6.025 | 11.547 | 13 | 13 |
| aplicacao base | 2.224 | 2.472 | 5.220 | 5.316 |
| biblioteca C | 9.856 | 9.856 | 408 | 408 |
| reserva de heap e pilha | 0 | 0 | 4.612 | 4.612 |
| outros e alinhamento | 177 | 180 | 36 | 39 |
| **total** | **440.668** | **450.228** | **31.648** | **34.992** |

- Flash: 450.228 B de 524.288 (85,9 %), 9.560 B a mais que em 22/10; livres 74.060 B.
- RAM: 34.992 B de 131.072 (26,7 %), 3.344 B a mais; livres 96.080 B.
- Soma das partes; o linker ("Memory region") dá 450.236 B de flash e 34.992 B de RAM: 8 B de alinhamento depois do `.isr_vector`, como em 22/10 (440.676 B pelo linker).

## Pilha

- Estática (`tools/pilha_estatica.py`, pior caminho com as interrupções): **1708 B** (1144 B do `main` e 564 B de interrupções); só o modo de recepção, a partir do `main`, 1588 B.
- Medida na placa (marca d'água): boot e demonstração, **1228 B**; modo de recepção (pilha pintada de novo depois da `med inicio`), **1124 B**.
- Reserva do linker: 4096 B. Uso: 42 % no pior dos dois.

## (a) Áudio da flash, do int16 a 51,2 kHz ao escore

- Decimação 115,93 ms + cadeia 58,49 ms = **174,42 ms** por segundo de áudio (17.441.807 ciclos).
- Saída: o `$SEG` (46 B) leva 497 µs na linha. Processamento e E/S: **17,5 %** do tempo real.
- Por amostra de entrada: 341 ciclos, contra 1.953 do período de amostragem a 51,2 kHz. **Critério de tempo real cumprido.**
- Cadeia dos clipes no mesmo boot (min, mediana, máx): 5.858.224, 5.859.338, 5.861.418 ciclos.

## (b) Segmento recebido pela serial (335 segmentos, 0 erros, 0 com estouro descartados)

| Grandeza (ms) | mediana (mín., máx.) |
|---|---|
| espera do PC (fim do `$PED` ao 1º byte) | 2,31 (mín. 0,33, máx. 2,67) |
| linha (1º ao último byte) | 306,3 (mín. 306,3, máx. 306,4) |
| linha teórica (25.618 B × 10 bits / 925.925) | 276,7 |
| lacunas (linha parada) | 30,2 (mín. 30,1, máx. 30,3) |
| maior lacuna | 0,07 (mín. 0,07, máx. 0,07) |
| CPU com o quadro (`consumir()`) | 121,1 (mín. 121,1, máx. 121,2) |
| ... dos quais, recepção byte a byte (inclui o custo de cada chamada ao `consumir()`) | 62,64 (mín. 62,63, máx. 62,65) |
| maior chamada ao `consumir()` | 0,60 (mín. 0,60, máx. 0,61) |
| resposta (LDA, `rx` e `$SEG`) | 2,18 (mín. 2,13, máx. 2,19) |
| pedido (`$PED`, 15 B na linha) | 0,16 |
| ciclo por segmento (`$PED` + espera + linha + resposta; sem a `med`, que o firmware normal não manda) | 311,0 (mín. 309,0, máx. 311,4) |
| campo `ms` da `rx` (resolução de 1 ms) | 306 (mín. 306, máx. 307) |

- MFCC e LDA (campo `ciclos` da `rx`): mediana 5.850.024 ciclos (58,50 ms).
- Cada chamada ao `consumir()` que achou bytes pegou 1,24 bytes (mediana), com 302 ciclos fora do MFCC: a recepção byte a byte é, em boa parte, o custo fixo de cada leitura do anel, e cairia se a CPU estivesse ocupada.
- A CPU fica ocupada 39,5 % do tempo de linha (um teto do laço como está escrito, não o mínimo necessário); trabalha 473 ciclos por byte recebido, contra 1080 de um byte na linha.
- A maior chamada ao `consumir()` (0,61 ms) fica abaixo dos 22,1 ms em que o anel de 2 KB enche.
- Um segmento de 1 s de áudio a cada 311,0 ms: **3,2 vezes o tempo real**. **Critério cumprido** no período que vale neste modo (o de um byte na linha).
