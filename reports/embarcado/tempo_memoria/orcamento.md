Orcamento de memoria (build/Debug/frente1.map)

| Parte | Flash (B) | RAM (B) |
|---|---:|---:|
| clipes a 12,8 kHz | 256.215 | 0 |
| recorte a 51,2 kHz | 103.284 | 0 |
| cadeia (MFCC) | 1.432 | 10.788 |
| cadeia completa na placa (cadeia_clipes.c) | 3.569 | 3.446 |
| decimacao (decimacao.c; a FIR esta na CMSIS-DSP) | 5.988 | 7.125 |
| recepcao pela serial (recepcao.c e recepcao_serial.c) | 3.055 | 3.245 |
| CRC-32 comum (crc32.c) | 200 | 0 |
| marco e LDA | 37.804 | 0 |
| pacote da serial | 1.598 | 0 |
| CMSIS-DSP | 13.028 | 0 |
| HAL e CubeMX | 11.547 | 13 |
| aplicacao base | 2.472 | 5.316 |
| biblioteca C | 9.856 | 408 |
| alinhamento | 36 | 11 |
| outros | 144 | 28 |
| reserva de heap e pilha (linker script: heap 512 B, pilha 4.096 B) | 0 | 4.612 |
| **total** | **450.228** | **34.992** |

- Flash: 450.228 B de 524.288 B (85.9 %); livres 74.060 B (72,3 KiB).
- Audio na flash (clipes + recorte): 359.499 B; o resto do firmware: 90.729 B (88,6 KiB), contra a reserva de 64 KB do orcamento.
- RAM: 34.992 B de 131.072 B (26.7 %), dos quais 4.612 B sao a reserva minima de heap e pilha; livres 96.080 B (93,8 KiB).
- Com o detector de novidade (~40.960 B a mais, estimativa do card): sobrariam 55.120 B (53,8 KiB) de RAM.
- A pilha reservada e so o minimo do linker script (4.096 B); a pilha real pode crescer para a RAM livre, sem aviso do linker.
