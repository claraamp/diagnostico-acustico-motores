# Decimação — definição da taxa e validação da assinatura BPFI/BPFO

- Sinal de origem: 51200 Hz, 5 clipes, 60.0 s por clipe.
- Condição de carga: **0Nm**.
- Critério primário: preservar ≥ 95% da energia discriminante em **todas** as classes de falha.
- Diagnóstico secundário (envelope): banda 2700–4700 Hz, pico considerado presente com SNR ≥ 6 dB.

## Critério primário — energia discriminante preservada

Para cada classe de falha, `D(f) = max(0, PSD_falha(f) − PSD_normal(f))` é a energia que ela tem a mais que a condição normal. A tabela dá a fração dessa diferença que sobrevive a cada taxa. Por classe, e não na média: as classes severas têm muito mais energia e esconderiam a classe incipiente, que é justamente a que decide.

| taxa | bpfi_0.3mm | bpfi_1.0mm | bpfo_0.3mm | bpfo_1.0mm | pior caso |
|---|---|---|---|---|---|
| 6.4 kHz | 82.2% | 99.1% | 83.3% | 97.7% | **82.2%** (bpfi_0.3mm) |
| 8.0 kHz | 89.1% | 99.4% | 86.5% | 98.4% | **86.5%** (bpfo_0.3mm) |
| 12.8 kHz | 96.1% | 99.9% | 96.1% | 99.5% | **96.1%** (bpfi_0.3mm) |
| 16.0 kHz | 97.8% | 99.9% | 96.8% | 99.7% | **96.8%** (bpfo_0.3mm) |
| 25.6 kHz | 99.4% | 100.0% | 99.9% | 99.9% | **99.4%** (bpfi_0.3mm) |

Frequência que concentra 95 % e 99 % da energia discriminante, por classe:

- **bpfi_0.3mm**: 95 % → 5634 Hz · 99 % → 9155 Hz
- **bpfi_1.0mm**: 95 % → 2260 Hz · 99 % → 3006 Hz
- **bpfo_0.3mm**: 95 % → 5999 Hz · 99 % → 9151 Hz
- **bpfo_1.0mm**: 95 % → 2387 Hz · 99 % → 4220 Hz

## Baseline (51,2 kHz)

| alvo | melhor SNR (dB) | clipe | f do pico (Hz) | presente |
|---|---|---|---|---|
| eixo (1x) | 37.74 | bpfi_0.3mm | 50.25 | sim |
| BPFO (1x) | 7.31 | bpfo_1.0mm | 178.5 | sim |
| BPFO (2x) | 9.62 | bpfo_0.3mm | 357.25 | sim |
| BPFO (3x) | 6.42 | bpfo_0.3mm | 535.75 | sim |
| BPFI (1x) | 9.88 | bpfi_0.3mm | 273.5 | sim |
| BPFI (2x) | 7.01 | bpfi_0.3mm | 545.0 | sim |
| BPFI (3x) | 7.73 | bpfi_0.3mm | 815.25 | sim |

SNR por clipe, para separar severidades:

- **BPFI (1x)**: bpfi_0.3mm = 9.88 dB, bpfi_1.0mm = 0.74 dB
- **BPFO (1x)**: bpfo_0.3mm = 2.44 dB, bpfo_1.0mm = 7.31 dB

## Comparativo por taxa

| taxa | fator | taps | atenuação (dB) | aliasing (dB) | SNR BPFI(1x) | SNR BPFO(1x) | harm. BPFI | harm. BPFO | amostras/25 ms | N FFT | RAM 1 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.4 kHz | /8 | 293 | 60.4 | -0.09 | 8.4 | 0.7 | 1/3 | 2/3 | 160 | 256 | 12.5 KiB |
| 8.0 kHz | ×5/32 | 1163 | 60.4 | -0.01 | 11.1 | 10.6 | 2/3 | 2/3 | 200 | 256 | 15.6 KiB |
| 12.8 kHz | /4 | 147 | 60.4 | -0.17 | 9.7 | 6.6 | 3/3 | 3/3 | 320 | 512 | 25.0 KiB |
| 16.0 kHz | ×5/16 | 583 | 60.4 | -0.03 | 9.7 | 6.9 | 3/3 | 3/3 | 400 | 512 | 31.2 KiB |
| 25.6 kHz | /2 | 75 | 59.3 | -0.02 | 9.9 | 7.3 | 3/3 | 3/3 | 640 | 1024 | 50.0 KiB |

## Decisão

**Taxa recomendada: 12800 Hz** (faixa-alvo 8000–16000 Hz; critérios atendidos: todos os critérios).

- Menor taxa dentro da faixa-alvo que preserva ao menos 95% da energia discriminante em todas as classes — o pior caso é bpfi_0.3mm com 96.1%.
- Energia espúria por aliasing dentro de +1 dB na banda útil.
- 51.200 / 4 é decimação por fator inteiro: um único FIR passa-baixa seguido de descarte de amostras — exatamente o que `arm_fir_decimate_f32` (CMSIS-DSP) implementa no STM32F411, sem estágio de interpolação.

## Acurácia do classificador — por que não entra na decisão

A acurácia balanceada (MFCC + LDA, split temporal contíguo) deu 1.0 no baseline e 6.4 kHz → 1.0, 8.0 kHz → 1.0, 12.8 kHz → 1.0, 16.0 kHz → 1.0, 25.6 kHz → 1.0.

São 5 gravações contínuas, uma por classe, e o split temporal tira treino e teste da **mesma** gravação. O classificador pode separar por características do registro — ganho, ruído de fundo, ponto de operação — sem usar a assinatura de falha. Métrica que não varia entre condições não discrimina nada, e um número assim não deve ir para o relatório como evidência de desempenho. Fica registrada como diagnóstico, e a avaliação honesta do classificador depende de validação entre registros distintos (ou de aumento de dados com partição por registro).

## Figuras

- `reports/decimation/fig1_psd_por_classe.png`
- `reports/decimation/fig2_envelope_por_taxa.png`
- `reports/decimation/fig3_filtros_antialiasing.png`
- `reports/decimation/fig4_assinatura_vs_custo.png`
