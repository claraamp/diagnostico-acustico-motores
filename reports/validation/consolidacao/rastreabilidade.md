# Rastreabilidade dos números consolidados

Gerado por `scripts/validation/run_consolidacao.py`. Cada rodada usada nas tabelas e figuras de `reports/validation/consolidacao/` tem linha no `experiments/registry.csv`, com o mesmo script (`validation/run_protocol.py`) e a mesma partição, e as métricas da linha batem com o `resumo` do `metrics.json` (tolerância de 6e-5, porque o registry grava 4 casas). Se alguma não bater, o script aborta sem gravar nada.

## Rodadas por papel

| papel | rodadas | métricas conferidas | resultado |
|---|---|---:|---|
| Protocolo B, referência (meta) | exp015 | 9 | todas batem |
| Protocolo B, reprodução | exp013 | 9 | todas batem |
| Protocolo A binário (limite otimista) | exp027 | 3 | todas batem |
| Protocolo A multiclasse (limite otimista) | exp028 | 3 | todas batem |
| Ablação: sem c0 | exp029–exp030 | 12 | todas batem |
| Ablação: normalização RMS por segmento | exp014, exp219 | 12 | todas batem |
| Permutação por bloco | exp031–exp050, exp059–exp138 | 900 | todas batem |
| Curva de aprendizado | exp139–exp218 | 480 | todas batem |
| Comparação de taxas (5/6): 12,8 × 25,6 kHz | exp234–exp235 | 18 | todas batem |

## Rodadas citadas uma a uma (exceto permutação e curva)

| rodada | pasta | métrica | registry | metrics.json |
|---|---|---|---:|---:|
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp015 | `exp015_protocoloB_binario_lda` | `sens_media` | 0.7500 | 0.7500 |
| exp015 | `exp015_protocoloB_binario_lda` | `espec_media` | 1.0000 | 1.0000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp015 | `exp015_protocoloB_binario_lda` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp013 | `exp013_protocoloB_binario_lda` | `sens_media` | 0.7500 | 0.7500 |
| exp013 | `exp013_protocoloB_binario_lda` | `espec_media` | 1.0000 | 1.0000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp013 | `exp013_protocoloB_binario_lda` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp027 | `exp027_protocoloA_binario_lda` | `acc_bal_media` | 1.0000 | 1.0000 |
| exp027 | `exp027_protocoloA_binario_lda` | `acc_bal_desvio` | 0.0000 | 0.0000 |
| exp027 | `exp027_protocoloA_binario_lda` | `acc_bal_min` | 1.0000 | 1.0000 |
| exp028 | `exp028_protocoloA_multiclasse_lda` | `acc_bal_media` | 1.0000 | 1.0000 |
| exp028 | `exp028_protocoloA_multiclasse_lda` | `acc_bal_desvio` | 0.0000 | 0.0000 |
| exp028 | `exp028_protocoloA_multiclasse_lda` | `acc_bal_min` | 1.0000 | 1.0000 |
| exp029 | `exp029_protocoloA_binario_lda_semc0` | `acc_bal_media` | 1.0000 | 1.0000 |
| exp029 | `exp029_protocoloA_binario_lda_semc0` | `acc_bal_desvio` | 0.0000 | 0.0000 |
| exp029 | `exp029_protocoloA_binario_lda_semc0` | `acc_bal_min` | 1.0000 | 1.0000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `sens_media` | 0.7500 | 0.7500 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `espec_media` | 1.0000 | 1.0000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp030 | `exp030_protocoloB_binario_lda_semc0` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp219 | `exp219_protocoloA_binario_lda_normclipe` | `acc_bal_media` | 1.0000 | 1.0000 |
| exp219 | `exp219_protocoloA_binario_lda_normclipe` | `acc_bal_desvio` | 0.0000 | 0.0000 |
| exp219 | `exp219_protocoloA_binario_lda_normclipe` | `acc_bal_min` | 1.0000 | 1.0000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `sens_media` | 0.7500 | 0.7500 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `espec_media` | 1.0000 | 1.0000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp014 | `exp014_protocoloB_binario_lda_normclipe` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp234 | `exp234_protocoloB_binario_lda` | `sens_media` | 0.7500 | 0.7500 |
| exp234 | `exp234_protocoloB_binario_lda` | `espec_media` | 1.0000 | 1.0000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp234 | `exp234_protocoloB_binario_lda` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_media` | 0.8704 | 0.8704 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `sens_media` | 0.7408 | 0.7408 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `espec_media` | 1.0000 | 1.0000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_bpfi_0.3mm` | 0.9816 | 0.9816 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp235 | `exp235_protocoloB_binario_lda_25600hz` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
