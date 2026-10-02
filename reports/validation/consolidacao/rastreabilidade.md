# Rastreabilidade dos números consolidados

Gerado por `scripts/validation/run_consolidacao.py`. Cada rodada usada nas tabelas e figuras de `reports/validation/consolidacao/` tem linha no `experiments/registry.csv`, com o mesmo script e a mesma partição, e as métricas da linha batem com o `metrics.json` (tolerância de 6e-5 para as gravadas com 4 casas e de 6e-4 para as de 3 casas da decomposição dos harmônicos). As rodadas do `run_protocol.py` são conferidas pelo `resumo`; o controle dos harmônicos (`inspect_lda_harmonicos.py`, que grava fora das pastas `expNNN_*`), pelo `resumo` e pelo bloco `intervencao`, e a referência da intervenção tem que ser o B da meta. Se alguma não bater, o script aborta sem gravar nada.

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
| Aumento (4/6): três técnicas, 10 sementes | exp224–exp233 | 90 | todas batem |
| Aumento (4/6): técnica isolada e modo velocidade | exp019–exp022 | 36 | todas batem |
| Aumento (4/6): rótulos permutados | exp017 | 9 | todas batem |
| Harmônicos do eixo: decomposição (inspect_lda_harmonicos.py) | exp220 | 9 | todas batem |
| Harmônicos do eixo: intervenção (inspect_lda_harmonicos.py) | exp221 | 15 | todas batem |

## Rodadas citadas uma a uma (exceto permutação, curva e as 10 sementes do aumento)

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
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `sens_media` | 0.7500 | 0.7500 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `espec_media` | 1.0000 | 1.0000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp019 | `exp019_protocoloB_binario_lda_aum4-desl` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `sens_media` | 0.7500 | 0.7500 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `espec_media` | 1.0000 | 1.0000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp020 | `exp020_protocoloB_binario_lda_aum4-estir` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_media` | 0.8750 | 0.8750 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `sens_media` | 0.7500 | 0.7500 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `espec_media` | 1.0000 | 1.0000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_bpfi_0.3mm` | 1.0000 | 1.0000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp021 | `exp021_protocoloB_binario_lda_aum4-ruido` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_media` | 0.7878 | 0.7878 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `sens_media` | 0.5756 | 0.5756 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `espec_media` | 1.0000 | 1.0000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_pior_falha` | 0.5000 | 0.5000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_pior_fold` | 0.5000 | 0.5000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_bpfi_0.3mm` | 0.6511 | 0.6511 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_bpfi_1.0mm` | 1.0000 | 1.0000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_bpfo_0.3mm` | 0.5000 | 0.5000 |
| exp022 | `exp022_protocoloB_binario_lda_aum4-desl-estir-ruido-vel` | `acc_bal_bpfo_1.0mm` | 1.0000 | 1.0000 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_media` | 0.5279 | 0.5279 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `sens_media` | 0.5720 | 0.5720 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `espec_media` | 0.4838 | 0.4838 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_pior_falha` | 0.3861 | 0.3861 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_pior_fold` | 0.0000 | 0.0000 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_bpfi_0.3mm` | 0.6014 | 0.6014 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_bpfi_1.0mm` | 0.4852 | 0.4852 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_bpfo_0.3mm` | 0.6390 | 0.6390 |
| exp017 | `exp017_protocoloB_binario_lda_aum4-desl-estir-ruido_permutado` | `acc_bal_bpfo_1.0mm` | 0.3861 | 0.3861 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_mediana` | -0.2010 | -0.2007 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_min` | -0.4220 | -0.4216 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_max` | 0.0000 | 0.0001 |
| exp220 | `controle_harmonicos` | `frac_bandas_mediana` | 0.2000 | 0.2000 |
| exp220 | `controle_harmonicos` | `frac_desvio_mfcc_mediana` | 0.0240 | 0.0240 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_bpfi_0.3mm` | -0.1990 | -0.1986 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_bpfi_1.0mm` | -0.1930 | -0.1934 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_bpfo_0.3mm` | 0.0000 | 0.0000 |
| exp220 | `controle_harmonicos` | `frac_sep_harm_bpfo_1.0mm` | -0.3490 | -0.3485 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_mediana` | -0.2010 | -0.2007 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_min` | -0.4220 | -0.4216 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_max` | 0.0000 | 0.0001 |
| exp221 | `controle_harmonicos/intervencao` | `frac_bandas_mediana` | 0.2000 | 0.2000 |
| exp221 | `controle_harmonicos/intervencao` | `frac_desvio_mfcc_mediana` | 0.0240 | 0.0240 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_bpfi_0.3mm` | -0.1990 | -0.1986 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_bpfi_1.0mm` | -0.1930 | -0.1934 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_bpfo_0.3mm` | 0.0000 | 0.0000 |
| exp221 | `controle_harmonicos/intervencao` | `frac_sep_harm_bpfo_1.0mm` | -0.3490 | -0.3485 |
| exp221 | `controle_harmonicos/intervencao` | `acc_bal_referencia` | 0.8750 | 0.8750 |
| exp221 | `controle_harmonicos/intervencao` | `acc_bal_sem_harm` | 0.8750 | 0.8750 |
| exp221 | `controle_harmonicos/intervencao` | `sens_sem_harm` | 0.7500 | 0.7500 |
| exp221 | `controle_harmonicos/intervencao` | `espec_sem_harm` | 1.0000 | 1.0000 |
| exp221 | `controle_harmonicos/intervencao` | `acc_bal_sorteios_min` | 0.8542 | 0.8542 |
| exp221 | `controle_harmonicos/intervencao` | `acc_bal_sorteios_max` | 0.8750 | 0.8750 |
