# Resultados consolidados do classificador

Gerado por `scripts/validation/run_consolidacao.py` a partir dos `metrics.json` e do `registry.csv`; não treina nem registra. LDA sobre média e desvio dos 13 MFCC, partição `772c8e1eb1`, 12800 Hz, sem aumento. Versões LaTeX das tabelas nos `.tex` desta pasta; origem de cada número em `rastreabilidade.md`.

## O número da meta

Acurácia balanceada média do Protocolo B: **0.875** (meta ≥ 0.85: atingida), com sensibilidade média 0.750, especificidade média 1.000, pior falha `bpfo_0.3mm` (0.500) e pior fold `B_bpfo_0.3mm_bloco0` (0.500). Rodada exp015, reproduzida por exp013.

## Protocolo B, fold a fold (acurácia balanceada)

| falha de fora | bloco 1 | bloco 2 | bloco 3 | bloco 4 | bloco 5 | bloco 6 | sensib. | especif. | acc. bal. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpfi_0.3mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bpfi_1.0mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bpfo_0.3mm | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 1.000 | 0.500 |
| bpfo_1.0mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **média** | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | **0.750** | **1.000** | **0.875** |

## Protocolo A — limite superior otimista (não conta para a meta)

Binário (exp027): 1.000 ± 0.000 (mín. 1.000). Multiclasse (exp028): 1.000 ± 0.000 (mín. 1.000). Treino e teste vêm das mesmas gravações. Fold a fold em `tab_protocoloA_folds.tex`.

## Matriz de confusão do B (somada nos 24 folds)

Cada segmento de falha é testado 6 vezes (uma por bloco normal) e cada segmento normal 4 vezes (uma por falha deixada de fora).

| classe verdadeira | testes | previsto normal | previsto falha |
|---|---:|---:|---:|
| normal | 236 | 236 (100.0 %) | 0 (0.0 %) |
| bpfi_0.3mm | 354 | 0 (0.0 %) | 354 (100.0 %) |
| bpfi_1.0mm | 354 | 0 (0.0 %) | 354 (100.0 %) |
| bpfo_0.3mm | 354 | 354 (100.0 %) | 0 (0.0 %) |
| bpfo_1.0mm | 354 | 0 (0.0 %) | 354 (100.0 %) |

VP = 1062, FN = 354, VN = 236, FP = 0. Figura: `fig_matriz_confusao_B.png`.

## Controles

| rodada | A (acc. bal.) | B (acc. bal.) | B, sensib. | B, especif. | B, bpfo_0.3mm | registro |
|---|---:|---:|---:|---:|---:|---|
| referência | 1.000 | 0.875 | 0.750 | 1.000 | 0.500 | exp027, exp015 |
| sem c0 | 1.000 | 0.875 | 0.750 | 1.000 | 0.500 | exp029, exp030 |
| normalização RMS por segmento | 1.000 | 0.875 | 0.750 | 1.000 | 0.500 | exp219, exp014 |
| permutação por bloco (100 sorteios) | — | 0.450 ± 0.060 | — | — | — | exp031–exp050, exp059–exp138 |
| curva: 2 s por classe | 0.840 ± 0.050 | 0.757 ± 0.036 | 0.568 | 0.946 | 0.682 | exp139–exp218 |
| curva: 5 s por classe | 0.977 ± 0.017 | 0.836 ± 0.014 | 0.676 | 0.996 | 0.584 | exp139–exp218 |
| curva: 10 s por classe | 0.997 ± 0.003 | 0.800 ± 0.031 | 0.603 | 0.997 | 0.570 | exp139–exp218 |
| curva: 30 s por classe | 1.000 ± 0.000 | 0.867 ± 0.014 | 0.734 | 1.000 | 0.500 | exp139–exp218 |

Permutação: p empírico da referência 0.0099 (0 de 100 sorteios ≥ 0.875); mín. 0.329, máx. 0.571. Figura: `fig_controles.png`.

## Taxa de amostragem (tarefa 5/6)

Mesma LDA e mesmo MFCC, features no mesmo commit; as 20 bandas Mel vão até o Nyquist de cada taxa. Partições conferidas em segundos pelo `run_tabela_taxas.py` (tabela completa em `reports/validation/tabela_taxas.md`).

| taxa | rodada | acc. bal. | sensib. | especif. | bpfi_0.3mm | bpfi_1.0mm | bpfo_0.3mm | bpfo_1.0mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 12800 Hz | exp234 | 0.875 | 0.750 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| 25600 Hz | exp235 | 0.870 | 0.741 | 1.000 | 0.982 | 1.000 | 0.500 | 1.000 |
