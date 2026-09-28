# Protocolos A e B: MFCC provisório × MFCC oficial

Gerado por `scripts/validation/run_tabela_comparativa.py` a partir dos `metrics.json`. Partição `772c8e1eb1`, LDA, sem aumento de dados.

| rodada | provisório | oficial |
|---|---|---|
| A binário | exp007 (`mfcc_dsp_media_desvio_provisorio`) | exp027 (`mfcc_dsp_media_desvio`) |
| A multiclasse | exp008 (`mfcc_dsp_media_desvio_provisorio`) | exp028 (`mfcc_dsp_media_desvio`) |
| B | exp009 (`mfcc_dsp_media_desvio_provisorio`) | exp015 (`mfcc_dsp_media_desvio`) |

## Protocolo A — limite otimista, não conta para a meta

| tarefa | provisório | oficial |
|---|---|---|
| binário | 1.000 ± 0.000 (mín. 1.000) | 1.000 ± 0.000 (mín. 1.000) |
| multiclasse | 1.000 ± 0.000 (mín. 1.000) | 1.000 ± 0.000 (mín. 1.000) |

## Protocolo B — resultado principal (meta: acc. bal. média ≥ 0,85)

| falha deixada de fora | sensib. prov. | especif. prov. | acc. bal. prov. | sensib. ofic. | especif. ofic. | acc. bal. ofic. |
|---|---:|---:|---:|---:|---:|---:|
| bpfi_0.3mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bpfi_1.0mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| bpfo_0.3mm | 0.000 | 1.000 | 0.500 | 0.000 | 1.000 | 0.500 |
| bpfo_1.0mm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **média** | 0.750 | 1.000 | 0.875 | 0.750 | 1.000 | 0.875 |
| pior fold | | | 0.500 (B_bpfo_0.3mm_bloco0) | | | 0.500 (B_bpfo_0.3mm_bloco0) |
