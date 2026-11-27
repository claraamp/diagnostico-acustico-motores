# Pipeline e scripts: documentação técnica

Este documento reúne os detalhes técnicos que saíram do README: as decisões fixadas da cadeia, o que cada etapa do pipeline grava e confere, a estrutura completa do repositório e todos os comandos. O README traz só a visão geral; os resultados e a discussão estão no relatório final.

## Decisões técnicas fixadas

| decisão | valor | onde está a justificativa |
|---|---|---|
| Taxa de amostragem de trabalho | **12.800 Hz** (decimação por fator inteiro 4 a partir de 51,2 kHz) | `reports/decimation/decimation_report.md` |
| Anti-aliasing | FIR Kaiser, 147 taps, corte em 5.760 Hz, 60 dB de atenuação | idem |
| Representação de features | MFCC (25 ms, hop 10 ms, 20 mels, 13 coeficientes) | proposta, seção de Metodologia |
| Plataforma | STM32F411CEU6 (Cortex-M4F, CMSIS-DSP; o X-CUBE-AI previsto na proposta não foi usado) | proposta, seção de Infraestrutura |
| Unidade de classificação | segmento de 1 s (12.800 amostras); 59 por gravação, em blocos de 10, 10, 10, 10, 10 e 9 | Notion, Registro de Decisões (protocolo de validação) |
| Protocolo de validação | **A** — blocos temporais (limite otimista); **B** — gravação de falha deixada de fora (resultado principal) | idem; implementação em `scripts/validation/particao.py` |
| Meta de desempenho | acurácia balanceada média do Protocolo B ≥ 85 %, sempre com sensibilidade, especificidade e pior caso | idem; resultados em `reports/validation/` |
| Classificador | **LDA** sobre média e desvio dos 13 MFCC (26 valores), priors uniformes; na validação interna ao treino do B, empata com a CNN 2D e fica pelo custo, e supera o MLP raso e a CNN 1D | `scripts/exploration/compare_classifiers.py`; Notion, estudo comparativo da tarefa Classificador 3/6 |
| Aumento de dados | só no treino de cada fold: deslocamento de janela (±0,4 s), estiramento temporal (taxa 0,95–1,05, phase vocoder) e ruído branco (SNR 20–35 dB); variante só entra se ler apenas segmentos de treino do fold | proposta (Metodologia); `scripts/augmentation/` |

A taxa de 12.800 Hz é a única com fator de decimação inteiro dentro da faixa de 8–16 kHz, requisito do `arm_fir_decimate_f32` da CMSIS-DSP, e preserva 96,1 % da energia discriminante no pior caso (classes de defeito incipiente, 0,3 mm). A comparação completa está em `experiments/registry.csv` (`exp001`–`exp006`).

## Dados

O conjunto principal é o subconjunto acústico (microfone) de Jung et al. (2023), *Data in Brief* 48, 109049, Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6): 5 classes (`0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10`), 60 s cada, 51,2 kHz, formato `.mat`. Esses cinco arquivos são **todo o áudio do dataset**. Os ensaios com carga (2 e 4 Nm), os defeitos de 3,0 mm e as falhas de desbalanceamento e desalinhamento têm só vibração, corrente e temperatura: os autores não gravaram o microfone com carga porque o freio, resfriado a ar, contaminaria o canal acústico (seção 3.1 do artigo).

Para confirmar as frequências de falha com um segundo sensor, usa-se também a **vibração** da mesma condição (0 Nm; `.mat` com 4 canais a 25,6 kHz; pela ordem das colunas no artigo, 0–1 = x e y do mancal A, 2–3 = mancal B). Ela não entra no classificador. Os arquivos de vibração têm os **mesmos nomes** dos de áudio, então ficam numa subpasta: `data/raw/vibracao/`.

Os arquivos `.mat` **não são versionados** neste repositório (ver `.gitignore`) — são grandes e já têm DOI fixo. O mesmo vale para os `.bin` gerados a partir deles; apenas os `manifest.json` e o `splits.json` (a partição dos protocolos de validação) entram no Git.

O HUSTmotor, usado no Protocolo C, tem download, formato e licença descritos em [`hustmotor.md`](./hustmotor.md).

## Estrutura completa do repositório

```
diagnostico-acustico-motores/
├── README.md
├── CONVENTIONS.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                  # .mat originais (baixados manualmente, não versionados)
│   │   ├── vibracao/         # .mat de vibração 0 Nm (só para confirmação; mesmos nomes dos de áudio)
│   │   └── hustmotor/        # .txt do HUSTmotor, H e BF nas 4 velocidades (Protocolo C; docs/hustmotor.md)
│   └── processed/
│       ├── pcm_raw/          # saída do 01 — .bin não versionados, manifest.json versionado
│       ├── pcm_decimated/    # saída do 02, um subdiretório por taxa (ex.: 12800/)
│       ├── splits/
│       │   ├── splits.json   # saída do 03 — partição dos protocolos A e B, VERSIONADA
│       │   └── splits_25600.json   # a mesma partição a 25,6 kHz (03 --fs 25600), para comparar taxas
│       ├── features/         # saída do 04 — mfcc_features.npz + manifest_features.json, NÃO versionados
│       │                     # (outra taxa: features/<fs>/)
│       └── hustmotor/
│           ├── pcm_decimated/12800/  # saída do 10 — .bin não versionados, manifest.json versionado
│           └── divisao_rotacao.json  # saída do 11 — divisão do C2 e da rotação deixada de fora, VERSIONADA
│
├── scripts/
│   ├── config.py             # parâmetros compartilhados entre etapas
│   ├── dsp.py                # blocos de processamento de sinais (PSD, MFCC, decimação)
│   ├── pcm_io.py             # leitura e escrita dos PCM e dos manifests
│   ├── experimentos.py       # escrita do registro de experimentos
│   ├── augmentation/         # aumento de dados do treino
│   │   ├── transformacoes.py              # estiramento (phase vocoder / reamostragem), ruído, recorte
│   │   └── variantes.py                   # sorteio reprodutível, janela lida e filtro por fold
│   ├── interface/            # interface de diagnóstico no computador (leitura serial)
│   │   ├── pacotes.py                     # leitura dos $CAB, $SEG e $ERR (formato v1) e estado do painel, só a biblioteca padrão
│   │   └── run_painel.py                  # executável: painel (Tkinter) pela porta, seguindo a captura da recepção ou de uma captura salva
│   ├── models/               # modelo final do classificador
│   │   └── lda.py                         # treino com todos os segmentos e parâmetros em JSON (escore ≥ 0 → falha)
│   ├── exploration/          # estudos e inspeções, sem numeração
│   │   ├── inspect_mat_keys.py
│   │   ├── inspect_signal.py              # árvore de campos de um .mat de áudio
│   │   ├── inspect_signal_fields.py       # campos do Signal (x_values, y_values, function_record)
│   │   ├── fig_psd_por_classe.py          # figura: PSD por classe no sinal original a 51,2 kHz, em Pa
│   │   ├── inspect_pcm.py                 # sanidade da conversão + caráter do sinal
│   │   ├── inspect_class_spectra.py       # PSD por classe e banda necessária
│   │   ├── compare_decimation_rates.py    # estudo que definiu a taxa de trabalho
│   │   ├── compare_classifiers.py         # estudo que escolheu o classificador (validação interna ao B)
│   │   ├── inspect_lda_harmonicos.py      # controle: peso da LDA nas bandas dos harmônicos do eixo
│   │   ├── inspect_comb_features.py       # pente harmônico (BPFO/BPFI) junto ao MFCC: entra na cadeia? (validação interna)
│   │   ├── inspect_deteccao_novidade.py   # detecção de novidade para a bpfo_0.3mm (D1–D4, sensibilidade no B; docs/criterio_deteccao_novidade.md)
│   │   ├── inspect_left_out_fault.py      # posição da falha deixada de fora (Protocolo B)
│   │   ├── inspect_acumulacao_float32.py  # média e desvio do MFCC em float32, por método (contrato numérico, A3)
│   │   ├── inspect_tolerancias_estagio.py # erro de cada estágio do MFCC em float32 e tolerâncias do porte (contrato numérico, A4)
│   │   ├── inspect_signature_spectra.py   # PSD assinada falha × normal (excesso e déficit)
│   │   ├── identify_tonal_peaks.py        # velocidades e BPFI/BPFO medidas por série harmônica
│   │   ├── inspect_vibration_mat.py       # árvore de um .mat (formato da vibração)
│   │   ├── confirm_bpf_envelope.py        # envelope com banda por curtose espectral, vibração + áudio
│   │   └── inspect_signature_mel.py       # o que o log-Mel oficial enxerga da bpfo_0.3mm
│   ├── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│   │   ├── 01_convert_mat_to_pcm.py
│   │   ├── 02_decimate_pcm.py             # aplica a taxa definida em config.py
│   │   ├── 03_make_splits.py              # segmenta e gera a partição, uma única vez
│   │   ├── 04_extract_features.py         # MFCC por segmento da partição (+ variantes do aumento) → .npz
│   │   ├── 05_train_classifier.py         # modelo final para o firmware (não mede desempenho) → reports/modelo_final/
│   │   ├── 06_export_lda_header.py        # .h da LDA final (float32), conferido contra os escores de referência
│   │   ├── 07_export_lda_test_vectors.py  # .h com features e escores de referência para o marco na placa
│   │   ├── 08_export_stage_references.py  # referências por estágio (float64) e tolerâncias do A4 para o harness em C
│   │   ├── 09_export_clips.py             # clipes int16 para a flash da placa (12,8 kHz e um recorte a 51,2 kHz) → .h
│   │   ├── 10_prepare_hustmotor.py        # HUSTmotor (Protocolo C): som de H e BF, 4 velocidades, decimado → data/processed/hustmotor/
│   │   └── 11_divisao_hustmotor.py        # divisão do HUSTmotor por rotação (C2 e rotação deixada de fora) → divisao_rotacao.json
│   └── validation/           # protocolo de validação do classificador
│       ├── particao.py                    # segmentos, folds A e B, verificação
│       ├── metricas.py                    # sensibilidade, especificidade, acurácia balanceada
│       ├── selecao.py                     # validação interna ao B e regra de escolha (escolher), sem torch
│       ├── run_protocol.py                # executável: roda A ou B e registra o resultado
│       ├── run_protocol_c.py              # executável: Protocolo C no HUSTmotor (C1, C2 e recalibração da LDA)
│       ├── matriz_clipes_placa.py         # executável: matriz embarcada nos 10 segmentos dos clipes da flash
│       ├── matriz_serial_placa.py         # executável: segmentos pela serial à placa e matriz embarcada contra o Python
│       ├── run_tempo_memoria.py           # executável: tempo (dois modos, E/S separada), memória e pilha do firmware final
│       ├── run_tabela_comparativa.py      # executável: tabela provisório × oficial a partir dos metrics.json
│       ├── run_tabela_controles.py        # executável: tabela dos controles (sem c0, permutação, curva)
│       ├── run_tabela_aumento.py          # executável: tabela do B com e sem aumento de dados
│       ├── run_tabela_taxas.py            # executável: tabela do B em duas taxas de amostragem
│       ├── run_tabela_protocolo_c.py      # executável: tabela única do Protocolo C, conferida contra o registry
│       ├── run_varredura_snr.py           # executável: robustez ao ruído (varredura de SNR no B e análise de margem)
│       └── run_consolidacao.py            # executável: tabelas .tex e figuras finais para o relatório
│
├── tests/                    # pytest; não dependem de data/
│   ├── conftest.py
│   ├── test_contrato_numerico.py      # cada valor do docs/contrato_numerico.md conferido contra o código
│   ├── test_dsp.py
│   ├── test_experimentos.py           # registry: linha nova nunca é colada na anterior
│   ├── test_fim_de_linha.py           # LF no repositório, no checkout e nos arquivos gerados
│   ├── augmentation/
│   │   ├── test_montar_treino.py          # montagem do treino com aumento no run_protocol
│   │   ├── test_transformacoes.py
│   │   └── test_variantes.py              # inclui: variante aceita nunca lê teste/descarte
│   ├── interface/
│   │   ├── test_pacotes.py                # leitura dos pacotes e estado do painel, sem placa e sem janela
│   │   └── test_run_painel.py             # fontes de linhas (captura salva, captura sendo gravada) e modo texto
│   ├── models/
│   │   └── test_lda.py                    # o JSON reproduz o sklearn; escore ≥ 0 é falha; a descrição da cadeia refaz o MFCC do dsp
│   ├── pipeline/
│   │   ├── test_train_classifier.py       # o 05 só aceita as features de referência da partição
│   │   ├── test_export_lda_header.py      # o .h é paramétrico em N_FEATURES, traz o float32 do JSON e, compilado, reproduz o escore
│   │   ├── test_export_lda_test_vectors.py # os vetores de teste ficam dentro da tolerância, a seleção não olha o escore, recusa .h de outro modelo
│   │   ├── test_export_stage_references.py # o formato volta igual do arquivo; as referências versionadas são as do dsp atual
│   │   ├── test_export_clips.py           # o clipe recalculado do .h dá o escore gravado; o recorte decimado é o clipe a 12,8 kHz
│   │   ├── test_prepare_hustmotor.py      # blocos do HUST lidos e decimados um a um, um segmento de 1 s por bloco
│   │   └── test_divisao_hustmotor.py      # o divisao_rotacao.json versionado é o que o manifest gera; papéis sem repetição
│   ├── exploration/
│   │   ├── test_deteccao_novidade.py      # D1b, amplitude da série sintética, D4 só com normais de treino, controle e regra de escolha
│   │   ├── test_comb_features.py          # candidatos A/B/C do card (A inviável); acha a série fora da razão cinemática; decisão pela validação interna
│   │   └── test_compare_classifiers.py    # critério de escolha: mais barato na margem, em qualquer ordem (pula sem torch)
│   └── validation/
│       ├── test_curva_aprendizado.py      # subamostra só treino, n por classe, reprodutível
│       ├── test_tabela_aumento.py         # tabela do aumento: comparabilidade e fator de expansão
│       ├── test_tabela_protocolo_c.py     # tabela única do Protocolo C
│       ├── test_tabela_taxas.py           # tabela das taxas: comparabilidade e mesma partição em segundos
│       ├── test_consolidacao.py           # consolidação: número sem registry ou rodada no papel errado aborta
│       ├── test_matriz_clipes_placa.py    # matriz dos clipes com uma captura sintética
│       ├── test_matriz_serial_placa.py    # envio pela serial e matriz, sem data/ e sem a placa
│       ├── test_particao.py               # inclui: outra taxa dá os mesmos segmentos em segundos
│       ├── test_run_protocol_opcoes.py    # combinações de opções que o run_protocol recusa
│       ├── test_protocol_c.py             # Protocolo C: limites binomiais, D2 do C2, recalibração (variantes, critério 12/15, só o bias), divisão conferida
│       ├── test_selecao.py                # folds internos não leem o teste externo; escolher fica com o mais barato na margem
│       ├── test_tempo_memoria.py          # análise de tempo e memória do firmware, sem a placa
│       └── test_varredura_snr.py          # SNR exata no segmento, ruído rosa 1/f, recorte decimado igual ao 02, limite E e as duas regras
│
├── notebooks/                # notebooks de análise/visualização
│
├── firmware/                 # pastas vazias: o firmware está no repositório lab3-firmware-stm32
│
├── experiments/
│   └── registry.csv          # registro estruturado de cada rodada de experimento
│
├── reports/                  # figuras, tabelas e outputs para os relatórios
│   ├── c_reference/          # reference_data.h: entrada int16 + MFCC esperado, para o porte em C;
│   │                         #   estagios/: referências por estágio e tolerâncias do A4 (08), lidas pelo harness em C;
│   │                         #   clipes_demo.h: clipes de áudio para a flash da placa (09)
│   ├── classifier/           # estudo de escolha do classificador (compare_classifiers.json; SVM linear: compare_classifiers_svm_lin.json)
│   ├── decimation/           # métricas, figuras e relatório da escolha da taxa
│   ├── embarcado/            # resultados da placa
│   │   ├── capturas/         #   logs da serial como chegaram (rodadas, demonstrações, testes no Windows e no Linux)
│   │   ├── clipes_22_10/     #   matriz nos clipes da flash (matriz_clipes_placa.py)
│   │   ├── serial_07_11/     #   295 + 40 segmentos pela serial contra o Python (matriz_serial_placa.py)
│   │   ├── serial_ruido_15_11/  # robustez ao ruído na placa (matriz_serial_placa.py --lista)
│   │   └── tempo_memoria/    #   tempo, memória e pilha do firmware final (run_tempo_memoria.py)
│   ├── exploration/          # figuras dos scripts exploratórios
│   ├── modelo_final/         # lda_final.json (parâmetros para o firmware) e escores_referencia.csv, do 05; lda_modelo.h, do 06
│   ├── signature/            # caracterização da assinatura acústica
│   └── validation/           # uma pasta por rodada (expNNN_*): metrics.json e folds.csv; tabelas .md na raiz
│       ├── consolidacao/     # tabelas .tex, figuras e rastreabilidade do relatório (run_consolidacao.py)
│       ├── controle_harmonicos/  # controle dos harmônicos do eixo (inspect_lda_harmonicos.py)
│       ├── protocolo_c/      # Protocolo C: tabela única, recalibração e rotação deixada de fora
│       └── varredura_snr/    # robustez ao ruído: curva, margem e o recorte para a placa (placa/)
│
└── docs/                     # proposta, documentação técnica complementar
    ├── pipeline.md           # este documento
    ├── contrato_numerico.md  # o que o C tem que reproduzir da extração de características (Fase 2)
    ├── hustmotor.md          # HUSTmotor para o Protocolo C: licença, download, blocos, geometria e rotação
    └── criterio_deteccao_novidade.md  # candidatos, limiar e regra de escolha da detecção de novidade (bpfo_0.3mm)
```

## Instalação e testes

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O `requirements.txt` instala o PyTorch **só para CPU**, pelo índice do PyTorch (linha `--extra-index-url` no topo do arquivo). É proposital: as redes do estudo do classificador têm menos de 2.400 parâmetros, a GPU não acelera nada nesse tamanho, e a versão CUDA ocupa alguns GB. Quem tiver GPU e quiser a versão CUDA pode instalá-la no próprio venv, sem mudar o `requirements.txt`.

Os testes não dependem de `data/` e devem passar logo depois de clonar:

```bash
pytest tests/
```

## Pipeline

Baixe os 5 arquivos `.mat` do Mendeley e coloque em `data/raw/`. Depois, o pipeline na ordem numérica:

```bash
python scripts/pipeline/01_convert_mat_to_pcm.py   # .mat → PCM int16 a 51,2 kHz
python scripts/pipeline/02_decimate_pcm.py         # decima para a taxa de trabalho
python scripts/pipeline/03_make_splits.py          # confere que a partição versionada bate
python scripts/pipeline/04_extract_features.py     # MFCC de cada segmento → data/processed/features/
python scripts/pipeline/05_train_classifier.py --sem-registro   # modelo final → reports/modelo_final/teste/
python scripts/pipeline/06_export_lda_header.py --sem-registro  # .h da LDA → reports/modelo_final/teste/
python scripts/pipeline/07_export_lda_test_vectors.py                 # vetores de teste → reports/modelo_final/lda_vetores_teste.h
python scripts/pipeline/08_export_stage_references.py                 # referências por estágio → reports/c_reference/estagios/
python scripts/pipeline/09_export_clips.py                            # clipes para a flash → reports/c_reference/clipes_demo.h
python scripts/pipeline/10_prepare_hustmotor.py                       # HUSTmotor (data/raw/hustmotor/) → data/processed/hustmotor/
python scripts/pipeline/11_divisao_hustmotor.py                       # divisão do HUSTmotor por rotação → data/processed/hustmotor/divisao_rotacao.json
```

Para reproduzir, o `05` e o `06` rodam com `--sem-registro`: gravam em `reports/modelo_final/teste/` e não tocam no `registry.csv` nem nos arquivos versionados. Com `--responsavel <nome>`, eles registram uma rodada nova e sobrescrevem o modelo versionado: o `05` troca o id e o SHA-256 do `lda_final.json`, e aí o `06` aborta, os testes que conferem o `.h` versionado falham e o painel acusa modelo diferente do da placa. Só se registra um modelo novo por decisão da equipe.

O `03` é determinístico: num clone novo, ele reconstrói exatamente o `splits.json` versionado e avisa que "já existe e é idêntico". Se disser que o arquivo é **diferente**, os dados reconstruídos não são os mesmos das rodadas registradas — pare e investigue antes de seguir.

### `05`: modelo final

O `05` treina a LDA uma única vez, com os 295 segmentos e sem aumento, e grava em `reports/modelo_final/lda_final.json` os parâmetros nas duas formas equivalentes de `models/lda.py`: a padronizada (média, escala, coeficientes e intercepto) e a dobrada, `escore = pesos · x + bias`, que é a que o firmware usa. A convenção é **escore ≥ 0 → falha**: o empate exato segue o sklearn, que o classifica como falha (campo `escore_zero` do JSON). As 26 features seguem a ordem do `04` (as 13 médias, depois os 13 desvios, populacionais). Antes de gravar, o script confere que o JSON reproduz o sklearn em todos os segmentos. O `escores_referencia.csv` traz o escore de cada segmento para conferir a implementação em C. Esses segmentos são os do próprio treino: **o `05` não mede desempenho**, que continua sendo o do Protocolo B (`exp015`). O modelo final classifica corretamente os 295 segmentos, inclusive os da `bpfo_0.3mm`, porque todos estão no treino; **uma demonstração com trechos dessas gravações reproduz o treino, não o Protocolo B**, e não deve ser apresentada como desempenho. O tamanho do escore (±1.300 a ±1.600 nesses segmentos) também não é medida de confiança: a margem vem de o treino conter as cinco gravações, e não diz nada sobre uma gravação nova. **O firmware usa só o sinal do escore.** O JSON é o contrato com o firmware: além dos pesos, ele descreve a cadeia que produz as 26 features nos pontos em que um porte em C costuma divergir sem erro (bloco `entrada`): a origem do sinal a 12,8 kHz (FIR Kaiser de 147 taps, corte em 5.760 Hz, decimação por 4, com o atraso de grupo descartado), escala da entrada (`int16 / 32767`, e não `/ 32768` como a conversão Q15 da CMSIS), janela de Hann simétrica (`np.hanning`), FFT de 512 pontos com espectro de potência, banco de Mel HTK de 20 Hz a fs/2 com os bins já calculados, log natural com `+1e-10`, DCT-II ortonormal e desvio populacional. Um teste refaz o MFCC só a partir dessa descrição e compara com o `dsp.mfcc`. Ao comparar o C com o `escores_referencia.csv`, use tolerância relativa (~1e-3) no escore, porque o MFCC em `float32` difere um pouco do Python; a previsão tem que ser idêntica. Ele exige as features de referência (sem `--norm-clipe`); se o `04` foi rodado com outra opção, rode-o de novo sem opções antes.

### `06`: modelo em C

O `06` lê o `lda_final.json` e gera o `reports/modelo_final/lda_modelo.h` com a forma dobrada em float32: `LDA_N_FEATURES`, `lda_pesos`, `lda_bias`, `lda_escore()` e `lda_e_falha()`, com a regra de empate do JSON (escore ≥ 0 → falha). O número de características sai do JSON, sem 26 fixo: o `.h` é regenerado sempre que o modelo final mudar. O cabeçalho usa `float` (o `float32_t` da CMSIS) e não inclui a CMSIS, para compilar também no host. Ele traz o id da rodada do modelo, o SHA-256 do JSON e o commit do gerador; numa rodada registrada, o script aborta se o JSON ou o próprio script tiverem mudanças não commitadas. Antes de gravar, ele lê o `.h` de volta e calcula o escore dos 295 segmentos em float32, em Python e compilando o `.h` no host, e aborta se a previsão mudar em algum segmento ou se o escore se afastar do `escores_referencia.csv` mais que o limite do arredondamento da soma em float32, (N + 3)·2⁻²⁴·(Σ|pesos·x| + |bias|), calculado segmento a segmento. Como as features são as mesmas do Python, esse arredondamento é o único erro esperado; a tolerância relativa de 1e-3 do JSON é a do firmware, que calcula também o MFCC em float32, e aqui deixaria passar um erro de ~1,3 no escore. No Cortex-M4 o gcc funde multiplicação e soma (`vfma.f32`), então o escore da placa não é bit a bit o do host, mas fica dentro do mesmo limite. A conferência completa precisa das features do `04` sem opções. Sem `data/`, os testes conferem o `.h` versionado: o SHA-256 escrito nele tem que ser o do `lda_final.json` (senão o `05` rodou de novo e o `06` não), e, compilado, ele tem que dar o escore do segmento 0 do CSV a partir do MFCC do `reference_data.h`. Como no `05`, os segmentos são os do treino: isso confere o porte, não mede desempenho.

### `07`: vetores de teste

O `07` gera o `reports/modelo_final/lda_vetores_teste.h`, que o firmware ([`lab3-firmware-stm32`](https://github.com/claraamp/lab3-firmware-stm32)) usa no marco de 13/10: as 26 características do `04` de cada segmento (todos, ou `--por-gravacao K` igualmente espaçados), o escore e a decisão do `escores_referencia.csv` e a tolerância de cada segmento. A tolerância é o limite do `06` mais o arredondamento do próprio escore de referência para float32. O `.h` também lista, por gravação, o segmento de menor |escore|, que a placa mostra em laço nos LEDs. Antes de gravar, o script confere que o `lda_modelo.h` versionado é o deste `lda_final.json` e que a soma em float32, em Python e com o `.h` compilado no host, a partir dos valores lidos de volta do arquivo gerado, fica dentro da tolerância e dá a decisão do CSV em todos os segmentos. Ele traz o bias do modelo (`VT_LDA_BIAS`), que o firmware compara com o do `lda_modelo.h` para recusar cabeçalhos de modelos diferentes. Como as características vêm do Python, o marco confere o porte do classificador, não o desempenho. Não há linha no registry: o arquivo é uma fixture de teste, e o resultado é o log da placa. Como o `.h` grava o commit do gerador, o script aborta, ao gravar no caminho versionado, se ele mesmo, o `lda_final.json`, o `escores_referencia.csv` ou o `lda_modelo.h` tiverem mudanças não commitadas; com `--out` em outro caminho, só avisa e marca o commit com `+mudancas`. Precisa das features do `04` sem opções.

### `08`: referências por estágio

O `08` gera as referências por estágio do porte em C, em `reports/c_reference/estagios/`: para o segmento do meio de cada gravação, a saída em float64 de cada estágio da cadeia (escala, janela, espectro, Mel, log-Mel, MFCC, média, desvio e escore), em binários little-endian com um `indice.txt` por segmento, e o `tolerancias.txt` com as tolerâncias do A4 (`dsp.ESTAGIOS_PORTE`). É o que o harness em C do firmware ([`lab3-firmware-stm32`](https://github.com/claraamp/lab3-firmware-stm32), `tests/host/harness`) lê para conferir o porte estágio a estágio, no PC. Os intermediários saem de `dsp.estagios_mfcc`, que usa o mesmo `dsp.mfcc` da extração; o script aborta se o MFCC ou o escore não forem os do `04` e do `05`. As tolerâncias, a métrica de cada estágio e como foram fixadas estão no `docs/contrato_numerico.md` (seção "Tolerâncias por estágio (A4)", `inspect_tolerancias_estagio.py`). Como o `07`, grava o commit e aborta, no caminho versionado, se o script, o `dsp.py`, o `lda_final.json` ou o `escores_referencia.csv` tiverem mudanças não commitadas. Sem `data/`, o teste confere que as referências versionadas são as do `dsp` atual, recalculando-as a partir do `int16` gravado nelas.

### `09`: clipes para a flash

O `09` gera o `reports/c_reference/clipes_demo.h`, com o áudio que o firmware grava na flash para rodar a cadeia inteira na placa, do som à decisão:
- **Clipes a 12,8 kHz:** por gravação, 2 segmentos consecutivos de 1 s (`--segundos`), a partir do segmento do meio, o mesmo das referências do `08`. Vêm em `int16` do PCM decimado, com o id, a classe, o escore e a decisão de cada segmento.
- **Recorte a 51,2 kHz:** um segmento de 1 s do PCM original, para demonstrar a decimação na placa. Tem as margens do decimador causal de fase 0 (75 amostras antes e 73 depois; descartar as 37 primeiras saídas) e os 147 coeficientes do FIR.

O recorte decimado em float64 e requantizado é, bit a bit, o primeiro segmento do clipe da `normal`. Na placa, a decimação é em float32, e uma amostra perto de um empate pode arredondar para o outro lado: o script emula o `arm_fir_decimate_f32` (sem e com FMA), aborta se o resultado diferir do clipe em mais de 1 LSB ou se o escore sair da tolerância do A4, e registra no `.h` quantas amostras diferem (1 sem FMA e 2 com FMA no recorte atual). Na placa, compare o recorte com folga de 1 LSB e o escore com tolerância. Como a entrada do MFCC é `int16`, a placa requantiza a saída do decimador antes do MFCC (`lrintf(y · 32767.0f)`, saturado em [−32768, 32767] antes do cast; `docs/contrato_numerico.md`, "Decimação", item 4): **a referência da placa é o `RECORTE_ESCORE_REF_INT16`**, o escore do CSV. O `RECORTE_ESCORE_REF`, da saída em float, é só diagnóstico; as duas diferem em ~7e-6, perto da tolerância de 1e-5.

**No firmware:** o `.h` só declara os vetores (`extern const`). Eles são definidos no único `.c` que fizer `#define CLIPES_DEMO_DADOS` antes do `#include "clipes_demo.h"`; os outros `.c` só incluem. Com `static const` no `.h`, cada `.c` que o usasse levaria uma cópia do áudio, e dois já passariam dos 512 KB da flash. Com o firmware atual (vetores do `07` incluídos), a flash fica em cerca de 420 KB de 512 KB. O recorte inteiro em float não cabe na RAM (200 KB, contra 128 KB): a placa decima em blocos de 512 amostras (128 saídas, um passo de quadro). Como 51.348 = 100 × 512 + 148, depois dos blocos cheios vem uma última chamada com 148 amostras (`RECORTE_ULTIMO_BLOCO`); sem ela, as 37 últimas saídas se perdem sem erro. O laço está no comentário do `.h`.

O padrão ocupa cerca de 350 KB, dentro do orçamento de flash do firmware (cerca de 434 KB para áudio); o script aborta acima dele (`--orcamento-kb`). Antes de gravar, ele confere que o escore de cada segmento, recalculado do `int16`, é o do `escores_referencia.csv`.

**Todos os segmentos dos clipes estão no treino do modelo final:** a demonstração na placa mede a fidelidade do porte (se a placa reproduz o Python), não o desempenho do classificador, que continua sendo o do Protocolo B. Como o `07` e o `08`, o script grava o commit e aborta, no caminho versionado, se houver mudanças não commitadas. Sem `data/`, o teste do clipe lê o `.h` versionado, recalcula as características de cada segmento e confere o escore e a decisão gravados. Ele também decima o recorte em float64 e confere que sai o clipe a 12,8 kHz, e em float32, como a CMSIS, que fica a no máximo 1 LSB dele. Se houver compilador C, confere que o `.h` compila, que os dados são definidos num único `.c` e, com o `arm-none-eabi-gcc`, que só esse `.c` leva o áudio para a flash.

Para conferir que a reconversão reproduziu a original (o `manifest.json` é versionado e a conversão é determinística, então número de amostras e pico PCM devem bater):

```bash
python scripts/exploration/inspect_pcm.py
```

## Validação, placa e estudos

Os comandos abaixo com `--responsavel <nome>` são os das rodadas registradas da equipe: cada um grava uma linha nova no `registry.csv` e as saídas nas pastas versionadas. Para reproduzir um resultado sem mexer no registro, troque `--responsavel <nome>` por `--sem-registro`.

Os protocolos de validação seguem o fluxo `04` → `mfcc_features.npz` → `run_protocol`. O `run_protocol` não extrai features, lê as do `04` e aborta se o `manifest_features.json` faltar ou tiver sido gerado com outra partição ou outros parâmetros de MFCC — nesse caso, rode o `04` de novo.

```bash
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/validation/run_protocol.py --protocolo A --tarefa multiclasse --responsavel <nome>
python scripts/validation/run_protocol.py --protocolo B --sem-c0      # ablação do ganho
python scripts/validation/run_protocol.py --protocolo B --permutar    # controle de permutação
python scripts/validation/run_protocol.py --protocolo B --permutar --semente 1   # uma rodada por semente
python scripts/validation/run_protocol.py --protocolo B --segundos-treino 5 --semente 1   # curva de aprendizado (mín. 2 s; sorteio repartido entre as gravações)
python scripts/validation/run_protocol.py --protocolo B --sem-registro   # teste, não registra

# Protocolo C no HUSTmotor (precisa do 10 e do 11; lê o data/processed/hustmotor/)
python scripts/validation/run_protocol_c.py --sintetico                    # auto-teste, sem registro
python scripts/validation/run_protocol_c.py --responsavel <nome>           # LDA sem retreino, C1 e C2
python scripts/validation/run_protocol_c.py --recalibracao --responsavel <nome>   # LDA recalibrada pela normal
python scripts/validation/run_protocol_c.py --recalibracao --sem-registro  # teste: grava em reports/validation/teste_protocolo_c/
python scripts/validation/run_protocol_c.py --rotacao-fora --sintetico   # rotação deixada de fora: auto-teste
python scripts/validation/run_protocol_c.py --rotacao-fora --responsavel <nome>   # D2 nos 4 folds; D1 a 20 e 30 Hz
python scripts/validation/run_tabela_protocolo_c.py   # tabela única do Protocolo C (só lê; confere o registry)

# robustez ao ruído: ruído no int16 a 51,2 kHz, B sem e com aumento, margem do escore e emulação em float32
python scripts/validation/run_varredura_snr.py --sem-registro --rapido   # teste: 1 semente, 30, 10 e 0 dB
python scripts/validation/run_varredura_snr.py --sem-registro            # teste: a grade inteira, 10 sementes (~10 min)
python scripts/validation/run_varredura_snr.py --responsavel <nome>      # uma linha por ponto da grade
python scripts/validation/run_varredura_snr.py --exportar-placa          # recorte para a placa; sha256 do .bin: 11f26bb8bfe42b5dfaf71523afe5c5d6bc690187d6791cb52880cd0f424066f8

# sistema embarcado: os 295 + 40 segmentos e o recorte da robustez ao ruído pela serial, e a matriz contra o Python (precisa do pyserial)
python scripts/validation/matriz_serial_placa.py --simulada                       # sem a placa (placa simulada em Python)
# emular_placa: do repo lab3-firmware-stm32 (Linux/macOS). Lá: cmake -S tests/host/harness -B build/harness -G Ninja,
# cmake --build build/harness e ./build/harness/emular_placa &, que imprime a porta /dev/pts/N usada abaixo
python scripts/validation/matriz_serial_placa.py --porta /dev/pts/N --sem-registro   # contra o emular_placa do firmware
python scripts/validation/matriz_serial_placa.py --porta COM5 --sem-registro      # ensaio com a placa (saídas fora do Git)
python scripts/validation/matriz_serial_placa.py --porta COM5 --responsavel <nome>   # a rodada registrada
python scripts/validation/matriz_serial_placa.py --captura reports/embarcado/capturas/serial-DD-MM.log --sem-registro   # refaz a matriz
python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --sem-registro --limite 10   # robustez na placa: ensaio (10 por ruído)
python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --responsavel <nome>         # robustez na placa: os 1.788 (precisa do .bin do --exportar-placa)
python scripts/validation/matriz_serial_placa.py --lista --porta COM5 --ids 3:48,3:3,3:57,3:59 --pausa 4   # teste dos LEDs: só esses, na ordem dada, 4 s em cada um com a cor esperada (ensaio)
python scripts/validation/matriz_serial_placa.py --captura reports/embarcado/capturas/serial-ruido-15-11-2.log --lista --sem-registro   # refaz a matriz da robustez (15/11)

# painel de diagnóstico (interface/run_painel.py; precisa do Tkinter, que vem com o Python no Windows; no Linux,
# sudo apt install python3-tk): na demonstração, a porta da placa (aperte o RESET depois de abrir); na recepção,
# a porta é do matriz_serial_placa.py e o painel segue a captura que ele grava (a pasta das saídas de ensaio)
python scripts/interface/run_painel.py --porta COM5                                    # demonstração (origens C e D)
python scripts/interface/run_painel.py --seguir reports/embarcado/teste_serial          # recepção: abra antes do envio
python scripts/interface/run_painel.py --arquivo reports/embarcado/capturas/serial-ruido-15-11-2.log --ritmo 0.3   # sem a placa
python scripts/interface/run_painel.py --arquivo reports/embarcado/capturas/serial-07-11.log --sem-janela          # texto

# tempo e memória do firmware final (seção 3.6 do relatório final): a captura é a de uma rodada dos 295 + 40
# com o firmware gravado com -DMEDIDA=ON (linhas med); o orçamento e a pilha estática vêm do build padrão,
# pelos tools/orcamento_memoria.py --markdown e tools/pilha_estatica.py do firmware; --linker é o "Memory
# region" do build padrão, e --comparar-captura confere os escores bit a bit contra outra rodada
python scripts/validation/run_tempo_memoria.py --captura reports/embarcado/capturas/tempo-memoria-DD-MM.log \
    --orcamento reports/embarcado/tempo_memoria/orcamento.md --pilha reports/embarcado/tempo_memoria/pilha_estatica.txt \
    --linker 450236 34992 --comparar-captura reports/embarcado/capturas/serial-07-11.log \
    --firmware-commit <hash> --responsavel <nome>    # resumo.md, tabelas.tex e segmentos.csv em reports/embarcado/tempo_memoria/

# tabelas (leem os metrics.json; não treinam nem registram)
python scripts/validation/run_tabela_comparativa.py   # provisório × oficial
python scripts/validation/run_tabela_controles.py     # sem c0, normalização, permutação e curva de aprendizado
python scripts/validation/run_tabela_aumento.py       # B com e sem aumento: médias, pior caso e fator de expansão

# controle dos harmônicos do eixo: peso da LDA nas bandas de Mel em que a normal
# tem harmônicos mais fortes (lê o picos_metrics.json do identify_tonal_peaks.py)
python scripts/exploration/inspect_lda_harmonicos.py --sintetico   # auto-teste
python scripts/exploration/inspect_lda_harmonicos.py --responsavel <nome>

# pente harmônico (BPFO/BPFI): candidatos de resolução A/B/C, decisão pela validação interna
python scripts/exploration/inspect_comb_features.py --sintetico    # auto-teste
python scripts/exploration/inspect_comb_features.py --responsavel <nome>

# ablação da normalização RMS por segmento: reextrai e roda de novo
python scripts/pipeline/04_extract_features.py --norm-clipe
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/pipeline/04_extract_features.py       # volta ao padrão (sem normalização)

# estudo da escolha do classificador: LDA × MLP × CNN 1D × CNN 2D por validação
# interna ao treino do B (~33 min completo; --blocos-externos 1 leva ~1/6 disso)
python scripts/exploration/compare_classifiers.py --sintetico       # auto-teste, sem registro
python scripts/exploration/compare_classifiers.py --responsavel <nome>
python scripts/exploration/compare_classifiers.py --blocos-externos 1 --sem-registro   # conferência rápida
python scripts/exploration/compare_classifiers.py --modelos lda,svm_lin --especificidade-minima 0.95 \
    --saida reports/classifier/compare_classifiers_svm_lin.json --responsavel <nome>   # SVM linear (exp264)

# referência para o porte em C (Fase 2)
python scripts/pipeline/04_extract_features.py --ref-c

# aumento de dados: o 04 gera N variantes de cada segmento e o run_protocol,
# com --aumento, soma ao treino de cada fold só as que leem segmentos de treino
python scripts/pipeline/04_extract_features.py --aumento 4
python scripts/validation/run_protocol.py --protocolo B --aumento --responsavel <nome>
python scripts/validation/run_protocol.py --protocolo B --aumento --permutar   # controle
# ablação por técnica (deslocamento | estiramento | ruido) e modo do estiramento
python scripts/pipeline/04_extract_features.py --aumento 4 --tecnicas ruido
python scripts/pipeline/04_extract_features.py --aumento 4 --modo-estiramento velocidade
# outro sorteio das variantes, com os mesmos parâmetros (uma rodada por semente)
python scripts/pipeline/04_extract_features.py --aumento 4 --semente-aumento 1
python scripts/validation/run_protocol.py --protocolo B --aumento --responsavel <nome>

# Protocolo B em outra taxa, sem mudar a de produção: cada etapa com --fs grava
# em caminhos próprios (pcm_decimated/25600/, splits_25600.json, features/25600/)
python scripts/pipeline/02_decimate_pcm.py --fs 25600
python scripts/pipeline/03_make_splits.py --fs 25600      # confere a partição versionada
python scripts/pipeline/04_extract_features.py --fs 25600
python scripts/validation/run_protocol.py --protocolo B --fs 25600 --responsavel <nome>
python scripts/pipeline/04_extract_features.py            # a referência, a 12,8 kHz
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/validation/run_tabela_taxas.py --rodadas <exp 12,8 kHz> <exp 25,6 kHz>

# resultados consolidados para o relatório: tabelas .tex (vírgula decimal), matriz de
# confusão do B, figura dos controles, aumento, harmônicos, taxas e rastreabilidade (cada
# número → rodada → registry). Aborta se alguma rodada não tiver linha no registry ou se
# as métricas não baterem. Não treina nem registra.
python scripts/validation/run_consolidacao.py
python scripts/validation/run_consolidacao.py --sem-taxas    # sem a tabela 12,8 × 25,6 kHz
```

O teste nunca é aumentado: todo fold é avaliado nos segmentos originais. Rodar o `04` sem `--aumento` apaga o `mfcc_aumento.npz` de uma rodada anterior, para que ele não seja lido como se fosse da extração atual.

Commite o código **antes** de uma rodada registrada: o `registry.csv` grava o hash do commit, e uma rodada feita com código não commitado aponta para um estado que não existe.

### Capturas `ch340-ruido-22-11*`

As capturas do teste de 22/11 no Windows com o grupo misto da robustez (`ch340-ruido-22-11*.log`, em `reports/embarcado/capturas/`, feitas com `--lista`, `--ids` e `--pausa`) têm no cabeçalho a lista `bcc8925a416a`. O `--exportar-placa` regravou o `manifest.json` do recorte com o commit em que rodou (`4a0558a`), e para refazer a análise de uma dessas capturas com `--captura ... --lista` o `git_commit` do manifesto tem de ser esse.

## Estudos exploratórios

O pipeline **aplica** decisões, não as toma. A comparação entre taxas candidatas, que definiu os 12.800 Hz, é um estudo pontual e está em `scripts/exploration/`; ele fica versionado para a decisão continuar auditável, mas não precisa ser reexecutado a cada rodada:
 
```bash
python scripts/exploration/compare_decimation_rates.py --condicao 0Nm
```

A caracterização da assinatura acústica segue a mesma lógica. Cada script tem um auto-teste com sinal sintético de resposta conhecida (`--sintetico`, que não registra e grava em `reports/signature/sintetico/`, fora do Git). O `confirm_bpf_envelope.py` e o `inspect_signature_mel.py` leem as frequências medidas do `reports/signature/picos_metrics.json`, então o `identify_tonal_peaks.py` roda antes deles. O `confirm_bpf_envelope.py` precisa da vibração em `data/raw/vibracao/`, e o `inspect_signature_mel.py` aborta se as features do `04` não forem as de referência (sem `--norm-clipe`, parâmetros de MFCC do `config.py`):

```bash
python scripts/exploration/identify_tonal_peaks.py --sintetico       # auto-teste
python scripts/exploration/inspect_signature_spectra.py
python scripts/exploration/identify_tonal_peaks.py
python scripts/exploration/confirm_bpf_envelope.py              # os 4 canais de vibração + microfone
python scripts/exploration/inspect_signature_mel.py
```
 
Os scripts de `scripts/exploration/` e `scripts/validation/` não escrevem em `data/`. Os módulos na raiz de `scripts/` (`config`, `dsp`, `pcm_io`, `experimentos`) não são executáveis: são importados pelas etapas e pelos estudos, para que todos usem a mesma implementação e os mesmos parâmetros. Nas pastas por assunto, como `validation/`, os executáveis começam com `run_` e o resto é importado.
