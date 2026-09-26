# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

Sistema embarcado que classifica, em tempo real e a partir de sinal acústico, o estado de operação de um motor rotativo (normal ou com defeito de rolamento — BPFI/BPFO) usando uma cadeia de processamento digital de sinais e um classificador leve, portados para um microcontrolador STM32F411CEU6. Um protótipo em Python valida a cadeia de processamento antes do porte para o hardware, permitindo comparar a solução em software e em firmware.

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

**Documentação do projeto:**
- Proposta de trabalho (aprovada em 14/09) — `docs/proposta-trabalho.pdf`
- Espaço de gerenciamento (tarefas, cronograma, decisões): [Notion](https://app.notion.com/p/04d931873a6c83b2b5a3017fae7db29a)
- Convenções do projeto: [`CONVENTIONS.md`](./CONVENTIONS.md)

## Status

Fase 1 — Protótipo em Python (em andamento). Conversão, decimação e protocolo de validação concluídos; extração oficial de features e escolha do classificador em andamento. Ver o quadro de tarefas no Notion para o estado detalhado de cada etapa.

## Decisões técnicas fixadas
 
| decisão | valor | onde está a justificativa |
|---|---|---|
| Taxa de amostragem de trabalho | **12.800 Hz** (decimação por fator inteiro 4 a partir de 51,2 kHz) | `reports/decimation/decimation_report.md` |
| Anti-aliasing | FIR Kaiser, 147 taps, corte em 5.760 Hz, 60 dB de atenuação | idem |
| Representação de features | MFCC (25 ms, hop 10 ms, 20 mels, 13 coeficientes) | proposta, seção de Metodologia |
| Plataforma | STM32F411CEU6 (Cortex-M4F, CMSIS-DSP, X-CUBE-AI) | proposta, seção de Infraestrutura |
| Unidade de classificação | segmento de 1 s (12.800 amostras); 59 por gravação, em blocos de 10, 10, 10, 10, 10 e 9 | Notion, Registro de Decisões (protocolo de validação) |
| Protocolo de validação | **A** — blocos temporais (limite otimista); **B** — gravação de falha deixada de fora (resultado principal) | idem; implementação em `scripts/validation/particao.py` |
| Meta de desempenho | acurácia balanceada média do Protocolo B ≥ 85 %, sempre com sensibilidade, especificidade e pior caso | idem; resultados em `reports/validation/` |
 
A escolha de 12.800 Hz é a única taxa com fator de decimação inteiro dentro da faixa de 8–16 kHz — requisito de `arm_fir_decimate_f32` do CMSIS-DSP — e preserva 96,1 % da energia discriminante no pior caso (classes de defeito incipiente, 0,3 mm). O registro completo da comparação está em `experiments/registry.csv` (`exp001`–`exp006`).

O protocolo de validação existe porque o dataset tem **uma única gravação por classe**: qualquer divisão treino/teste dentro de uma gravação deixa os dois no mesmo registro, e o classificador pode separar as classes pela identidade da gravação em vez da falha (foi o que deu acurácia 1,0 no estudo de decimação). O Protocolo B testa cada gravação de falha sem que ela apareça no treino; é o único resultado que conta para a meta. A primeira rodada, provisória, está em `exp007`–`exp012`.

## Dataset

Jung, W.; Kim, S.-H.; Yun, S.-H.; Bae, J.; Park, Y.-H. (2023), *Data in Brief* 48, 109049. Subconjunto acústico (microfone), 5 classes — `0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10` — 60 s cada, 51,2 kHz, formato `.mat`.
Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6).

Esses cinco arquivos são **todo o áudio do dataset**. Os ensaios com carga (2 e 4 Nm), os defeitos de 3,0 mm e as falhas de desbalanceamento e desalinhamento têm só vibração, corrente e temperatura: os autores não gravaram o microfone com carga porque o freio, resfriado a ar, contaminaria o canal acústico (seção 3.1 do artigo).

Os arquivos `.mat` **não são versionados** neste repositório (ver `.gitignore`) — são grandes e já têm DOI fixo. O mesmo vale para os `.bin` gerados a partir deles; apenas os `manifest.json` e o `splits.json` (a partição dos protocolos de validação) entram no Git.

Isso significa que **um clone novo vem com `data/` vazio**. Para reconstruir, ver "Como rodar" abaixo.

## Estrutura do repositório

```
diagnostico-acustico-motores/
├── README.md
├── CONVENTIONS.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                  # .mat originais (baixados manualmente, não versionados)
│   └── processed/
│       ├── pcm_raw/          # saída do 01 — .bin não versionados, manifest.json versionado
│       ├── pcm_decimated/    # saída do 02, um subdiretório por taxa (ex.: 12800/)
│       ├── splits/
│       │   └── splits.json   # saída do 03 — partição dos protocolos A e B, VERSIONADA
│       └── features/         # saída do 04 — mfcc_features.npz + manifest_features.json, NÃO versionados
│
├── scripts/
│   ├── config.py             # parâmetros compartilhados entre etapas
│   ├── dsp.py                # blocos de processamento de sinais (PSD, MFCC, decimação)
│   ├── pcm_io.py             # leitura e escrita dos PCM e dos manifests
│   ├── experimentos.py       # escrita do registro de experimentos
│   ├── exploration/          # estudos e inspeções, sem numeração
│   │   ├── inspect_mat_keys.py
│   │   ├── inspect_signal_data.py
│   │   ├── inspect_pcm.py                 # sanidade da conversão + caráter do sinal
│   │   ├── inspect_class_spectra.py       # PSD por classe e banda necessária
│   │   └── compare_decimation_rates.py    # estudo que definiu a taxa de trabalho
│   ├── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│   │   ├── 01_convert_mat_to_pcm.py
│   │   ├── 02_decimate_pcm.py             # aplica a taxa definida em config.py
│   │   ├── 03_make_splits.py              # segmenta e gera a partição, uma única vez
│   │   ├── 04_extract_features.py         # MFCC por segmento da partição → .npz
│   │   └── 05_train_classifier.py         # (previsto) modelo final para o firmware
│   └── validation/           # protocolo de validação do classificador
│       ├── particao.py                    # segmentos, folds A e B, verificação
│       ├── metricas.py                    # sensibilidade, especificidade, acurácia balanceada
│       └── run_protocol.py                # executável: roda A ou B e registra o resultado
│
├── tests/                    # pytest; não dependem de data/
│   ├── conftest.py
│   ├── test_dsp.py
│   └── validation/
│       └── test_particao.py
│
├── notebooks/                # notebooks de análise/visualização
│
├── firmware/                 # Fase 2 — porte para o STM32F411CEU6
│   ├── Core/
│   └── Drivers/
│
├── experiments/
│   └── registry.csv          # registro estruturado de cada rodada de experimento
│
├── reports/                  # figuras, tabelas e outputs para os relatórios
│   ├── c_reference/          # reference_data.h: entrada int16 + MFCC esperado, para o porte em C
│   ├── decimation/           # métricas, figuras e relatório da escolha da taxa
│   ├── exploration/          # figuras dos scripts exploratórios
│   └── validation/           # uma pasta por rodada: metrics.json e folds.csv
│
└── docs/                     # proposta, documentação técnica complementar
```

Ver [`CONVENTIONS.md`](./CONVENTIONS.md) para as convenções de nomenclatura de scripts, classes e commits, e para o formato do registro de experimentos.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Baixe os 5 arquivos `.mat` do Mendeley e coloque em `data/raw/`. Depois, o pipeline na ordem numérica:

```bash
python scripts/pipeline/01_convert_mat_to_pcm.py   # .mat → PCM int16 a 51,2 kHz
python scripts/pipeline/02_decimate_pcm.py         # decima para a taxa de trabalho
python scripts/pipeline/03_make_splits.py          # confere que a partição versionada bate
python scripts/pipeline/04_extract_features.py     # MFCC de cada segmento → data/processed/features/
```

O `03` é determinístico: num clone novo, ele reconstrói exatamente o `splits.json` versionado e avisa que "já existe e é idêntico". Se disser que o arquivo é **diferente**, os dados reconstruídos não são os mesmos das rodadas registradas — pare e investigue antes de seguir.

Os testes não dependem de `data/` e devem passar logo depois de clonar:

```bash
pytest tests/
```

Para rodar os protocolos de validação. O fluxo é `04` → `mfcc_features.npz` → `run_protocol`: o `run_protocol` não extrai features, lê as do `04` e aborta se o `manifest_features.json` faltar ou tiver sido gerado com outra partição ou outros parâmetros de MFCC — nesse caso, rode o `04` de novo.

```bash
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/validation/run_protocol.py --protocolo A --tarefa multiclasse --responsavel <nome>
python scripts/validation/run_protocol.py --protocolo B --sem-c0      # ablação do ganho
python scripts/validation/run_protocol.py --protocolo B --permutar    # controle de permutação
python scripts/validation/run_protocol.py --protocolo B --sem-registro   # teste, não registra

# ablação da normalização RMS por segmento: reextrai e roda de novo
python scripts/pipeline/04_extract_features.py --norm-clipe
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/pipeline/04_extract_features.py       # volta ao padrão (sem normalização)

# referência para o porte em C (Fase 2)
python scripts/pipeline/04_extract_features.py --ref-c
```

Commite o código **antes** de uma rodada registrada: o `registry.csv` grava o hash do commit, e uma rodada feita com código não commitado aponta para um estado que não existe.

Para conferir que a reconversão reproduziu a original — o `manifest.json` é versionado e a conversão é determinística, então número de amostras e pico PCM devem bater:

```bash
python scripts/exploration/inspect_pcm.py
```

O pipeline **aplica** decisões, não as toma. A comparação entre taxas candidatas, que definiu os 12.800 Hz, é um estudo pontual e está em `scripts/exploration/`; ele fica versionado para a decisão continuar auditável, mas não precisa ser reexecutado a cada rodada:
 
```bash
python scripts/exploration/compare_decimation_rates.py --condicao 0Nm
```
 
Os scripts de `scripts/exploration/` e `scripts/validation/` não escrevem em `data/`. Os módulos na raiz de `scripts/` (`config`, `dsp`, `pcm_io`, `experimentos`) não são executáveis: são importados pelas etapas e pelos estudos, para que todos usem a mesma implementação e os mesmos parâmetros. Nas pastas por assunto, como `validation/`, os executáveis começam com `run_` e o resto é importado.

## Fluxo de trabalho
 
Cada tarefa em uma branch própria a partir de `main`, com prefixo por tipo de trabalho (`feature/`, `fix/`, `refactor/`), e merge para `main` ao concluir. As regras completas — branches, mensagens de commit e títulos de PR — estão em [`CONVENTIONS.md`](./CONVENTIONS.md), seção 6.