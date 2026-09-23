# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

Sistema embarcado que classifica, em tempo real e a partir de sinal acústico, o estado de operação de um motor rotativo (normal ou com defeito de rolamento — BPFI/BPFO) usando uma cadeia de processamento digital de sinais e um classificador leve, portados para um microcontrolador STM32F411CEU6. Um protótipo em Python valida a cadeia de processamento antes do porte para o hardware, permitindo comparar a solução em software e em firmware.

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

**Documentação do projeto:**
- Proposta de trabalho (aprovada em 14/09) — `docs/proposta-trabalho.pdf`
- Espaço de gerenciamento (tarefas, cronograma, decisões): [Notion](https://app.notion.com/p/04d931873a6c83b2b5a3017fae7db29a)
- Convenções do projeto: [`CONVENTIONS.md`](./CONVENTIONS.md)

## Status

Fase 1 — Protótipo em Python (em andamento). Ver o quadro de tarefas no Notion para o estado detalhado de cada etapa.

## Decisões técnicas fixadas
 
| decisão | valor | onde está a justificativa |
|---|---|---|
| Taxa de amostragem de trabalho | **12.800 Hz** (decimação por fator inteiro 4 a partir de 51,2 kHz) | `reports/decimation/decimation_report.md` |
| Anti-aliasing | FIR Kaiser, 147 taps, corte em 5.760 Hz, 60 dB de atenuação | idem |
| Representação de features | MFCC (25 ms, hop 10 ms, 20 mels, 13 coeficientes) | proposta, seção de Metodologia |
| Plataforma | STM32F411CEU6 (Cortex-M4F, CMSIS-DSP, X-CUBE-AI) | proposta, seção de Infraestrutura |
 
A escolha de 12.800 Hz é a única taxa com fator de decimação inteiro dentro da faixa de 8–16 kHz — requisito de `arm_fir_decimate_f32` do CMSIS-DSP — e preserva 96,1 % da energia discriminante no pior caso (classes de defeito incipiente, 0,3 mm). O registro completo da comparação está em `experiments/registry.csv` (`exp001`–`exp006`).

## Dataset

Jung, W.; Kim, J.; Park, Y.-H. et al. (2023), subconjunto acústico (microfone), 5 classes — `0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10` — 60 s cada, 51,2 kHz, formato `.mat`.
Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6).

Os arquivos `.mat` **não são versionados** neste repositório (ver `.gitignore`) — são grandes e já têm DOI fixo. O mesmo vale para os `.bin` gerados a partir deles; apenas os `manifest.json` entram no Git.

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
│       └── pcm_decimated/    # saída do 02, um subdiretório por taxa (ex.: 12800/)
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
│   └── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│       ├── 01_convert_mat_to_pcm.py
│       └── 02_decimate_pcm.py             # aplica a taxa definida em config.py
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
│   ├── decimation/           # métricas, figuras e relatório da escolha da taxa
│   └── exploration/          # figuras dos scripts exploratórios
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
```

Para conferir que a reconversão reproduziu a original — o `manifest.json` é versionado e a conversão é determinística, então número de amostras e pico PCM devem bater:

```bash
python scripts/exploration/inspect_pcm.py
```

O pipeline **aplica** decisões, não as toma. A comparação entre taxas candidatas, que definiu os 12.800 Hz, é um estudo pontual e está em `scripts/exploration/`; ele fica versionado para a decisão continuar auditável, mas não precisa ser reexecutado a cada rodada:
 
```bash
python scripts/exploration/compare_decimation_rates.py --condicao 0Nm
```
 
Os scripts de `scripts/exploration/` não escrevem em `data/`. Os módulos em `scripts/` (`config`, `dsp`, `pcm_io`, `experimentos`) não são executáveis: são importados pelas etapas e pelos estudos, para que todos usem a mesma implementação e os mesmos parâmetros.

## Fluxo de trabalho
 
Cada tarefa em uma branch própria a partir de `main`, com prefixo por tipo de trabalho (`feature/`, `fix/`, `refactor/`), e merge para `main` ao concluir. As regras completas — branches, mensagens de commit e títulos de PR — estão em [`CONVENTIONS.md`](./CONVENTIONS.md), seção 6.