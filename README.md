# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

Sistema embarcado que classifica, em tempo real e a partir de sinal acústico, o estado de operação de um motor rotativo (normal ou com defeito de rolamento — BPFI/BPFO) usando uma cadeia de processamento digital de sinais e um classificador leve, portados para um microcontrolador STM32F411CEU6. Um protótipo em Python valida a cadeia de processamento antes do porte para o hardware, permitindo comparar a solução em software e em firmware.

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

**Documentação do projeto:**
- Proposta de trabalho (aprovada em 14/09) — `docs/proposta-trabalho.pdf`
- Espaço de gerenciamento (tarefas, cronograma, decisões): [Notion](https://app.notion.com/p/3d906cf573128129adfbc99d69e19645)
- Convenções do projeto: [`CONVENTIONS.md`](./CONVENTIONS.md)

## Status

Fase 1 — Protótipo em Python (em andamento). Ver o quadro de tarefas no Notion para o estado detalhado de cada etapa.

## Dataset

Jung, W.; Kim, J.; Park, Y.-H. et al. (2023), subconjunto acústico (microfone), 5 classes — `0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10` — 60 s cada, 51,2 kHz, formato `.mat`.
Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6).

Os arquivos `.mat` **não são versionados** neste repositório (ver `.gitignore`) — são grandes e já têm DOI fixo. Para reproduzir:

1. Baixe os 5 arquivos `.mat` do link acima e coloque em `data/raw/`.
2. Rode o pipeline a partir de `scripts/pipeline/`, na ordem numérica (ver seção abaixo).

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
│       └── pcm_raw/          # saída do pipeline — .bin não versionados, manifest.json versionado
│
├── scripts/
│   ├── exploration/          # scripts exploratórios (investigação do formato .mat, etc.)
│   └── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│       └── 01_convert_mat_to_pcm.py
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
├── reports/                  # figuras, tabelas e outputs para os relatórios (1º/2º parcial, final)
│
└── docs/                     # proposta, documentação técnica complementar
```

Ver [`CONVENTIONS.md`](./CONVENTIONS.md) para as convenções de nomenclatura de scripts, classes e commits, e para o formato do registro de experimentos.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# depois de colocar os .mat em data/raw/:
python scripts/pipeline/01_convert_mat_to_pcm.py
```

## Fluxo de branches

Trabalho de cada tarefa em uma branch própria (`feature/<nome-da-tarefa>`), aberta a partir de `main`. Merge para `main` ao concluir a tarefa, para que `main` reflita sempre o estado real e comparável do projeto — importante para os marcos de entrega (1º parcial, 2º parcial, final).