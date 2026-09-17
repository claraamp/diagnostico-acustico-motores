# Convenções do Projeto — Diagnóstico Acústico de Falhas em Motores Rotativos

Fixadas no início da Fase 1, conforme a Metodologia de Gerenciamento da proposta (item "Padronização"). Objetivo: garantir que os resultados dos experimentos sejam comparáveis entre si ao longo do semestre, independentemente de quem rodou o quê.

## 1. Estrutura de diretórios

```
diagnostico-acustico-motores/
├── README.md
├── CONVENTIONS.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                  # .mat originais do dataset Jung et al. — NÃO versionado (ver .gitignore)
│   └── processed/
│       └── pcm_raw/          # saída do 01_convert_mat_to_pcm.py — .bin NÃO versionados, manifest.json É versionado
│
├── scripts/
│   ├── exploration/          # scripts exploratórios, um-off, sem numeração
│   └── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│
├── notebooks/                # notebooks de análise/visualização (se usados)
│
├── firmware/                 # Fase 2 em diante — porte para STM32F411CEU6
│   ├── Core/
│   └── Drivers/
│
├── experiments/
│   └── registry.csv          # registro estruturado de cada rodada de experimento (ver seção 4)
│
├── reports/                  # figuras, tabelas e outputs usados nos relatórios (1º parcial, 2º parcial, final)
│
└── docs/                     # documentação técnica complementar (ex.: notas sobre o formato do .mat)
```

**Por que `data/raw/` e os `.bin` de `data/processed/` não são versionados:** os 5 arquivos `.mat` somam ~120 MB e os PCM outros ~60 MB — grande demais para um repositório Git normal e, além disso, redundante: o dataset já está publicado com DOI fixo (Mendeley `10.17632/ztmf3m7h5x.6`). O `README.md` deve trazer o link de download e o script `01_convert_mat_to_pcm.py` para qualquer um reconstruir `data/` do zero. O que **é** versionado é o código e os metadados leves (`manifest.json`, `registry.csv`), que são o que realmente precisa ter histórico rastreável.

## 2. Nomenclatura de arquivos e scripts

**Scripts exploratórios** (`scripts/exploration/`): sem prefixo numérico, nome descritivo do que investigam.
Exemplos já usados: `inspect_mat_keys.py`, `inspect_signal_data.py`.

**Scripts de pipeline** (`scripts/pipeline/`): prefixo numérico de duas casas indicando a ordem de execução, seguido do verbo da ação em `snake_case`.
Exemplos: `01_convert_mat_to_pcm.py`, `02_decimate_pcm.py`, `03_extract_features.py`, `04_train_classifier.py`.
Regra: se um script novo precisa rodar *entre* dois existentes, renumerar em vez de usar sufixos como `01b_`.

**Classes/rótulos**: sempre em `snake_case`, minúsculo, consistente entre código, nomes de arquivo e Notion:
`normal`, `bpfi_0.3mm`, `bpfi_1.0mm`, `bpfo_0.3mm`, `bpfo_1.0mm` (5 classes) — e `normal` / `falha` (binário).

**Arquivos de dados processados**: `<classe>.<extensão>` dentro do diretório da etapa do pipeline que os gerou (ex.: `data/processed/pcm_raw/bpfi_0.3mm.bin`).

**Commits Git**: mensagens no imperativo, em português, prefixadas pela frente de trabalho quando ajudar a rastrear: `dsp: converte .mat para PCM int16`, `firmware: implementa FFT em Q15`, `docs: atualiza convenções`.

## 3. Ambiente Python

`requirements.txt` na raiz, com versões fixadas (`pip freeze > requirements.txt` depois de montar o ambiente). Isso evita que um resultado deixe de ser reprodutível por causa de uma versão diferente do `scipy` ou do `numpy` entre integrantes.

## 4. Formato de registro de experimentos

Cada rodada de um experimento (extração de features, treino de classificador, comparação de taxa de decimação, etc.) ganha uma linha em `experiments/registry.csv`. Colunas:

| coluna | descrição |
|---|---|
| `id` | identificador curto e sequencial, ex. `exp001` |
| `data` | data da rodada (AAAA-MM-DD) |
| `etapa` | qual etapa do pipeline foi exercitada, ex. `decimacao`, `extracao_features`, `treino_classificador` |
| `script` | script executado, ex. `02_decimate_pcm.py` |
| `git_commit` | hash curto do commit em que o script estava (`git rev-parse --short HEAD`) — garante que dá pra reproduzir exatamente aquela rodada |
| `parametros` | parâmetros relevantes da rodada, em formato `chave=valor;chave=valor` (ex. `fator_decimacao=4;filtro=fir_lowpass_order8`) |
| `dataset` | qual versão/subset dos dados foi usada |
| `metricas` | resultado quantitativo, em formato `chave=valor;chave=valor` (ex. `acuracia=0.91;f1=0.89`) |
| `responsavel` | quem rodou |
| `notas` | observações livres (ex. algo estranho no resultado, decisão tomada a partir dele) |

Por que CSV e não só um texto corrido: permite comparar experimentos entre si (abrir no pandas/Excel, filtrar por etapa, plotar métrica x parâmetro) — é exatamente o "resultados comparáveis entre si" que a proposta promete. Toda rodada que gerar um número que vá para o relatório parcial deveria ter uma linha aqui, mesmo que o resultado seja negativo.