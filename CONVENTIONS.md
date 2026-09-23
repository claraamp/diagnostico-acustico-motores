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
│   ├── raw/                  # .mat originais (baixados manualmente, não versionados)
│   └── processed/
│       ├── pcm_raw/          # saída do 01 — .bin não versionados, manifest.json versionado
│       └── pcm_decimated/    # saída do 02, um subdiretório por taxa (ex.: 12800/)
│
├── scripts/
│   ├── exploration/          # scripts exploratórios, sem numeração
│   │   ├── inspect_mat_keys.py
│   │   ├── inspect_signal_data.py
│   │   ├── inspect_pcm.py            # sanidade da conversão + caráter do sinal por classe
│   │   └── inspect_class_spectra.py  # PSD por classe e banda necessária por classe
│   └── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│       ├── 01_convert_mat_to_pcm.py
│       └── 02_decimate_pcm.py
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
    └── proposta-trabalho.pdf/
```

**Por que `data/raw/` e os `.bin` de `data/processed/` não são versionados:** os 5 arquivos `.mat` somam ~120 MB e os PCM outros ~60 MB — grande demais para um repositório Git normal e, além disso, redundante: o dataset já está publicado com DOI fixo (Mendeley `10.17632/ztmf3m7h5x.6`). O `README.md` deve trazer o link de download e o script `01_convert_mat_to_pcm.py` para qualquer um reconstruir `data/` do zero. O que **é** versionado é o código e os metadados leves (`manifest.json`, `registry.csv`), que são o que realmente precisa ter histórico rastreável.

Consequência prática dessa escolha, aprendida na marra: **depois de clonar o repositório, `data/` vem vazio**. É preciso baixar o dataset e rodar o `01` de novo antes de qualquer análise. Como o `manifest.json` é versionado e a conversão é determinística, ele serve de referência para conferir se a reconversão reproduziu a original — o `inspect_pcm.py` compara número de amostras e pico PCM de cada `.bin` com o manifest e avisa se divergir.

## 2. Nomenclatura de arquivos e scripts
 
**Scripts exploratórios** (`scripts/exploration/`): sem prefixo numérico, nome descritivo do que investigam.
Exemplos já usados: `inspect_mat_keys.py`, `inspect_signal_data.py`, `inspect_pcm.py`, `inspect_class_spectra.py`.
 
**Scripts de pipeline** (`scripts/pipeline/`): prefixo numérico de duas casas indicando a ordem de execução, seguido do verbo da ação em `snake_case`.
Exemplos: `01_convert_mat_to_pcm.py`, `02_decimate_pcm.py`, `03_extract_features.py`, `04_train_classifier.py`.
Regra: se um script novo precisa rodar *entre* dois existentes, renumerar em vez de usar sufixos como `01b_`.
 
**Classes/rótulos**: sempre em `snake_case`, minúsculo, consistente entre código, nomes de arquivo e Notion:
`normal`, `bpfi_0.3mm`, `bpfi_1.0mm`, `bpfo_0.3mm`, `bpfo_1.0mm` (5 classes) — e `normal` / `falha` (binário).
 
**Arquivos de dados processados**: `<classe>.<extensão>` dentro do diretório da etapa do pipeline que os gerou (ex.: `data/processed/pcm_raw/bpfi_0.3mm.bin`, `data/processed/pcm_decimated/12800/bpfi_0.3mm.bin`).
 
Para branches e mensagens de commit, ver a seção 6.

## 3. Ambiente Python
 
`requirements.txt` na raiz, com versões fixadas. Isso evita que um resultado deixe de ser reprodutível por causa de uma versão diferente do `scipy` ou do `numpy` entre integrantes.
 
Trabalhe sempre dentro do venv do projeto, e gere o arquivo com `--local`:
 
```bash
source .venv/bin/activate
pip freeze --local > requirements.txt
```
 
O `--local` restringe a lista ao que está instalado no próprio venv. Sem ele, um freeze pode arrastar pacotes de fora — de outro ambiente ativo ou de um diretório em `PYTHONPATH` — e o arquivo deixa de descrever o ambiente que reproduz os resultados. Confira antes de commitar:
 
```bash
python -c "import sys; print(sys.prefix)"    # tem que apontar para o .venv do projeto
wc -l requirements.txt                       # ~15–25 linhas; muito mais que isso é contaminação
```

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

Três regras práticas, fixadas depois da primeira rodada real:
 
- **O script escreve a própria linha.** Preencher à mão depende de alguém lembrar, e foi por isso que a coluna `git_commit` existe: o script lê `git rev-parse --short HEAD` sozinho. Rodadas de teste (trechos curtos, verificação de ambiente) usam a flag que desliga o registro, para não sujar o arquivo.
- **Uma varredura de parâmetro gera uma linha por ponto**, todas com o mesmo `git_commit` e a mesma data. A rodada de decimação, por exemplo, gerou seis linhas: o baseline de 51,2 kHz e uma para cada taxa candidata. É isso que permite plotar métrica × parâmetro, que é a justificativa do formato.
- **Resultado negativo se registra; medição inválida se descarta.** As duas coisas não são iguais. Uma taxa que preserva pouco da assinatura é resultado e entra no arquivo. Uma rodada cujo método estava errado — banda de análise mal escolhida, critério que não se aplica ao sinal — não mede o que diz medir, e manter a linha só contamina comparações futuras. Nesse caso a rodada é refeita e a linha inválida não entra.

## 5. Parâmetros compartilhados entre etapas
 
Parâmetros que atravessam mais de uma etapa do pipeline — taxa de amostragem alvo, tamanho e hop da janela, tipo de janela, número de filtros de Mel, número de coeficientes MFCC — ficam num único módulo de configuração, importado por quem precisar. Nenhum script declara os seus próprios.
 
O motivo é concreto: a validação da taxa de decimação foi feita medindo o efeito sobre MFCC com uma configuração específica. Se a etapa de extração de features usar outra, a validação deixa de valer para as features que o classificador realmente consome — e a divergência não gera erro nenhum, só resultados incomparáveis. O mesmo vale para o porte em C da Fase 2: os valores do firmware têm que sair da mesma fonte, senão a comparação bloco a bloco entre C e Python não fecha.
 
Esses valores também entram na coluna `parametros` do `registry.csv` de cada rodada.

## 6. Fluxo de trabalho no Git
 
Trabalho de cada tarefa em uma branch própria, aberta a partir de `main`. Merge para `main` ao concluir, para que `main` reflita sempre o estado real e comparável do projeto — importante para os marcos de entrega (1º parcial, 2º parcial, final).
 
### Branches
 
Prefixo pelo tipo de trabalho:
 
- `feature/` — nova etapa do pipeline, nova funcionalidade, novo estudo.
- `fix/` — correção de algo que já está na `main`.
- `refactor/` — reorganização de código sem mudança de comportamento; os resultados numéricos antes e depois têm que ser idênticos.
O que separa um `fix/` de um `feature/` é o estado de partida, não o tamanho: se o que está sendo corrigido já foi mergeado, é `fix/`. E um `fix/` carrega só a correção — mudanças aproveitadas "já que estou aqui" vão para branch própria, senão a revisão deixa de ser possível.
 
Quando uma tarefa depende de outra cujo PR ainda não foi aprovado, a branch é aberta a partir da branch anterior em vez de `main`, e isso fica registrado na descrição do PR — senão o trabalho parte de um estado que não contém a dependência.
 
Evite rebase em branch cujo trabalho já esteja registrado no `registry.csv`: o rebase reescreve os hashes dos commits, e a coluna `git_commit` passa a apontar para um commit que não existe mais. Prefira merge, ou atualize as linhas afetadas.
 
### Commits
 
Mensagens no imperativo, em português, prefixadas pela frente de trabalho quando ajudar a rastrear: `dsp: converte .mat para PCM int16`, `firmware: implementa FFT em Q15`, `docs: atualiza convenções`.
 
Note que o prefixo do commit (frente de trabalho) e o prefixo da branch (tipo de trabalho) são eixos diferentes e não precisam coincidir: uma branch `fix/` pode perfeitamente conter um commit `dsp:`.
 
### Títulos de PR
 
Substantivos, descrevendo a entrega e o resultado quando couber — `Conversão .mat para PCM`, `Definição da taxa de decimação (12,8 kHz)`. Quem varre a lista de PRs meses depois precisa entender o que foi feito sem abrir cada um.