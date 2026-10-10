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
│   │   └── vibracao/         # .mat de vibração (0 Nm), só para confirmação; mesmos nomes dos de áudio
│   └── processed/
│       ├── pcm_raw/          # saída do 01_convert_mat_to_pcm.py — .bin NÃO versionados, manifest.json É versionado
│       ├── pcm_decimated/    # saída do 02_decimate_pcm.py, subdiretório pela taxa de trabalho (12800/)
│       │                     # mesma regra: .bin fora do Git, manifest.json versionado
│       ├── splits/
│       │   └── splits.json   # saída do 03_make_splits.py — partição dos protocolos A e B, VERSIONADA
│       └── features/         # saída do 04_extract_features.py — .npz e manifest_features.json, NÃO versionados
│
├── scripts/
│   ├── config.py             # parâmetros compartilhados (ver seção 5)
│   ├── dsp.py                # blocos de processamento de sinais
│   ├── pcm_io.py             # leitura e escrita de PCM e manifests
│   ├── experimentos.py       # escrita do registry (ver seção 4)
│   ├── exploration/          # estudos e inspeções, um-off, sem numeração
│   ├── pipeline/             # pipeline reprodutível, numerado pela ordem de execução
│   ├── models/               # pasta por assunto: modelo final do classificador (ver seção 5)
│   └── validation/           # pasta por assunto: protocolo de validação do classificador (ver seção 5)
│
├── tests/                    # testes automáticos (pytest), espelhando scripts/ (ver seção 7)
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
├── reports/                  # figuras, tabelas e outputs usados nos relatórios
│   ├── decimation/           # saída do estudo de taxas: métricas, figuras e relatório
│   ├── exploration/          # figuras dos scripts de scripts/exploration/
│   ├── signature/            # caracterização da assinatura acústica (BPFI/BPFO medidas, envelope, Mel)
│   ├── c_reference/          # reference_data.h do 04 --ref-c, VERSIONADO (referência da Fase 2)
│   ├── modelo_final/         # lda_final.json e escores_referencia.csv do 05 e lda_modelo.h do 06, VERSIONADOS (referência da Fase 2)
│   └── validation/           # uma pasta por rodada dos protocolos: metrics.json e folds.csv
│
└── docs/                     # documentação técnica complementar (ex.: notas sobre o formato do .mat)
```

**Por que `data/raw/` e os `.bin` de `data/processed/` não são versionados:** os 5 arquivos `.mat` somam ~120 MB e os PCM outros ~60 MB — grande demais para um repositório Git normal e, além disso, redundante: o dataset já está publicado com DOI fixo (Mendeley `10.17632/ztmf3m7h5x.6`). O `README.md` deve trazer o link de download e o script `01_convert_mat_to_pcm.py` para qualquer um reconstruir `data/` do zero. O que **é** versionado é o código e os metadados leves (`manifest.json`, `registry.csv`), que são o que realmente precisa ter histórico rastreável.

O `splits.json` é versionado pelo mesmo motivo que o `manifest.json`, e por mais um: ele **define** quais segmentos são treino e quais são teste em cada fold. Se cada integrante gerasse a própria partição, os números de rodadas diferentes deixariam de ser comparáveis. Por isso ele é gerado uma única vez, e o `03_make_splits.py` se recusa a sobrescrever um arquivo diferente. Uma partição nova invalida a comparação com todas as rodadas já registradas: só se usa `--sobrescrever` depois de registrar essa decisão no Notion.

As features do `04_extract_features.py` (`data/processed/features/`), ao contrário, **não** são versionadas: são reconstruíveis a partir dos `.bin` e do `splits.json`, e mudam a cada extração. O que garante a rastreabilidade é o `manifest_features.json` gravado ao lado do `.npz`, com o commit, o hash do `splits.json`, o `norm_clipe` e os parâmetros de MFCC usados. O `run_protocol.py` aborta se esse manifesto faltar ou divergir da partição e do `config.py` atuais, e grava no registry os valores do manifesto, não os do `config.py`. As pastas `reports/validation/teste_*`, geradas com `--sem-registro`, também ficam fora do Git.

Consequência prática dessa escolha, aprendida na marra: **depois de clonar o repositório, `data/` vem vazio**. É preciso baixar o dataset e rodar o `01` de novo antes de qualquer análise. Como o `manifest.json` é versionado e a conversão é determinística, ele serve de referência para conferir se a reconversão reproduziu a original — o `inspect_pcm.py` compara número de amostras e pico PCM de cada `.bin` com o manifest e avisa se divergir.

## 2. Nomenclatura de arquivos e scripts

**Scripts exploratórios** (`scripts/exploration/`): sem prefixo numérico, nome descritivo do que investigam.
Exemplos já usados: `inspect_mat_keys.py`, `inspect_signal_data.py`, `inspect_pcm.py`, `inspect_class_spectra.py`, `compare_decimation_rates.py`, `compare_classifiers.py`, `inspect_left_out_fault.py`, `identify_tonal_peaks.py`, `confirm_bpf_envelope.py`.

Um estudo cujo resultado vai para o relatório grava em `reports/<assunto>/` (ex.: `reports/signature/`) e escreve a própria linha no registry, como qualquer rodada (seção 4).

O estudo de escolha do classificador seguiu a mesma regra: nasceu como `validation/piloto_modelos.py`, mas não é executável do protocolo (não começa com `run_`) nem mede desempenho. Responde uma vez qual modelo usar, então foi para `exploration/compare_classifiers.py`, grava em `reports/classifier/` e registra a rodada com etapa `escolha_classificador`.

A linha entre as duas pastas é o que o script **faz**, não o seu tamanho: `pipeline/` é transformação que roda de novo toda vez que o dado muda; `exploration/` responde uma pergunta uma vez. O caso que fixou a regra: a escolha da taxa de decimação nasceu misturada com a decimação em si, num arquivo de 1.200 linhas. O estudo — varredura de cinco taxas, métricas, figuras e relatório — foi para `exploration/compare_decimation_rates.py`, e a etapa que aplica a taxa escolhida ficou em `pipeline/02_decimate_pcm.py`, com 105 linhas. Um estudo fica versionado para a decisão continuar auditável, não para ser reexecutado.

**Scripts de pipeline** (`scripts/pipeline/`): prefixo numérico de duas casas indicando a ordem de execução, seguido do verbo da ação em `snake_case`.
Sequência atual: `01_convert_mat_to_pcm.py`, `02_decimate_pcm.py`, `03_make_splits.py`, `04_extract_features.py`, `05_train_classifier.py`, `06_export_lda_header.py`.
Regra: se um script novo precisa rodar *entre* dois existentes, renumerar em vez de usar sufixos como `01b_`. Foi o que aconteceu com a partição: ela entrou como `03` porque a extração de features opera sobre os segmentos que ela define, e a extração e o treino passaram a `04` e `05`.

O `05_train_classifier.py` produz o modelo que vai para o firmware; ele **não** mede desempenho. Medir desempenho é papel de `scripts/validation/run_protocol.py`, que treina e descarta um modelo por fold. As duas coisas ficam em scripts separados de propósito.

**Pastas por assunto** (`scripts/validation/`, e as que vierem: `augmentation/`, `models/`): reúnem os módulos e os executáveis de um mesmo assunto. Cada uma tem um `__init__.py`, para poder ser importada (`from validation import particao`). Dentro delas, **executáveis começam com `run_`**; o resto é módulo importado e não roda sozinho. A raiz de `scripts/` fica reservada aos módulos que praticamente toda etapa usa (ver seção 5).

**Terminologia**: no código, `pcm_io.Clip` é a **gravação** inteira de uma classe (~60 s). A unidade de classificação de 1 s se chama **segmento** (`config.SEGMENTO_S`). Não chamar segmentos de "clipes", para os dois conceitos não se confundirem.

**Classes/rótulos**: sempre em `snake_case`, minúsculo, consistente entre código, nomes de arquivo e Notion:
`normal`, `bpfi_0.3mm`, `bpfi_1.0mm`, `bpfo_0.3mm`, `bpfo_1.0mm` (5 classes) — e `normal` / `falha` (binário).

**Arquivos de dados processados**: `<classe>.<extensão>` dentro do diretório da etapa do pipeline que os gerou (ex.: `data/processed/pcm_raw/bpfi_0.3mm.bin`, `data/processed/pcm_decimated/12800/bpfi_0.3mm.bin`).

Para branches e mensagens de commit, ver a seção 6.

## 3. Ambiente Python

`requirements.txt` na raiz, com versões fixadas. Isso evita que um resultado deixe de ser reprodutível por causa de uma versão diferente do `scipy` ou do `numpy` entre integrantes.

Trabalhe sempre dentro do venv do projeto, e gere o arquivo com `--local`:

```bash
source .venv/bin/activate
python -m pip freeze --local > requirements.txt
```

Use `python -m pip`, não `pip` sozinho: com o conda `(base)` ativo junto do venv, o `pip` do PATH pode ser o do conda, e o pacote vai para o ambiente errado.

**PyTorch só para CPU.** O torch vem do índice do PyTorch (`torch==X.Y.Z+cpu`), que não está no PyPI. O `pip freeze` não guarda de onde o pacote veio, então, **depois de todo freeze**, recoloque no topo do arquivo:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

Sem essa linha, um `pip install -r requirements.txt` num venv novo falha com `No matching distribution found for torch==X.Y.Z+cpu`. É `--extra-index-url`, e não `--index-url`, para o resto continuar vindo do PyPI. A escolha pela CPU está justificada no README ("Como rodar").

O `--local` restringe a lista ao que está instalado no próprio venv. Sem ele, um freeze pode arrastar pacotes de fora — de outro ambiente ativo ou de um diretório em `PYTHONPATH` — e o arquivo deixa de descrever o ambiente que reproduz os resultados. Confira antes de commitar:

```bash
python -c "import sys; print(sys.prefix)"    # tem que apontar para o .venv do projeto
wc -l requirements.txt                       # ~30–35 linhas; muito mais que isso é contaminação
```

Desde o torch, o arquivo tem ~30 linhas: ele traz dependências próprias (`filelock`, `fsspec`, `Jinja2`, `MarkupSafe`, `mpmath`, `networkx`, `setuptools`, `sympy`, `typing_extensions`). Antes de commitar um `requirements.txt` novo, teste num venv limpo:

```bash
python -m venv /tmp/teste-req && /tmp/teste-req/bin/python -m pip install -r requirements.txt
```

## 4. Formato de registro de experimentos

Cada rodada de um experimento (extração de features, treino de classificador, comparação de taxa de decimação, etc.) ganha uma linha em `experiments/registry.csv`. Colunas:

| coluna | descrição |
|---|---|
| `id` | identificador curto e sequencial, ex. `exp001` |
| `data` | data da rodada (AAAA-MM-DD) |
| `etapa` | qual etapa do pipeline foi exercitada, ex. `decimacao`, `extracao_features`, `validacao_classificador`, `escolha_classificador`, `treino_classificador`, `exportacao_firmware`, `caracterizacao_assinatura`, `contrato_numerico` |
| `script` | script executado, ex. `02_decimate_pcm.py` |
| `git_commit` | hash curto do commit em que o script estava (`git rev-parse --short HEAD`) — garante que dá pra reproduzir exatamente aquela rodada |
| `parametros` | parâmetros relevantes da rodada, em formato `chave=valor;chave=valor` (ex. `fator_decimacao=4;filtro=fir_lowpass_order8`) |
| `dataset` | qual versão/subset dos dados foi usada |
| `metricas` | resultado quantitativo, em formato `chave=valor;chave=valor` (ex. `acuracia=0.91;f1=0.89`) |
| `responsavel` | quem rodou |
| `notas` | observações livres (ex. algo estranho no resultado, decisão tomada a partir dele) |

Por que CSV e não só um texto corrido: permite comparar experimentos entre si (abrir no pandas/Excel, filtrar por etapa, plotar métrica x parâmetro) — é exatamente o "resultados comparáveis entre si" que a proposta promete. Toda rodada que gerar um número que vá para o relatório parcial deveria ter uma linha aqui, mesmo que o resultado seja negativo.

Regras práticas, fixadas a partir das primeiras rodadas reais:

- **O script escreve a própria linha.** Preencher à mão depende de alguém lembrar, e foi por isso que a coluna `git_commit` existe: o script lê `git rev-parse --short HEAD` sozinho. Rodadas de teste (trechos curtos, verificação de ambiente) usam a flag que desliga o registro, para não sujar o arquivo.
- **Uma varredura de parâmetro gera uma linha por ponto**, todas com o mesmo `git_commit` e a mesma data. A rodada de decimação, por exemplo, gerou seis linhas: o baseline de 51,2 kHz e uma para cada taxa candidata. É isso que permite plotar métrica × parâmetro, que é a justificativa do formato.
- **Rodada de classificação registra protocolo e partição.** Toda linha de `validacao_classificador` traz em `parametros` o `protocolo` (A ou B), o `splits` (hash do `splits.json` usado) e as `features`. É o que permite saber, meses depois, se duas rodadas são comparáveis: com hash diferente, não são. E o número que vale para a meta é o do Protocolo B — o A é limite otimista e não deve ser citado como desempenho.
- **Commit antes de rodar.** O `git_commit` só serve se o código daquele commit for o que rodou. Rodada registrada com mudanças não commitadas aponta para um estado que não existe.
- **Resultado negativo se registra; medição inválida se descarta.** As duas coisas não são iguais. Uma taxa que preserva pouco da assinatura é resultado e entra no arquivo. Uma rodada cujo método estava errado — banda de análise mal escolhida, critério que não se aplica ao sinal — não mede o que diz medir, e manter a linha só contamina comparações futuras. Nesse caso a rodada é refeita e a linha inválida não entra.
- **Escolha de modelo não usa o teste.** Arquitetura e hiperparâmetros se escolhem por validação interna ao treino de cada fold (etapa `escolha_classificador`), nunca olhando o resultado do Protocolo B. O número interno não é desempenho e não vai para a meta; o desempenho do modelo escolhido é o do `run_protocol.py`.
- **Auto-teste sintético antes do dado real.** Um estudo que mede algo novo (frequências, SNR, posição num eixo) roda primeiro com `--sintetico`: um sinal montado com a resposta conhecida, para conferir que o método a recupera. O `--sintetico` implica `--sem-registro` e grava em `reports/<assunto>/sintetico/`, que fica fora do Git (`.gitignore`). Foi o que pegou, por exemplo, o pente harmônico travando em f_eixo/2 antes de qualquer número real ser lido.

## 5. Código compartilhado entre etapas

Módulos importáveis ficam na raiz de `scripts/` quando praticamente toda etapa os usa, e numa pasta por assunto quando pertencem a um assunto só. Nenhum script reimplementa o que eles oferecem:

| módulo | responsabilidade |
|---|---|
| `config.py` | parâmetros que atravessam etapas: taxa de trabalho, janela, hop, banco de Mel, rótulos, segmentação e partição |
| `dsp.py` | blocos de sinal: PSD, espectro de envelope, banco de Mel, MFCC, projeto de decimação; para o porte em C, os coeficientes do FIR (`taps_decimacao`), o recorte que o decimador do firmware lê (`recorte_para_decimar`) e a média e o desvio em float32 (`media_desvio_float32`) |
| `pcm_io.py` | leitura e escrita dos `.bin` e dos `manifest.json` |
| `experimentos.py` | numeração, hash do commit e escrita do `registry.csv` |
| `validation/particao.py` | segmentação das gravações, folds dos protocolos A e B, verificação das garantias, leitura e escrita do `splits.json` |
| `validation/metricas.py` | sensibilidade, especificidade, acurácia balanceada e resumos dos protocolos |
| `augmentation/transformacoes.py` | transformações de sinal do aumento: estiramento temporal (phase vocoder ou reamostragem), ruído com SNR fixada, recorte central |
| `models/lda.py` | modelo final: o mesmo `novo_modelo` do `run_protocol` treinado com todos os segmentos, parâmetros em JSON nas formas padronizada e dobrada (escore ≥ 0 → falha, o empate do sklearn), descrição da cadeia de features para o firmware (`cadeia_entrada`, tirada do `config` e do `dsp`) e conferência contra o sklearn |
| `augmentation/variantes.py` | variantes por segmento (sorteio reprodutível por segmento e cópia), intervalo lido da gravação e filtro que aceita, em cada fold, só as variantes que leem segmentos de treino |

A regra vale em particular para a segmentação: qualquer etapa que opere sobre segmentos — extração de features, aumento de dados, treino — obtém os segmentos de `particao.segmentos_de(particao.carregar(...))`, nunca recorta o sinal por conta própria. Uma segunda segmentação poderia divergir da partição sem erro nenhum, e o teste deixaria de estar separado do treino.

Como os scripts de pipeline têm prefixo numérico, eles não podem ser importados como módulo (nome de módulo não começa com dígito) e o `python -m` não se aplica. Cada script — de pipeline, de exploração ou executável de uma pasta por assunto — abre com:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/
```

Sem isso, `import config` falha quando o script é chamado da raiz do repositório. Como as três pastas estão no mesmo nível, `parents[1]` aponta para `scripts/` em todas, e daí `from validation import particao` também funciona.

No `config.py` ficam **escolhas**, não **resultados**: valores que são consequência de uma escolha — número de coeficientes do FIR, tamanho da FFT — são calculados a partir dela, nunca copiados como literal. Congelados, viram mentira silenciosa no dia em que alguém mudar a taxa de trabalho.

As funções de `dsp.py` recebem seus parâmetros na assinatura, com valor padrão vindo do `config`. Um estudo que varre taxas precisa calcular alguns por taxa sob teste, e não pode herdar os de produção.

O motivo é concreto: a validação da taxa de decimação foi feita medindo o efeito sobre MFCC com uma configuração específica. Se a etapa de extração de features usar outra, a validação deixa de valer para as features que o classificador realmente consome — e a divergência não gera erro nenhum, só resultados incomparáveis. O mesmo vale para o porte em C da Fase 2: os valores do firmware têm que sair da mesma fonte, senão a comparação bloco a bloco entre C e Python não fecha.

Esses valores também entram na coluna `parametros` do `registry.csv` de cada rodada.

**Refactor que mexe nessa camada** não pode mudar resultado. O contrato é: guardar `reports/` e `registry.csv` antes, mover o código, reexecutar, e conferir que `metrics.json`, o relatório e as figuras saem idênticos. `git status` limpo em `reports/` é a forma mais forte de verificar, porque cobre os PNG.

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

Mensagens no padrão [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/):

```
<tipo>(<escopo opcional>): <descrição no imperativo, em português, minúscula, sem ponto final>

<corpo opcional: o porquê da mudança, não o como>
```

| tipo | quando usar | exemplo |
|---|---|---|
| `feat` | etapa, funcionalidade ou estudo novo | `feat(aumento): adiciona aumento de dados ao treino` |
| `fix` | correção de comportamento errado | `fix(exploracao): impede que a f_e trave em 2·f_motor` |
| `refactor` | reorganização sem mudar resultado (regra da seção 5: saídas idênticas antes e depois) | `refactor(pipeline): isola extração de MFCC` |
| `docs` | README, CONVENTIONS, docstrings, e o commit que registra rodadas (`registry.csv` + `reports/`) | `docs: registra rodadas do aumento de dados (exp015–exp022)` |
| `test` | testes novos ou alterados, sem mudar o código testado | `test(validacao): cobre a faixa de descarte` |
| `chore` | manutenção que não é código nem documentação: `requirements.txt`, `.gitignore` | `chore: ignora saídas do modo sintético` |

**Escopo** é opcional e diz a frente de trabalho, pelo nome da pasta ou do módulo: `dsp`, `pipeline`, `validacao`, `aumento`, `exploracao`, `firmware`. Use quando ajudar a rastrear; commit que atravessa várias frentes fica sem escopo.

**Mudança que quebra a comparação com rodadas registradas** — partição nova, parâmetros de MFCC, taxa de trabalho — leva `!` depois do tipo/escopo e uma linha `BREAKING CHANGE:` no corpo dizendo quais rodadas deixam de ser comparáveis. Exemplo: `feat(pipeline)!: muda o banco de Mel para 40 filtros`.

O prefixo do commit (tipo de mudança) e o prefixo da branch (tipo de trabalho) são eixos diferentes e não precisam coincidir: uma branch `feature/` normalmente tem commits `feat:`, `fix:` e `docs:`.

Os commits mais antigos, de antes da adoção desse padrão, usam o formato anterior (`dsp:`, `validacao:`, `exploracao:`, com a frente de trabalho como prefixo). Eles não são reescritos: isso exigiria rebase, e o parágrafo sobre rebase, acima, explica por que evitá-lo em trabalho que já está no registry.

### Títulos de PR

Substantivos, descrevendo a entrega e o resultado quando couber — `Conversão .mat para PCM`, `Definição da taxa de decimação (12,8 kHz)`. Quem varre a lista de PRs meses depois precisa entender o que foi feito sem abrir cada um.
## 7. Testes

Testes automáticos ficam em `tests/`, com a mesma estrutura de `scripts/` (`tests/validation/` testa `scripts/validation/`), e rodam com:

```bash
pytest tests/
```

O `tests/conftest.py` coloca `scripts/` no caminho de importação, do mesmo jeito que os scripts fazem, então os testes importam `config`, `pcm_io` e `validation.particao` diretamente.

Três regras:

- **Testes não dependem de `data/`.** Eles usam dados sintéticos com o mesmo formato dos reais (por exemplo, gravações de 767.982 amostras a 12,8 kHz), para rodar num clone recém-feito, antes de qualquer download.
- **Rodar antes de commitar** qualquer mudança em módulo importado, e antes de abrir PR. Um teste que falha é motivo para não mergear.
- **O que um teste protege é uma garantia, não um número.** Os testes da partição conferem que nenhum segmento está em treino e teste no mesmo fold, que a faixa de descarte é respeitada e que a falha deixada de fora não aparece no treino — as condições sem as quais os resultados de classificação não valem. Resultados numéricos vão para o `registry.csv`, não para os testes.

Se `pytest` falhar ao iniciar com erro de importação vindo de fora do projeto (um caminho como `/opt/ros/...`), é o `PYTHONPATH` do sistema trazendo plugins de pytest de outro ambiente. Limpe-o no terminal do projeto (`unset PYTHONPATH`) ou acrescente essa linha ao final de `.venv/bin/activate`. É a mesma contaminação que o `--local` evita no `pip freeze` (seção 3).