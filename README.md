# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

Sistema embarcado que classifica, em tempo real e a partir de sinal acústico, o estado de operação de um motor rotativo (normal ou com defeito de rolamento — BPFI/BPFO) usando uma cadeia de processamento digital de sinais e um classificador leve, portados para um microcontrolador STM32F411CEU6. Um protótipo em Python valida a cadeia de processamento antes do porte para o hardware, permitindo comparar a solução em software e em firmware.

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

**Documentação do projeto:**
- Proposta de trabalho (aprovada em 14/09) — `docs/proposta-trabalho.pdf`
- Espaço de gerenciamento (tarefas, cronograma, decisões): [Notion](https://app.notion.com/p/04d931873a6c83b2b5a3017fae7db29a)
- Convenções do projeto: [`CONVENTIONS.md`](./CONVENTIONS.md)

## Status

Fase 1 — Protótipo em Python (em andamento). Conversão, decimação, protocolo de validação, extração de MFCC, aumento de dados, caracterização da assinatura acústica e estudo de escolha do classificador concluídos (fica a LDA: empata com a CNN 2D na validação interna e sai mais barata; ver "Escolha do classificador" abaixo). Próximo passo: modelo final e exportação dos pesos para o firmware. Ver o quadro de tarefas no Notion para o estado detalhado de cada etapa.

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
| Classificador | **LDA** sobre média e desvio dos 13 MFCC (26 valores), priors uniformes; na validação interna ao treino do B, empata com a CNN 2D e fica pelo custo, e supera o MLP raso e a CNN 1D | `scripts/exploration/compare_classifiers.py`; Notion, estudo comparativo da tarefa Classificador 3/6 |
| Aumento de dados | só no treino de cada fold: deslocamento de janela (±0,4 s), estiramento temporal (taxa 0,95–1,05, phase vocoder) e ruído branco (SNR 20–35 dB); variante só entra se ler apenas segmentos de treino do fold | proposta (Metodologia); `scripts/augmentation/` |
 
A escolha de 12.800 Hz é a única taxa com fator de decimação inteiro dentro da faixa de 8–16 kHz — requisito de `arm_fir_decimate_f32` do CMSIS-DSP — e preserva 96,1 % da energia discriminante no pior caso (classes de defeito incipiente, 0,3 mm). O registro completo da comparação está em `experiments/registry.csv` (`exp001`–`exp006`). O diagnóstico por espectro de envelope desse relatório (SNRs de BPFI/BPFO por taxa) **não vale**: a busca do pico era centrada nas frequências nominais e não alcançava as linhas reais. Ele foi refeito no sinal original pelo `confirm_bpf_envelope.py` (ver "Assinatura acústica" abaixo). A decisão de 12.800 Hz não muda, porque nunca dependeu do envelope.

O protocolo de validação existe porque o dataset tem **uma única gravação por classe**: qualquer divisão treino/teste dentro de uma gravação deixa os dois no mesmo registro, e o classificador pode separar as classes pela identidade da gravação em vez da falha (foi o que deu acurácia 1,0 no estudo de decimação). O Protocolo B testa cada gravação de falha sem que ela apareça no treino; é o único resultado que conta para a meta. A primeira rodada, provisória, está em `exp007`–`exp012`. Com o MFCC oficial, sem aumento, o A está em `exp027` (binário) e `exp028` (multiclasse) e o B em `exp013`/`exp015`, e os números são idênticos aos da rodada provisória: A com 1,000 em todos os folds; B com média 0,875, a `bpfo_0.3mm` em 0,5 e as outras três falhas em 1,0. A comparação está em `reports/validation/tabela_provisorio_vs_oficial.md`. Os controles obrigatórios com o MFCC oficial estão em `reports/validation/tabela_controles.md`. Sem o c0 e com normalização RMS por segmento nada muda (`exp029`, `exp030`, `exp014`, `exp219`). Com os rótulos permutados, o B fica em 0,450 ± 0,060 em 100 sementes (`exp031`–`exp050`, `exp059`–`exp138`; p empírico 0,0099). Na curva de aprendizado com o sorteio repartido entre as gravações (`exp139`–`exp218`, 10 sementes por ponto), o B vai de 0,757 com 2 s a 0,867 com 30 s por classe. A curva não descarta que o modelo tenha aprendido "diferente da gravação normal = falha", porque há uma só gravação normal e o teste dela vem da mesma gravação. O `inspect_lda_harmonicos.py` mostra que a LDA não depende das bandas de Mel em que a normal tem harmônicos do eixo mais fortes: apagar essas bandas (log-Mel trocado pela média do treino) e treinar o B de novo deixa o resultado idêntico, 0,875, enquanto apagar o mesmo número de bandas sorteadas fora dos harmônicos dá 0,854–0,875 (`exp221`). A decomposição da separação por banda (`exp220`, −0,20 nas bandas dos harmônicos) é complemento, porque bandas vizinhas são correlacionadas. As rodadas `exp051`–`exp058` são da curva sem estratificação e ficam só no registry.

### Escolha do classificador

A proposta previa escolher entre um MLP raso e uma CNN pequena. O `compare_classifiers.py` comparou quatro modelos (LDA como referência, MLP 26 → 16 → 2, CNN 1D sobre a matriz MFCC 13 × 98 e CNN 2D sobre o log-Mel 20 × 98) **sem usar o teste do Protocolo B**. Em cada fold externo do B, o treino é dividido de novo pela mesma lógica: cada falha de treino deixada de fora, combinada com cada bloco normal de treino (15 treinos internos por fold, 24 folds, 3 sementes por rede). A escolha é feita separadamente para cada falha deixada de fora (escolha aninhada), pelo critério do `escolher()`: dentro de uma margem de 0,02 da melhor acurácia interna (ou da dispersão entre sementes, se for maior), fica o modelo mais barato em MACs. O critério foi fechado depois da primeira rodada (`exp222`), que desempatava pela ordem da lista de modelos; aplicado aos mesmos números, ele dá a mesma escolha, e a rodada seguinte já o registra.

A LDA e a CNN 2D empatam exatamente nas quatro escolhas aninhadas (0,875 de acurácia balanceada interna média), e a LDA fica pelo custo: a CNN 2D custa ~13.000× mais MACs e ~700× mais RAM de ativação. O MLP (0,828) e a CNN 1D (0,824) perdem na `bpfi_0.3mm`, onde a sensibilidade média cai para ~0,6. Nenhum modelo recupera a `bpfo_0.3mm`, nem a CNN 2D, que vê as bandas de Mel onde está a diferença: o pior caso vem de deslocamento de distribuição, não de capacidade do modelo. Em todos os modelos, os erros são falsos negativos (especificidade 1,000). Os números internos são pessimistas, porque o treino interno tem só 2 falhas contra 3 no B externo; o que conta é a comparação entre modelos. Cada rede foi avaliada numa configuração fixa (120 épocas, taxa de aprendizado 3e-3, 16 neurônios ocultos no MLP, 16 filtros na CNN 1D, 8 e 16 na CNN 2D), sem busca de hiperparâmetros: a conclusão de que modelos não lineares não trouxeram ganho vale para essas configurações. O estudo é só binário, porque no B a falha de fora não tem exemplos de treino e um modelo multiclasse não poderia prever essa classe. O custo estimado no STM32F411CEU6 (parâmetros, MACs e pico de ativação) sai do próprio script. Saída em `reports/classifier/compare_classifiers.json`.

### Aumento de dados com o modelo escolhido

Com a LDA, o Protocolo B com aumento aplicado só no treino de cada fold dá o mesmo resultado que sem aumento: 0,875 de acurácia balanceada média (sensibilidade 0,750, especificidade 1,000), com a `bpfo_0.3mm` como pior falha (0,500) e o pior fold em 0,500. Isso vale para o aumento completo em 10 sorteios diferentes das variantes (`exp224`–`exp233`, desvio 0,000 entre as sementes; o `exp224` reproduz o `exp016`) e para cada técnica isolada (`exp019`–`exp021`). O fator de expansão do treino é 5 nominal (o segmento e 4 variantes) e 4,98 efetivo, porque o filtro por fold recusa as variantes que encostam no teste ou na faixa de descarte (889 a 903 aceitas por fold). O estiramento por reamostragem é o único que muda o resultado, para pior (0,788, `exp022`), por deslocar as linhas estreitas de falha; por isso o padrão é o phase vocoder. O ganho aleatório não foi implementado: a ablação do nível do sinal (`exp014`, `exp029`, `exp030`) mostra que o modelo não usa o nível. A tabela está em `reports/validation/tabela_aumento.md`. Nas rodadas com aumento, os `parametros` (e o registry) trazem duas sementes: `semente` é a do `run_protocol` (permutação e curva de aprendizado; nestas rodadas, sempre a do config) e `aumento_semente` é a do sorteio das variantes, a única que varia entre o `exp224` e o `exp233`.

### Decimação conferida com o classificador

Com a LDA, o Protocolo B a 25,6 kHz não melhora o de 12,8 kHz: 0,870 de acurácia balanceada média contra 0,875 (`exp235` × `exp234`, no mesmo commit e com a mesma partição em segundos; a de 25,6 kHz é a `splits_25600.json`). A `bpfo_0.3mm` continua sem ser detectada nas duas taxas (sensibilidade 0,000). A única diferença é a `bpfi_0.3mm`, que a 25,6 kHz perde 13 dos 59 segmentos num dos seis folds (acurácia balanceada 0,982). O classificador não indica perda com a decimação; somado ao `exp023` (nenhuma energia a mais que a normal entre 6,4 e 12,8 kHz) e ao `exp026` (linhas de BPFO abaixo de ~3,8 kHz, que o MFCC a 12,8 kHz recebe), não há indício de que a faixa cortada carregue a informação da `bpfo_0.3mm`. O `exp235` sozinho não isola a banda alta: a 25,6 kHz as mesmas 20 bandas Mel se redistribuem até 12,8 kHz, e as bandas com centro abaixo de 3,8 kHz, onde ficam as linhas de BPFO, caem de 16 para 13, enquanto só 4 cobrem a faixa de 6,4 a 12,8 kHz. A comparação mede, portanto, a taxa junto com o MFCC padrão; isolar a banda alta pediria uma rodada a 25,6 kHz com as bandas abaixo de 6,4 kHz iguais às de 12,8 kHz e bandas extras na faixa alta. A tabela está em `reports/validation/tabela_taxas.md`.

### Resultados consolidados para o relatório

O `run_consolidacao.py` reúne, a partir dos `metrics.json` e do registry, as tabelas e figuras finais do classificador em `reports/validation/consolidacao/`, com o LaTeX já no formato do relatório (vírgula decimal, `tabularx`/`booktabs`). O número da meta é o do `exp015` (reproduzido pelo `exp013` e pelo `exp234`): acurácia balanceada média de 0,875 no Protocolo B, com sensibilidade 0,750, especificidade 1,000 e pior caso na `bpfo_0.3mm` (0,500 em todos os 6 folds). Somados os 24 folds, a matriz de confusão tem 0 falsos positivos em 236 testes de segmentos normais e 354 falsos negativos, todos da `bpfo_0.3mm`; as outras três falhas são detectadas em todos os testes. O Protocolo A (`exp027`, `exp028`) sai rotulado como limite otimista, e a comparação de taxas (`exp234` × `exp235`) entra com as mesmas conferências do `run_tabela_taxas.py`. O script confere, antes de gravar qualquer saída, que cada rodada tem linha no registry com as mesmas métricas (`rastreabilidade.md`, 190 rodadas).

### Assinatura acústica

As frequências de falha **medidas** no áudio são BPFI ≈ 268,3 Hz e BPFO ≈ 182,7–183,4 Hz (eixo a 50,20 Hz). A BPFI e a BPFO da falha de 1,0 mm foram confirmadas por um segundo sensor: no espectro de envelope, a vibração as mostra a menos de 0,25 Hz do áudio. Na `bpfo_0.3mm` a confirmação é **parcial**: a vibração tem energia em 182,65 Hz, mas o pico do envelope dela fica em 181,5 Hz, e o envelope do áudio não mostra a linha. As medidas diferem das frequências **cinemáticas** da Tabela 1 do artigo (272,1 e 179,4 Hz), calculadas para ângulo de contato θ = 0°. Use as medidas quando precisar de uma frequência de falha, e chame as do artigo de "cinemáticas", não de "frequências da bancada". Abaixo de 6,4 kHz, cada falha aparece no espectro como uma série harmônica estreita da própria pista, e essas linhas sobrevivem à decimação. A impulsividade clássica, no áudio, fica acima de 6,4 kHz (9–22 kHz nas classes de falha), e a decimação a remove.

A `bpfo_0.3mm` tem essas linhas (+30 dB sobre a normal) e o log-Mel oficial as enxerga (+6 a +14 dB em 0,7–1,3 kHz), mas sem a elevação larga das outras falhas. No Protocolo B, o eixo da LDA aprendido com as outras três falhas a coloca a ~20 % do caminho entre a normal e as falhas de treino, do lado da normal. O `inspect_left_out_fault.py` chega ao mesmo número por um cálculo independente. Scripts em `scripts/exploration/` (`inspect_signature_spectra.py`, `identify_tonal_peaks.py`, `confirm_bpf_envelope.py`, `inspect_signature_mel.py`), saídas em `reports/signature/`, rodadas `exp023`–`exp026`.

## Dataset

Jung, W.; Kim, S.-H.; Yun, S.-H.; Bae, J.; Park, Y.-H. (2023), *Data in Brief* 48, 109049. Subconjunto acústico (microfone), 5 classes — `0Nm_Normal`, `0Nm_BPFI_03`, `0Nm_BPFI_10`, `0Nm_BPFO_03`, `0Nm_BPFO_10` — 60 s cada, 51,2 kHz, formato `.mat`.
Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6).

Esses cinco arquivos são **todo o áudio do dataset**. Os ensaios com carga (2 e 4 Nm), os defeitos de 3,0 mm e as falhas de desbalanceamento e desalinhamento têm só vibração, corrente e temperatura: os autores não gravaram o microfone com carga porque o freio, resfriado a ar, contaminaria o canal acústico (seção 3.1 do artigo).

Para confirmar as frequências de falha com um segundo sensor, usa-se também a **vibração** da mesma condição (0 Nm; `.mat` com 4 canais a 25,6 kHz; pela ordem das colunas no artigo, 0–1 = x e y do mancal A, 2–3 = mancal B). Ela não entra no classificador. Os arquivos de vibração têm os **mesmos nomes** dos de áudio, então ficam numa subpasta: `data/raw/vibracao/`.

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
│   │   └── vibracao/         # .mat de vibração 0 Nm (só para confirmação; mesmos nomes dos de áudio)
│   └── processed/
│       ├── pcm_raw/          # saída do 01 — .bin não versionados, manifest.json versionado
│       ├── pcm_decimated/    # saída do 02, um subdiretório por taxa (ex.: 12800/)
│       ├── splits/
│       │   ├── splits.json   # saída do 03 — partição dos protocolos A e B, VERSIONADA
│       │   └── splits_25600.json   # a mesma partição a 25,6 kHz (03 --fs 25600), para comparar taxas
│       └── features/         # saída do 04 — mfcc_features.npz + manifest_features.json, NÃO versionados
│                             # (outra taxa: features/<fs>/)
│
├── scripts/
│   ├── config.py             # parâmetros compartilhados entre etapas
│   ├── dsp.py                # blocos de processamento de sinais (PSD, MFCC, decimação)
│   ├── pcm_io.py             # leitura e escrita dos PCM e dos manifests
│   ├── experimentos.py       # escrita do registro de experimentos
│   ├── augmentation/         # aumento de dados do treino
│   │   ├── transformacoes.py              # estiramento (phase vocoder / reamostragem), ruído, recorte
│   │   └── variantes.py                   # sorteio reprodutível, janela lida e filtro por fold
│   ├── exploration/          # estudos e inspeções, sem numeração
│   │   ├── inspect_mat_keys.py
│   │   ├── inspect_signal_data.py
│   │   ├── inspect_pcm.py                 # sanidade da conversão + caráter do sinal
│   │   ├── inspect_class_spectra.py       # PSD por classe e banda necessária
│   │   ├── compare_decimation_rates.py    # estudo que definiu a taxa de trabalho
│   │   ├── compare_classifiers.py         # estudo que escolheu o classificador (validação interna ao B)
│   │   ├── inspect_lda_harmonicos.py      # controle: peso da LDA nas bandas dos harmônicos do eixo
│   │   ├── inspect_left_out_fault.py      # posição da falha deixada de fora (Protocolo B)
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
│   │   └── 05_train_classifier.py         # (previsto) modelo final para o firmware
│   └── validation/           # protocolo de validação do classificador
│       ├── particao.py                    # segmentos, folds A e B, verificação
│       ├── metricas.py                    # sensibilidade, especificidade, acurácia balanceada
│       ├── run_protocol.py                # executável: roda A ou B e registra o resultado
│       ├── run_tabela_comparativa.py      # executável: tabela provisório × oficial a partir dos metrics.json
│       ├── run_tabela_controles.py        # executável: tabela dos controles (sem c0, permutação, curva)
│       ├── run_tabela_aumento.py          # executável: tabela do B com e sem aumento de dados
│       ├── run_tabela_taxas.py            # executável: tabela do B em duas taxas de amostragem
│       └── run_consolidacao.py            # executável: tabelas .tex e figuras finais para o relatório
│
├── tests/                    # pytest; não dependem de data/
│   ├── conftest.py
│   ├── test_dsp.py
│   ├── test_experimentos.py           # registry: linha nova nunca é colada na anterior
│   ├── augmentation/
│   │   ├── test_transformacoes.py
│   │   └── test_variantes.py              # inclui: variante aceita nunca lê teste/descarte
│   ├── exploration/
│   │   └── test_compare_classifiers.py    # critério de escolha: mais barato na margem, em qualquer ordem (pula sem torch)
│   └── validation/
│       ├── test_curva_aprendizado.py      # subamostra só treino, n por classe, reprodutível
│       ├── test_tabela_aumento.py         # tabela do aumento: comparabilidade e fator de expansão
│       ├── test_tabela_taxas.py           # tabela das taxas: comparabilidade e mesma partição em segundos
│       ├── test_consolidacao.py           # consolidação: número sem registry ou rodada no papel errado aborta
│       └── test_particao.py               # inclui: outra taxa dá os mesmos segmentos em segundos
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
│   ├── classifier/           # estudo de escolha do classificador (compare_classifiers.json)
│   ├── decimation/           # métricas, figuras e relatório da escolha da taxa
│   ├── exploration/          # figuras dos scripts exploratórios
│   ├── signature/            # caracterização da assinatura acústica
│   └── validation/           # uma pasta por rodada: metrics.json e folds.csv
│       └── consolidacao/     # tabelas .tex, figuras e rastreabilidade do relatório (run_consolidacao.py)
│
└── docs/                     # proposta, documentação técnica complementar
```

Ver [`CONVENTIONS.md`](./CONVENTIONS.md) para as convenções de nomenclatura de scripts, classes e commits, e para o formato do registro de experimentos.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O `requirements.txt` instala o PyTorch **só para CPU**, pelo índice do PyTorch (linha `--extra-index-url` no topo do arquivo). É proposital: as redes do estudo do classificador têm menos de 2.400 parâmetros, a GPU não acelera nada nesse tamanho, e a versão CUDA ocupa alguns GB. Quem tiver GPU e quiser a versão CUDA pode instalá-la no próprio venv, sem mudar o `requirements.txt`.

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
python scripts/validation/run_protocol.py --protocolo B --permutar --semente 1   # uma rodada por semente
python scripts/validation/run_protocol.py --protocolo B --segundos-treino 5 --semente 1   # curva de aprendizado (mín. 2 s; sorteio repartido entre as gravações)
python scripts/validation/run_protocol.py --protocolo B --sem-registro   # teste, não registra

# tabelas (leem os metrics.json; não treinam nem registram)
python scripts/validation/run_tabela_comparativa.py   # provisório × oficial
python scripts/validation/run_tabela_controles.py     # sem c0, normalização, permutação e curva de aprendizado
python scripts/validation/run_tabela_aumento.py       # B com e sem aumento: médias, pior caso e fator de expansão

# controle dos harmônicos do eixo: peso da LDA nas bandas de Mel em que a normal
# tem harmônicos mais fortes (lê o picos_metrics.json do identify_tonal_peaks.py)
python scripts/exploration/inspect_lda_harmonicos.py --sintetico   # auto-teste
python scripts/exploration/inspect_lda_harmonicos.py --responsavel <nome>

# ablação da normalização RMS por segmento: reextrai e roda de novo
python scripts/pipeline/04_extract_features.py --norm-clipe
python scripts/validation/run_protocol.py --protocolo B --responsavel <nome>
python scripts/pipeline/04_extract_features.py       # volta ao padrão (sem normalização)

# estudo da escolha do classificador: LDA × MLP × CNN 1D × CNN 2D por validação
# interna ao treino do B (~33 min completo; --blocos-externos 1 leva ~1/6 disso)
python scripts/exploration/compare_classifiers.py --sintetico       # auto-teste, sem registro
python scripts/exploration/compare_classifiers.py --responsavel <nome>
python scripts/exploration/compare_classifiers.py --blocos-externos 1 --sem-registro   # conferência rápida

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
# confusão do B, figura dos controles, comparação de taxas e rastreabilidade (cada
# número → rodada → registry). Aborta se alguma rodada não tiver linha no registry ou se
# as métricas não baterem. Não treina nem registra.
python scripts/validation/run_consolidacao.py
python scripts/validation/run_consolidacao.py --sem-taxas    # sem a tabela 12,8 × 25,6 kHz
```

O teste nunca é aumentado: todo fold é avaliado nos segmentos originais. Rodar o `04` sem `--aumento` apaga o `mfcc_aumento.npz` de uma rodada anterior, para que ele não seja lido como se fosse da extração atual.

Commite o código **antes** de uma rodada registrada: o `registry.csv` grava o hash do commit, e uma rodada feita com código não commitado aponta para um estado que não existe.

Para conferir que a reconversão reproduziu a original — o `manifest.json` é versionado e a conversão é determinística, então número de amostras e pico PCM devem bater:

```bash
python scripts/exploration/inspect_pcm.py
```

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

## Fluxo de trabalho
 
Cada tarefa em uma branch própria a partir de `main`, com prefixo por tipo de trabalho (`feature/`, `fix/`, `refactor/`), e merge para `main` ao concluir. As regras completas — branches, mensagens de commit e títulos de PR — estão em [`CONVENTIONS.md`](./CONVENTIONS.md), seção 6.