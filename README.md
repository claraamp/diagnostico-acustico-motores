# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

Sistema embarcado que classifica, em tempo real e a partir de sinal acústico, o estado de operação de um motor rotativo (normal ou com defeito de rolamento — BPFI/BPFO) usando uma cadeia de processamento digital de sinais e um classificador leve, portados para um microcontrolador STM32F411CEU6. Um protótipo em Python valida a cadeia de processamento antes do porte para o hardware, permitindo comparar a solução em software e em firmware.

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

**Documentação do projeto:**
- Proposta de trabalho (aprovada em 14/09) — `docs/proposta-trabalho.pdf`
- Espaço de gerenciamento (tarefas, cronograma, decisões): [Notion](https://app.notion.com/p/04d931873a6c83b2b5a3017fae7db29a)
- Convenções do projeto: [`CONVENTIONS.md`](./CONVENTIONS.md)

## Status

Fase 1 — Protótipo em Python (em andamento). Conversão, decimação, protocolo de validação, extração de MFCC, aumento de dados, caracterização da assinatura acústica e estudo de escolha do classificador concluídos (fica a LDA: empata com a CNN 2D na validação interna e sai mais barata; ver "Escolha do classificador" abaixo). O modelo final (LDA treinada com os 295 segmentos) sai do `05_train_classifier.py` em `reports/modelo_final/`; o `06_export_lda_header.py` gera dele o `.h` para o firmware, e o próximo passo é o porte, na Fase 2. Ver o quadro de tarefas no Notion para o estado detalhado de cada etapa.

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

O `run_consolidacao.py` reúne, a partir dos `metrics.json` e do registry, as tabelas e figuras finais do classificador em `reports/validation/consolidacao/`, com o LaTeX já no formato do relatório (vírgula decimal, `tabularx`/`booktabs`). O número da meta é o do `exp015` (reproduzido pelo `exp013` e pelo `exp234`): acurácia balanceada média de 0,875 no Protocolo B, com sensibilidade 0,750, especificidade 1,000 e pior caso na `bpfo_0.3mm` (0,500 em todos os 6 folds). Somados os 24 folds, a matriz de confusão tem 0 falsos positivos em 236 testes de segmentos normais e 354 falsos negativos, todos da `bpfo_0.3mm`; as outras três falhas são detectadas em todos os testes. O Protocolo A (`exp027`, `exp028`) sai rotulado como limite otimista. O aumento com 10 sementes (`exp224`–`exp233`, mais as ablações `exp019`–`exp022` e o controle permutado `exp017`) e a comparação de taxas (`exp234` × `exp235`) entram com as mesmas conferências do `run_tabela_aumento.py` e do `run_tabela_taxas.py`. O controle dos harmônicos do eixo (`exp220`, `exp221`), que é do `inspect_lda_harmonicos.py` e grava fora das pastas `expNNN_*`, é conferido pelo bloco `intervencao` do seu `metrics.json`, e a referência da intervenção tem que ser o B da meta. O script confere, antes de gravar qualquer saída, que cada rodada tem linha no registry com as mesmas métricas (`rastreabilidade.md`, 207 rodadas).

### Assinatura acústica

As frequências de falha **medidas** no áudio são BPFI ≈ 268,3 Hz e BPFO ≈ 182,7–183,4 Hz (eixo a 50,20 Hz). A BPFI e a BPFO da falha de 1,0 mm foram confirmadas por um segundo sensor: no espectro de envelope, a vibração as mostra a menos de 0,25 Hz do áudio. Na `bpfo_0.3mm` a confirmação é **parcial**: a vibração tem energia em 182,65 Hz, mas o pico do envelope dela fica em 181,5 Hz, e o envelope do áudio não mostra a linha. As medidas diferem das frequências **cinemáticas** da Tabela 1 do artigo (272,1 e 179,4 Hz), calculadas para ângulo de contato θ = 0°. Use as medidas quando precisar de uma frequência de falha, e chame as do artigo de "cinemáticas", não de "frequências da bancada". Abaixo de 6,4 kHz, cada falha aparece no espectro como uma série harmônica estreita da própria pista, e essas linhas sobrevivem à decimação. A impulsividade clássica, no áudio, fica acima de 6,4 kHz (9–22 kHz nas classes de falha), e a decimação a remove.

A `bpfo_0.3mm` tem essas linhas (+30 dB sobre a normal) e o log-Mel oficial as enxerga (+6 a +14 dB em 0,7–1,3 kHz), mas sem a elevação larga das outras falhas. No Protocolo B, o eixo da LDA aprendido com as outras três falhas a coloca a ~20 % do caminho entre a normal e as falhas de treino, do lado da normal. O `inspect_left_out_fault.py` chega ao mesmo número por um cálculo independente. Scripts em `scripts/exploration/` (`inspect_signature_spectra.py`, `identify_tonal_peaks.py`, `confirm_bpf_envelope.py`, `inspect_signature_mel.py`), saídas em `reports/signature/`, rodadas `exp023`–`exp026`.

### Pente harmônico (linhas estreitas): fica fora da cadeia

O `inspect_comb_features.py` avaliou 4 features de pente harmônico de BPFO e BPFI (pontuação e fração de dentes ≥ 6 dB de cada série, com o f₀ buscado a ±3 % da razão cinemática e as harmônicas do eixo mascaradas) junto ao MFCC, com a mesma LDA, nos três candidatos de resolução do card: A, espectro médio dos 98 quadros do MFCC (40 Hz); B, FFT de 4.096 pontos com média de 3 blocos (3,125 Hz, ~40 KB de RAM estimados); C, FFT de 1 s (1 Hz, só referência). A decisão é pela validação interna ao treino do B (`validation.selecao`, a mesma regra do estudo do classificador: o pente só entra se ganhar por mais que a margem de 0,02, sem perder especificidade, para as quatro falhas externas). **O pente fica fora e a cadeia congela com MFCC e LDA** (`exp248`–`exp252`, que substituem o `exp243`–`exp247` com os mesmos números e mais diagnósticos). O A é inviável: com 40 Hz de resolução, nenhum dente da BPFO fica longe das harmônicas do eixo, que são mais fortes na normal. No B e no C, a validação interna é a mesma com e sem o pente (0,833 nas falhas externas em que a `bpfo_0.3mm` sai dentro do treino, 1,000 na outra), e o B externo também (0,875; `bpfo_0.3mm` em 0,500). O pente enxerga a falha: com 1 Hz (C), o f₀ da BPFO na `bpfo_0.3mm` sai em 182,65 Hz e a pontuação é 21,7 dB, contra 9,9 dB na normal; no B, que caberia no chip, 14,4 contra 5,4 dB com a rotação do artigo e 9,4 ± 4,6 contra 5,4 dB com a estimada. Mas a LDA treinada com as outras falhas dá pouco peso a essas colunas (1–7 % da soma dos |coef| nas features padronizadas), e a posição da `bpfo_0.3mm` no eixo da LDA não muda: 0,80 só com o MFCC e 0,79–0,82 com o pente (0 = centro das falhas de treino, 1 = centro da normal, a definição do `inspect_left_out_fault.py`). No B, a rotação estimada no segmento erra nas gravações de BPFO (~48,8 Hz, contra ~50,2 Hz medidos), e o pente só acha a série com a rotação do artigo (`EIXO_ARTIGO_HZ` = 50,17 Hz, Jung et al., 2023); o erro é silencioso, porque o f₀ fica dentro da faixa de busca (só 2 % dos segmentos com o f₀ no extremo da grade). A decisão é a mesma nos dois modos. Como achado, fora da decisão: a regra "LDA OU pente acima de média + 4 desvios da normal do treino" detecta a `bpfo_0.3mm` (0,992–1,000 no C) com especificidade 1,000, mas muda o modelo (é detecção de novidade, assunto do card do Mahalanobis) e a especificidade dela foi medida na mesma gravação normal de onde sai o limiar: só o Protocolo C pode confirmá-la.

### Detecção de novidade para a `bpfo_0.3mm`: sensibilidade no Protocolo B

O `inspect_deteccao_novidade.py` roda, no Protocolo B, os candidatos do `docs/criterio_deteccao_novidade.md` (versão 4) (`exp254`–`exp263`). **A rodada foi feita antes da aprovação formal do critério.** A versão 4 só mudou o controle do D3, que não altera nenhum veredito (o D3 sai descartado por não superar o D2); as regras do D1, do D2 e do D4 são as da versão 3 revisada. Rotação nominal; a estimada é só reporte. **Nenhum é adotado aqui: a especificidade em outra máquina sai do Protocolo C (HUSTmotor).**

- **D2, Mahalanobis no MFCC (OU a LDA): candidato à adoção.** Detecta a `bpfo_0.3mm` em todos os segmentos (1,000), mantém as outras falhas e a especificidade em 1,000, e não dá nenhum alarme na normal fora do bloco. É o mais barato depois do D4 (~1,6 KB de flash). Ressalva: ele mede "diferente da normal" por construção, e não há controle por rótulo no B; com uma única gravação normal, o resultado não separa a falha da identidade da gravação. O Protocolo C decide.
- **D1, pente com limiar da normal (OU a LDA): não elegível.** Detecta a `bpfo_0.3mm` (1,000), mas o controle reprova: pentes fixos em frequências que não são de rolamento também a detectam (0,555 em média com a FFT de 4.096 e 0,603 com a D1b), e a margem exigida é 0,5. O controle não separa as linhas de rolamento de outras diferenças em relação à normal: no auto-teste sintético, em que a falha tem só linhas de BPFO, os pentes sorteados também a detectaram (0,44 e 0,15). A pergunta "o detector depende da gravação normal?" continua aberta, para o D1 e para o D2, e só o Protocolo C responde. Com a rotação estimada, a `bpfo_0.3mm` cai para 0,712 e 0,729.
- **D3, Mahalanobis no MFCC + pente: descartado.** Não supera o D2, que já detecta tudo.
- **D4, LDA retreinada com falha sintética de linhas estreitas: não elegível.** Passa a detectar a `bpfo_0.3mm` (1,000 nas cinco sementes) e passa na permutação (0,535), mas a `bpfi_0.3mm` cai de 1,000 para 0,732, abaixo do piso de 0,95 para as outras falhas.

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
│   ├── models/               # modelo final do classificador
│   │   └── lda.py                         # treino com todos os segmentos e parâmetros em JSON (escore ≥ 0 → falha)
│   ├── exploration/          # estudos e inspeções, sem numeração
│   │   ├── inspect_mat_keys.py
│   │   ├── inspect_signal_data.py
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
│   │   └── 08_export_stage_references.py  # referências por estágio (float64) e tolerâncias do A4 para o harness em C
│   └── validation/           # protocolo de validação do classificador
│       ├── particao.py                    # segmentos, folds A e B, verificação
│       ├── metricas.py                    # sensibilidade, especificidade, acurácia balanceada
│       ├── selecao.py                     # validação interna ao B e regra de escolha (escolher), sem torch
│       ├── run_protocol.py                # executável: roda A ou B e registra o resultado
│       ├── run_tabela_comparativa.py      # executável: tabela provisório × oficial a partir dos metrics.json
│       ├── run_tabela_controles.py        # executável: tabela dos controles (sem c0, permutação, curva)
│       ├── run_tabela_aumento.py          # executável: tabela do B com e sem aumento de dados
│       ├── run_tabela_taxas.py            # executável: tabela do B em duas taxas de amostragem
│       └── run_consolidacao.py            # executável: tabelas .tex e figuras finais para o relatório
│
├── tests/                    # pytest; não dependem de data/
│   ├── conftest.py
│   ├── test_contrato_numerico.py      # cada valor do docs/contrato_numerico.md conferido contra o código
│   ├── test_dsp.py
│   ├── test_experimentos.py           # registry: linha nova nunca é colada na anterior
│   ├── augmentation/
│   │   ├── test_transformacoes.py
│   │   └── test_variantes.py              # inclui: variante aceita nunca lê teste/descarte
│   ├── models/
│   │   └── test_lda.py                    # o JSON reproduz o sklearn; escore ≥ 0 é falha; a descrição da cadeia refaz o MFCC do dsp
│   ├── pipeline/
│   │   ├── test_train_classifier.py       # o 05 só aceita as features de referência da partição
│   │   ├── test_export_lda_header.py      # o .h é paramétrico em N_FEATURES, traz o float32 do JSON e, compilado, reproduz o escore
│   │   ├── test_export_lda_test_vectors.py # os vetores de teste ficam dentro da tolerância, a seleção não olha o escore, recusa .h de outro modelo
│   │   └── test_export_stage_references.py # o formato volta igual do arquivo; as referências versionadas são as do dsp atual
│   ├── exploration/
│   │   ├── test_deteccao_novidade.py      # D1b, amplitude da série sintética, D4 só com normais de treino, controle e regra de escolha
│   │   ├── test_comb_features.py          # candidatos A/B/C do card (A inviável); acha a série fora da razão cinemática; decisão pela validação interna
│   │   └── test_compare_classifiers.py    # critério de escolha: mais barato na margem, em qualquer ordem (pula sem torch)
│   └── validation/
│       ├── test_curva_aprendizado.py      # subamostra só treino, n por classe, reprodutível
│       ├── test_tabela_aumento.py         # tabela do aumento: comparabilidade e fator de expansão
│       ├── test_tabela_taxas.py           # tabela das taxas: comparabilidade e mesma partição em segundos
│       ├── test_consolidacao.py           # consolidação: número sem registry ou rodada no papel errado aborta
│       ├── test_particao.py               # inclui: outra taxa dá os mesmos segmentos em segundos
│       └── test_selecao.py                # folds internos não leem o teste externo; escolher fica com o mais barato na margem
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
│   ├── c_reference/          # reference_data.h: entrada int16 + MFCC esperado, para o porte em C;
│   │                         #   estagios/: referências por estágio e tolerâncias do A4 (08), lidas pelo harness em C
│   ├── classifier/           # estudo de escolha do classificador (compare_classifiers.json)
│   ├── decimation/           # métricas, figuras e relatório da escolha da taxa
│   ├── exploration/          # figuras dos scripts exploratórios
│   ├── modelo_final/         # lda_final.json (parâmetros para o firmware) e escores_referencia.csv, do 05; lda_modelo.h, do 06
│   ├── signature/            # caracterização da assinatura acústica
│   └── validation/           # uma pasta por rodada: metrics.json e folds.csv
│       └── consolidacao/     # tabelas .tex, figuras e rastreabilidade do relatório (run_consolidacao.py)
│
└── docs/                     # proposta, documentação técnica complementar
    ├── contrato_numerico.md  # o que o C tem que reproduzir da extração de características (Fase 2)
    └── criterio_deteccao_novidade.md  # candidatos, limiar e regra de escolha da detecção de novidade (bpfo_0.3mm)
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
python scripts/pipeline/05_train_classifier.py --responsavel <nome>   # modelo final → reports/modelo_final/
python scripts/pipeline/06_export_lda_header.py --responsavel <nome>  # .h da LDA → reports/modelo_final/lda_modelo.h
python scripts/pipeline/07_export_lda_test_vectors.py                 # vetores de teste → reports/modelo_final/lda_vetores_teste.h
python scripts/pipeline/08_export_stage_references.py                 # referências por estágio → reports/c_reference/estagios/
```

O `05` treina a LDA uma única vez, com os 295 segmentos e sem aumento, e grava em `reports/modelo_final/lda_final.json` os parâmetros nas duas formas equivalentes de `models/lda.py`: a padronizada (média, escala, coeficientes e intercepto) e a dobrada, `escore = pesos · x + bias`, que é a que o firmware usa. A convenção é **escore ≥ 0 → falha**: o empate exato segue o sklearn, que o classifica como falha (campo `escore_zero` do JSON). As 26 features seguem a ordem do `04` (as 13 médias, depois os 13 desvios, populacionais). Antes de gravar, o script confere que o JSON reproduz o sklearn em todos os segmentos. O `escores_referencia.csv` traz o escore de cada segmento para conferir a implementação em C. Esses segmentos são os do próprio treino: **o `05` não mede desempenho**, que continua sendo o do Protocolo B (`exp015`). O modelo final classifica corretamente os 295 segmentos, inclusive os da `bpfo_0.3mm`, porque todos estão no treino; **uma demonstração com trechos dessas gravações reproduz o treino, não o Protocolo B**, e não deve ser apresentada como desempenho. O tamanho do escore (±1.300 a ±1.600 nesses segmentos) também não é medida de confiança: a margem vem de o treino conter as cinco gravações, e não diz nada sobre uma gravação nova. **O firmware usa só o sinal do escore.** O JSON é o contrato com o firmware: além dos pesos, ele descreve a cadeia que produz as 26 features nos pontos em que um porte em C costuma divergir sem erro (bloco `entrada`): a origem do sinal a 12,8 kHz (FIR Kaiser de 147 taps, corte em 5.760 Hz, decimação por 4, com o atraso de grupo descartado), escala da entrada (`int16 / 32767`, e não `/ 32768` como a conversão Q15 da CMSIS), janela de Hann simétrica (`np.hanning`), FFT de 512 pontos com espectro de potência, banco de Mel HTK de 20 Hz a fs/2 com os bins já calculados, log natural com `+1e-10`, DCT-II ortonormal e desvio populacional. Um teste refaz o MFCC só a partir dessa descrição e compara com o `dsp.mfcc`. Ao comparar o C com o `escores_referencia.csv`, use tolerância relativa (~1e-3) no escore, porque o MFCC em `float32` difere um pouco do Python; a previsão tem que ser idêntica. Ele exige as features de referência (sem `--norm-clipe`); se o `04` foi rodado com outra opção, rode-o de novo sem opções antes.

O `06` lê o `lda_final.json` e gera o `reports/modelo_final/lda_modelo.h` com a forma dobrada em float32: `LDA_N_FEATURES`, `lda_pesos`, `lda_bias`, `lda_escore()` e `lda_e_falha()`, com a regra de empate do JSON (escore ≥ 0 → falha). O número de características sai do JSON, sem 26 fixo: o `.h` é regenerado sempre que o modelo final mudar. O cabeçalho usa `float` (o `float32_t` da CMSIS) e não inclui a CMSIS, para compilar também no host. Ele traz o id da rodada do modelo, o SHA-256 do JSON e o commit do gerador; numa rodada registrada, o script aborta se o JSON ou o próprio script tiverem mudanças não commitadas. Antes de gravar, ele lê o `.h` de volta e calcula o escore dos 295 segmentos em float32, em Python e compilando o `.h` no host, e aborta se a previsão mudar em algum segmento ou se o escore se afastar do `escores_referencia.csv` mais que o limite do arredondamento da soma em float32, (N + 3)·2⁻²⁴·(Σ|pesos·x| + |bias|), calculado segmento a segmento. Como as features são as mesmas do Python, esse arredondamento é o único erro esperado; a tolerância relativa de 1e-3 do JSON é a do firmware, que calcula também o MFCC em float32, e aqui deixaria passar um erro de ~1,3 no escore. No Cortex-M4 o gcc funde multiplicação e soma (`vfma.f32`), então o escore da placa não é bit a bit o do host, mas fica dentro do mesmo limite. A conferência completa precisa das features do `04` sem opções. Sem `data/`, os testes conferem o `.h` versionado: o SHA-256 escrito nele tem que ser o do `lda_final.json` (senão o `05` rodou de novo e o `06` não), e, compilado, ele tem que dar o escore do segmento 0 do CSV a partir do MFCC do `reference_data.h`. Como no `05`, os segmentos são os do treino: isso confere o porte, não mede desempenho.

O `07` gera o `reports/modelo_final/lda_vetores_teste.h`, que o firmware ([`lab3-firmware-stm32`](https://github.com/claraamp/lab3-firmware-stm32)) usa no marco de 13/10: as 26 características do `04` de cada segmento (todos, ou `--por-gravacao K` igualmente espaçados), o escore e a decisão do `escores_referencia.csv` e a tolerância de cada segmento. A tolerância é o limite do `06` mais o arredondamento do próprio escore de referência para float32. O `.h` também lista, por gravação, o segmento de menor |escore|, que a placa mostra em laço nos LEDs. Antes de gravar, o script confere que o `lda_modelo.h` versionado é o deste `lda_final.json` e que a soma em float32, em Python e com o `.h` compilado no host, a partir dos valores lidos de volta do arquivo gerado, fica dentro da tolerância e dá a decisão do CSV em todos os segmentos. Ele traz o bias do modelo (`VT_LDA_BIAS`), que o firmware compara com o do `lda_modelo.h` para recusar cabeçalhos de modelos diferentes. Como as características vêm do Python, o marco confere o porte do classificador, não o desempenho. Não há linha no registry: o arquivo é uma fixture de teste, e o resultado é o log da placa. Como o `.h` grava o commit do gerador, o script aborta, ao gravar no caminho versionado, se ele mesmo, o `lda_final.json`, o `escores_referencia.csv` ou o `lda_modelo.h` tiverem mudanças não commitadas; com `--out` em outro caminho, só avisa e marca o commit com `+mudancas`. Precisa das features do `04` sem opções.

O `08` gera as referências por estágio do porte em C, em `reports/c_reference/estagios/`: para o segmento do meio de cada gravação, a saída em float64 de cada estágio da cadeia (escala, janela, espectro, Mel, log-Mel, MFCC, média, desvio e escore), em binários little-endian com um `indice.txt` por segmento, e o `tolerancias.txt` com as tolerâncias do A4 (`dsp.ESTAGIOS_PORTE`). É o que o harness em C do firmware ([`lab3-firmware-stm32`](https://github.com/claraamp/lab3-firmware-stm32), `tests/host/harness`) lê para conferir o porte estágio a estágio, no PC. Os intermediários saem de `dsp.estagios_mfcc`, que usa o mesmo `dsp.mfcc` da extração; o script aborta se o MFCC ou o escore não forem os do `04` e do `05`. As tolerâncias, a métrica de cada estágio e como foram fixadas estão no `docs/contrato_numerico.md` (seção "Tolerâncias por estágio (A4)", `inspect_tolerancias_estagio.py`). Como o `07`, grava o commit e aborta, no caminho versionado, se o script, o `dsp.py`, o `lda_final.json` ou o `escores_referencia.csv` tiverem mudanças não commitadas. Sem `data/`, o teste confere que as referências versionadas são as do `dsp` atual, recalculando-as a partir do `int16` gravado nelas.

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