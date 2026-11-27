# Diagnóstico Acústico de Falhas em Motores Rotativos com Processamento Digital de Sinais Embarcado

Projeto da disciplina ENGG54 — Laboratório Integrado III-A (UFBA, Escola Politécnica, 2026.2).

**Equipe:** Amanda Bastos de Melo Correia, Júlia Teixeira Gonçalves, Maria Clara Andrade Magalhães Paternostro D'Oliveira.

## O projeto

Um sistema embarcado que escuta um motor rotativo e diz, a cada segundo de áudio, se ele está **normal** ou **com defeito de rolamento** (pista interna ou externa). A cadeia de processamento e o classificador foram desenvolvidos e validados primeiro em Python, neste repositório, e depois portados para um microcontrolador STM32F411CEU6, no repositório do firmware ([`lab3-firmware-stm32`](https://github.com/claraamp/lab3-firmware-stm32)). O protótipo em Python é a referência: a placa tem que reproduzir as mesmas decisões.

A descrição completa (metodologia, protocolos de validação, resultados e discussão) está no **relatório final**. Este README só apresenta o projeto e mostra como rodar o código; os detalhes técnicos de cada etapa e todos os comandos estão em [`docs/pipeline.md`](./docs/pipeline.md).

## Como funciona

```
áudio a 51,2 kHz → decimação para 12,8 kHz → segmentos de 1 s → MFCC (13 coef.) → média e desvio (26 valores) → LDA → normal / falha
```

1. **Decimação:** FIR (Kaiser, 147 taps) e decimação por 4, de 51,2 para 12,8 kHz — a taxa que cabe no `arm_fir_decimate_f32` da CMSIS-DSP e preserva a banda útil.
2. **Características:** MFCC com quadros de 25 ms e passo de 10 ms; o segmento de 1 s vira a média e o desvio dos 13 coeficientes.
3. **Classificação:** uma LDA binária; o sinal do escore dá a decisão (escore ≥ 0 → falha). A LDA empatou com uma CNN 2D na escolha do classificador e custa milhares de vezes menos.
4. **Validação:** como o conjunto de dados tem uma única gravação por classe, o resultado principal vem de um protocolo em que cada gravação de falha fica fora do treino (Protocolo B). A generalização para outra máquina foi testada no HUSTmotor (Protocolo C).
5. **Placa:** o modelo final é exportado para cabeçalhos em C; a placa processa áudio gravado na flash ou segmentos enviados pela serial, sinaliza a decisão nos LEDs e manda o resultado para um painel no computador.

## Resultados principais

- **Protocolo B:** acurácia balanceada média de 0,875 (sensibilidade 0,750, especificidade 1,000). O pior caso é a falha de pista externa de 0,3 mm, que não é detectada.
- **Placa × Python:** 295/295 segmentos do conjunto principal e 40/40 do HUSTmotor com a mesma decisão do Python, e o escore dentro da tolerância. Isso mede a **fidelidade do porte**, não o desempenho: os 295 segmentos estão no treino do modelo final, e o desempenho é o do Protocolo B.
- **Outra máquina (HUSTmotor):** a LDA treinada no conjunto principal **não transfere**. Sem retreino, ela decide "falha" nos 40 segmentos, inclusive nos 20 normais.
- **Tempo real e memória:** a cadeia inteira usa 17,5 % do tempo real; o firmware ocupa 85,9 % da flash e 26,7 % da RAM.

Os números, as tabelas e as condições de cada resultado estão no relatório final. Cada rodada está registrada em `experiments/registry.csv`, com o commit em que rodou.

## Dataset

- **Conjunto principal:** Jung et al. (2023), *Data in Brief* 48, 109049 — Mendeley Data, DOI [`10.17632/ztmf3m7h5x.6`](https://doi.org/10.17632/ztmf3m7h5x.6). Usa-se o áudio (microfone) a 0 Nm: 5 gravações de 60 s a 51,2 kHz (normal, BPFI e BPFO com defeitos de 0,3 e 1,0 mm).
- **HUSTmotor** (Zhao et al., 2025), para o Protocolo C: download e detalhes em [`docs/hustmotor.md`](./docs/hustmotor.md).

Os arquivos de áudio **não são versionados**: um clone novo vem com `data/` vazio. Baixe os 5 `.mat` do Mendeley para `data/raw/` e rode o pipeline abaixo. Os `manifest.json` e a partição (`splits.json`) são versionados, para conferir que a reconstrução bate com a original.

## Estrutura do repositório

```
scripts/
  config.py, dsp.py, ...   módulos compartilhados (parâmetros e blocos de DSP)
  pipeline/                pipeline numerado, do .mat aos cabeçalhos em C
  validation/              protocolos de validação, envio à placa e tabelas do relatório
  exploration/             estudos que embasaram as decisões (taxa, classificador, assinatura acústica, ...)
  augmentation/            aumento de dados
  models/                  modelo final (LDA)
  interface/               painel de diagnóstico no computador
tests/                     pytest; não dependem de data/
data/                      dados brutos e processados (só os manifestos são versionados)
experiments/registry.csv   registro de todas as rodadas
reports/                   saídas: figuras, tabelas, modelo final, capturas da placa
docs/                      proposta e documentação técnica (pipeline, contrato numérico, HUSTmotor, detecção de novidade)
```

## Como rodar

Requer Python 3. O `requirements.txt` instala o PyTorch só para CPU (as redes do estudo do classificador são pequenas e não ganham nada com GPU).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest tests/                     # não precisa de data/
```

Com os `.mat` em `data/raw/`, o pipeline roda na ordem numérica:

```bash
python scripts/pipeline/01_convert_mat_to_pcm.py    # .mat → PCM int16 a 51,2 kHz
python scripts/pipeline/02_decimate_pcm.py          # decimação para 12,8 kHz
python scripts/pipeline/03_make_splits.py           # confere a partição versionada
python scripts/pipeline/04_extract_features.py      # MFCC de cada segmento
python scripts/pipeline/05_train_classifier.py --sem-registro   # modelo final (LDA) → reports/modelo_final/teste/
python scripts/pipeline/06_export_lda_header.py --sem-registro  # modelo em C → reports/modelo_final/teste/
```

Use sempre `--sem-registro` para reproduzir: ele grava em pastas de teste e não toca no `registry.csv` nem nos arquivos versionados. Com `--responsavel <nome>`, o script registra uma rodada nova, e no `05` isso troca o id e o SHA-256 do `lda_final.json`: o `06` passa a abortar, testes falham e o painel acusa modelo diferente do da placa.

Os passos `07` a `11` geram as referências e os clipes para a placa e preparam o HUSTmotor. O que cada etapa grava e confere está em [`docs/pipeline.md`](./docs/pipeline.md).

Alguns pontos de entrada (o que cada um precisa está no comentário):

```bash
python scripts/validation/run_protocol.py --protocolo B --sem-registro   # resultado principal; precisa do 01 ao 04
python scripts/validation/run_protocol_c.py --sem-registro               # HUSTmotor; precisa do 01 ao 06 e do HUSTmotor com o 10
python scripts/validation/matriz_serial_placa.py --simulada                 # envio à placa simulada (não registra); precisa do 01 ao 06 e do HUSTmotor com o 10
python scripts/interface/run_painel.py --arquivo reports/embarcado/capturas/linux-demo-24-11.log   # painel com uma captura salva; não precisa de data/
python scripts/validation/run_consolidacao.py                            # tabelas e figuras do relatório; só lê os resultados versionados
```

O HUSTmotor vai em `data/raw/hustmotor/` (download em [`docs/hustmotor.md`](./docs/hustmotor.md)) e é preparado pelo `10_prepare_hustmotor.py`. As opções de cada script estão no `--help` dele, e a lista completa de comandos, em [`docs/pipeline.md`](./docs/pipeline.md). As rodadas que entram no `registry.csv` (com `--responsavel`) são da equipe: o registro grava o commit, então commite o código antes de rodar.

## Convenções

Branches, mensagens de commit, nomes de scripts e formato do registro de experimentos: [`CONVENTIONS.md`](./CONVENTIONS.md).
