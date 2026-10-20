# HUSTmotor: dados para o Protocolo C

Card "HUSTmotor — preparar os dados (licença, download, decimação e geometria)" (2º parcial, 26/10). Primeira parte do Protocolo C: deixa o áudio do HUSTmotor no formato do pipeline para medir a especificidade dos detectores de novidade (semana de 21 a 27/10). Nenhum modelo roda aqui.

## Resumo

| Item do card | Situação |
|---|---|
| Licença | **O repositório não declara licença** (a API do GitHub devolve `license: null`). A descrição diz que o dataset é público, para validar algoritmos de diagnóstico, e pede a citação do artigo (abaixo). |
| Download | Só saudável (`H`) e falha de rolamento (`BF`), nas 4 velocidades: 8 arquivos `.txt`, ~78 MB. O SHA-256 de cada um fica no manifest. |
| Decimação | 25,6 → 12,8 kHz pelo `scripts/pipeline/10_prepare_hustmotor.py`, com a mesma regra de projeto do filtro do Jung. Manifest versionado em `data/processed/hustmotor/pcm_decimated/12800/`. |
| Geometria | Rolamento **6203** (8 esferas), **inferido**: BPFO = 3,053 × e BPFI = 4,947 × a rotação. Os autores não publicam a geometria; os dados batem com ela a 1 a 3 %. |
| Rotação | A nominal de cada arquivo (5, 10, 20 e 30 Hz), como pede o critério v4. A medida na vibração é 1 a 3 % menor (escorregamento). |
| Segmentos | **5 por classe e velocidade**, 20 por classe, e não ~25: cada arquivo são 5 aquisições separadas de 1,28 s. |
| Risco a 5 Hz | **O pente é inviável a 5 e a 10 Hz** com a resolução de 3,125 Hz. Afeta só o D1, que já não é elegível; o D2 não usa o pente. |

## Fonte e licença

- Repositório: [CHAOZHAO-1/HUSTmotor-multi-modal-dataset](https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset). Os dados ficam no Google Drive indicado no README dele (pasta `Raw data`), com um PDF de descrição (`Description file/Read me for details.pdf`).
- Citação pedida: Chao Zhao, Weiming Shen, Enrico Zio, Hui Ma. *Multimodal unified generalization and translation network for intelligent fault diagnosis under dynamic environments*. Engineering Applications of Artificial Intelligence, v. 162, parte C, 2025, 112559.
- **Licença:** não há arquivo de licença no repositório nem menção no README ou no PDF. Sem licença declarada, o uso fica no que os autores escrevem: dados públicos para validar algoritmos, com citação. Por isso os `.txt` e os `.bin` **não são versionados** aqui (como os `.mat` do Jung); só o manifest, com o SHA-256 de cada arquivo de origem.
- Bancada: simulador SpectraQuest, motor Marathon (Cat. No. D393A) de 1/2 HP. Microfone AWA14004; acelerômetro triaxial TREA331. As falhas foram feitas pelo fabricante.

## Download

1. Abrir a pasta do Google Drive do README do repositório e baixar, de `Raw data/`, os 8 arquivos `H_5HZ.txt`, `H_10HZ.txt`, `H_20HZ.txt`, `H_30HZ.txt`, `BF_5HZ.txt`, `BF_10HZ.txt`, `BF_20HZ.txt` e `BF_30HZ.txt`. As outras 4 classes (`BOW`, `BROKEN`, `MISAL`, `UNBAL`) não entram.
2. Copiar para `data/raw/hustmotor/`.
3. `python scripts/pipeline/10_prepare_hustmotor.py`. O SHA-256 de cada `.txt` vai para o manifest; uma reconversão com os mesmos arquivos dá o mesmo manifest.

## Estrutura dos arquivos

Cada `.txt` tem um cabeçalho e 163.840 linhas com 5 colunas: tempo, vibração X, Y e Z e **som** (a 5ª, a única usada). A descrição fala em "6,4 s por amostragem", mas o cabeçalho diz `Number of Blocks 5`, e o tempo volta a zero a cada 32.768 amostras. **São 5 aquisições de 1,28 s, sem continuidade entre elas.** O script confere isso em cada arquivo e recusa um arquivo em que o tempo não volte a zero na emenda.

Consequência: cada bloco é decimado sozinho (o filtro não atravessa a emenda) e dá **um** segmento de 1 s, o do meio do bloco, longe dos transitórios do filtro. Segmentos sobrepostos no mesmo bloco seriam quase cópias e inflariam a contagem.

| Classe | 5 Hz | 10 Hz | 20 Hz | 30 Hz | Total |
|---|---:|---:|---:|---:|---:|
| saudável (`H`) | 5 | 5 | 5 | 5 | 20 |
| falha de rolamento (`BF`) | 5 | 5 | 5 | 5 | 20 |

O card estimava ~25 s por classe; o que se aproveita são 20 segmentos de 1 s.

## Conversão e decimação

A mesma cadeia do Jung, nas duas etapas:

- **Quantização como no `01`:** um fator único para os 8 arquivos, 0,98 × 32767 / pico global (0,1295 V, no `H_30HZ`), e o sinal em `int16` à taxa original.
- **Decimação como no `02`:** `dsp.design_decimation` e `dsp.resample_clip`, com a mesma regra (corte em 0,45 × 12,8 kHz = 5.760 Hz, transição de 640 Hz, janela de Kaiser para 60 dB). O fator é 2, e não 4, então os coeficientes não são os do Jung: são 75 taps, contra 147, e a atenuação medida fica em 59,3 dB, contra 60,4 dB no Jung.

Os `.bin` têm 64.000 amostras (5 segmentos de 12.800, um por bloco, em sequência). O `pcm_io.carregar_clipes` os lê como os do Jung.

**Unidade:** o HUST está em volts na saída do microfone, sem a sensibilidade publicada, e o Jung em pascal. Os níveis absolutos não são comparáveis, e o MFCC (o c0 em especial) depende do nível. Isso pesa no **C1** (detectores do Jung aplicados ao HUST sem recalibrar) e não no **C2** (a normal aprendida no próprio HUST).

## Rotação

A rotação de cada arquivo é a **nominal** da tabela 2 da descrição: 5, 10, 20 e 30 Hz. É a que o critério v4 manda usar (seção 4), porque é o que um sistema instalado sabe da máquina.

Medida na vibração (eixo Y, maior pico entre 0,85 e 1,02 da nominal, 5 blocos), a rotação do motor saudável é 4,88, 9,67, 19,63 e 29,64 Hz: 1 a 3 % abaixo da nominal, o escorregamento esperado de um motor de indução. É só referência; no motor BF a 5 Hz o pico não é confiável (4,30 Hz).

## Geometria do rolamento

A descrição não traz o rolamento, e as falhas foram feitas pelo fabricante. O que se sabe:

- O kit de motor com rolamentos com falha da SpectraQuest (`M-FBM`) é um motor de 1/2 HP "com um rolamento com falha na pista interna e outro na pista externa". **O `BF` deve ter as duas falhas**, e não só uma.
- O rolamento do motor Marathon D393 é o **6203**: 8 esferas de 6,747 mm, diâmetro primitivo de 28,5 mm, contato de 0°. Daí **BPFO = 3,053 ×** e **BPFI = 4,947 ×** a rotação.

Conferência nos dados (envelope do som entre 2 e 8 kHz, 5 blocos): no `BF`, aparecem linhas 20 a 30 dB acima do saudável perto das duas frequências, em todas as velocidades de 10 a 30 Hz. Em relação à rotação medida, ficam em ~4,82 a 4,86 × (BPFI) e ~3,04 a 3,09 × (BPFO), com bandas laterais: a 1 a 3 % do 6203. Isso sustenta a geometria, mas não a prova. Ela fica registrada como **inferida**, e a busca de ±3 % do pente cobre essa diferença.

## Risco do pente a 5 Hz

No 6203, BPFO ≈ 3,05 × e BPFI ≈ 4,95 × a rotação: os dentes do pente caem quase em cima das harmônicas do eixo (3 × e 5 ×). O pente do critério mascara ±2 resoluções em volta de cada harmônica do eixo, ±6,25 Hz com 3,125 Hz de resolução. Quando as harmônicas estão a menos de 12,5 Hz umas das outras, a máscara cobre o espectro inteiro:

| Rotação | Distância entre harmônicas | Dentes fora da máscara, D1 (até 5,6 kHz) | D1b (até 2,8 kHz) |
|---|---:|---|---|
| 5 Hz | 5 Hz | **nenhum: inviável** | **nenhum: inviável** |
| 10 Hz | 10 Hz | **nenhum: inviável** | **nenhum: inviável** |
| 20 Hz | 20 Hz | BPFO 35/91, BPFI 21/56 | BPFO 16/45, BPFI 11/28 |
| 30 Hz | 30 Hz | BPFO 34/61, BPFI 22/37 | BPFO 19/30, BPFI 11/18 |

O pente só pode rodar a 20 e a 30 Hz. Isso afeta só o D1, que já saiu não elegível no Protocolo B (`exp254`–`exp263`). O D2 (Mahalanobis no MFCC), o candidato à adoção, não usa o pente.

## O que isto muda no Protocolo C

- **Poucos segmentos.** No C2 (comissionamento), a normal é aprendida no próprio HUST: são 5 segmentos saudáveis por velocidade, 20 no total, para uma covariância de 26 dimensões. O Ledoit-Wolf permite estimar, mas a distância de Mahalanobis fica pouco estável e o limiar "maior pontuação da normal fora do bloco" sai de poucos pontos. A taxa de alarme falso medida terá resolução grossa (1 alarme em 5 segmentos é 20 %). Vale decidir antes de rodar se a normal do C2 é por velocidade ou junta.
- **Nível.** No C1, a diferença de unidade e de ganho entre os dois datasets muda o MFCC por inteiro; o resultado esperado é alarme em quase tudo, como o critério já prevê.
- **Falha mista.** O `BF` deve ter falha nas duas pistas. A detecção é reportada por velocidade, mas não dá para separar BPFO de BPFI.
