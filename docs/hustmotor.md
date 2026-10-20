# HUSTmotor: dados para o Protocolo C

Card "HUSTmotor — preparar os dados (licença, download, decimação e geometria)" (2º parcial, 26/10). Primeira parte do Protocolo C: deixa o áudio do HUSTmotor no formato do pipeline para medir a especificidade dos detectores de novidade (semana de 21 a 27/10). Nenhum modelo roda aqui.

## Resumo

| Item do card | Situação |
|---|---|
| Licença | **O repositório não declara licença** (a API do GitHub devolve `license: null`). A descrição diz que o dataset é público, para validar algoritmos de diagnóstico, e pede a citação do artigo (abaixo). |
| Download | Só saudável (`H`) e falha de rolamento (`BF`), nas 4 velocidades: 8 arquivos `.txt`, ~78 MB. O SHA-256 de cada um fica no manifest. |
| Decimação | 25,6 → 12,8 kHz pelo `scripts/pipeline/10_prepare_hustmotor.py`, com a mesma regra de projeto do filtro do Jung. Manifest versionado em `data/processed/hustmotor/pcm_decimated/12800/`. |
| Geometria | Rolamento **6203** (8 esferas), **inferido**: BPFO = 3,053 × e BPFI = 4,947 × a rotação. Os autores não publicam a geometria; no `BF`, o BPFO aparece a até ~1 % e o BPFI 2 a 3 % abaixo. |
| Rotação | A nominal de cada arquivo (5, 10, 20 e 30 Hz), como pede o critério v4. A medida na vibração é 1,2 a 3,3 % menor (escorregamento). |
| Segmentos | **5 por classe e velocidade**, 20 por classe, e não ~25: cada arquivo são 5 aquisições separadas de 1,28 s. |
| Divisão do C2 | Em cada arquivo da normal, os blocos 1 e 2 calibram, o 3 é descartado e o 4 e o 5 medem; a falha mede com os 5 blocos (seção "Divisão do C2"). **O C2 só consegue reprovar um detector**: com 8 segmentos medidos, zero alarmes ainda é compatível com até 31 % de alarme falso. |
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

**Como ler o `.bin`:** os 5 segmentos estão em sequência, mas são independentes. Corte em segmentos de 1 s, sem sobreposição (`cortar_segmentos()` do `10`), e nunca filtre nem decime o `.bin` inteiro: o filtro cruzaria as emendas. Isso vale, por exemplo, para o D1b, que no Jung decima a gravação inteira; no HUST, ele tem de decimar segmento a segmento.

| Classe | 5 Hz | 10 Hz | 20 Hz | 30 Hz | Total |
|---|---:|---:|---:|---:|---:|
| saudável (`H`) | 5 | 5 | 5 | 5 | 20 |
| falha de rolamento (`BF`) | 5 | 5 | 5 | 5 | 20 |

O card estimava ~25 s por classe; o que se aproveita são 20 segmentos de 1 s.

## Conversão e decimação

A mesma cadeia do Jung, nas duas etapas:

- **Quantização como no `01`:** um fator único para os 8 arquivos, 0,98 × 32767 / pico, e o sinal em `int16` à taxa original. O pico é só o da normal (0,1295 V, no `H_30HZ`), para os dados de falha não definirem o ganho; hoje dá o mesmo que o pico de todos.
- **Decimação como no `02`:** `dsp.design_decimation` e `dsp.resample_clip`, com a mesma regra (corte em 0,45 × 12,8 kHz = 5.760 Hz, transição de 640 Hz, janela de Kaiser para 60 dB). O fator é 2, e não 4, então os coeficientes não são os do Jung: são 75 taps, contra 147, e a atenuação medida fica em 59,3 dB, contra 60,4 dB no Jung. **Fica 0,7 dB abaixo do alvo de 60 dB** (`config.FIR_ATTENUATION`): a estimativa de taps da janela de Kaiser erra por pouco com menos taps. O efeito é desprezível, porque o pior ponto (6.475 Hz) dobra para 6.325 Hz, dentro da faixa de transição, e não para a banda útil abaixo de 5.760 Hz. Acrescentar taps à mão faria o filtro sair da regra do `02`, então ficou assim.

Os `.bin` têm 64.000 amostras (5 segmentos de 12.800, um por bloco, em sequência). O `pcm_io.carregar_clipes` os lê como os do Jung.

**Unidade:** o HUST está em volts na saída do microfone, sem a sensibilidade publicada, e o Jung em pascal. Os níveis absolutos não são comparáveis, e o MFCC (o c0 em especial) depende do nível. Isso pesa no **C1** (detectores do Jung aplicados ao HUST) e não no **C2** (a normal aprendida no próprio HUST). No C1, "sem recalibrar" vale para o modelo e o limiar, não para o nível: a escala pelo pico, feita em cada dataset como no `01`, já põe os dois na mesma faixa de `int16`. É uma normalização de ganho por dataset, e o relatório deve dizer isso.

## Rotação

A rotação de cada arquivo é a **nominal** da tabela 2 da descrição: 5, 10, 20 e 30 Hz. É a que o critério v4 manda usar (seção 4), porque é o que um sistema instalado sabe da máquina.

Medida na vibração (eixo Y, maior pico entre 0,85 e 1,02 da nominal, 5 blocos), a rotação do motor saudável é 4,88, 9,67, 19,63 e 29,64 Hz: 2,3, 3,3, 1,9 e 1,2 % abaixo da nominal, o escorregamento esperado de um motor de indução. É só referência; no motor BF a 5 Hz o pico não é confiável (4,30 Hz).

## Geometria do rolamento

A descrição não traz o rolamento, e as falhas foram feitas pelo fabricante. O que se sabe:

- O kit de motor com rolamentos com falha da SpectraQuest (`M-FBM`) é um motor de 1/2 HP "com um rolamento com falha na pista interna e outro na pista externa". **O `BF` deve ter as duas falhas**, e não só uma.
- O rolamento do motor Marathon D393 é o **6203**: 8 esferas de 6,747 mm, diâmetro primitivo de 28,5 mm, contato de 0°. Daí **BPFO = 3,053 ×** e **BPFI = 4,947 ×** a rotação.

Conferência nos dados (envelope do som entre 2 e 8 kHz, 5 blocos): no `BF`, aparecem linhas 20 a 30 dB acima do saudável perto das duas frequências, em todas as velocidades de 10 a 30 Hz. Em relação à rotação medida, ficam em ~3,04 a 3,09 × (BPFO, a até ~1 % do 6203) e ~4,82 a 4,86 × (BPFI, 2 a 3 % abaixo do 6203), com bandas laterais. Isso sustenta a geometria, mas não a prova. Ela fica registrada como **inferida**, e a busca de ±3 % do pente cobre essa diferença.

## Risco do pente a 5 Hz

No 6203, BPFO ≈ 3,05 × e BPFI ≈ 4,95 × a rotação: os dentes do pente caem quase em cima das harmônicas do eixo (3 × e 5 ×). O pente do critério mascara ±2 resoluções em volta de cada harmônica do eixo, ±6,25 Hz com 3,125 Hz de resolução. Quando as harmônicas estão a menos de 12,5 Hz umas das outras, a máscara cobre o espectro inteiro:

| Rotação | Distância entre harmônicas | Dentes fora da máscara, D1 (até 5,6 kHz) | D1b (até 2,8 kHz) |
|---|---:|---|---|
| 5 Hz | 5 Hz | **nenhum: inviável** | **nenhum: inviável** |
| 10 Hz | 10 Hz | **nenhum: inviável** | **nenhum: inviável** |
| 20 Hz | 20 Hz | BPFO 35/91, BPFI 21/56 | BPFO 16/45, BPFI 11/28 |
| 30 Hz | 30 Hz | BPFO 34/61, BPFI 22/37 | BPFO 19/30, BPFI 11/18 |

O pente só pode rodar a 20 e a 30 Hz. Isso afeta só o D1, que já saiu não elegível no Protocolo B (`exp254`–`exp263`). O D2 (Mahalanobis no MFCC), o candidato à adoção, não usa o pente.

## Divisão do C2

A seção 5 do critério v4 pede para anotar a divisão entre calibração e medida do C2 "assim que o dataset for baixado, antes de olhar qualquer gravação de falha". **Aqui o `BF` foi visto antes da divisão ser escrita**: o envelope e a rotação da seção anterior, porque o card exigia a geometria. A divisão é mecânica, pela posição do bloco, e não depende do conteúdo de nenhum bloco, então o dano é pequeno, mas fica registrado.

A regra do critério (metade calibra, 1 de descarte, metade mede) vale só para a normal, que é o que calibra. Aplicada aos 5 blocos de cada arquivo `H`, e fixada no `10` (`BLOCOS_CALIBRACAO`, `BLOCOS_DESCARTE`, `BLOCOS_MEDIDA`, `BLOCOS_MEDIDA_FALHA`) e no manifest (`divisao_c2`):

| Classe | Blocos | Papel | Segmentos (por rotação / nas 4) |
|---|---|---|---|
| `H` | 1 e 2 | calibração | 2 / 8 |
| `H` | 3 | descarte | 1 / 4 |
| `H` | 4 e 5 | medida do alarme falso | 2 / 8 |
| `BF` | 1 a 5 | medida da detecção | 5 / 20 |

A falha não calibra nada, então não perde blocos: os 5 medem a detecção.

Os blocos são aquisições separadas, então o descarte do bloco 3 não é necessário contra vazamento no tempo, como no Jung. Ele fica para seguir a regra do critério ao pé da letra.

**Normal por rotação não funciona:** seriam 2 segmentos de calibração para 26 dimensões. A proposta, de Clara na revisão, é **juntar as 4 rotações na normal de calibração** (8 segmentos, com Ledoit-Wolf) e ler o resultado por rotação, com intervalo binomial, em vez de aprovado ou reprovado contra X. Como muda o protocolo, essa proposta precisa de uma página no Registro de Decisões antes de rodar o C2.

**O C2 só consegue reprovar.** Com zero alarmes, o limite superior de 95 % do alarme falso fica em 31 % com 8 segmentos medidos (1 − 0,05^(1/8)) e em 14 % com 20. Para mostrar alarme falso ≤ 5 %, seriam precisos 59 segmentos sem alarme, e o HUST não tem esse tanto de normal. O C2 pode mostrar que um detector dispara demais, mas não que ele dispara pouco o bastante. Isso tem de ir para o relatório.

## O que isto muda no Protocolo C

- **Poucos segmentos, e o C2 só reprova.** Ver a seção "Divisão do C2": por rotação, sobram 2 segmentos para calibrar uma covariância de 26 dimensões, e nenhuma divisão destes dados consegue mostrar alarme falso ≤ 5 %.
- **Nível.** No C1, a escala pelo pico iguala só o nível de pico entre os datasets; o microfone, a sala e o motor continuam diferentes e mudam o MFCC por inteiro. O resultado esperado é alarme em quase tudo, como o critério já prevê.
- **Falha mista.** O `BF` deve ter falha nas duas pistas. A detecção é reportada por velocidade, mas não dá para separar BPFO de BPFI.
