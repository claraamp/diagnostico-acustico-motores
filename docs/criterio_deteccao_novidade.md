# Critério da detecção de novidade

*Especificação técnica do critério. A aprovação é feita por comentário na página "Critério da detecção de novidade", no Registro de Decisões do Notion; este arquivo é o que os scripts seguem, como o `contrato_numerico.md` para o porte. Situação: **Pendente**. Responsável: Júlia. Versão 4 (17/10): revisões da Clara e da Amanda no PR #24.*

## Objetivo

Detectar a `bpfo_0.3mm` no Protocolo B, a falha que a LDA não detecta (0,500), **sem perder a especificidade**, com um custo que caiba no STM32F411 junto da cadeia atual. O que já se sabe:

- A falha está no sinal: são linhas estreitas de BPFO, que o MFCC dilui.
- Como entrada extra da LDA, o pente não muda nada (`exp249`–`exp252`).
- Como detector separado, com limiar tirado só da normal, ele detectou a falha no B. A especificidade, porém, foi medida na mesma gravação normal de onde sai o limiar.

**Ressalva de honestidade.** A equipe já viu os resultados da variante "LDA OU pente" no B (0,729 com a rotação estimada e 1,000 com a nominal). As escolhas abaixo não podem ser justificadas por esses números. Cada uma tem uma razão independente, escrita ao lado.

## 1. Candidatos (lista fechada)

**D1 a D3 são detectores de novidade:** aprendem só a normal e entram combinados com a LDA atual por **OU** (o segmento é falha se a LDA disser falha ou se o detector disparar). Neles, a LDA e o `lda_modelo.h` não mudam. **O D4 é diferente:** não é um detector à parte, e sim um retreino da própria LDA, que troca os pesos do `lda_modelo.h` (seção 8).

| # | Candidato | O que aprende | Custo adicional no STM32 (estimado) |
|---|---|---|---|
| D1 | **Pente com limiar da normal** | pontuação do pente de BPFO e de BPFI nos segmentos normais | FFT de 4.096 pontos: ~39 KB de flash (tabelas da CMSIS) e ~40 KB de RAM; ~1,1 milhão de ciclos por segmento (~11 ms por segundo de áudio). **D1b:** ~20 KB de flash e ~20 KB de RAM, com ~30 % mais tempo (~1,5 milhão de ciclos, ~15 ms por segundo de áudio) |
| D2 | **Mahalanobis no MFCC** (26 características) | média e covariância da normal | ~1,6 KB de flash (fator de Cholesky, triangular, mais média e escala), ~400 operações; sem RAM a mais |
| D3 | **Mahalanobis no MFCC + pente** (30 características) | idem, com as 4 do pente | a FFT do D1, mais ~2 KB de flash |
| D4 | **LDA retreinada com falha sintética de linhas estreitas** | a mesma LDA, treinada também com segmentos normais acrescidos de uma série harmônica sintética, rotulados como falha | **zero no firmware**: só os pesos do `.h` mudam; refaz os artefatos da seção 9 |

**D1b, variante de memória do pente.** Decimar de 12,8 para 6,4 kHz e usar uma FFT de 2.048 pontos: resolução de 3,125 Hz, a mesma do candidato B do card do pente, com dentes até 2,8 kHz (cerca de 15 ordens de BPFO, das 21 identificadas na caracterização). Gasta metade da memória (RAM e flash), mas não economiza tempo: o filtro da decimação (75 coeficientes) custa mais do que a FFT menor economiza, e o total fica ~30 % acima do D1. **Como a placa decima:** de forma contínua e causal, com o `arm_fir_decimate_f32`, ao longo da gravação. O Python imita isso decimando a gravação inteira de uma vez e cortando o segmento depois, e não cada segmento separado, o que criaria um transitório no começo de cada um. O atraso do filtro causal só desloca o trecho em algumas dezenas de amostras, sem efeito no espectro de potência médio. O D1b entra porque a memória importa, não porque já tenha sido testado: não foi.

**D4: por que pode funcionar, se o pente como entrada não funcionou.** As duas coisas falham por motivos diferentes. O pente como entrada não ajudou porque a LDA, treinada só com falhas de banda larga, não tinha motivo para usar as colunas novas. A `bpfo_0.3mm` aparece no MFCC: quatro bandas de Mel entre 685 e 1.220 Hz sobem de 6 a 14 dB, de forma consistente. O que falta é a LDA ter aprendido que essa direção também é falha. O D4 ensina isso ao acrescentar ao treino uma falha com a mesma *natureza* (linhas estreitas), sem usar os dados da `bpfo_0.3mm`. É uma aposta. Se o resultado for negativo, ele é registrado como tal. Como custa refazer os artefatos do `05` ao `08`, ele só é adotado se passar nos critérios da seção 6.

**D4: especificação da injeção (fixada antes de rodar).**
- **Onde:** na forma de onda, no segmento em ponto flutuante (`int16 / 32767`), antes de toda a cadeia (janela, FFT, Mel, log, DCT). A série sintética passa pela mesma extração que o firmware faria.
- **Em quais segmentos:** só nos segmentos normais de treino de cada fold. Cada segmento normal de treino gera uma cópia "falha sintética". O teste e o descarte do fold nunca são usados.
- **Série:**
  - família BPFO ou BPFI, sorteada com a mesma chance;
  - f₀ sorteado em ±3 % da razão cinemática × a rotação nominal;
  - de 10 a 25 harmônicos, até 5,6 kHz;
  - fase aleatória por harmônico.
- **Amplitude:** cada dente é posto entre 10 e 40 dB (sorteio uniforme em dB) acima do fundo local do próprio segmento. O fundo é a mediana do espectro de 1 s entre 3 e 10 Hz em torno do dente, o mesmo estimador do pente. A faixa vem da literatura e da diferença entre as falhas de 1,0 mm e a normal, e não da `bpfo_0.3mm`.
- **Semente:** fixa (`config.SEMENTE`). Reportar também 5 sementes para medir a dispersão.

## 2. Regra de limiar (uma só, para D1, D2 e D3)

As duas regras escritas não batem: a proposta usa média + 4 desvios, e o card do Mahalanobis, o percentil 97,5. Proposta: **uma regra única, sem suposição de distribuição e medida fora do bloco.**

- Em cada fold do B, os ~47 segmentos normais de treino estão em 5 blocos. Para cada bloco, o detector é ajustado nos outros 4 (~38 segmentos) e pontua o bloco deixado de fora (~9 segmentos). Ao fim, há ~47 pontuações "fora da amostra".
- **Limiar = maior dessas pontuações.** Com ~47 valores, isso corresponde a um alarme falso de cerca de 2 % por pontuação, numa normal parecida com a de treino. O detector final é ajustado nos 5 blocos.
- **Por que esta regra.**
  - O "média + 4σ" supõe uma distribuição gaussiana, que a pontuação do pente não tem.
  - O percentil 97,5 aceita, por construção, 2,5 % de alarme falso por pontuação.
  - A pontuação fora do bloco mede o que acontece com uma normal que o detector não viu. Pontuar os próprios dados de ajuste subestima o alarme falso.
- **O alarme falso acumula.** No D1 há dois limiares (BPFO e BPFI), e todos os detectores entram por OU com a LDA. O alarme falso de cada parte se soma, e os ~2 % por pontuação **não controlam o total**. Por isso a elegibilidade (seção 6) mede a especificidade do **detector combinado**, e não de cada pontuação.
- **Mahalanobis (D2, D3):** ajustado em ~38 segmentos para 26 a 30 dimensões. Os segmentos de um mesmo bloco são vizinhos no tempo e correlacionados, então o número efetivo de amostras é menor que 38. A covariância amostral fica mal condicionada. Usar covariância com encolhimento (Ledoit–Wolf), com a intensidade estimada pelo próprio método, sem escolher olhando o teste. O resultado do D2 e do D3 deve ser lido com essa ressalva.
- **Pente (D1):** a pontuação do pente não tem ajuste (sai direto do espectro), então não há o que deixar fora do bloco: **o limiar de cada série é a maior pontuação dos segmentos normais de treino do fold**. Há um limiar para BPFO e outro para BPFI, e o detector dispara se qualquer uma passar.
- **Mahalanobis na placa (D2, D3):** exportar o fator de Cholesky da covariância (triangular, 351 valores no D2), e não a inversa inteira (676): é metade das operações e da flash, e é mais estável em float32. A distância sai de uma substituição direta. Como o limiar é a maior pontuação da normal, a decisão da placa precisa bater com a do Python. Isso exige vetores de teste com tolerância, como os do `06` e do `07` para a LDA.

## 3. Rotação do eixo

**Os detectores que usam o pente na placa (D1, D3) recebem a rotação nominal como parâmetro de configuração**, como um tacômetro ou o ajuste do inversor. A rotação estimada no segmento é reportada à parte.

- **Por quê:** no HUSTmotor, a rotação de cada arquivo é conhecida (5, 10, 20 e 30 Hz). Num monitoramento real, a rotação nominal da máquina é um dado de instalação. A estimativa no segmento com 3,1 Hz de resolução erra nas gravações de pista externa (~48,8 Hz), um problema de resolução e não do detector.
- **Ressalva:** já sabemos que a rotação nominal favorece o resultado no B. Por isso, os dois modos são reportados lado a lado, e o relatório diz que o sistema depende da rotação conhecida.
- **Quem depende da rotação, e onde:**
  - D1 e D3: no treino, na placa e no Protocolo C;
  - D4: **só no treino**, para posicionar a série sintética. Na placa e no Protocolo C, ele é a LDA e não usa rotação;
  - D2: não usa rotação.

## 4. Taxa máxima de alarme falso (o X %)

**X = 5 % por segmento de 1 s**, medido no Protocolo C (seção 5).

- **Em termos práticos:** 5 % por segmento é **um alarme falso a cada ~20 s**. Com a regra de decisão "2 de 3 segmentos consecutivos", barata no firmware (um contador), cai para ~0,7 % por decisão, **um alarme a cada 2 a 3 minutos**. Esse número supõe segmentos independentes, o que não vale (segmentos vizinhos são correlacionados), e por isso é otimista. O relatório deve dizer nesses termos.
- A decisão usa o número por segmento. A regra de 2 de 3 é reportada como secundária.
- No Protocolo B, a especificidade continua com o piso de 0,95 de antes.

## 5. Protocolo C: duas leituras, porque a transferência entre máquinas é o risco principal

**Esta seção muda o critério do card do HUSTmotor**, que previa o limiar calibrado no Jung, sem recalibrar. O card tem de ser atualizado junto.

O limiar sem recalibrar testa se o detector **transfere de uma máquina para outra**, o que é muito exigente. O MFCC de um motor diferente, outro microfone, outra sala e outra rotação vai cair longe da "normal" do Jung. **É provável que o D2 e o D3 disparem em quase tudo no HUST**, e isso não diria nada sobre a utilidade deles numa máquina instalada. O pente é mais robusto a essa troca, porque a pontuação é uma razão entre o pico e o fundo local. Mesmo assim, ele precisa da **geometria do rolamento do HUSTmotor** para as razões de BPFO e BPFI. A verificação faz parte do card "HUSTmotor: preparar os dados". Se a geometria não estiver documentada, o D1 e o D3 não se aplicam ao C.

Proposta: reportar duas leituras, decididas agora.

- **C1, transferência (o critério original do card):** detectores e limiares do Jung aplicados ao HUST, sem recalibrar. Critério: alarme falso ≤ X.
- **C2, comissionamento:** a "normal" é aprendida no próprio HUST, como faria um sistema instalado.
  - **Divisão, fixada agora:** para cada rotação, as gravações normais são cortadas em segmentos de 1 s. Os **primeiros 50 %** (no tempo) calibram o detector, com a regra da seção 2 aplicada aos blocos dessa metade. Os **últimos 50 %**, depois de uma faixa de descarte de 1 segmento, medem o alarme falso. Se houver mais de um arquivo normal por rotação, os primeiros arquivos calibram e os demais medem.
  - Os nomes dos arquivos de cada lado são anotados assim que o dataset for baixado, **antes de olhar qualquer gravação de falha**.
  - Critério: alarme falso ≤ X. A detecção das falhas do HUST na mesma rotação é reportada.
- **"Deixar uma velocidade de fora"** (Fase 3, até 09/11): calibrar em três rotações e medir na quarta. É uma leitura intermediária entre o C1 e o C2.

**A escolha usa o C2.** O C1 é reportado como limite de transferência. Um detector que só passa no C2 é aceitável, desde que o relatório diga que exige calibração na máquina.

## 6. Regra de escolha

1. **Elegíveis:** os candidatos que, no Protocolo B, mantêm a **especificidade do detector combinado** em ≥ 0,95 e passam no controle da seção 7. O **D3** só segue se detectar a `bpfo_0.3mm` mais de 0,02 acima do D2. Se não, o pente não acrescenta nada ao MFCC, e o D3 é descartado pelo custo (~40 KB a mais para fazer o mesmo que o D2 com ~3 KB). A especificidade aqui é a da normal fora do bloco, como na seção 2. As outras três falhas ficam ≥ 0,95 quase automaticamente, porque o OU só acrescenta detecções. Elas são conferidas, mas a restrição real é a especificidade. No D4, que troca a LDA, as outras três falhas são, sim, uma restrição real.
2. **Sensibilidade:** acurácia balanceada da `bpfo_0.3mm` no B, reportada seja qual for. Para ser adotado, um candidato precisa de ≥ 0,80 nela.
3. **Especificidade externa:** alarme falso ≤ X no C2.
4. **Desempate:** dentro de 0,02 na `bpfo_0.3mm`, fica o mais barato em memória (flash, depois RAM). A ordem de custo no firmware é D4 < D2 < D1b < D1 < D3. O D4 tem o custo extra de refazer os artefatos (seção 9).
5. **Nenhum passa:** a cadeia segue com MFCC e LDA, e o resultado negativo é registrado e reportado.

A rodada de sensibilidade no B pode ser feita já, antes do HUST. A **adoção** só acontece depois do C2.

## 7. Controles

A permutação de rótulos só faz sentido para quem aprende com rótulos. Por isso, o controle muda conforme o candidato:

- **D4 (supervisionado):** permutação por bloco dos rótulos de treino, 20 sorteios. As cópias sintéticas continuam "falha". **Passa com a acurácia balanceada média ≤ 0,55.** Acima disso, as cópias sintéticas carregam informação que não deveriam (por exemplo, a identidade da gravação normal), e o D4 fica inelegível até investigar.
- **D1 e D3 (pente): pente fixo em frequências sorteadas, controle relativo.** Repetir a rodada com 20 pentes em frequências que não são de rolamento:
  - razões entre 2 e 7 × a rotação, fora de ±6 % de BPFO, de BPFI e das frações simples delas (p/q, com p e q até 3), e a mais de 0,15 de um múltiplo inteiro do eixo;
  - **sem a busca de ±3 %**: o f₀ fica fixo.

  **D1: passa se o pente de rolamento detectar a `bpfo_0.3mm` (detector sozinho, sem a LDA) pelo menos 0,5 acima da média dos pentes sorteados.**

  **D3: só o ganho sobre o D2 conta.** O D3 junta o MFCC e o pente. Se o MFCC sozinho (D2) já detecta, trocar o pente por um sorteado não muda nada, e a regra do D1 nunca passaria. Por isso, o D3 passa se tiver ganho sobre o D2 e **pelo menos metade desse ganho sumir** com os pentes sorteados: detecção sorteada ≤ D2 + 0,5 × (D3 real − D2).

  **Só como referência**, o D1 e o D3 também reportam o pente de rolamento **fixo**, sem a busca, para comparar pentes do mesmo tipo. Ele não entra na regra: a BPFO real fica 1,8 % acima da cinemática, e um pente fixo na cinemática erra a série a partir do 2º harmônico. Ele detectaria pouco por esse motivo, e não por falta de falha.

  *Por que mudou depois do auto-teste sintético:* a versão 2 pedia que a detecção com o pente sorteado ficasse "perto da taxa de alarme falso". No sinal sintético, em que o detector funciona por construção, essa regra o reprovava. Com linhas de falha fortes, mesmo um pente em frequência errada esbarra em algumas delas por acaso, e a média sobe. A busca de ±3 % e as frações simples da BPFO (um pente em 2/3 dela acerta um dente em cada três harmônicos) pioravam isso. No sintético, o detector correto dá 1,00 com o pente de rolamento e 0,44 (FFT de 4.096) e 0,15 (D1b) com os sorteados, e passa na regra relativa. O controle continua respondendo à pergunta da Clara: o detector pega a falha **muito mais** nas frequências de rolamento que em outras?
- **D2 (Mahalanobis no MFCC):** não há controle por rótulo. Por construção, ele mede "diferente da normal", que é exatamente o atalho que os controles do 1º parcial não descartaram. O controle dele é o Protocolo C, e a leitura do B deve dizer isso.

## 8. Memória e tempo (números da Frente 1)

**O gargalo é a flash, não a RAM.**

- **RAM:** o firmware usa hoje ~7 KB (28 KB com o teste da Clara), de 128 KB. Os ~40 KB do D1 cabem.
- **Flash:** o firmware ocupa ~61 KB e os clipes, ~350 KB, de 512 KB. Sobram ~100 KB. As tabelas da FFT da CMSIS ficam na flash: **~39 KB para 4.096 pontos e ~20 KB para 2.048**. O D1 sozinho levaria cerca de 40 % do que sobra; o D1b, cerca de 20 %. O D2 (~1,6 KB) e o D4 (zero) praticamente não pesam.
- **Tempo:** a estimativa foi calibrada pela FFT de 512 medida na placa (35,7 mil ciclos, ~3× a fórmula teórica). O D1 fica em ~1,1 milhão de ciclos por segmento, ~11 ms por segundo de áudio (~1 % do tempo real), e o D1b gasta ~30 % mais (~15 ms por segundo, ~1,5 % do tempo real), por causa do FIR de decimação. Nenhum candidato é problema de tempo real.
- **Decimação e reaproveitamento de buffer:** dependem de como a Frente 1 organizar o firmware (decimação em streaming, por exemplo). A conta final sai do `.map` e do DWT.

## 9. Impacto nos artefatos do porte

As referências por estágio e as tolerâncias do A4 (`08_export_stage_references.py`, `exp253`) e os cabeçalhos do firmware dependem do modelo. O efeito de adotar cada candidato é diferente:

| Candidato | O que muda | O que tem de ser refeito |
|---|---|---|
| D4 | os pesos da LDA (modelo novo, outro `id`) | o `05`, o `06` (`lda_modelo.h`), o `07` (`lda_vetores_teste.h`), o `08` (o `escore.f64` e o `modelo.txt` de cada segmento) e o `09` (os escores do `clipes_demo.h`), além dos arquivos do firmware que conferem o modelo, como o `clipes_demo.h` copiado para o firmware, que o receptor dos blocos 1 a 3 confere pelo modelo; as tolerâncias do MFCC continuam valendo, porque o MFCC não muda |
| D2 | um cabeçalho novo com a média, a escala, o fator de Cholesky e o limiar | um gerador novo (no estilo do `06`) e vetores de teste do escore de Mahalanobis com tolerância (no estilo do `07`); a LDA e as referências do MFCC não mudam |
| D1, D1b, D3 | uma cadeia nova (FFT dedicada, pente) | referências e tolerâncias próprias para os estágios do pente, no `08` e no harness, além do gerador dos limiares |

## 10. Onde cada parte fica

- **A decisão:** no Notion, no Registro de Decisões. Este arquivo guarda a especificação.
- **Rodadas no B, no C1 e no C2, e o gerador dos parâmetros:** no repositório do Python (`diagnostico-acustico-motores`), com o script de exploração, o registry e os testes.
- **Se um detector for adotado:** o cabeçalho com os parâmetros, gerado pelo Python, vai para o repositório do Python. O código C do detector e a integração ficam no repositório do firmware (`lab3-firmware-stm32`), com a Clara.

## Pronto quando

A página tem a Situação "Definida", com os valores acima (ou os que a equipe trocar) aprovados por comentário, e o card do HUSTmotor está atualizado com a seção 5. Só depois disso a rodada de sensibilidade no Protocolo B começa.
