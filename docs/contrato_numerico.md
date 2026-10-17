# Contrato numérico da extração de características

O que a implementação em C (Fase 2) tem que reproduzir para chegar às mesmas 26
características do Python e, portanto, ao mesmo escore da LDA. Cada linha diz o
que o Python faz hoje, onde está no código e onde um porte "certo" costuma
divergir.

A fonte da verdade é o código (`scripts/dsp.py`, `scripts/pcm_io.py`,
`scripts/pipeline/04_extract_features.py`). Do `config.py`, o `dsp.py` lê o
`MFCC_FMIN` (piso do banco de Mel), mas não os outros valores que descrevem a
cadeia:

- `MFCC_FMAX` não é lido de propósito: o teto do banco de Mel é sempre o Nyquist
  da taxa da rodada (`fs/2`). O config fixa 6.400 Hz, que só vale a 12,8 kHz, e
  a comparação a 25,6 kHz (`exp235`) depende de o teto acompanhar a taxa.
- `MFCC_JANELA` e `N_FFT` descrevem o que o código faz (`np.hanning` e a
  próxima potência de 2 acima da janela), mas não são lidos.

O que garante que esses valores e o código coincidem é o teste
`test_valores_do_config_coincidem_com_o_dsp`. Os demais testes de
`tests/test_contrato_numerico.py` conferem, contra o código, cada valor desta
página: se alguém mudar a cadeia, o teste falha e esta página tem que mudar
junto.

Referências de linha: os arquivos da `main` depois do PR da Frente 3D. Nele, o
`log_mel` e o `mfcc` do `dsp.py` ganharam o parâmetro opcional `estagios`, que
guarda os intermediários para as referências por estágio sem mudar a conta: as
features do `04` saem idênticas, byte a byte, às de antes.

**Relação com o `lda_final.json`.** Desde o #17, o bloco `entrada` do
`reports/modelo_final/lda_final.json`, gerado por `models/lda.py`
(`cadeia_entrada`), é o contrato que acompanha o modelo até o firmware. Ele
descreve a mesma cadeia em forma compacta: divisor 32767, quadros, Hann
simétrica, FFT, bins do Mel, log, DCT, resumo por segmento e a origem do sinal
(FIR de 147 taps e descarte das 73 amostras do atraso). Esta página não o
substitui. Ela detalha cada estágio, diz onde está no código e onde um porte
costuma divergir, e acrescenta o que o JSON não cobre: a fase do decimador e o
recorte dos clipes (a dúvida que o JSON deixa em aberto), a quantização em C, o
empacotamento da FFT e a acumulação em float32. O teste
`test_contrato_coincide_com_o_bloco_entrada` confere que os valores desta
página, o `cadeia_entrada()` e o JSON versionado coincidem; quem mudar a cadeia
atualiza os três.

**Opção: tabelas em vez de recálculo.** A janela de Hann, o banco de Mel e a
matriz da DCT são constantes: dependem só dos parâmetros, não do sinal. Em vez
de o C recalculá-los, eles podem ser exportados do `dsp.py` como tabelas
`const float` na flash (Hann[320]; banco de Mel 20 × 257, ou esparso, com os
bins de início e fim e os pesos de cada filtro; DCT 13 × 20), como o `06` faz
com o `lda_modelo.h`. Com isso, as divergências das linhas 4, 7, 8 e 11 da
tabela (Hann periódica, normalização de Slaney, `round` nas bordas, escala da
DCT) deixariam de ser possíveis. O gerador, no estilo do `06`, fica para o
bloco B; até lá, e para quem recalcular no C, o contrato desses estágios é o
descrito abaixo.

Escopo: a tabela abaixo cobre a cadeia a partir do PCM decimado a 12,8 kHz
(`int16`). A decimação (filtro, atraso de grupo, recorte dos clipes e
quantização para `int16`) está na seção "Decimação", e a forma de acumular a
média e o desvio em float32, na seção "Média e desvio em float32". As
tolerâncias de comparação por estágio estão na seção "Tolerâncias por estágio
(A4)", e as referências com que o C é comparado, em "Referência numérica".

## Cadeia, estágio a estágio

| # | Estágio | O que o Python faz | Onde | Onde o porte costuma divergir |
|---|---|---|---|---|
| 1 | Entrada | `x = int16 / 32767` (`config.INT16_FULL`), em float64 | `pcm_io.py:75`; `config.py:152` | Q15 da CMSIS (`arm_q15_to_float`) divide por **32768**. A diferença é um fator de 1,00003, que soma uma constante a todo o log-Mel e mexe só no c0, mas tem que ser a mesma dos dois lados. |
| 2 | Segmento | 1 s = 12.800 amostras, alinhado à partição (`splits.json`); nenhum quadro atravessa o limite do segmento | `dsp.py:183-187`; `04_extract_features.py` (laço sobre `particao.segmentos_de`) | Buffer circular contínuo, com quadros que atravessam segmentos. |
| 3 | Quadros | 320 amostras (25 ms), passo de 128 (10 ms), **98 quadros**; o quadro k começa em `128·k`. As amostras 12.736 a 12.799 do segmento **não entram** em nenhum quadro | `dsp.py:173-174, 183-187` | Contar 99 quadros, ou completar o último com zeros. |
| 4 | Janela | Hann **simétrica**, `np.hanning(320)`: `w[n] = 0,5 − 0,5·cos(2πn/319)`, com `w[0] = w[319] = 0` | `dsp.py:182` | A Hann periódica (`cos(2πn/320)`, a de `scipy.signal.get_window('hann', N)` e de muitas bibliotecas) é outra janela. Se usar uma função de janela de biblioteca, conferir qual é. |
| 5 | FFT | `rfft` de **512 pontos**: as 320 amostras janeladas e 192 zeros **no fim**; 257 bins (0 a 256) | `dsp.py:175, 189` | Zeros no começo ou centrados. No `arm_rfft_fast_f32`, a saída é empacotada: `out[0]` é a parte real do bin 0 e `out[1]` é a parte real do bin 256 (Nyquist), não a imaginária do bin 0. |
| 6 | Espectro | **Potência**, `|X|²`, **sem** dividir por N nem pela energia da janela | `dsp.py:189` | Magnitude `|X|` em vez de potência (`arm_cmplx_mag_f32` em vez de `arm_cmplx_mag_squared_f32`); normalizar por N. |
| 7 | Banco de Mel | 20 filtros triangulares, escala Mel **HTK**: `mel = 2595·log10(1 + f/700)`; 22 pontos igualmente espaçados em Mel de **20 Hz a fs/2 = 6.400 Hz**; bin de cada ponto = `floor(513·f/12800)` | `dsp.py:137-159` | Fórmula de Slaney (a padrão do librosa), `fmin = 0`, ou bin com `round` em vez de `floor`. |
| 8 | Forma dos filtros | Pico **1** no bin central, **sem normalização por área**; rampa de subida `(k − l)/(c − l)` em `[l, c)` e de descida `(r − k)/(r − c)` em `[c, r)`; o bin `r` fica fora | `dsp.py:152-159` | Normalização por área (`norm='slaney'` do librosa), incluir o bin `r`, triângulo em frequência contínua em vez de por bin. |
| 9 | Mel | `mel = espectro @ fbᵀ` (soma ponderada da **potência**) | `dsp.py:190-191` | — |
| 10 | Log | **Logaritmo natural**, `log(mel + 1e-10)`: o épsilon é **somado**, não um piso | `dsp.py:194` | `log10` ou `10·log10` (dB); piso `max(mel, eps)` em vez de soma; outro épsilon. Num quadro todo em zero, o valor é `ln(1e-10) = −23,0259`. |
| 11 | DCT | DCT-II **ortonormal** (`norm="ortho"`), 13 primeiros coeficientes (c0 incluído) | `dsp.py:210` | DCT sem a escala ortonormal; descartar o c0. A CMSIS não tem DCT-II pronta (só DCT-IV): o caminho direto é uma matriz 13 × 20 pré-calculada (fórmula abaixo). |
| 12 | Resumo | Média e desvio-padrão de cada coeficiente ao longo dos 98 quadros, desvio **populacional** (`ddof=0`, o padrão do `np.std`) | `04_extract_features.py:105` | Desvio amostral (`ddof=1`, divide por 97); acumular por soma e soma dos quadrados em float32 (ver "Média e desvio em float32"). |
| 13 | Ordem | `x = [média c0…c12, desvio c0…c12]`, 26 valores; é a ordem dos pesos no `lda_modelo.h` | `04_extract_features.py:105`; `reports/modelo_final/lda_modelo.h` | Intercalar média e desvio por coeficiente. |
| 14 | Precisão | Tudo em float64 no Python | — | O C em float32 não reproduz bit a bit; a comparação é por tolerância (seção "Tolerâncias por estágio (A4)"). |

## O que a cadeia não tem

Nenhum destes passos existe no Python, e o C não deve acrescentá-los:
pré-ênfase, *dither*, remoção da média do quadro, *liftering*, coeficientes
delta, normalização da média cepstral (CMN), conversão para dB, normalização do
segmento pelo RMS (existe no código só como ablação, `--norm-clipe`, e o modelo
final não a usa).

## Espectro no C: empacotamento da FFT

O `arm_rfft_fast_f32` devolve os 257 bins empacotados em 512 floats: `out[0]` é
a parte real do bin 0 (DC), `out[1]` é a parte real do bin 256 (Nyquist), e os
bins 1 a 255 vêm como pares (real, imaginária) a partir de `out[2]`. Chamar
`arm_cmplx_mag_squared_f32(out, P, 256)` direto soma DC e Nyquist em `P[0]` e
nunca produz `P[256]`. A forma explícita:

```c
P[0]   = out[0] * out[0];                    /* DC */
P[256] = out[1] * out[1];                    /* Nyquist */
arm_cmplx_mag_squared_f32(out + 2, P + 1, 255);
```

No MFCC isso não faz diferença, porque os bins 0 e 256 têm peso zero em todos
os filtros. Faz diferença na referência do espectro por estágio
(`espectro.f64`, seção "Referência numérica"), que tem os 257 bins. Além disso, o `arm_rfft_fast_f32` **sobrescreve o buffer de
entrada**: o quadro janelado não sobrevive à FFT. Se ele for comparado com a
referência, a cópia ou a comparação tem que vir antes da FFT.

## Valores exatos para conferir

**Bordas do banco de Mel** (a 12,8 kHz, FFT de 512): os 22 bins, de onde saem
`l`, `c` e `r` do filtro m como `bins[m]`, `bins[m+1]` e `bins[m+2]`:

```
0, 4, 7, 11, 16, 21, 27, 33, 40, 48, 57, 67, 78, 90, 104, 119, 136, 155, 177, 200, 227, 256
```

Com essa configuração, a correção de filtros degenerados do código (`c == l` ou
`r == c`, `dsp.py:154-157`) nunca é acionada. O bin 0 (DC) e o bin 256 (Nyquist)
têm peso zero em todos os filtros: o primeiro filtro tem pesos nos bins 1 a 6, e
o último, nos bins 201 a 255.

**DCT-II ortonormal**, com N = 20 bandas e k = 0…12:

```
c[k] = s(k) · Σ_{n=0}^{19} L[n] · cos(π·k·(2n + 1) / 40),   s(0) = √(1/20),   s(k>0) = √(2/20)
```

**Janela:** `w[0] = w[319] = 0`, e o máximo, `0,99997575…`, fica em `w[159] = w[160]`
(a janela simétrica de comprimento par não tem amostra igual a 1).

## Referência numérica

O `reports/c_reference/reference_data.h` (gerado por `04 --ref-c`) traz as
12.800 amostras `int16` do primeiro segmento da classe `normal` e a matriz
98 × 13 de MFCC esperada para ele. É a referência que cabe na placa.

**Referências por estágio** (`scripts/pipeline/08_export_stage_references.py`,
em `reports/c_reference/estagios/`). Para um segmento de cada gravação (o do
meio, escolhido pela posição), a saída em float64 de cada estágio, de
`dsp.estagios_mfcc`: `entrada.i16` (o segmento `int16`), `escala.f64`,
`janela.f64` (98 × 320), `espectro.f64` (98 × 257), `mel.f64` e `logmel.f64`
(98 × 20), `mfcc.f64` (98 × 13), `media.f64` e `desvio.f64` (13 cada) e
`escore.f64` (o escore da LDA final em float64). São binários little-endian sem
cabeçalho; o `indice.txt` de cada pasta dá o tipo, as linhas e as colunas de
cada arquivo. Na mesma pasta ficam o `tolerancias.txt` (a tabela do A4), o
`modelo.txt` (id, SHA-256 do JSON e bias da LDA) e o `manifest.json` (commit e
SHA-256 de cada arquivo). É o que o harness em C do repositório do firmware lê.
O 08 aborta se o MFCC por estágio não for o do `dsp.mfcc` ou se o escore não for
o do `escores_referencia.csv`.

## Decimação

Vale para o firmware que decima a bordo, a partir de clipes a 51,2 kHz. Se os
clipes forem gravados já decimados, basta reproduzir a quantização (item 4).

**1. O filtro.** FIR passa-baixa de Kaiser com **147 coeficientes**, corte em
5.760 Hz (ponto de −6 dB do `firwin`), β = 5,65326 (do `kaiserord` com 60 dB
de atenuação e 640 Hz de transição), fator **M = 4** (51,2 → 12,8 kHz), ganho 1
em DC. Os coeficientes saem de `dsp.taps_decimacao`, os mesmos que o
`resample_clip` aplica (`dsp.py:258-274`; projeto em `dsp.py:216-255`, num só
`_fir_kaiser` para os dois). O filtro
é **simétrico**: a ordem invertida dos coeficientes que a CMSIS espera não
muda nada.

**2. O que o Python faz.** `y = lfilter(taps, x)`, com estado inicial zerado;
descarta as **73** primeiras saídas (o atraso de grupo, (147 − 1)/2) e fica com
uma a cada 4 (`dsp.py:270-273`). A amostra decimada j é, portanto,

```
y[j] = Σ_{k=0}^{146} b[k] · x[4j + 73 − k]      →   usa x[4j − 73 … 4j + 73]
```

isto é, a saída fica centrada no instante 4j da gravação original, e não
atrasada.

**3. O que o decimador do firmware precisa ler.** Um decimador FIR causal, com
estado inicial zerado, calcula a saída m com a entrada até a amostra M·m + p do
trecho lido: `y[m] = Σ b[k]·x[M·m + p − k]`. A **fase p** depende da
implementação. O `arm_fir_decimate_f32` tem **p = 0**: copia as M amostras
novas para o estado e calcula a saída com a janela que termina na primeira
delas (conferido no código-fonte da CMSIS-DSP na revisão do PR). Um decimador que espera o bloco
inteiro tem p = M − 1 = 3. Para que ele produza as amostras decimadas
`j = início … início + n − 1` do Python, `dsp.recorte_para_decimar(início, n,
spec, fase)` devolve o trecho a ler e quantas saídas descartar:

| | Fórmula | p = 0 (CMSIS) | p = 3 |
|---|---|---|---|
| Saídas a descartar | `ceil((L − 1 − p)/M)` | **37** | 36 |
| Começo do trecho | `M·início + d − p − M·descartar` | `4·início − 75` | `4·início − 74` |
| Fim do trecho (exclusivo) | `começo + M·(descartar + n)` | `4·(início + n) + 73` | `4·(início + n) + 70` |

com L = 147 coeficientes, d = 73 de atraso e M = 4. O tamanho do trecho,
`M·(descartar + n)`, é sempre múltiplo de M, como o `arm_fir_decimate_f32`
exige do bloco. As saídas descartadas são as que ainda têm os zeros do estado
inicial.

Para um segmento de 1 s (n = 12.800), com a CMSIS: **75 amostras antes e 73
depois** do trecho de 51.200 amostras que o segmento cobre, 51.348 no total
(≈ 100,3 KB em `int16`). O mínimo que o Python usa depois do trecho é 70; as 3
a mais são lidas para completar o bloco e não entram em nenhuma saída
aproveitada. Isso corrige a estimativa inicial da tarefa ("146 amostras a mais
antes do início"): com a compensação do atraso que o Python faz, a margem se
divide entre os dois lados.

**Primeiro segmento de cada gravação.** Com `início = 0`, o trecho começa em
−75: o `lfilter` do Python preencheu esse começo com zeros, e as amostras
decimadas 0 a 18 dependem deles. Para reproduzir esse segmento, acrescentar 75
zeros antes do clipe; o mais simples é não usar o segmento 0 como clipe.

**Conferir a fase na placa.** Um impulso confirma qual é: com `x = [1, 0, 0, …]`,
a saída é `b[0], b[4], b[8], …` com p = 0 e `b[3], b[7], b[11], …` com p = 3
(teste `test_impulso_identifica_a_fase`). O recorte das duas fases é conferido
em `test_recorte_reproduz_o_segmento_decimado`.

**4. Quantização para `int16`.** O `02` decima em float64 a partir do PCM
original (`int16 / 32767`), arredonda o resultado para `int16`
(`round(y·32767)`, com **arredondamento meio-para-par** do `np.round`, e
saturação em −32768…32767; `pcm_io.py:291`) e grava. O `04` lê esse
`int16`. Para reproduzir o Python, o firmware que decima a bordo também
arredonda para `int16` antes do MFCC, com esta linha:

```c
int16_t q = (int16_t)__SSAT(lrintf(y * 32767.0f), 16);
```

O `lrintf` segue o modo de arredondamento corrente, que é meio-para-par por
padrão, e o `__SSAT` satura em −32768…32767, como o `pcm_io`. O `roundf`
arredonda o meio para longe de zero e diverge só nos empates exatos.
**Não usar `arm_float_to_q15`**: ele multiplica por 32768, e não por 32767, e
trunca (ou, com `ARM_MATH_ROUNDING`, arredonda o meio para longe de zero), o que
diverge do `np.round(y·32767)` em quase toda amostra, e não só nos empates.
Pular a requantização dá uma diferença de até meio LSB por amostra: relativa à
escala do segmento, passa da tolerância do estágio `escala` (10⁻⁶), então não é
uma simplificação admitida.

**Testes** (`tests/test_contrato_numerico.py`): o filtro (147 coeficientes,
simétrico, ganho 1); a janela de dependência `x[4j − 73 … 4j + 73]`, mudando uma
amostra de cada vez; o impulso que identifica a fase; o recorte, com um
decimador causal de fase 0 e de fase 3 reproduzindo dois segmentos inteiros do
`resample_clip` (erro < 1e-12); o primeiro segmento, com os zeros; e a
quantização meio-para-par com saturação.

## Média e desvio em float32

O Python calcula a média e o desvio de cada coeficiente com a matriz 98 × 13
inteira, em float64. O firmware, em float32, pode acumular quadro a quadro ou
guardar a matriz. A `dsp.media_desvio_float32` emula, operação por operação em
float32, as três formas:

- **Welford**, quadro a quadro: `n += 1; d = x − média; média += d/n;
  m2 += d·(x − média)`; no fim, `desvio = √(m2/98)`.
- **Soma e soma dos quadrados**, quadro a quadro: `s += x; q += x²`; no fim,
  `desvio = √(q/98 − (s/98)²)`.
- **Dois passos**, com a matriz guardada (98 × 13 floats = 5 KB): primeiro a
  média, depois `Σ(x − média)²/98`.

**Medição** (`exp242`, `scripts/exploration/inspect_acumulacao_float32.py`,
resultado em `reports/exploration/acumulacao_float32.json`). Nos 295 segmentos,
com os quadros de MFCC do Python arredondados para float32, contra a média e o
desvio desses mesmos valores em float64. A medição isola o erro da acumulação; o
erro do próprio MFCC em float32 está na seção "Tolerâncias por estágio (A4)". O efeito no escore é
`|pesos · Δx|` com os pesos da LDA final (`exp239`).

| Método | Erro da média | Erro do desvio | Erro relativo do desvio | Efeito no escore |
|---|---:|---:|---:|---:|
| Welford | 5,1 × 10⁻⁶ | 1,2 × 10⁻⁶ | 2,1 × 10⁻⁶ | 0,0013 |
| Soma e soma dos quadrados | 4,1 × 10⁻⁶ | 1,2 × 10⁻⁴ | 2,4 × 10⁻⁴ | 0,0056 |
| Dois passos (matriz) | 4,1 × 10⁻⁶ | 2,6 × 10⁻⁷ | 3,3 × 10⁻⁷ | 0,0012 |

Máximos sobre os 295 segmentos e os 13 coeficientes. A maior razão
|média|/desvio de um coeficiente num segmento é 28,8.

**Leitura.** Para a decisão do classificador, o método é indiferente: o efeito
no escore fica abaixo de 0,006, e o menor |escore| dos 295 segmentos é 1.297,2. A
escolha importa para a comparação estágio a estágio do porte (A4: a tolerância
do desvio é 5 × 10⁻⁵, e a soma dos quadrados passa dela). A soma dos
quadrados erra o desvio cerca de 100 vezes mais que Welford, porque calcula a
variância pela diferença de dois números grandes (`q/98` e `(s/98)²`); um erro
dessa ordem esconderia uma divergência real do porte no estágio do resumo. O
problema cresce com a razão média/desvio: num caso sintético com média 1.000 e
desvio 0,1, o desvio sai com erro de cerca de 500% (teste
`test_soma_dos_quadrados_perde_precisao_com_media_grande`).

**Decisão (17/10, Registro de Decisões).** **Welford**, que mantém o cálculo
quadro a quadro, sem guardar a matriz, com erro relativo do desvio de 2 × 10⁻⁶.
É o que o firmware usa (`mfcc_resumo_acumular` no `mfcc_estagios.c`); no harness,
contra o Python, o desvio fica em 0,7% da tolerância. A soma dos quadrados fica
descartada, pelo erro acima. Os dois passos ficam só como comparação: são um
pouco mais precisos, mas exigem guardar a matriz 98 × 13 e quebram o cálculo
quadro a quadro, e a precisão de Welford já sobra. A emulação não reproduz a
fusão de multiplicação e soma (FMA) do Cortex-M4, que muda o último bit de cada
operação, mas não a ordem de grandeza desses erros.

## Tolerâncias por estágio (A4)

Define quando um estágio portado "bateu" com o Python. O C calcula em float32 e
o Python em float64: a diferença esperada é só de arredondamento, e a tolerância
tem que ficar acima dela e bem abaixo de qualquer erro de porte.

**Dois modos de comparar.**

- **Isolado:** o estágio recebe a referência do estágio anterior, arredondada
  para float32. Mede o erro do próprio estágio e funciona com a cadeia ainda
  incompleta: é como se confere cada bloco à medida que é portado.
- **Encadeado:** a cadeia em C desde o `int16`, como na placa. O erro de um
  estágio inclui o de todos os anteriores, por isso a tolerância encadeada nunca
  é menor que a isolada.

**Métricas** (`dsp.erro_estagio`; o harness faz a mesma conta), com `c` o
calculado e `r` a referência:

- `rel_quadro`: em cada quadro (linha), `max|c − r| / max|r|`; vale o pior
  quadro. Um vetor conta como um quadro só. Mede o erro contra a escala do
  próprio quadro, para que um quadro silencioso não seja julgado pela escala de
  um quadro forte. Usada nos estágios lineares: escala, janela, espectro e Mel.
- `abs`: `max|c − r|`. Usada no log-Mel, no MFCC e na média, que já estão em
  escala logarítmica e passam por zero.
- `rel`: `max|c − r| / |r|`, elemento a elemento. Usada no desvio e no escore,
  que ficam longe de zero.

**Como os valores foram fixados** (`exp253`,
`scripts/exploration/inspect_tolerancias_estagio.py`, resultado em
`reports/exploration/tolerancias_estagio.json`). A cadeia foi emulada em
float32, na ordem do C, nos 295 segmentos:

- escala por divisão;
- janela com o cosseno em float32;
- FFT de precisão simples do scipy;
- Mel e DCT por soma em sequência;
- média e desvio por Welford;
- escore na ordem do `lda_escore()`.

A tolerância é **10 vezes o maior erro medido, arredondada para cima na
sequência 1, 2, 5**. Os valores estão em `dsp.ESTAGIOS_PORTE`, e o 08 os copia
para o `tolerancias.txt` que o harness lê.

| Estágio | Métrica | Maior erro, isolado | Tolerância, isolado | Maior erro, encadeado | Tolerância, encadeado |
|---|---|---:|---:|---:|---:|
| escala | rel_quadro | 5,1 × 10⁻⁸ | 1 × 10⁻⁶ | 5,1 × 10⁻⁸ | 1 × 10⁻⁶ |
| janela | rel_quadro | 3,1 × 10⁻⁷ | 5 × 10⁻⁶ | 3,1 × 10⁻⁷ | 5 × 10⁻⁶ |
| espectro | rel_quadro | 3,9 × 10⁻⁷ | 5 × 10⁻⁶ | 4,1 × 10⁻⁷ | 5 × 10⁻⁶ |
| mel | rel_quadro | 4,8 × 10⁻⁷ | 5 × 10⁻⁶ | 5,0 × 10⁻⁷ | 5 × 10⁻⁶ |
| logmel | abs | 5,5 × 10⁻⁷ | 1 × 10⁻⁵ | 1,7 × 10⁻⁵ | 2 × 10⁻⁴ |
| mfcc | abs | 3,2 × 10⁻⁶ | 5 × 10⁻⁵ | 5,9 × 10⁻⁶ | 1 × 10⁻⁴ |
| media | abs | 5,1 × 10⁻⁶ | 1 × 10⁻⁴ | 5,1 × 10⁻⁶ | 1 × 10⁻⁴ |
| desvio | rel | 2,1 × 10⁻⁶ | 5 × 10⁻⁵ | 1,8 × 10⁻⁶ | 5 × 10⁻⁵ |
| escore | rel | 4,2 × 10⁻⁷ | 5 × 10⁻⁶ | 9,9 × 10⁻⁷ | 1 × 10⁻⁵ |

**Leitura.** Até o Mel, o erro fica na ordem do arredondamento de uma operação
em float32 (cerca de 6 × 10⁻⁸), multiplicado por poucas operações. O salto está
no log encadeado: o erro relativo que o FFT deixa nas bandas de pouca energia
vira erro absoluto no log (`Δln(m) ≈ Δm/m`), e por isso a tolerância encadeada
do log-Mel é 20 vezes a isolada. Mesmo assim, o efeito no escore fica em
10⁻⁶ relativo, cerca de 0,0015 num escore de 1.500. Esse valor é mil vezes
menor que o 1e-3 relativo do `uso_no_firmware` do JSON. O 1e-3 continua valendo
como conferência grosseira da placa inteira, e o A4 é a referência da
comparação estágio a estágio. A decisão (falha ou normal) tem que ser sempre a
mesma do Python.

**O que as tolerâncias pegam.** Cada divergência da coluna "Onde o porte
costuma divergir" passa da tolerância do seu estágio por várias ordens de
grandeza (teste `test_tolerancias_pegam_erros_de_porte`):

- dividir por 32768 dá 3,05 × 10⁻⁵ na escala, 30 vezes a tolerância;
- a Hann periódica, a magnitude em vez da potência e o `log10` em vez do `ln`;
- a DCT sem a escala ortonormal e o desvio amostral (`ddof=1`);
- a soma dos quadrados no desvio, que erra 2,4 × 10⁻⁴ (seção "Média e desvio em
  float32").

**Limites.** A FFT emulada é a do scipy, não a da CMSIS, e a emulação não faz a
fusão de multiplicação e soma (FMA) do Cortex-M4. O harness mede a CMSIS de
verdade, no PC. Nos 5 segmentos das referências, todos os estágios (do `int16`
ao escore, com o `lda_modelo.h` do firmware) ficaram entre 0,6% e 7,8% da
tolerância, nos dois modos. A menor margem é a do Mel encadeado (7,8%, no
`bpfo_0.3mm_seg0206`), seguido do espectro isolado (7,5%); a maior é a do
desvio (0,6% a 0,7%). O escore encadeado erra 3,4 × 10⁻⁷ relativo, 3,4% da
tolerância. A placa, com FMA, muda o último bit de cada operação, mas não a
ordem de grandeza; a margem de 10 vezes cobre isso. Se um estágio passar da
tolerância por pouco, confira a margem antes de procurar erro no porte.

**Harness no PC.** Fica no repositório do firmware (`lab3-firmware-stm32`,
`tests/host/harness`; instruções no README de lá). Ele lê
`reports/c_reference/estagios/`, roda os estágios de `Core/Src/mfcc_estagios.c`
com a implementação genérica em C da CMSIS-DSP e imprime, por estágio e nos dois
modos, o maior erro absoluto, o maior erro relativo à escala do estágio, a
métrica do A4, a tolerância e OK ou FALHOU. Todos os estágios estão portados;
um estágio que entre depois aparece como "não portado" até existir em C. No modo
encadeado, ele também confere que o `mfcc_segmento()`, a chamada única que a
placa usa (`int16` → 26 características), dá bit a bit o mesmo que o laço
estágio a estágio. Ele recusa (código 2) referências de outro modelo, sem o
`modelo.txt`, com forma diferente da esperada, com zero onde a métrica não se
aplica, como o `dsp.erro_estagio`, ou com uma métrica desconhecida no
`tolerancias.txt` (só valem `rel_quadro`, `abs` e `rel`; um nome errado não vira
`rel` sem aviso).
