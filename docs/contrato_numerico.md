# Contrato numérico da extração de características

O que a implementação em C (Fase 2) tem que reproduzir para chegar às mesmas 26
características do Python e, portanto, ao mesmo escore da LDA. Cada linha diz o
que o Python faz hoje, onde está no código e onde um porte "certo" costuma
divergir.

A fonte da verdade é o código (`scripts/dsp.py`, `scripts/pcm_io.py`,
`scripts/pipeline/04_extract_features.py`), não o `config.py`: alguns valores do
config (`MFCC_FMIN`, `MFCC_FMAX`, `MFCC_JANELA`, `N_FFT`) descrevem a cadeia, mas
o `dsp.py` não os lê. Hoje eles coincidem. O teste `tests/test_contrato_numerico.py`
confere, contra o código, cada valor desta página e a coincidência com o config;
se alguém mudar a cadeia, o teste falha e esta página tem que mudar junto.

Referências de linha: `main` em `340d37d`.

Escopo: a tabela abaixo cobre a cadeia a partir do PCM decimado a 12,8 kHz
(`int16`). A decimação (filtro, atraso de grupo, recorte dos clipes e
quantização para `int16`) está na seção "Decimação", e a forma de acumular a
média e o desvio em float32, na seção "Média e desvio em float32". As
tolerâncias de comparação por estágio são a tarefa A4.

## Cadeia, estágio a estágio

| # | Estágio | O que o Python faz | Onde | Onde o porte costuma divergir |
|---|---|---|---|---|
| 1 | Entrada | `x = int16 / 32767` (`config.INT16_FULL`), em float64 | `pcm_io.py:75`; `config.py:152` | Q15 da CMSIS (`arm_q15_to_float`) divide por **32768**. A diferença é um fator de 1,00003, que soma uma constante a todo o log-Mel e mexe só no c0, mas tem que ser a mesma dos dois lados. |
| 2 | Segmento | 1 s = 12.800 amostras, alinhado à partição (`splits.json`); nenhum quadro atravessa o limite do segmento | `dsp.py:177-181`; `04_extract_features.py` (laço sobre `particao.segmentos_de`) | Buffer circular contínuo, com quadros que atravessam segmentos. |
| 3 | Quadros | 320 amostras (25 ms), passo de 128 (10 ms), **98 quadros**; o quadro k começa em `128·k`. As amostras 12.736 a 12.799 do segmento **não entram** em nenhum quadro | `dsp.py:167-168, 177-181` | Contar 99 quadros, ou completar o último com zeros. |
| 4 | Janela | Hann **simétrica**, `np.hanning(320)`: `w[n] = 0,5 − 0,5·cos(2πn/319)`, com `w[0] = w[319] = 0` | `dsp.py:176` | A Hann periódica (`cos(2πn/320)`, a de `scipy.signal.get_window('hann', N)` e de muitas bibliotecas) é outra janela. Se usar uma função de janela de biblioteca, conferir qual é. |
| 5 | FFT | `rfft` de **512 pontos**: as 320 amostras janeladas e 192 zeros **no fim**; 257 bins (0 a 256) | `dsp.py:169, 183` | Zeros no começo ou centrados. No `arm_rfft_fast_f32`, a saída é empacotada: `out[0]` é a parte real do bin 0 e `out[1]` é a parte real do bin 256 (Nyquist), não a imaginária do bin 0. |
| 6 | Espectro | **Potência**, `|X|²`, **sem** dividir por N nem pela energia da janela | `dsp.py:183` | Magnitude `|X|` em vez de potência (`arm_cmplx_mag_f32` em vez de `arm_cmplx_mag_squared_f32`); normalizar por N. |
| 7 | Banco de Mel | 20 filtros triangulares, escala Mel **HTK**: `mel = 2595·log10(1 + f/700)`; 22 pontos igualmente espaçados em Mel de **20 Hz a fs/2 = 6.400 Hz**; bin de cada ponto = `floor(513·f/12800)` | `dsp.py:133-155` | Fórmula de Slaney (a padrão do librosa), `fmin = 0`, ou bin com `round` em vez de `floor`. |
| 8 | Forma dos filtros | Pico **1** no bin central, **sem normalização por área**; rampa de subida `(k − l)/(c − l)` em `[l, c)` e de descida `(r − k)/(r − c)` em `[c, r)`; o bin `r` fica fora | `dsp.py:148-155` | Normalização por área (`norm='slaney'` do librosa), incluir o bin `r`, triângulo em frequência contínua em vez de por bin. |
| 9 | Mel | `mel = espectro @ fbᵀ` (soma ponderada da **potência**) | `dsp.py:184-185` | — |
| 10 | Log | **Logaritmo natural**, `log(mel + 1e-10)`: o épsilon é **somado**, não um piso | `dsp.py:188` | `log10` ou `10·log10` (dB); piso `max(mel, eps)` em vez de soma; outro épsilon. Num quadro todo em zero, o valor é `ln(1e-10) = −23,0259`. |
| 11 | DCT | DCT-II **ortonormal** (`norm="ortho"`), 13 primeiros coeficientes (c0 incluído) | `dsp.py:200` | DCT sem a escala ortonormal; descartar o c0. A CMSIS não tem DCT-II pronta (só DCT-IV): o caminho direto é uma matriz 13 × 20 pré-calculada (fórmula abaixo). |
| 12 | Resumo | Média e desvio-padrão de cada coeficiente ao longo dos 98 quadros, desvio **populacional** (`ddof=0`, o padrão do `np.std`) | `04_extract_features.py:105` | Desvio amostral (`ddof=1`, divide por 97); acumular por soma e soma dos quadrados em float32 (ver "Média e desvio em float32"). |
| 13 | Ordem | `x = [média c0…c12, desvio c0…c12]`, 26 valores; é a ordem dos pesos no `lda_modelo.h` | `04_extract_features.py:105`; `reports/modelo_final/lda_modelo.h` | Intercalar média e desvio por coeficiente. |
| 14 | Precisão | Tudo em float64 no Python | — | O C em float32 não reproduz bit a bit; a comparação é por tolerância (A4). |

## O que a cadeia não tem

Nenhum destes passos existe no Python, e o C não deve acrescentá-los:
pré-ênfase, *dither*, remoção da média do quadro, *liftering*, coeficientes
delta, normalização da média cepstral (CMN), conversão para dB, normalização do
segmento pelo RMS (existe no código só como ablação, `--norm-clipe`, e o modelo
final não a usa).

## Valores exatos para conferir

**Bordas do banco de Mel** (a 12,8 kHz, FFT de 512): os 22 bins, de onde saem
`l`, `c` e `r` do filtro m como `bins[m]`, `bins[m+1]` e `bins[m+2]`:

```
0, 4, 7, 11, 16, 21, 27, 33, 40, 48, 57, 67, 78, 90, 104, 119, 136, 155, 177, 200, 227, 256
```

Com essa configuração, a correção de filtros degenerados do código (`c == l` ou
`r == c`, `dsp.py:150-153`) nunca é acionada. O bin 0 (DC) e o bin 256 (Nyquist)
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
98 × 13 de MFCC esperada para ele. As referências por estágio (janela,
espectro, Mel, log, DCT, resumo e escore) são a tarefa B2.

## Decimação

Vale para o firmware que decima a bordo, a partir de clipes a 51,2 kHz. Se os
clipes forem gravados já decimados, basta reproduzir a quantização (item 4).

**1. O filtro.** FIR passa-baixa de Kaiser com **147 coeficientes**, corte em
5.760 Hz (ponto de −6 dB do `firwin`), β = 5,65326 (do `kaiserord` com 60 dB
de atenuação e 640 Hz de transição), fator **M = 4** (51,2 → 12,8 kHz), ganho 1
em DC. Os coeficientes saem de `dsp.taps_decimacao`, os mesmos que o
`resample_clip` aplica (`dsp.py:240-250`; projeto em `dsp.py:203-237`). O filtro
é **simétrico**: a ordem invertida dos coeficientes que a CMSIS espera não
muda nada.

**2. O que o Python faz.** `y = lfilter(taps, x)`, com estado inicial zerado;
descarta as **73** primeiras saídas (o atraso de grupo, (147 − 1)/2) e fica com
uma a cada 4 (`dsp.py:247-250`). A amostra decimada j é, portanto,

```
y[j] = Σ_{k=0}^{146} b[k] · x[4j + 73 − k]      →   usa x[4j − 73 … 4j + 73]
```

isto é, a saída fica centrada no instante 4j da gravação original, e não
atrasada.

**3. O que o decimador do firmware precisa ler.** O `arm_fir_decimate_f32` é
causal: com estado inicial zerado, a saída m usa a entrada até a amostra M·m do
trecho lido, `y[m] = Σ b[k]·x[M·m − k]`. Para que ele produza as amostras
decimadas `j = início … início + n − 1` do Python, `dsp.recorte_para_decimar`
devolve o trecho a ler e quantas saídas descartar:

| | Valor | Por quê |
|---|---|---|
| Começo do trecho | `4·início − 75` | 37 saídas a descartar × 4 − 73 de atraso |
| Fim do trecho (exclusivo) | `4·(início + n) + 73` | a última saída usa até `4·(início+n−1) + 73`; o bloco múltiplo de 4 acrescenta 3 amostras lidas e não usadas |
| Saídas a descartar | **37** | `ceil(146/4)`: até lá, o estado ainda tem os zeros iniciais |
| Tamanho do trecho | `4·(37 + n)`, múltiplo de 4 | o `arm_fir_decimate_f32` exige bloco múltiplo de M |

Para um segmento de 1 s (n = 12.800): **75 amostras antes e 73 depois** do
trecho de 51.200 amostras que o segmento cobre, 51.348 no total (≈ 100,3 KB em
`int16`). Isso corrige a estimativa inicial da tarefa ("146 amostras a mais
antes do início"): com a compensação do atraso que o Python faz, a margem se
divide entre os dois lados.

**Primeiro segmento de cada gravação.** Com `início = 0`, o trecho começa em
−75: o `lfilter` do Python preencheu esse começo com zeros, e as amostras
decimadas 0 a 18 dependem deles. Para reproduzir esse segmento, acrescentar 75
zeros antes do clipe; o mais simples é não usar o segmento 0 como clipe.

**Conferir a convenção na placa.** Tudo acima supõe `y[m] = Σ b[k]·x[M·m − k]`,
a convenção documentada do `arm_fir_decimate_f32`. Um impulso confirma: com
`x = [1, 0, 0, …]`, a saída tem que ser `b[0], b[4], b[8], …`. Se sair
`b[3], b[7], …`, o decimador usa a última amostra de cada bloco, e o recorte
muda (o teste `test_recorte_reproduz_o_segmento_decimado` documenta a
convenção usada aqui).

**4. Quantização para `int16`.** O `02` decima em float64 a partir do PCM
original (`int16 / 32767`), arredonda o resultado para `int16`
(`round(y·32767)`, com **arredondamento meio-para-par** do `np.round`, e
saturação em −32768…32767; `pcm_io.py:291-293`) e grava. O `04` lê esse
`int16`. Para reproduzir o Python, o firmware que decima a bordo também
arredonda para `int16` antes do MFCC. No C, `lrintf` segue o modo de
arredondamento corrente (meio-para-par por padrão); `roundf` arredonda o meio
para longe de zero e diverge nos casos de empate exato, que são raros. Pular a
requantização dá uma diferença de até meio LSB por amostra, a avaliar nas
tolerâncias da A4.

**Testes** (`tests/test_contrato_numerico.py`): o filtro (147 coeficientes,
simétrico, ganho 1); a janela de dependência `x[4j − 73 … 4j + 73]`, mudando uma
amostra de cada vez; o recorte, com um decimador causal na convenção acima
reproduzindo dois segmentos inteiros do `resample_clip` (erro < 1e-12); o
primeiro segmento, com os zeros; e a quantização meio-para-par com saturação.

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
erro do próprio MFCC em float32 é assunto da A4. O efeito no escore é
`|pesos · Δx|` com os pesos da LDA final (`exp239`).

| Método | Erro da média | Erro do desvio | Erro relativo do desvio | Efeito no escore |
|---|---:|---:|---:|---:|
| Welford | 5,1 × 10⁻⁶ | 1,2 × 10⁻⁶ | 2,1 × 10⁻⁶ | 0,0013 |
| Soma e soma dos quadrados | 4,1 × 10⁻⁶ | 1,2 × 10⁻⁴ | 2,4 × 10⁻⁴ | 0,0056 |
| Dois passos (matriz) | 4,1 × 10⁻⁶ | 2,6 × 10⁻⁷ | 3,3 × 10⁻⁷ | 0,0012 |

Máximos sobre os 295 segmentos e os 13 coeficientes. A maior razão
|média|/desvio de um coeficiente num segmento é 28,8.

**Leitura.** Para a decisão do classificador, o método é indiferente: o efeito
no escore fica abaixo de 0,006, e o menor |escore| dos 295 segmentos é 1.297. A
escolha importa para a comparação estágio a estágio do porte (A4). A soma dos
quadrados erra o desvio cerca de 100 vezes mais que Welford, porque calcula a
variância pela diferença de dois números grandes (`q/98` e `(s/98)²`); um erro
dessa ordem esconderia uma divergência real do porte no estágio do resumo. O
problema cresce com a razão média/desvio: num caso sintético com média 1.000 e
desvio 0,1, o desvio sai com erro de cerca de 500% (teste
`test_soma_dos_quadrados_perde_precisao_com_media_grande`).

**Recomendação.** **Welford**, que mantém o cálculo quadro a quadro, sem guardar
a matriz, com erro relativo do desvio de 2 × 10⁻⁶. Se a Frente 1 aceitar os 5 KB
da matriz, os dois passos são ainda mais precisos e permitem comparar o MFCC
quadro a quadro com a referência do `reference_data.h`. Evitar a soma dos
quadrados. A emulação não reproduz a fusão de multiplicação e soma (FMA) do
Cortex-M4, que muda o último bit de cada operação, mas não a ordem de grandeza
desses erros.
