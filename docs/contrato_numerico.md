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

Escopo: a cadeia a partir do PCM decimado a 12,8 kHz (`int16`). A decimação
(filtro, atraso de grupo, margens, quantização para `int16`) fica na seção
"Decimação", a preencher na tarefa A2. Média e desvio quadro a quadro em float32
(Welford ou soma e soma dos quadrados) são a tarefa A3, e as tolerâncias de
comparação por estágio, a A4.

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
| 12 | Resumo | Média e desvio-padrão de cada coeficiente ao longo dos 98 quadros, desvio **populacional** (`ddof=0`, o padrão do `np.std`) | `04_extract_features.py:105` | Desvio amostral (`ddof=1`, divide por 97). |
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

A preencher na tarefa A2: filtro FIR de 147 coeficientes, compensação do atraso
de grupo (73 amostras a 51,2 kHz), margens de cada clipe e quantização para
`int16` antes da extração.
