# Protocolo C no HUSTmotor: tabela única

Gerada pelo `run_tabela_protocolo_c.py` a partir dos `metrics.json` do `run_protocol_c.py`; 125 contagens conferidas contra o `registry.csv`. Alarme falso nos segmentos normais e detecção nos de falha, por rotação; a decisão é a do alarme (Clopper–Pearson unilateral de 95 %, `decidir()`), que só consegue reprovar.

| Etapa | Leitura | Papel | Alarme falso 5 / 10 / 20 / 30 Hz | Detecção 5 / 10 / 20 / 30 Hz | Alarme total | Detecção total | Decisão | Rodada |
|---|---|---|---|---|---:|---:|---|---|
| Sem retreino | LDA do Jung | transferência | 5/5 · 5/5 · 5/5 · 5/5 | (5/5) · (5/5) · (5/5) · (5/5) | 20/20 | — | reprova | `exp274` |
| C1, limiar do Jung | D2 | reporte | 5/5 · 5/5 · 5/5 · 5/5 | (5/5) · (5/5) · (5/5) · (5/5) | 20/20 | — | reprova | `exp275` |
| C1, limiar do Jung | D2 OU LDA | reporte | 5/5 · 5/5 · 5/5 · 5/5 | (5/5) · (5/5) · (5/5) · (5/5) | 20/20 | — | reprova | `exp275` |
| C1, limiar do Jung | D1, rotação nominal | reporte | — · — · 0/5 · 0/5 | — · — · 0/5 · 5/5 | 0/10 | 5/10 | inconclusivo | `exp276` |
| C1, limiar do Jung | D1, rotação estimada | reporte | — · — · 0/5 · 0/5 | — · — · 1/5 · 3/5 | 0/10 | 4/10 | inconclusivo | `exp277` |
| C1, limiar do Jung | D4 | reporte | 5/5 · 5/5 · 5/5 · 5/5 | (5/5) · (5/5) · (5/5) · (5/5) | 20/20 | — | reprova | `exp278` |
| C2, normal do HUST | D2 | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 5/5 · 5/5 · 5/5 | 0/8 | 15/20 | inconclusivo | `exp279` |
| C2, normal do HUST | D2 OU LDA | reporte | 2/2 · 2/2 · 2/2 · 2/2 | (5/5) · (5/5) · (5/5) · (5/5) | 8/8 | — | reprova | `exp279` |
| C2, normal do HUST | D1, rotação nominal | reporte | — · — · 2/2 · 1/2 | — · — · (5/5) · (5/5) | 3/4 | — | reprova | `exp280` |
| C2, normal do HUST | D1, rotação estimada | reporte | — · — · 1/2 · 1/2 | — · — · (5/5) · (5/5) | 2/4 | — | reprova | `exp281` |
| LDA recalibrada (C2) | só as 13 médias | decide | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | não funciona, com o controle sem o c0 (0/15 a 10–30 Hz; 5 Hz à parte: 0/5) | `exp287` |
| LDA recalibrada (C2) | sem o c0 | decide junto | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | 0/15 a 10–30 Hz; 5 Hz à parte: 0/5 | `exp288` |
| LDA recalibrada (C2) | as 26 médias | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | 0/15 a 10–30 Hz; 5 Hz à parte: 0/5 | `exp289` |
| LDA recalibrada (C2) | com o desvio | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | 0/15 a 10–30 Hz; 5 Hz à parte: 0/5 | `exp290` |
| LDA recalibrada (C2) | D2 OU LDA recalibrada | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 5/5 · 5/5 · 5/5 | 0/8 | 15/20 | 15/15 a 10–30 Hz; 5 Hz à parte: 0/5 | `exp291` |
| Rotação deixada de fora | D2 | decide (adoção do D2) | [5/5] · 0/5 · 5/5 · [5/5] | (5/5) · 5/5 · (5/5) · (5/5) | 15/20 | 5/5 | reprova a 20 Hz | `exp296` |
| Rotação deixada de fora | D2 sem o c0 | reporte | [5/5] · 0/5 · 5/5 · [5/5] | (5/5) · 5/5 · (5/5) · (5/5) | 15/20 | 5/5 | reprova a 20 Hz | `exp297` |
| Rotação deixada de fora | D2 OU LDA | reporte | 5/5 · 5/5 · 5/5 · 5/5 | (5/5) · (5/5) · (5/5) · (5/5) | 20/20 | — | reprova em todas | `exp298` |
| Rotação deixada de fora | D1, rotação nominal | reporte | — · — · 3/5 · 1/5 | — · — · (5/5) · 5/5 | 4/10 | 5/5 | reprova a 20 Hz | `exp299` |
| Rotação deixada de fora | D1, rotação estimada | reporte | — · — · 1/5 · 1/5 | — · — · 5/5 · 5/5 | 2/10 | 10/10 | inconclusivo | `exp300` |

Entre colchetes, alarmes em rotação fora da faixa calibrada (extrapolação): ficam fora da decisão e são reportados à parte; no D2 OU LDA não há colchetes, porque a LDA do Jung alarma em qualquer rotação, sem depender da calibração. Entre parênteses, detecção onde o detector alarma os normais (na rotação deixada de fora, rotação a rotação; nas outras etapas, pela decisão do total): não é lida como acerto. No total, a detecção entre parênteses fica de fora; traço quando não sobra nenhuma.

- **Sem retreino e C1:** com o limiar do Jung, a LDA, o D2 e o D4 disparam nos normais do HUST; é o limite de transferência que o critério já previa. O D1 não dispara, mas detecta pouco.
- **C2:** calibrado com a normal do HUST na mesma rotação, o D2 fica inconclusivo no alarme (o C2 só consegue reprovar) e detecta a 10–30 Hz. O combinado com a LDA do Jung herda os alarmes dela.
- **LDA recalibrada:** zera os alarmes, mas não detecta a falha, nem sem o c0: a diferença entre as máquinas não é só de nível e microfone. A detecção do combinado com o D2 vem toda do D2.
- **Rotação deixada de fora:** sem a normal da própria rotação, o D2 reprova a 20 Hz, dentro da faixa calibrada. Na decisão, só ele conta (página "Adoção do D2"); o resto é reporte. A 5 e a 30 Hz, os alarmes são extrapolação e não decidem; o achado é que o D2 não generaliza para fora da faixa calibrada.

**Decisão (reunião de 03/11, página "Adoção do D2" Definida):** opção 1, o D2 não é adotado. Ele fica só como resultado em Python no relatório.

## Fora do registry

Checagem do retreino da LDA (Amanda, 29/10), **fora do registry**: está só no Registro de Decisões, na página "Retreinar a LDA no HUSTmotor" (descartada) e na decisão 4 da página "Recalibrar a LDA pela normal da máquina nova". Rodou antes de o protocolo ser aprovado e serviu só para descartar o retreino. Acurácia balanceada com uma rotação deixada de fora: LDA 0,65 (1,00 / 1,00 / 0,50 / 0,10 a 5 / 10 / 20 / 30 Hz), SVM linear 0,88. No controle com rótulos trocados por arquivo, a LDA tem média 0,48 e percentil 95 de 0,82; nas 14 trocas possíveis, 4 igualam ou superam a LDA (valor-p ≈ 0,33) e 2 igualam o SVM (≈ 0,2). As trocas formam 7 pares com o mesmo valor (inverter todos os rótulos não muda a acurácia balanceada), e o valor-p nunca cai abaixo de 1/15 ≈ 0,07: o controle não teria como mostrar significância a 5 %. Dividindo dentro do arquivo, 0,94 a 1,00, que mede reconhecer a gravação.
