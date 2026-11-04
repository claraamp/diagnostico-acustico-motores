# Protocolo C no HUSTmotor: tabela única

Gerada pelo `run_tabela_protocolo_c.py` a partir dos `metrics.json` do `run_protocol_c.py`; 125 contagens conferidas contra o `registry.csv`. Alarme falso nos segmentos normais e detecção nos de falha, por rotação; a decisão é a do alarme (Clopper–Pearson unilateral de 95 %, `decidir()`), que só consegue reprovar.

| Etapa | Leitura | Papel | Alarme falso 5 / 10 / 20 / 30 Hz | Detecção 5 / 10 / 20 / 30 Hz | Alarme total | Detecção total | Decisão | Rodada |
|---|---|---|---|---|---:|---:|---|---|
| Sem retreino | LDA do Jung | transferência | 5/5 · 5/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 20/20 | 20/20 | reprova | `exp274` |
| C1, limiar do Jung | D2 | reporte | 5/5 · 5/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 20/20 | 20/20 | reprova | `exp275` |
| C1, limiar do Jung | D2 OU LDA | reporte | 5/5 · 5/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 20/20 | 20/20 | reprova | `exp275` |
| C1, limiar do Jung | D1, rotação nominal | reporte | — · — · 0/5 · 0/5 | — · — · 0/5 · 5/5 | 0/10 | 5/10 | inconclusivo | `exp276` |
| C1, limiar do Jung | D1, rotação estimada | reporte | — · — · 0/5 · 0/5 | — · — · 1/5 · 3/5 | 0/10 | 4/10 | inconclusivo | `exp277` |
| C1, limiar do Jung | D4 | reporte | 5/5 · 5/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 20/20 | 20/20 | reprova | `exp278` |
| C2, normal do HUST | D2 | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 5/5 · 5/5 · 5/5 | 0/8 | 15/20 | inconclusivo | `exp279` |
| C2, normal do HUST | D2 OU LDA | reporte | 2/2 · 2/2 · 2/2 · 2/2 | 5/5 · 5/5 · 5/5 · 5/5 | 8/8 | 20/20 | reprova | `exp279` |
| C2, normal do HUST | D1, rotação nominal | reporte | — · — · 2/2 · 1/2 | — · — · 5/5 · 5/5 | 3/4 | 10/10 | reprova | `exp280` |
| C2, normal do HUST | D1, rotação estimada | reporte | — · — · 1/2 · 1/2 | — · — · 5/5 · 5/5 | 2/4 | 10/10 | reprova | `exp281` |
| LDA recalibrada (C2) | só as 13 médias | decide | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | alarme inconclusivo; detecção não passa (0/15 a 10–30 Hz) | `exp287` |
| LDA recalibrada (C2) | sem o c0 | decide junto | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | alarme inconclusivo; detecção não passa (0/15 a 10–30 Hz) | `exp288` |
| LDA recalibrada (C2) | as 26 médias | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | alarme inconclusivo; detecção não passa (0/15 a 10–30 Hz) | `exp289` |
| LDA recalibrada (C2) | com o desvio | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 0/5 · 0/5 · 0/5 | 0/8 | 0/20 | alarme inconclusivo; detecção não passa (0/15 a 10–30 Hz) | `exp290` |
| LDA recalibrada (C2) | D2 OU LDA recalibrada | reporte | 0/2 · 0/2 · 0/2 · 0/2 | 0/5 · 5/5 · 5/5 · 5/5 | 0/8 | 15/20 | alarme inconclusivo; detecção passa (15/15 a 10–30 Hz) | `exp291` |
| Rotação deixada de fora | D2 | decide (adoção do D2) | 5/5 · 0/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 15/20 | 20/20 | reprova a 20 Hz | `exp296` |
| Rotação deixada de fora | D2 sem o c0 | reporte | 5/5 · 0/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 15/20 | 20/20 | reprova a 20 Hz | `exp297` |
| Rotação deixada de fora | D2 OU LDA | reporte | 5/5 · 5/5 · 5/5 · 5/5 | 5/5 · 5/5 · 5/5 · 5/5 | 20/20 | 20/20 | reprova em todas | `exp298` |
| Rotação deixada de fora | D1, rotação nominal | reporte | — · — · 3/5 · 1/5 | — · — · 5/5 · 5/5 | 4/10 | 10/10 | reprova a 20 Hz | `exp299` |
| Rotação deixada de fora | D1, rotação estimada | reporte | — · — · 1/5 · 1/5 | — · — · 5/5 · 5/5 | 2/10 | 10/10 | inconclusivo | `exp300` |

- **Sem retreino e C1:** com o limiar do Jung, a LDA, o D2 e o D4 disparam nos normais do HUST; é o limite de transferência que o critério já previa. O D1 não dispara, mas detecta pouco.
- **C2:** calibrado com a normal do HUST na mesma rotação, o D2 fica inconclusivo no alarme (o C2 só consegue reprovar) e detecta a 10–30 Hz. O combinado com a LDA do Jung herda os alarmes dela.
- **LDA recalibrada:** zera os alarmes, mas não detecta a falha; a direção da falha no HUST não é a do Jung.
- **Rotação deixada de fora:** sem a normal da própria rotação, o D2 reprova a 20 Hz, dentro da faixa calibrada. Na decisão, só ele conta (página "Adoção do D2"); o resto é reporte.

**Decisão (reunião de 03/11, página "Adoção do D2" Definida):** opção 1, o D2 não é adotado. Ele fica só como resultado em Python no relatório.

## Fora do registry

Checagem do retreino da LDA (Amanda, 29/10), **fora do registry**: está só na página "Retreinar a LDA no HUSTmotor" do Registro de Decisões, descartada. Acurácia balanceada com uma rotação deixada de fora: LDA 0,65 (1,00 / 1,00 / 0,50 / 0,10 a 5 / 10 / 20 / 30 Hz), SVM linear 0,88; o controle com rótulos trocados por arquivo chega a 0,82 no percentil 95 (só para a LDA). Dividindo dentro do arquivo, 0,94 a 1,00, que mede reconhecer a gravação.
