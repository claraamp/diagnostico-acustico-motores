"""
Garantias do scripts/validation/run_tempo_memoria.py, sem a placa.

A leitura da captura junta cada linha med à rx do mesmo segmento e descarta a que
teve estouro; o orçamento é lido da saída do orcamento_memoria.py e recusa o build
de medida; a pilha sai da saída do pilha_estatica.py; as contas da E/S (linha
teórica, trabalho com os bytes, ciclo por segmento) e do critério de tempo real
batem com valores feitos à mão; e as partes do 2º parcial somam o total da Tabela
tab:res_flash do relatório.
"""

import pytest

from validation import run_tempo_memoria as T

ORCAMENTO = """Orcamento de memoria (build/Debug/frente1.map)

| Parte | Flash (B) | RAM (B) |
|---|---:|---:|
| clipes a 12,8 kHz | 256.215 | 0 |
| HAL e CubeMX | 11.547 | 13 |
| aplicacao base | 2.472 | 5.316 |
| alinhamento | 36 | 11 |
| outros | 144 | 28 |
| reserva de heap e pilha (linker script: heap 512 B, pilha 4.096 B) | 0 | 4.612 |
| **total** | **450.228** | **34.992** |
"""

PILHA = """main (boot e laco): 1144 B
Pior caso (main + SysTick interrompido por USART1 ou DMA2, cada entrada com 108 B de quadro de excecao)
  main 1144 B + interrupcoes 564 B = 1708 B, contra a reserva de 4096 B (42 %); folga 2388 B
  so o modo de recepcao: 928 B + 564 B = 1492 B
"""

BOOT = [
    "hello frente1",
    "$CAB,1,exp239,ee0c3e088366,26,100*3B",
    "cadeia resumo ok=10/10 ciclos min=5854382 med=5855382 max=5857468 (med 58.55 ms, max 58.57 ms)",
    "decim ciclos total=17436612 (174.37 ms, 17.4% do tempo real) decimacao=11593270 (115.93 ms, 11.6%) "
    "fir=10209646 (102.10 ms) resto=1383624 (13.84 ms) cadeia=5843342 (58.43 ms)",
    "$SEG,1,0,D,0,0,N,0,-1331.769,0,,,17436609*69",
    "recepcao inicio modelo=exp239 json=ee0c3e088366 quadro=25618 anel=2048 baud=921600 silencio_ms=200",
    "med inicio baud_real=925925 ciclos_por_byte=1080 pilha_boot=1180 pilha_pintada=16384 pilha_reserva=4096",
]


def segmento(id_, estouros=0, consumir=7_345_552, pilha=1404):
    return [
        f"$PED,1,{id_ + 1}*00",
        f"rx grupo=1 id={id_} bytes=25618 ms=306 ignorados=0 a_mais=0 erros_uart=0 ciclos=5845552 "
        f"escore_hex=0xC4C53EA7 voltas=12 estouros={estouros}",
        f"$SEG,1,{id_},S,1,{id_},N,0,-1577.959,0,,,5845552*00",
        f"med grupo=1 id={id_} espera_us=2000 linha_us=303000 lacunas_us=26000 lacuna_max_us=1500 "
        f"consumir={consumir} consumir_max_us=900 resposta_us=2300 pilha={pilha}",
    ]


@pytest.fixture
def arquivos(tmp_path):
    cap = tmp_path / "cap.log"
    linhas = BOOT + segmento(0) + segmento(1, pilha=1412) + segmento(2, estouros=1, pilha=3000)
    linhas += ["rx erro=crc grupo=1 id=3 bytes=25618 descartados=0", "$ERR,1,1,3,crc*00"]
    cap.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    orc = tmp_path / "orcamento.md"
    orc.write_text(ORCAMENTO, encoding="utf-8")
    pil = tmp_path / "pilha.txt"
    pil.write_text(PILHA, encoding="utf-8")
    return cap, orc, pil


def test_p2_soma_o_total_do_relatorio():
    assert tuple(sum(v[i] for v in T.P2.values()) for i in (0, 1)) == T.P2_TOTAL


def test_orcamento_agrupa_e_recusa_o_build_de_medida(tmp_path, arquivos):
    _, orc, _ = arquivos
    partes = T.ler_orcamento(orc)
    assert partes["total"] == (450228, 34992)
    assert partes["outros e alinhamento"] == (180, 39)
    assert partes["reserva de heap e pilha"] == (0, 4612)
    ruim = tmp_path / "ruim.md"
    ruim.write_text(ORCAMENTO.replace("| clipes", "| build de medida (orcamento.c; nao vai para o "
                                                 "firmware final) | 516 | 8 |\n| clipes"), encoding="utf-8")
    with pytest.raises(SystemExit):
        T.ler_orcamento(ruim)


def test_captura_pareia_med_com_rx_e_descarta_estouro(arquivos):
    cap, _, _ = arquivos
    c = T.ler_captura(cap)
    assert [p["rx"]["id"] for p in c["pares"]] == ["0", "1"]
    assert c["estouros"] == 1 and c["erros"] == 1
    assert c["decim"] == {"total": 17436612, "decimacao": 11593270, "cadeia": 5843342}
    assert c["inicio"]["ciclos_por_byte"] == 1080


def test_captura_sem_med_e_recusada(tmp_path):
    cap = tmp_path / "padrao.log"
    cap.write_text("\n".join(BOOT[:-1] + segmento(0)[:3]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        T.ler_captura(cap)


def test_contas_dos_dois_modos(arquivos):
    cap, orc, pil = arquivos
    r = T.calcular(T.ler_captura(cap), T.ler_orcamento(orc), T.ler_pilha(pil))
    a, b = r["a"], r["b"]
    assert a["proc_ms"] == pytest.approx(174.36612)
    assert a["ciclos_por_amostra"] == pytest.approx(17436612 / 51200)
    assert a["orcamento_por_amostra"] == pytest.approx(100e6 / 51200)
    assert a["es_us"] == pytest.approx(1e6 * (len(BOOT[4]) + 2) * 10 / 925925)
    assert b["linha_teorica_ms"] == pytest.approx(1000 * 25618 * 10 / 925925)
    e = b["est"]
    assert e["bytes_cpu"]["med"] == pytest.approx(15.0)            # (7.345.552 - 5.845.552) / 100 MHz
    assert e["ciclo"]["med"] == pytest.approx(307.3)               # 2,0 + 303,0 + 2,3 ms
    assert b["cpu_por_byte"] == pytest.approx(7345552 / 25618)
    assert b["anel_ms"] == pytest.approx(1000 * 2048 * 10 / 925925)
    assert r["pilha"]["medida"] == 1412                             # o segmento com estouro não conta
    assert r["pilha"]["pior"] == 1708


def test_saidas_e_registry(tmp_path, arquivos):
    cap, orc, pil = arquivos
    r = T.calcular(T.ler_captura(cap), T.ler_orcamento(orc), T.ler_pilha(pil))
    meta = {"captura": "cap.log", "captura_sha": "x", "firmware_commit": "58da414", "data": "2026-11-16"}
    md = T.resumo_md(r, meta)
    assert "Critério de tempo real cumprido" in md and "Critério cumprido" in md
    tex = T.tabela_memoria_tex(r["mem"]) + T.tabela_tempo_tex(r)
    assert "tab:final_memoria" in tex and "tab:final_tempo" in tex and "450.228" in tex
    linha = T.linha_registry(r, meta, 320, "Clara", "")
    assert linha["id"] == "exp320" and linha["etapa"] == "caracterizacao_embarcada"
    assert "pilha_estatica_b=1708" in linha["metricas"] and "b_segmentos=2" in linha["metricas"]
