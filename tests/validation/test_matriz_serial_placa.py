"""
Garantias do scripts/validation/matriz_serial_placa.py, sem data/ e sem a placa.

O quadro tem o layout e o CRC do README do firmware; os checksums batem com os
exemplos de lá; a validação de uma resposta pega rx de outro segmento, estouro do
anel, rx corrompida, decisão incoerente e erro de conta. Contra a placa simulada:
a rodada limpa processa tudo; $ERR, timeout, estouro, $SEG corrompido, reinício e
resposta atrasada são reenviados como as regras mandam (3 tentativas, o reinício
não gasta tentativa, a duplicata é contada e tem de ser igual bit a bit); um modelo
diferente não deixa sair nenhum quadro; e a análise da captura salva dá o mesmo
que a rodada. Nas saídas, as matrizes ficam por conjunto e a fração da tolerância é
contra o float64.
"""

import csv
import time
import struct
import zlib

import numpy as np
import pytest

from validation import matriz_serial_placa as M

MODELO, JSON = "exp239", "ee0c3e088366"
RAPIDO = M.Tempos(folga_s=0.05, silencio_s=0.01, receber_s=0.01, max_modo_s=0.5, aviso_calada_s=0.1)

# (conjunto, id, classe, escore do Python)
PLANO = [("jung", 29, "normal", -1331.76767), ("jung", 88, "falha", 1361.83595),
         ("jung", 148, "falha", 1425.16537), ("hust", 0, "normal", 3694.2),
         ("hust", 25, "falha", 1784.5), ("hust", 39, "falha", 483.386511)]


def segmentos():
    rng = np.random.default_rng(0)
    return [M.Segmento(c, M.GRUPOS[c], i, f"{c}_{i}", cl,
                       rng.integers(-3000, 3000, M.N_AMOSTRAS).astype(np.int16), e, M.decisao_de(e))
            for c, i, cl, e in PLANO]


def rodar(tmp_path, falhas=None, escore=None, modelo=MODELO, **kw):
    segs = segmentos()
    ref = {s.chave: s.escore_py for s in segs}
    placa = M.PlacaSimulada(escore or (lambda g, i: ref[(g, i)]), modelo, JSON,
                            refs={29: ref[(1, 29)], 88: ref[(1, 88)]}, falhas=falhas,
                            ulps=lambda g, i: 1, demo_s=0.005, **kw)
    cap = M.Captura(tmp_path / "cap.log")
    for c in ("jung", "hust"):
        cap.pc(f"plano grupo={M.GRUPOS[c]} ids={M.faixas([s.id for s in segs if s.conjunto == c])}")
    msgs = []
    try:
        envio = M.Rodada(placa, cap, MODELO, JSON, 0, RAPIDO, mostrar=msgs.append).enviar(segs)
    finally:
        cap.fechar()
    a = M.analisar(M.ler_linhas_captura(tmp_path / "cap.log"), MODELO, JSON)
    return segs, placa, envio, a, M.resultado(a, segs), msgs


# --------------------------------------------------------------------------- #
def test_quadro_tem_o_layout_e_o_crc_do_readme():
    x = np.arange(M.N_AMOSTRAS, dtype=np.int16) - 6400
    q = M.montar_quadro(x, 2, 117)
    assert len(q) == M.TAM_QUADRO == 25618 and q[:4] == b"QSEG"
    assert struct.unpack("<BBHIH", q[4:14]) == (1, 1, 2, 117, 12800)
    assert struct.unpack("<I", q[14:18])[0] == zlib.crc32(q[4:14] + q[18:])     # cobre o cabeçalho
    assert np.array_equal(np.frombuffer(q[18:], "<i2"), x)                        # little-endian
    with pytest.raises(ValueError):
        M.montar_quadro(x.astype(np.int32), 2, 117)
    with pytest.raises(ValueError):
        M.montar_quadro(x[:-1], 2, 117)


def test_checksums_dos_exemplos_do_readme():
    assert M.assinar("PED,1,1") == "$PED,1,1*51"
    assert M.assinar("ERR,1,2,118,crc") == "$ERR,1,2,118,crc*0C"
    assert M.assinar("ERR,1,,,truncado") == "$ERR,1,,,truncado*60"
    assert M.ler_pacote("$PED,1,1*51") == ["PED", "1", "1"]
    assert M.ler_pacote("$PED,1,1*52") is None and M.ler_pacote("$PED,1,1") is None


def test_faixas_vao_e_voltam():
    ids = [0, 1, 2, 3, 7, 9, 10]
    assert M.faixas(ids) == "0-3,7,9-10" and M.de_faixas(M.faixas(ids)) == ids
    assert M.de_faixas(M.faixas(list(range(295)))) == list(range(295))


def _par(escore=1361.836, hexa=None, grupo=1, id_=88, rx_id=None, fim=" voltas=12 estouros=0",
         dec=None, disparou=None, txt=None):
    hexa = hexa or M.hex_de_f32(np.float32(escore))
    rx = (f"rx grupo={grupo} id={rx_id if rx_id is not None else id_} bytes=25618 ms=307 ignorados=0 "
          f"a_mais=0 erros_uart=0 ciclos=5843000 escore_hex=0x{hexa}{fim}")
    falha = escore >= 0
    corpo = (f"SEG,1,7,S,{grupo},{id_},{dec or ('F' if falha else 'N')},"
             f"{disparou or int(falha)},{txt or f'{escore:.3f}'},0,,,5843000")
    return rx, M.ler_pacote(M.assinar(corpo))


def test_validar_resposta():
    r, motivo = M.validar_resposta(*_par())
    assert motivo == "" and r.id == 88 and r.decisao == "falha" and r.escore == float(np.float32(1361.836))
    r, _ = M.validar_resposta(*_par(fim=""))                       # firmware sem voltas/estouros
    assert r is not None and r.estouros == 0 and r.voltas is None
    assert M.validar_resposta(*_par(fim=" voltas=12 estouros=1 anel=estourado"))[1] == "estouro"
    assert M.validar_resposta(*_par(rx_id=89))[1] == "rx_de_outro_segmento"
    assert M.validar_resposta(None, _par()[1])[1] == "sem_rx"
    assert M.validar_resposta(*_par(hexa=M.hex_de_f32(np.float32(1461.836))))[1] == "hex_nao_bate_com_o_seg"
    assert M.validar_resposta(*_par(escore=-0.0002, txt="0.000"))[1] == "hex_nao_bate_com_o_seg"   # sinal
    assert M.validar_resposta(*_par(dec="N"))[1] == "decisao_nao_bate"
    assert M.validar_resposta(*_par(disparou="4"))[1] == "erro_de_conta"


def test_inicio_com_outro_modelo_e_problema():
    assert M.conferir_inicio({"modelo": MODELO, "json": JSON, "quadro": "25618"}, MODELO, JSON) == []
    assert M.conferir_inicio({"modelo": MODELO, "json": "000000000000"}, MODELO, JSON)
    assert M.conferir_inicio({"modelo": MODELO, "json": JSON, "quadro": "25600"}, MODELO, JSON)


# --------------------------------------------------------------------------- #
def test_rodada_limpa(tmp_path):
    segs, placa, envio, a, res, _ = rodar(tmp_path)
    assert envio == {"ok": 6, "reinicios": 0} and a.problemas == []
    assert sum(a.envios.values()) == 6 and not a.duplicatas and not a.erros
    j, h = res["conjuntos"]["jung"], res["conjuntos"]["hust"]
    assert (j["processados"], j["concordancia"], h["processados"], h["concordancia"]) == (3, 3, 3, 3)
    assert j["m_placa"] == {("normal", "normal"): 1, ("normal", "falha"): 0,
                            ("falha", "normal"): 0, ("falha", "falha"): 2}
    # fração contra o float64: |float32 + 1 ulp - float64| / (1e-5 |float64|)
    x = next(l for l in res["linhas"] if l["id"] == 39)
    e32 = np.nextafter(np.float32(483.386511), np.float32(np.inf))
    assert x["frac_tol"] == pytest.approx(abs(float(e32) - 483.386511) / (1e-5 * 483.386511))
    assert res["clipes_iguais"] == 2 and len(res["clipes"]) == 2    # ref e serial com a mesma conta
    # a captura guarda o plano e cada envio, e a análise sai só dela
    linhas = (tmp_path / "cap.log").read_bytes().split(b"\r\n")
    assert b"#pc plano grupo=1 ids=29,88,148" in linhas and b"#pc plano grupo=2 ids=0,25,39" in linhas


def test_err_e_timeout_sao_reenviados(tmp_path):
    falhas = {(1, 29, 1): "crc", (1, 88, 1): "ignora", (2, 0, 1): "corrompe_seg"}
    _, _, envio, a, res, _ = rodar(tmp_path, falhas)
    assert all(x["estado"] == "processado" for x in res["linhas"])
    assert a.envios[(1, 29)] == 2 and a.envios[(1, 88)] == 2 and a.envios[(2, 0)] == 2
    assert a.erros["crc"] == 1 and a.falhas_pc["timeout"] == 2 and a.checksum_errado == 1
    assert res["conjuntos"]["jung"]["tentativas_extras"] == 2


def test_estouro_do_anel_rejeita_e_reenvia(tmp_path):
    _, _, _, a, res, _ = rodar(tmp_path, {(2, 25, 1): "estouro", (1, 148, 1): "hex_errado",
                                          (1, 29, 1): "erro_conta"})
    assert a.rejeitadas == {"estouro": 1, "hex_nao_bate_com_o_seg": 1, "erro_de_conta": 1}
    assert a.estouros == 1 and all(x["estado"] == "processado" for x in res["linhas"])
    assert a.envios[(2, 25)] == 2 and a.envios[(1, 148)] == 2 and a.envios[(1, 29)] == 2


def test_tres_falhas_viram_nao_processado(tmp_path):
    falhas = {(1, 88, k): "crc" for k in (1, 2, 3)}
    _, placa, envio, a, res, msgs = rodar(tmp_path, falhas)
    assert placa.tentativa[(1, 88)] == 3 and envio["nao_processados"] == 1
    j = res["conjuntos"]["jung"]
    assert j["nao_processados"] == ["jung_88"] and j["processados"] == 2
    assert (1, 88) in a.nao_processados_pc and any("NÃO PROCESSADO" in m for m in msgs)


def test_reinicio_pede_o_modo_de_novo_sem_gastar_tentativa(tmp_path):
    _, placa, envio, a, res, _ = rodar(tmp_path, {(2, 0, 1): "reinicia"})
    assert envio["reinicios"] == 1 and a.reinicios == 1 and len(a.inicios) == 2
    assert all(x["estado"] == "processado" for x in res["linhas"])
    assert placa.tentativa[(2, 0)] == 2


def test_resposta_atrasada_vira_duplicata_igual(tmp_path):
    _, _, _, a, res, _ = rodar(tmp_path, {(1, 88, 1): "atrasa"})
    assert a.problemas == [] and a.duplicatas and all(x["estado"] == "processado" for x in res["linhas"])
    x = next(l for l in res["linhas"] if l["id"] == 88)
    assert x["duplicatas"] == 1 and x["tentativas"] == 2
    assert next(l for l in res["linhas"] if l["id"] == 148)["tentativas"] == 1    # o $PED atrasado não reenvia


def test_duplicata_com_outro_escore_e_problema():
    rx1, s1 = _par(escore=1361.836)
    rx2, s2 = _par(escore=1361.837)
    linhas = ["#pc plano grupo=1 ids=88", f"recepcao inicio modelo={MODELO} json={JSON} quadro=25618",
              rx1, M.assinar(",".join(s1)), rx2, M.assinar(",".join(s2))]
    a = M.analisar(linhas, MODELO, JSON)
    assert any("duplicata" in p for p in a.problemas)


def test_modelo_diferente_nao_deixa_sair_quadro(tmp_path):
    with pytest.raises(M.Abortar, match="modelo"):
        rodar(tmp_path, modelo="exp999")


def test_placa_ja_no_modo_pede_o_reset(tmp_path):
    with pytest.raises(M.Abortar, match="não entrou no modo"):
        rodar(tmp_path, inicia_no_modo=True)


def test_rx_velha_nao_casa_com_o_seg_depois_do_ped():
    # a rx de uma tentativa cujo $SEG se perdeu não pode emprestar o escore ao $SEG seguinte
    rx, seg = _par()
    linhas = ["#pc plano grupo=1 ids=88", f"recepcao inicio modelo={MODELO} json={JSON} quadro=25618",
              rx, M.assinar("PED,1,2"), M.assinar(",".join(seg))]
    a = M.analisar(linhas, MODELO, JSON)
    assert a.rejeitadas == {"sem_rx": 1} and not a.respostas


class _PortaComLinhas:
    """Uma porta que só devolve as linhas dadas (para a rodada ver um reinício pelo marco)."""

    def __init__(self, linhas):
        self.dados = b"".join(l.encode("ascii") + b"\r\n" for l in linhas)

    def write(self, dados):
        pass

    def read(self):
        out, self.dados = self.dados, b""
        return out


def test_seg_do_marco_no_meio_da_rodada_e_reinicio(tmp_path):
    cap = M.Captura(tmp_path / "cap.log")
    porta = _PortaComLinhas(["seg seg=0 grav=normal ref=-1577.959 placa=-1577.959 dif=0.000000",
                             M.assinar("SEG,1,0,R,0,0,N,0,-1577.959,0,,,298")])
    rodada = M.Rodada(porta, cap, MODELO, JSON, 0, RAPIDO)
    assert rodada.esperar_resposta(segmentos()[0], time.monotonic() + 0.2) == "reinicio"
    cap.fechar()


def test_captura_sem_inicio_nao_passa():
    a = M.analisar(["#pc plano grupo=1 ids=88"], MODELO, JSON)
    assert any("recepcao inicio" in p for p in a.problemas)


def test_discordancia_e_fora_da_tolerancia_aparecem(tmp_path):
    ref = {(1, 29): -1331.76767, (1, 88): 1361.83595, (1, 148): 1425.16537,
           (2, 0): 3694.2, (2, 25): 1784.5, (2, 39): 483.386511}
    errado = {**ref, (1, 88): 1361.83595 * (1 + 3e-5), (2, 39): -483.386511}
    _, _, _, a, res, _ = rodar(tmp_path, escore=lambda g, i: errado[(g, i)])
    j, h = res["conjuntos"]["jung"], res["conjuntos"]["hust"]
    assert j["fora_da_tolerancia"] == ["jung_88"] and j["concordancia"] == 3
    assert h["discordantes"] == ["hust_39"] and h["concordancia"] == 2
    assert h["m_concordancia"][("falha", "normal")] == 1


def test_saidas_e_registro(tmp_path):
    segs, _, _, a, res, _ = rodar(tmp_path)
    meta = {"captura_nome": "cap.log", "captura_sha": M.sha12(tmp_path / "cap.log"), "modelo": MODELO,
            "json": JSON, "referencia_sha": "2507e68b5489", "divisao_sha": "acff87bde3", "baud": 921600,
            "commit": "abc1234", "proximo_exp": 303}
    M.escrever_csv(res, tmp_path / "segmentos.csv")
    M.escrever_md(res, a, meta, tmp_path / "resumo.md")
    M.escrever_tex(res, tmp_path / "tabelas.tex")
    with (tmp_path / "segmentos.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and {r["conjunto"] for r in rows} == {"jung", "hust"}
    assert all(len(r["escore_hex"]) == 8 for r in rows)
    md = (tmp_path / "resumo.md").read_text(encoding="utf-8")
    assert "## Jung (3/3 processados)" in md and "## HUSTmotor (3/3 processados)" in md
    tex = (tmp_path / "tabelas.tex").read_text(encoding="utf-8")
    assert "\\label{tab:serial_placa_matriz}" in tex
    assert "\\texttt{hust\\_" in tex and "\\texttt{jung\\_" in tex      # o _ escapado no nome do pior
    novas = M.linhas_registry(res, a, meta, "Amanda", "")
    assert [n["id"] for n in novas] == ["exp303", "exp304"]
    assert "conjunto=jung" in novas[0]["parametros"] and "concordancia=3/3" in novas[0]["metricas"]
    assert "divisao_hust=acff87bde3" in novas[1]["parametros"] and ";" not in novas[1]["dataset"]
