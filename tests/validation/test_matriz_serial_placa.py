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
contra o float64. Com --lista: a lista é conferida (sha256 do .bin, modelo,
escores do CSV, grupos e ids) e o critério é |Δs| < E, com a fração relativa só como
informação.
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
    cap.pc(f"rodada inicio data=2026-11-05 porta=simulada baud=921600 modelo={MODELO} json={JSON} commit=abc1234")
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
    assert res["divergencias_pc"] == []                                # o PC e a captura concordam


def test_reinicio_pede_o_modo_de_novo_sem_gastar_tentativa(tmp_path):
    _, placa, envio, a, res, _ = rodar(tmp_path, {(2, 0, 1): "reinicia"})
    assert envio["reinicios"] == 1 and a.reinicios == 1 and len(a.inicios) == 2
    assert all(x["estado"] == "processado" for x in res["linhas"])
    assert placa.tentativa[(2, 0)] == 2 and a.envios[(2, 0)] == 2     # o quadro foi mandado duas vezes
    assert next(l for l in res["linhas"] if l["id"] == 0 and l["grupo"] == 2)["tentativas"] == 1
    assert res["conjuntos"]["hust"]["tentativas_extras"] == 0         # mas o reinício não gasta tentativa


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
            "commit": "abc1234", "data": "2026-11-05", "proximo_exp": 303}
    M.escrever_csv(res, tmp_path / "segmentos.csv")
    M.escrever_md(res, a, meta, tmp_path / "resumo.md")
    M.escrever_tex(res, tmp_path / "tabelas.tex")
    with (tmp_path / "segmentos.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and {r["conjunto"] for r in rows} == {"jung", "hust"}
    assert all(len(r["escore_hex"]) == 8 for r in rows)
    md = (tmp_path / "resumo.md").read_text(encoding="utf-8")
    assert "## Jung (3/3 processados)" in md and "## HUSTmotor (3/3 processados)" in md
    assert md.count("- Concordância placa × Python: **3/3**.") == 2 and "Critério" not in md
    assert_tabelas_md(md)
    tex = (tmp_path / "tabelas.tex").read_text(encoding="utf-8")
    assert "\\label{tab:serial_placa_matriz}" in tex
    assert "\\texttt{hust\\_" in tex and "\\texttt{jung\\_" in tex      # o _ escapado no nome do pior
    novas = M.linhas_registry(res, a, meta, "Amanda", "")
    assert [n["id"] for n in novas] == ["exp303", "exp304"]
    assert "conjunto=jung" in novas[0]["parametros"] and "concordancia=3/3" in novas[0]["metricas"]
    assert "divisao_hust=acff87bde3" in novas[1]["parametros"] and ";" not in novas[1]["dataset"]
    assert novas[0]["notas"] == M.NOTAS_PADRAO["jung"]                       # sem --notas: a padrão
    com = M.linhas_registry(res, a, meta, "Amanda", "rodada de teste")
    assert [n["notas"] for n in com] == [M.NOTAS_PADRAO[c] + "; rodada de teste" for c in ("jung", "hust")]


def _meta(tmp_path, a):
    return {"captura_nome": "cap.log", "captura_sha": M.sha12(tmp_path / "cap.log"), "modelo": MODELO,
            "json": JSON, "referencia_sha": "2507e68b5489", "divisao_sha": "acff87bde3", "baud": 921600,
            "commit": a.cabecalho["commit"], "data": a.cabecalho["data"]}


def test_registro_usa_o_commit_da_rodada_e_nao_registra_a_mesma_captura_duas_vezes(tmp_path):
    _, _, _, a, res, _ = rodar(tmp_path)
    reg = tmp_path / "registry.csv"
    novas = M.registrar_rodada(reg, res, a, _meta(tmp_path, a), "Amanda", "")
    assert [n["git_commit"] for n in novas] == ["abc1234", "abc1234"]           # o do cabeçalho
    assert {n["data"] for n in novas} == {"2026-11-05"}
    with pytest.raises(M.Abortar, match="já está no registry"):               # refazer com --captura
        M.registrar_rodada(reg, res, a, _meta(tmp_path, a), "Amanda", "")
    with reg.open(encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 2


def test_captura_interrompida_nao_e_registrada(tmp_path):
    _, _, _, a, res, _ = rodar(tmp_path)
    linhas = [l for l in M.ler_linhas_captura(tmp_path / "cap.log")]
    corte = max(i for i, l in enumerate(linhas) if l.startswith("#pc envio grupo=2 id=25"))
    a2 = M.analisar(linhas[:corte], MODELO, JSON)                 # parou antes dos dois últimos
    res2 = M.resultado(a2, segmentos())
    assert len(res2["divergencias_pc"]) == 2 and "captura interrompida" in res2["divergencias_pc"][0]
    with pytest.raises(M.Abortar, match="incompleta"):
        M.registrar_rodada(tmp_path / "registry.csv", res2, a2, _meta(tmp_path, a2), "Amanda", "")


# --------------------------------------------------------------------------- #
# --lista (recorte da robustez ao ruído)                                      #
# --------------------------------------------------------------------------- #
# (ruído, snr, semente, segmento); o id é o do run_varredura_snr.id_placa
LISTA = [("branco", 10, 20260925, 0), ("branco", 0, 20260925, 48), ("branco", 0, 5, 57),
         ("rosa", 5, 20260925, 3), ("rosa", 5, 9, 23)]


def _id(snr, seg, semente):
    return snr * 1000 + seg + (0 if semente == 20260925 else semente * 100000)


def criar_lista(pasta, pesos, bias, modelo=MODELO, json_=JSON, mexer=None):
    """Uma lista pequena, no formato do run_varredura_snr.exportar_placa."""
    rng = np.random.default_rng(1)
    pcm = rng.integers(-4000, 4000, (len(LISTA), M.N_AMOSTRAS)).astype("<i2")
    linhas = []
    for k, (ruido, snr, semente, seg) in enumerate(LISTA):
        e = M.escore_python(pcm[k].astype(np.int16), pesos, bias)
        linhas.append({"ordem": k, "grupo": {"branco": 3, "rosa": 4}[ruido], "id": _id(snr, seg, semente),
                       "nome": f"normal_seg{seg:04d}_{ruido}_{snr}db", "classe": "normal", "ruido": ruido,
                       "snr_db": snr, "semente": semente, "segmento": seg, "escore": f"{e:.9f}",
                       "decisao": M.decisao_de(e), "E": "0.410000", "margem_E": f"{abs(e) / 0.41:.1f}"})
    if mexer:
        mexer(linhas, pcm)
    pasta.mkdir(parents=True, exist_ok=True)
    corpo = pcm.tobytes()
    (pasta / "segmentos.bin").write_bytes(corpo)
    with (pasta / "referencias.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    import hashlib
    import json
    (pasta / "manifest.json").write_text(json.dumps({
        "bin": (pasta / "segmentos.bin").as_posix(), "sha256": hashlib.sha256(corpo).hexdigest(),
        "n_segmentos": len(linhas), "amostras_por_segmento": 12800, "dtype": "int16 little-endian",
        "fs_hz": 12800, "modelo": modelo, "json_sha12": json_, "grupos": {"branco": 3, "rosa": 4}}),
        encoding="utf-8")
    return pasta


PESOS = np.random.default_rng(2).normal(0, 50, 26)


def test_lista_carrega_e_confere(tmp_path):
    p = criar_lista(tmp_path / "l", PESOS, 10.0)
    lista = M.carregar_lista(p, MODELO, JSON, PESOS, 10.0)
    assert [s.chave for s in lista.segmentos] == [(3, 10000), (3, 48), (3, 500057), (4, 5003), (4, 905023)]
    assert [s.conjunto for s in lista.segmentos] == ["branco"] * 3 + ["rosa"] * 2
    assert all(s.E == 0.41 and s.pcm.dtype == np.int16 for s in lista.segmentos)
    assert lista.segmentos[2].extra == {"ruido": "branco", "snr_db": 0, "semente": 5, "id_jung": 57}
    assert M.montar_quadro(lista.segmentos[-1].pcm, 4, 905023)[4:14] == struct.pack("<BBHIH", 1, 1, 4,
                                                                                     905023, 12800)


@pytest.mark.parametrize("mexer, msg", [
    (lambda l, pcm: pcm.__setitem__((1, 7), pcm[1, 7] + 1), "sha256"),            # .bin de outra lista
    (lambda l, pcm: l[1].__setitem__("escore", "123.0"), "escore"),               # CSV de outro .bin
    (lambda l, pcm: l[4].update(grupo=3, id=_id(10, 0, 20260925), ruido="branco"), "repetido"),
    (lambda l, pcm: l[3].__setitem__("grupo", 1), "grupo 1"),                      # grupo do Jung
    (lambda l, pcm: l[0].__setitem__("E", "0"), "E = 0"),
])
def test_lista_recusa_o_que_nao_bate(tmp_path, mexer, msg):
    p = criar_lista(tmp_path / "l", PESOS, 10.0)
    if msg == "sha256":                         # o .bin muda depois do manifesto
        b = bytearray((p / "segmentos.bin").read_bytes())
        b[100] ^= 1
        (p / "segmentos.bin").write_bytes(bytes(b))
    else:
        p = criar_lista(tmp_path / "m", PESOS, 10.0, mexer=mexer)
    with pytest.raises(M.Abortar, match=msg):
        M.carregar_lista(p, MODELO, JSON, PESOS, 10.0)


def test_lista_de_outro_modelo_e_recusada(tmp_path):
    p = criar_lista(tmp_path / "l", PESOS, 10.0, modelo="exp238")
    with pytest.raises(M.Abortar, match="modelo exp238"):
        M.carregar_lista(p, MODELO, JSON, PESOS, 10.0)


def _rodar_lista(tmp_path, escore_placa=None):
    segs = M.carregar_lista(criar_lista(tmp_path / "l", PESOS, 10.0), MODELO, JSON, PESOS, 10.0).segmentos
    ref = {s.chave: s.escore_py for s in segs}
    placa = M.PlacaSimulada(escore_placa or (lambda g, i: ref[(g, i)]), MODELO, JSON, demo_s=0.005)
    cap = M.Captura(tmp_path / "cap.log")
    cap.pc(f"rodada inicio data=2026-11-14 porta=simulada baud=921600 modelo={MODELO} json={JSON} "
           "commit=abc1234 lista=0123456789ab")
    for g in (3, 4):
        cap.pc(f"plano grupo={g} ids={M.faixas([s.id for s in segs if s.grupo == g])}")
    try:
        M.Rodada(placa, cap, MODELO, JSON, 0, RAPIDO, mostrar=lambda m: None).enviar(segs)
    finally:
        cap.fechar()
    a = M.analisar(M.ler_linhas_captura(tmp_path / "cap.log"), MODELO, JSON)
    return segs, a, M.resultado(a, segs)


def test_lista_criterio_e_E_e_nao_a_tolerancia_relativa(tmp_path):
    segs, a, res = _rodar_lista(tmp_path)
    assert list(res["conjuntos"]) == ["branco", "rosa"] and a.cabecalho["lista"] == "0123456789ab"
    b = res["conjuntos"]["branco"]
    assert b["aprovado"] and b["motivos"] == [] and [q["snr_db"] for q in b["por_snr"]] == [10, 0]
    assert b["por_snr"][1]["n"] == 2 and b["por_snr"][1]["sementes"] == 2
    assert all(0 <= x["dif_E"] < 1 for x in res["linhas"])
    # perto do limiar (|s| = 0,5), 0,001 de erro passa da tolerância relativa e fica bem abaixo de E:
    # aprova; 0,6 de erro passa de E (0,41), mesmo com a decisão igual: reprova
    ref = {s.chave: s.escore_py for s in segs}
    python = {**ref, (3, 48): 0.5}
    placa = {**python, (3, 48): 0.501, (4, 5003): ref[(4, 5003)] + 0.6}
    segs2 = [M.Segmento(s.conjunto, s.grupo, s.id, s.nome, s.classe, s.pcm, python[s.chave],
                        M.decisao_de(python[s.chave]), s.E, s.extra) for s in segs]
    _, a2, _ = _rodar_lista(tmp_path / "b", escore_placa=lambda g, i: placa[(g, i)])
    res2 = M.resultado(a2, segs2)
    b2, r2 = res2["conjuntos"]["branco"], res2["conjuntos"]["rosa"]
    assert b2["aprovado"] and b2["fora_da_tolerancia"] == ["normal_seg0048_branco_0db"]   # só informação
    assert b2["pior_E"]["segmento"] == "normal_seg0048_branco_0db" and b2["pior_E"]["dif_E"] < 0.01
    assert not r2["aprovado"] and r2["fora_de_E"] == ["normal_seg0003_rosa_5db"]
    assert r2["motivos"] == ["1 com |Δs| ≥ E"] and r2["concordancia"] == 2               # mesma decisão


def assert_tabelas_md(md: str) -> None:
    """Cada tabela do Markdown tem o mesmo número de colunas em todas as linhas (| escapado não conta)."""
    import re
    blocos, atual = [], []
    for l in md.splitlines() + [""]:
        if l.startswith("|"):
            atual.append(l)
        elif atual:
            blocos.append(atual)
            atual = []
    assert blocos
    for b in blocos:
        colunas = {len(re.split(r"(?<!\\)\|", l)) for l in b}
        assert len(colunas) == 1, b[:2]


def test_lista_saidas_e_registro(tmp_path):
    _, a, res = _rodar_lista(tmp_path)
    meta = {"captura_nome": "cap.log", "captura_sha": M.sha12(tmp_path / "cap.log"), "modelo": MODELO,
            "json": JSON, "referencia_sha": "x", "divisao_sha": "y", "baud": 921600, "commit": "abc1234",
            "data": "2026-11-14", "proximo_exp": 320,
            "lista": {"manifesto_sha": "0123456789ab", "referencias_sha": "ba9876543210", "bin_sha256": "f" * 64}}
    M.escrever_csv(res, tmp_path / "segmentos.csv")
    M.escrever_md(res, a, meta, tmp_path / "resumo.md")
    M.escrever_tex(res, tmp_path / "tabelas.tex")
    with (tmp_path / "segmentos.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {"ruido", "snr_db", "semente", "id_jung", "E", "dif_E"} <= set(rows[0]) and len(rows) == 5
    assert rows[2]["segmento"] == "normal_seg0057_branco_0db" and rows[2]["id_jung"] == "57"
    md = (tmp_path / "resumo.md").read_text(encoding="utf-8")
    assert "**Critério da rodada: aprovado**" in md and "| 0 dB | 2 | 2 |" in md and "Clipes" not in md
    assert_tabelas_md(md)
    tex = (tmp_path / "tabelas.tex").read_text(encoding="utf-8")
    assert "\\label{tab:serial_placa_ruido}" in tex and "branco & 10~dB & 1 & 1/1" in tex
    novas = M.linhas_registry(res, a, meta, "Amanda", "")
    assert [n["id"] for n in novas] == ["exp320", "exp321"]
    assert "lista=0123456789ab" in novas[0]["parametros"] and "bin_sha256=ffffffffffff" in novas[0]["parametros"]
    assert "aprovado=True" in novas[0]["metricas"] and "fora_de_E=0" in novas[0]["metricas"]
    assert novas[1]["dataset"] == "jung_ruido_rosa_2seg_12800Hz" and novas[1]["notas"] == M.NOTA_LISTA


def test_jung_e_hust_nao_mudam_com_a_lista(tmp_path):
    """O CSV e o resumo do Jung e do HUST continuam com as mesmas colunas e linhas de antes."""
    _, _, _, a, res, _ = rodar(tmp_path)
    M.escrever_csv(res, tmp_path / "segmentos.csv")
    assert (tmp_path / "segmentos.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "conjunto,grupo,id,segmento,classe,estado,escore_python,escore_placa,escore_hex,dif,tol,frac_tol,"
        "decisao_python,decisao_placa,tentativas,duplicatas,n,ms,ciclos")
    assert all("aprovado" not in r for r in res["conjuntos"].values())


def test_main_simulada_com_lista_e_captura_de_outra_lista(tmp_path, monkeypatch, capsys):
    raiz = __import__("pathlib").Path(M.__file__).resolve().parents[2]
    monkeypatch.chdir(raiz)
    modelo_id, json12, pesos, bias = M.info_modelo(M.MODELO)
    p = criar_lista(tmp_path / "l", pesos, bias, modelo=modelo_id, json_=json12)
    out = tmp_path / "out"
    assert M.main(["--simulada", "--lista", str(p), "--out-dir", str(out)]) == 0
    texto = capsys.readouterr().out
    assert "critério (decisões iguais e |Δs| < E): APROVADO" in texto and "registry não alterado" in texto
    cap = next(out.glob("serial-*.log"))
    assert M.main(["--captura", str(cap), "--lista", str(p), "--out-dir", str(out), "--sem-registro"]) == 0
    p2 = criar_lista(tmp_path / "l2", pesos, bias, modelo=modelo_id, json_=json12,      # outro E: outra lista
                     mexer=lambda l, pcm: l[0].__setitem__("E", "0.500000"))
    assert M.main(["--captura", str(cap), "--lista", str(p2), "--out-dir", str(out), "--sem-registro"]) == 1
    assert "a captura é da lista" in capsys.readouterr().out
    # sem o .bin (em outra máquina): a análise sai igual, com o escore do CSV, e não registra
    (p / "segmentos.bin").rename(tmp_path / "guardado.bin")
    out2 = tmp_path / "out2"
    assert M.main(["--captura", str(cap), "--lista", str(p), "--out-dir", str(out2), "--sem-registro"]) == 0
    for f in ("segmentos.csv", "tabelas.tex"):
        assert (out2 / f).read_bytes() == (out / f).read_bytes()
    assert "sem o .bin" in (out2 / "resumo.md").read_text(encoding="utf-8")
    assert M.main(["--captura", str(cap), "--lista", str(p), "--out-dir", str(out2), "--responsavel", "A",
                   "--registry", str(tmp_path / "reg.csv")]) == 1
    assert "não é registrada" in capsys.readouterr().out and not (tmp_path / "reg.csv").exists()
    with pytest.raises(M.Abortar, match="não achei"):                       # para mandar, o .bin é obrigatório
        M.carregar_lista(p, modelo_id, json12, pesos, bias)
    with pytest.raises(SystemExit):                                          # a lista substitui os conjuntos
        M.main(["--captura", str(cap), "--lista", str(p), "--conjuntos", "jung"])


def test_lista_com_ruido_que_o_script_nao_conhece_e_recusada(tmp_path):
    import json
    p = criar_lista(tmp_path / "l", PESOS, 10.0)
    man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    man["grupos"] = {"branco": 3, "rosa": 4, "marrom": 5}         # sem nome nem linha de registry
    (p / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(M.Abortar, match="grupos do manifesto inválidos"):
        M.carregar_lista(p, MODELO, JSON, PESOS, 10.0)


# --------------------------------------------------------------------------- #
# --ids e --pausa: o teste dos LEDs (README do firmware, "Sinalização por LED")
def test_ler_ids_na_ordem_e_recusas():
    assert M.ler_ids("3:48, 3:3,4:5047") == [(3, 48), (3, 3), (4, 5047)]
    for ruim in ("", "3", "3:x", "3-48", "3:48,3:48"):
        with pytest.raises(ValueError):
            M.ler_ids(ruim)


def test_escolher_mantem_a_ordem_dada_e_recusa_o_que_falta():
    segs = segmentos()
    assert [s.chave for s in M.escolher(segs, [(2, 39), (1, 29)])] == [(2, 39), (1, 29)]
    with pytest.raises(M.Abortar, match="1:30"):
        M.escolher(segs, [(1, 29), (1, 30)])


def test_pausa_depois_de_cada_resposta_certa_com_a_cor_do_led(tmp_path):
    segs = segmentos()
    ref = {s.chave: s.escore_py for s in segs}
    # o 88 dá $ERR na primeira: só a resposta certa pausa
    placa = M.PlacaSimulada(lambda g, i: ref[(g, i)], MODELO, JSON, refs={29: ref[(1, 29)]},
                            falhas={(1, 88, 1): "crc"}, demo_s=0.005)
    cap = M.Captura(tmp_path / "cap.log")
    msgs, pausas = [], []
    try:
        envio = M.Rodada(placa, cap, MODELO, JSON, 0, RAPIDO, mostrar=msgs.append, pausa_s=3.0,
                         dormir=pausas.append).enviar(segs)
    finally:
        cap.fechar()
    assert envio["ok"] == 6 and envio["err"] == 1 and pausas == [3.0] * 6
    leds = [m for m in msgs if "-> LED" in m]
    assert len(leds) == 6
    assert leds[0].endswith("jung_29: Python normal, placa normal -> LED verde (3 s)")
    assert leds[1].endswith("Python falha, placa falha -> LED vermelho (3 s)")
    assert not any("DIVERGE" in m for m in leds)


def test_pausa_mostra_a_divergencia_na_hora_com_a_cor_da_placa(tmp_path):
    segs = segmentos()
    ref = {s.chave: s.escore_py for s in segs}
    # a placa dá normal no 88 (falha no Python): o LED segue a placa
    placa = M.PlacaSimulada(lambda g, i: -1.0 if (g, i) == (1, 88) else ref[(g, i)], MODELO, JSON,
                            refs={29: ref[(1, 29)]}, demo_s=0.005)
    cap = M.Captura(tmp_path / "cap.log")
    msgs = []
    try:
        M.Rodada(placa, cap, MODELO, JSON, 0, RAPIDO, mostrar=msgs.append, pausa_s=1.0,
                 dormir=lambda s: None).enviar(segs)
    finally:
        cap.fechar()
    leds = [m for m in msgs if "-> LED" in m]
    assert leds[1].endswith("jung_88: Python falha, placa normal -> LED verde (1 s)  <- DIVERGE do Python")
    assert sum("DIVERGE" in m for m in leds) == 1


def test_sem_pausa_nao_dorme(tmp_path):
    *_, msgs = rodar(tmp_path)
    assert not any("-> LED" in m for m in msgs)


def test_main_ids_e_pausa_sao_ensaio(tmp_path, monkeypatch, capsys):
    raiz = __import__("pathlib").Path(M.__file__).resolve().parents[2]
    monkeypatch.chdir(raiz)
    modelo_id, json12, pesos, bias = M.info_modelo(M.MODELO)
    p = criar_lista(tmp_path / "l", pesos, bias, modelo=modelo_id, json_=json12)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    out = tmp_path / "out"
    ids = f"4:{_id(5, 3, 20260925)},3:{_id(0, 48, 20260925)}"          # rosa primeiro: a ordem dada
    assert M.main(["--simulada", "--lista", str(p), "--ids", ids, "--pausa", "2", "--out-dir", str(out)]) == 0
    texto = capsys.readouterr().out
    assert "2 segmentos" in texto and "registry não alterado" in texto
    assert texto.index("normal_seg0003_rosa_5db") < texto.index("normal_seg0048_branco_0db")
    cap = next(out.glob("serial-*.log")).read_text(encoding="utf-8")
    assert "#pc plano grupo=3 ids=48" in cap and "#pc plano grupo=4 ids=5003" in cap
    with pytest.raises(SystemExit):
        M.main(["--simulada", "--lista", str(p), "--ids", "3:x"])
    assert M.main(["--simulada", "--lista", str(p), "--ids", "3:7", "--out-dir", str(out)]) == 1
    assert "3:7 não estão" in capsys.readouterr().out
    for ruim in (["--ids", ids, "--limite", "1"], ["--pausa", "-1"]):
        with pytest.raises(SystemExit):
            M.main(["--simulada", "--lista", str(p), *ruim])
    with pytest.raises(SystemExit):                                     # a captura não tem envio
        M.main(["--captura", str(out / "x.log"), "--lista", str(p), "--pausa", "2"])
