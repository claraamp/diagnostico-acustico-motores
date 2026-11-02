"""Garantias do scripts/validation/matriz_clipes_placa.py, com uma captura sintética."""

from validation import matriz_clipes_placa as mc

# (id, clipe, k, escore do CSV, decisão)
SEGS = [(29, "normal", 0, -1331.76767, "normal"), (30, "normal", 1, -1392.13252, "normal"),
        (88, "bpfi_0.3mm", 0, 1361.83595, "falha"), (89, "bpfi_0.3mm", 1, 1477.39064, "falha"),
        (147, "bpfi_1.0mm", 0, 1469.79876, "falha"), (148, "bpfi_1.0mm", 1, 1425.16537, "falha"),
        (206, "bpfo_0.3mm", 0, 1492.37453, "falha"), (207, "bpfo_0.3mm", 1, 1518.17746, "falha"),
        (265, "bpfo_1.0mm", 0, 1499.83921, "falha"), (266, "bpfo_1.0mm", 1, 1499.84737, "falha")]

REF = {i: {"gravacao": c, "classe": d, "escore": e, "previsao": d} for i, c, _, e, d in SEGS}


def assinar(corpo):
    return f"${corpo}*{mc.checksum(corpo):02X}"


def linha_seg(prefixo, i, c, k, e, d, placa=None):
    placa = f"{e:.3f}" if placa is None else placa
    tol = 1e-5 * abs(e)
    return (f"{prefixo} seg={i} clipe={c} k={k} ref={e:.3f} placa={placa} dif=0.000122 "
            f"tol={tol:.6f} dec={d} OK ciclos=5855300")


def captura(voltas=2, mexer=None):
    """Captura no formato da de 22/10; mexer(linhas) pode alterar as linhas antes."""
    linhas = ["hello frente1", assinar("CAB,1,exp239,ee0c3e088366,26,100")]
    linhas += [linha_seg("cadeia", *s) for s in SEGS]
    linhas += ["cadeia resumo ok=10/10", "CADEIA OK", assinar("CAB,1,exp239,ee0c3e088366,26,100")]
    n = 0
    for _ in range(voltas):
        linhas.append("ponta seg=29 clipe=normal ref=-1331.768 placa=-1331.769 dif=0.001099 "
                      "tol=0.013318 dec=normal OK ciclos=17446485 igual_boot=1")
        linhas.append(assinar(f"SEG,1,{n},D,0,0,N,0,-1331.769,0,,,17446485"))
        n += 1
        for i, c, k, e, d in SEGS:
            linhas.append(linha_seg("demo", i, c, k, e, d))
            linhas.append(assinar(f"SEG,1,{n},C,{mc.CLIPES.index(c)},{k},"
                                  f"{'F' if d == 'falha' else 'N'},{int(d == 'falha')},"
                                  f"{e:.3f},0,,,5855300"))
            n += 1
    if mexer:
        mexer(linhas)
    return [l + "\r\n" for l in linhas]


def test_captura_certa_da_matriz_diagonal():
    cap = mc.ler_captura(captura())
    assert mc.conferir(cap, REF) == []
    res = mc.resultado(cap, REF)
    assert res["concordancia"] == 10 and res["n_voltas"] == 2 and res["n_pacotes_c"] == 20
    assert res["m_placa"] == {("normal", "normal"): 2, ("normal", "falha"): 0,
                              ("falha", "normal"): 0, ("falha", "falha"): 8}
    assert res["m_placa"] == res["m_python"]


def test_volta_com_escore_diferente_do_boot_reprova():
    def mexer(linhas):
        i = next(j for j, l in enumerate(linhas) if l.startswith("demo seg=148"))
        linhas[i] = linhas[i].replace("placa=1425.165", "placa=1425.166")
    problemas = mc.conferir(mc.ler_captura(captura(mexer=mexer)), REF)
    assert any("no laço" in p for p in problemas)          # difere do boot
    assert any("difere da linha demo" in p for p in problemas)  # e do $SEG


def test_decisao_diferente_do_python_reprova():
    ref = {**REF, 88: {**REF[88], "previsao": "normal"}}
    assert any("o Python decidiu" in p for p in mc.conferir(mc.ler_captura(captura()), ref))


def test_segmento_faltando_no_boot_reprova():
    def mexer(linhas):
        linhas[:] = [l for l in linhas if not l.startswith("cadeia seg=266")]
    assert any("o boot tem" in p for p in mc.conferir(mc.ler_captura(captura(mexer=mexer)), REF))


def test_pacote_perdido_reprova():
    def mexer(linhas):
        i = next(j for j, l in enumerate(linhas) if l.startswith("$SEG,1,5,"))
        del linhas[i]
    problemas = mc.conferir(mc.ler_captura(captura(mexer=mexer)), REF)
    assert any("salto no contador" in p for p in problemas)


def test_checksum_errado_aborta():
    def mexer(linhas):
        i = next(j for j, l in enumerate(linhas) if l.startswith("$SEG,1,3,"))
        linhas[i] = linhas[i][:-2] + "00"
    try:
        mc.ler_captura(captura(mexer=mexer))
    except mc.Abortar as e:
        assert "checksum" in str(e)
    else:
        raise AssertionError("aceitou um pacote com checksum errado")


def test_ref_do_firmware_diferente_do_csv_reprova():
    ref = {**REF, 30: {**REF[30], "escore": -1392.2}}
    assert any("não é o escore do CSV" in p for p in mc.conferir(mc.ler_captura(captura()), ref))


def test_formatacao_brasileira():
    assert mc.br(-1331.76767, 3) == "-1.331,768"
    assert mc.tex_num(-1331.76767, 3) == "$-1.331{,}768$"


def test_hash_nao_depende_da_quebra_de_linha(tmp_path):
    """O git guarda o CSV e a captura com LF; o hash tem de ser o dos bytes originais, em CRLF."""
    crlf, lf = tmp_path / "crlf.csv", tmp_path / "lf.csv"
    crlf.write_bytes(b"segmento,escore\r\n29,-1331.76767\r\n")
    lf.write_bytes(b"segmento,escore\n29,-1331.76767\n")
    assert mc.sha12_crlf(crlf) == mc.sha12_crlf(lf) == mc.sha12(crlf)
