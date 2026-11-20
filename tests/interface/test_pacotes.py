"""
Garantias do scripts/interface/pacotes.py, sem placa e sem janela.

A leitura aceita e recusa as mesmas linhas que o leitor de referência do firmware
(tests/host/pacote/ler_pacotes.py, autoteste), mais o $ERR da recepção. O painel não
mostra decisão antes de conferir o modelo pelo $CAB, nem com outro modelo; um
reinício (hello) pede o $CAB de novo; o erro de conta nunca vira "normal"; o LED
segue a tabela do README do firmware ("Sinalização por LED"); o salto do n conta
pacotes perdidos, e um pacote de origem desconhecida é ignorado sem contar como
perdido nem como corrompido.
"""

from functools import reduce

import pytest

from interface import pacotes as P

MODELO = ("exp239", "ee0c3e088366")


def assinar(corpo: str) -> str:
    return f"${corpo}*{reduce(lambda x, c: x ^ ord(c), corpo, 0):02X}"


CAB = "$CAB,1,exp239,ee0c3e088366,26,100*3B"

# as linhas do autoteste do ler_pacotes.py (firmware), que o test_pacote_serial.c também usa
BOAS = [
    CAB,
    "$SEG,1,7,R,2,1043,F,1,231.475,0,,,412*69",
    "$SEG,1,7,R,2,1043,N,0,-1577.959,0,,,412*7A",
    "$SEG,1,7,C,0,1,F,2,-1400.000,3,812.250,61.500,412*6D",
    "$SEG,1,7,A,,0,F,1,10.000,2,3.000,4.000,412*4C",
    "$SEG,1,0,D,0,0,N,0,-1331.770,0,,,17453346*6C",
    "$SEG,1,812,S,2,117,N,0,-1.234,0,,,1043210*71",
    assinar("SEG,1,1,R,0,5,N,0,-0.000,0,,,0"),
    assinar("SEG,1,1,R,0,5,F,1,0.000,0,,,0"),
    assinar("SEG,1,1,C,0,5,N,0,-3.000,3,5.000,5.000,0"),
    assinar("SEG,1,1,R,0,5,F,4,nan,0,,,0"),
    assinar("SEG,1,1,R,0,5,F,4,-inf,0,,,0"),
    assinar("SEG,1,1,C,0,5,F,4,-1400.000,3,nan,61.500,0"),
    assinar("SEG,1,1,R,0,5,F,1,inf,0,,,0"),
    "$ERR,1,2,118,crc*0C",          # os exemplos do README do firmware
    "$ERR,1,,,truncado*60",
]
RUINS = {
    "checksum": "$SEG,1,7,R,2,1043,F,1,231.475,0,,,412*68",
    "campo a menos": assinar("SEG,1,7,R,2,1043,F,1,231.475,0,,"),
    "versao": assinar("SEG,2,7,R,2,1043,F,1,231.475,0,,,412"),
    "decisao x disparou": assinar("SEG,1,7,R,2,1043,N,1,231.475,0,,,412"),
    "sentido da LDA": assinar("SEG,1,7,R,2,1043,N,0,231.475,0,,,412"),
    "zero sem sinal com N": assinar("SEG,1,1,R,0,5,N,0,0.000,0,,,0"),
    "zero com sinal com F": assinar("SEG,1,1,R,0,5,F,1,-0.000,0,,,0"),
    "escore nan sem erro": assinar("SEG,1,1,R,0,5,N,0,nan,0,,,0"),
    "escore nan com F sem erro": assinar("SEG,1,1,R,0,5,F,1,nan,0,,,0"),
    "erro com N": assinar("SEG,1,1,R,0,5,N,4,nan,0,,,0"),
    "erro junto com LDA": assinar("SEG,1,1,R,0,5,F,5,nan,0,,,0"),
    "erro sem valor nao finito": assinar("SEG,1,1,R,0,5,F,4,231.475,0,,,0"),
    "pontuacao nan sem erro": assinar("SEG,1,1,C,0,5,F,1,1.000,3,nan,5.000,0"),
    "sentido da novidade": assinar("SEG,1,7,C,0,1,N,0,-1400.000,3,812.250,61.500,412"),
    "novidade sem detector": assinar("SEG,1,7,R,2,1043,F,3,231.475,0,,,412"),
    "pontuacao com codigo 0": assinar("SEG,1,7,R,2,1043,F,1,231.475,0,1.000,2.000,412"),
    "err motivo": assinar("ERR,1,2,118,outro"),
    "err so com grupo": assinar("ERR,1,2,,crc"),
    "err campos": assinar("ERR,1,2,118"),
}


@pytest.mark.parametrize("linha", BOAS)
def test_aceita_as_linhas_do_leitor_de_referencia(linha):
    P.ler_linha(linha + "\r\n")


@pytest.mark.parametrize("motivo", sorted(RUINS))
def test_recusa_as_linhas_do_leitor_de_referencia(motivo):
    with pytest.raises(P.ErroPacote):
        P.ler_linha(RUINS[motivo])


def test_origem_desconhecida_e_so_ignorada():
    with pytest.raises(P.OrigemDesconhecida) as e:
        P.ler_linha(assinar("SEG,1,7,X,2,1043,F,1,231.475,0,,,412"))
    assert e.value.n == 7


def test_classe_nunca_e_normal_com_erro():
    assert P.classe(P.ler_linha(assinar("SEG,1,1,R,0,5,F,4,nan,0,,,0"))[1]) == "erro"
    assert P.classe(P.ler_linha(assinar("SEG,1,1,R,0,5,N,0,-0.000,0,,,0"))[1]) == "normal"
    assert P.classe(P.ler_linha(assinar("SEG,1,1,R,0,5,F,1,0.000,0,,,0"))[1]) == "falha"


def seg(n, origem="C", clipe=0, segmento=0, escore="-1331.768", disparou=None):
    falha = disparou != 0 if disparou is not None else not escore.startswith("-")
    disp = disparou if disparou is not None else int(falha)
    return assinar(f"SEG,1,{n},{origem},{clipe},{segmento},{'F' if falha else 'N'},{disp},{escore},0,,,5850000")


def painel(**kw):
    t = iter(range(10 ** 6))
    return P.Painel(MODELO, relogio=lambda: float(next(t)), **kw)


def test_sem_cab_nao_mostra_decisao():
    p = painel()
    assert p.linha(seg(5)) == "seg"
    assert p.atual is None and p.contagem["sem_cab"] == 1 and p.contagem["segmentos"] == 0
    assert p.led == (False, False)
    p.linha(CAB)
    p.linha(seg(6, escore="1361.836", clipe=1))
    assert p.modelo_ok and p.atual.classe == "falha" and p.led == (False, True)
    assert p.atual.onde == "clipe bpfi_0.3mm" and "seg. 88 do Jung" in p.atual.segmento


def test_outro_modelo_nao_mostra_decisao():
    p = painel()
    p.linha(assinar("CAB,1,exp238,0123456789ab,26,100"))
    p.linha(seg(0))
    assert not p.modelo_ok and "exp238" in p.modelo_errado and p.atual is None


def test_hello_pede_o_cab_de_novo():
    p = painel()
    for l in (CAB, seg(0), seg(1), "hello frente1\r\n"):
        p.linha(l)
    assert p.contagem["reinicios"] == 1 and not p.modelo_ok and p.led == (False, False)
    p.linha(seg(0))
    assert p.contagem["sem_cab"] == 1 and p.contagem["segmentos"] == 2
    p.linha(CAB)
    p.linha(seg(1))
    assert p.contagem["segmentos"] == 3 and p.contagem["perdidos"] == 0


def test_salto_do_n_e_reinicio_sem_cab():
    p = painel()
    for l in (CAB, seg(10), seg(13), seg(0)):
        p.linha(l)
    assert p.contagem["perdidos"] == 2 and p.contagem["reinicios"] == 1
    assert [i.texto for i in p.historico if i.tipo == "aviso"] == [
        "2 pacotes perdidos (n 10 → 13)", "a placa reiniciou (n voltou a 0)"]


def test_cab_zera_o_contador_como_na_entrada_da_recepcao():
    # ao entrar no modo de recepção a placa manda o $CAB de novo, e o n segue o da demonstração
    p = painel()
    for l in (CAB, seg(3), CAB, seg(53, origem="S", clipe=3, segmento=48, escore="-0.677")):
        p.linha(l)
    assert p.contagem["perdidos"] == 0 and p.atual.onde == "grupo 3 (ruído branco)"


def test_origem_desconhecida_nao_conta_como_perdido_nem_corrompido():
    p = painel()
    for l in (CAB, seg(1), assinar("SEG,1,2,X,0,0,N,0,-1.000,0,,,0"), seg(3)):
        p.linha(l)
    assert p.contagem["origem_desconhecida"] == 1
    assert p.contagem["perdidos"] == 0 and p.contagem["descartadas"] == 0


def test_checksum_errado_descarta_e_o_salto_conta():
    p = painel()
    ruim = seg(2)[:-2] + "00"
    for l in (CAB, seg(1), ruim, seg(3)):
        p.linha(l)
    assert p.contagem["descartadas"] == 1 and p.contagem["perdidos"] == 1


def test_leds_seguem_a_tabela_do_firmware():
    p = painel()
    p.linha(CAB)
    assert p.led == (False, False)                        # antes do primeiro $SEG
    p.linha(seg(0, escore="-1331.768"))
    assert p.led == (True, False)                         # N: verde
    p.linha(seg(1, escore="1361.836"))
    assert p.led == (False, True)                         # F: vermelho
    p.linha(seg(2, escore="nan", disparou=4))
    assert p.led == (True, True) and p.atual.classe == "erro"   # erro de conta: os dois
    p.linha("$ERR,1,2,118,crc*0C")
    assert p.led == (False, False) and p.atual.tipo == "err"    # $ERR: apagados
    assert p.contagem["err"] == 1 and p.contagem["segmentos"] == 3


def test_consistencia_k_de_n_nao_conta_erro_como_falha():
    p = painel(n_consistencia=4)
    p.linha(CAB)
    for n, e in enumerate(("-1.0", "1.0", "1.0", "-1.0", "2.0")):
        p.linha(seg(n, escore=e))
    assert p.consistencia() == (3, 4)
    p.linha(seg(5, escore="nan", disparou=4))
    assert p.consistencia() == (2, 4)


def test_nomes_das_linhas_pc_envio_da_captura():
    p = painel()
    for l in ("#pc envio grupo=3 id=48 tentativa=1 nome=normal_seg0048_branco_0db\r\n", CAB,
              seg(53, origem="S", clipe=3, segmento=48, escore="-0.677")):
        p.linha(l)
    assert p.atual.nome == "normal_seg0048_branco_0db" and p.atual.escore == "-0.677"
    assert p.atual.classe == "normal"


def test_linhas_que_nao_sao_pacote_sao_ignoradas():
    p = painel()
    for l in ("demo seg=29 clipe=normal k=0", "rx grupo=3 id=48 bytes=25618", "$PED,1,3*52",
              "recepcao inicio modelo=exp239", "#pc rodada fim"):
        assert p.linha(l) is None
    assert p.contagem["descartadas"] == 0 and not p.avisos


def test_historico_tem_tamanho_maximo():
    p = painel(n_historico=3)
    p.linha(CAB)
    for n in range(5):
        p.linha(seg(n))
    assert [i.n for i in p.historico] == [2, 3, 4]


def test_modelo_python_e_o_do_lda_final(tmp_path):
    import hashlib
    j = tmp_path / "lda_final.json"
    j.write_bytes(b'{"id": "exp239", "x": 1}')
    assert P.modelo_python(j) == ("exp239", hashlib.sha256(j.read_bytes()).hexdigest()[:12])
