"""
Garantias do scripts/interface/run_painel.py sem placa: as fontes de linhas (captura
salva, com e sem ritmo; captura que outro programa está gravando, num arquivo ou numa
pasta) e o modo texto, com o código de saída. A janela (Tkinter) não é testada aqui; a
captura do matriz_serial_placa.py tem de chegar ao disco linha a linha, para o --seguir.
"""

import threading
import time
from functools import reduce

import pytest

from interface import run_painel as R
from interface.pacotes import Painel

MODELO = ("exp239", "ee0c3e088366")


def assinar(corpo: str) -> str:
    return f"${corpo}*{reduce(lambda x, c: x ^ ord(c), corpo, 0):02X}"


CAPTURA = ["hello frente1", "$CAB,1,exp239,ee0c3e088366,26,100*3B",
           assinar("SEG,1,0,D,0,0,N,0,-1331.769,0,,,17441807"),
           "demo seg=29 clipe=normal k=0 ...",
           assinar("SEG,1,1,C,0,0,N,0,-1331.768,0,,,5859338"),
           assinar("SEG,1,2,C,1,0,F,1,1361.836,0,,,5859338"),
           "$ERR,1,2,118,crc*0C"]


def test_arquivo_com_ritmo_dorme_depois_de_cada_seg_e_err(tmp_path):
    cap = tmp_path / "cap.log"
    cap.write_bytes(("\r\n".join(CAPTURA) + "\r\n").encode())
    sonos = []
    linhas = list(R.linhas_arquivo(cap, ritmo=0.5, dormir=sonos.append))
    assert len(linhas) == len(CAPTURA) and sonos == [0.5] * 4


def test_seguir_le_a_captura_enquanto_ela_cresce_com_linha_cortada(tmp_path):
    cap = tmp_path / "serial-20-11.log"
    cap.write_bytes(b"")
    parar = threading.Event()
    lidas = []

    def escrever():
        with open(cap, "ab") as fh:
            for l in CAPTURA[:3]:
                fh.write(l.encode() + b"\r\n")
                fh.flush()
                time.sleep(0.05)
            fh.write(b"$SEG,1,1,C,0,0,N,0,-13")          # metade de uma linha
            fh.flush()
            time.sleep(0.1)
            fh.write(CAPTURA[4][len("$SEG,1,1,C,0,0,N,0,-13"):].encode() + b"\r\n")
            fh.flush()

    t = threading.Thread(target=escrever)
    t.start()
    for l in R.linhas_seguir(cap, parar, intervalo=0.01, segundos=3):
        lidas.append(l)
        if len(lidas) == 5:
            parar.set()
    t.join()
    assert lidas[0] == f"#painel captura {cap.name}\n"
    assert [l.rstrip("\r\n") for l in lidas[1:]] == CAPTURA[:3] + [CAPTURA[4]]


def test_seguir_pasta_espera_captura_nova_e_ignora_as_antigas(tmp_path):
    velha = tmp_path / "serial-19-11.log"
    velha.write_bytes(CAPTURA[1].encode() + b"\r\n")
    antes = time.time() - 100
    import os
    os.utime(velha, (antes, antes))
    parar = threading.Event()

    def gravar_nova():
        time.sleep(0.1)
        (tmp_path / "serial-20-11.log").write_bytes(("\r\n".join(CAPTURA[:3]) + "\r\n").encode())

    threading.Thread(target=gravar_nova).start()
    lidas = []
    for l in R.linhas_seguir(tmp_path, parar, intervalo=0.01, segundos=3):
        lidas.append(l)
        if len(lidas) == 4:
            parar.set()
    assert lidas[0] == "#painel captura serial-20-11.log\n" and lidas[1].startswith("hello")


def test_modo_texto_e_codigo_de_saida(tmp_path, monkeypatch, capsys):
    cap = tmp_path / "cap.log"
    modelo = tmp_path / "lda_final.json"
    modelo.write_bytes(b'{"id": "exp239"}')
    import hashlib
    sha = hashlib.sha256(modelo.read_bytes()).hexdigest()[:12]
    cab = assinar(f"CAB,1,exp239,{sha},26,100")
    linhas = [cab] + CAPTURA[2:]
    cap.write_bytes(("\r\n".join(linhas) + "\r\n").encode())
    assert R.main(["--arquivo", str(cap), "--modelo", str(modelo), "--sem-janela"]) == 0
    saida = capsys.readouterr().out
    assert "NORMAL" in saida and "FALHA" in saida and "LED: vermelho" in saida
    assert "$ERR crc (grupo 2, id 118)  LED: apagados" in saida
    assert "3 segmentos (2 normal, 1 falha, 0 erro); 1 $ERR" in saida
    # uma linha corrompida: código 1
    cap.write_bytes(("\r\n".join(linhas + [linhas[-2][:-2] + "00"]) + "\r\n").encode())
    assert R.main(["--arquivo", str(cap), "--modelo", str(modelo), "--sem-janela"]) == 1
    # a placa com outro modelo: código 1, e nenhuma decisão aparece
    cap.write_bytes(("\r\n".join(CAPTURA) + "\r\n").encode())
    capsys.readouterr()
    assert R.main(["--arquivo", str(cap), "--modelo", str(modelo), "--sem-janela"]) == 1
    saida = capsys.readouterr().out
    assert "NÃO é o do Python" in saida and "NORMAL" not in saida


def test_opcoes_que_nao_combinam(tmp_path):
    cap = tmp_path / "cap.log"
    cap.write_bytes(b"")
    for args in (["--seguir", str(tmp_path), "--ritmo", "1"], ["--arquivo", str(cap), "--salvar", "x"],
                 ["--arquivo", str(cap), "--ritmo", "-1"], ["--arquivo", str(tmp_path / "nao.log")]):
        with pytest.raises(SystemExit):
            R.main(args + ["--sem-janela"])


def test_captura_do_matriz_chega_ao_disco_a_cada_linha(tmp_path):
    from validation import matriz_serial_placa as M
    cap = M.Captura(tmp_path / "cap.log")
    cap.placa(b"$SEG,1,0,S,3,48,N,0,-0.677,0,,,5843000*00\r")
    assert (tmp_path / "cap.log").read_bytes().endswith(b"*00\r\n")      # antes de fechar
    cap.fechar()


def test_linha_texto_sem_medida_de_tempo():
    p = Painel(MODELO)
    p.linha("$CAB,1,exp239,ee0c3e088366,26,100*3B")
    p.linha(assinar("SEG,1,0,S,1,0,N,0,-1577.958,0,,,0"))
    assert "(—)" in R.linha_texto(p) and "grupo 1 (Jung)" in R.linha_texto(p)


def test_janela_desenha_os_estados_quando_ha_tela():
    tk = pytest.importorskip("tkinter")
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sem tela (DISPLAY)")
    try:
        import queue
        p = Painel(MODELO)
        j = R.Janela(raiz, p, queue.Queue(), "teste", threading.Event())
        for l in CAPTURA[1:6]:
            p.linha(l)
        j.redesenhar()
        assert j.l_classe.cget("text") == "FALHA" and j.caixa.cget("bg") == R.CORES["falha"]
        p.linha(CAPTURA[6])
        j.redesenhar()
        assert j.l_classe.cget("text") == "$ERR" and len(j.arvore.get_children()) == 4
    finally:
        raiz.destroy()
