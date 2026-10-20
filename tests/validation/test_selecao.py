"""
Testes do validation/selecao.py, sem torch (o compare_classifiers importa o torch e
os testes dele pulam sem ele).

Garantias: os folds internos nunca leem o teste externo, deixam cada falha de
treino inteira fora uma vez e respeitam a faixa de descarte; o `escolher()` fica com
o mais barato dentro da margem, em qualquer ordem.
"""

import numpy as np
import pytest

import config
import pcm_io
from validation import particao
from validation.selecao import escolher, folds_internos

N_AMOSTRAS = 767_982
CUSTOS = {"barato": {"macs": 26, "parametros": 27}, "caro": {"macs": 30, "parametros": 31}}


@pytest.fixture(scope="module")
def particoes():
    clipes = [pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=config.FS_TRABALHO,
                          pcm=np.zeros(N_AMOSTRAS, dtype=config.PCM_DTYPE)) for r in config.CLASSES]
    return particao.gerar_particoes(clipes)


def test_folds_internos_nao_leem_o_teste_externo(particoes):
    segmentos = particao.segmentos_de(particoes)
    descarte = int(particoes["parametros"]["segmentos_descarte"])
    for fe in particao.folds_de(particoes, "B"):
        externo = set(fe.teste)
        internos = folds_internos(segmentos, fe.treino, descarte)
        falhas_treino = {segmentos[i].rotulo for i in fe.treino if segmentos[i].binario == "falha"}
        assert {f for _, _, f in internos} == falhas_treino
        for tr, te, falha in internos:
            assert not (set(tr) | set(te)) & externo
            assert not set(tr) & set(te)
            assert all(segmentos[i].rotulo != falha for i in tr)        # a falha interna fica inteira fora
            for i in tr:
                s = segmentos[i]
                vizinhos = [segmentos[j].indice for j in te if segmentos[j].rotulo == s.rotulo]
                assert all(abs(s.indice - k) > descarte for k in vizinhos)


@pytest.mark.parametrize("ordem", [("barato", "caro"), ("caro", "barato")])
def test_dentro_da_margem_fica_o_mais_barato(ordem):
    medias = {"barato": 0.900, "caro": 0.915}
    assert escolher({m: medias[m] for m in ordem}, CUSTOS, 0.02)[0] == "barato"


def test_mais_caro_entra_se_ganhar_por_mais_que_a_margem():
    assert escolher({"barato": 0.900, "caro": 0.950}, CUSTOS, 0.02)[0] == "caro"


def test_dispersao_entre_sementes_alarga_a_tolerancia():
    medias = {"barato": 0.800, "caro": 0.840}
    assert escolher(medias, CUSTOS, 0.02)[0] == "caro"
    assert escolher(medias, CUSTOS, 0.02, desvios={"barato": 0.0, "caro": 0.05})[0] == "barato"


def test_em_custo_igual_fica_a_referencia():
    # o SVM linear custa o mesmo que a LDA: dentro da margem, fica a LDA, mesmo
    # que o SVM tenha a melhor pior falha (que só desempata entre não referências)
    custos = {"lda": CUSTOS["barato"], "svm_lin": CUSTOS["barato"]}
    medias = {"lda": 0.900, "svm_lin": 0.915}
    pior = {"lda": 0.70, "svm_lin": 0.90}
    for ordem in (("lda", "svm_lin"), ("svm_lin", "lda")):
        m = {k: medias[k] for k in ordem}
        assert escolher(m, custos, 0.02, pior=pior, referencia="lda")[0] == "lda"
    assert escolher(medias, custos, 0.02, pior=pior)[0] == "svm_lin"       # sem referência, como antes
    # fora da margem, o rival de mesmo custo troca a referência
    assert escolher({"lda": 0.900, "svm_lin": 0.950}, custos, 0.02, referencia="lda")[0] == "svm_lin"
