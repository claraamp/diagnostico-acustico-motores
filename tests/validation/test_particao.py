"""
Testes da partição dos protocolos A e B.

Não dependem de data/: usam gravações sintéticas com o mesmo tamanho das reais
(767.982 amostras a 12,8 kHz), então rodam num clone recém-feito.

    pytest tests/
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import config
import pcm_io
from validation import particao

N_AMOSTRAS = 767_982          # tamanho real das gravações decimadas
N_SEG = N_AMOSTRAS // config.AMOSTRAS_POR_SEGMENTO   # 59


@pytest.fixture(scope="module")
def clipes():
    # int16 zerado: a partição só olha rótulo, classe binária e tamanho
    return [
        pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=config.FS_TRABALHO,
                    pcm=np.zeros(N_AMOSTRAS, dtype=config.PCM_DTYPE))
        for r in config.CLASSES
    ]


@pytest.fixture(scope="module")
def particoes(clipes):
    return particao.gerar_particoes(clipes)


@pytest.fixture(scope="module")
def segmentos(particoes):
    return particao.segmentos_de(particoes)


# --------------------------------------------------------------- segmentação
def test_59_segmentos_por_gravacao(segmentos):
    for r in config.CLASSES:
        assert sum(s.rotulo == r for s in segmentos) == N_SEG == 59


def test_blocos_10_10_10_10_10_9(segmentos):
    for r in config.CLASSES:
        blocos = [s.bloco for s in segmentos if s.rotulo == r]
        assert [blocos.count(b) for b in sorted(set(blocos))] == [10, 10, 10, 10, 10, 9]


def test_segmentos_contiguos_sem_sobreposicao(segmentos):
    for r in config.CLASSES:
        segs = sorted((s for s in segmentos if s.rotulo == r), key=lambda s: s.indice)
        assert segs[0].inicio == 0
        for a, b in zip(segs, segs[1:]):
            assert a.fim == b.inicio
        for s in segs:
            assert s.fim - s.inicio == config.AMOSTRAS_POR_SEGMENTO
        assert segs[-1].fim <= N_AMOSTRAS


# --------------------------------------------------------------- garantias gerais
def test_particao_gerada_passa_na_verificacao(particoes):
    assert particao.verificar(particoes) == []


def test_numero_de_folds(particoes):
    assert len(particao.folds_de(particoes, "A")) == 6
    assert len(particao.folds_de(particoes, "B")) == 4 * 6


def test_nenhum_segmento_em_treino_e_teste(particoes):
    for f in particao.folds_de(particoes):
        assert not set(f.treino) & set(f.teste), f.nome


def test_faixa_de_descarte(particoes, segmentos):
    """Nenhum segmento de treino encosta num de teste da mesma gravação."""
    for f in particao.folds_de(particoes):
        teste = {(segmentos[i].rotulo, segmentos[i].indice) for i in f.teste}
        for i in f.treino:
            s = segmentos[i]
            for d in range(-config.SEGMENTOS_DESCARTE, config.SEGMENTOS_DESCARTE + 1):
                assert (s.rotulo, s.indice + d) not in teste, f"{f.nome}: segmento {i}"


# --------------------------------------------------------------- Protocolo A
def test_a_cada_segmento_testado_exatamente_uma_vez(particoes, segmentos):
    vezes = np.zeros(len(segmentos), dtype=int)
    for f in particao.folds_de(particoes, "A"):
        vezes[f.teste] += 1
    assert (vezes == 1).all()


# --------------------------------------------------------------- Protocolo B
def test_b_falha_de_fora_inteira_no_teste_e_ausente_do_treino(particoes, segmentos):
    for f in particao.folds_de(particoes, "B"):
        falha = f.info["falha_de_fora"]
        da_falha = {s.id for s in segmentos if s.rotulo == falha}
        assert da_falha <= set(f.teste), f.nome
        assert not da_falha & set(f.treino), f.nome


def test_b_outras_falhas_inteiras_no_treino(particoes, segmentos):
    for f in particao.folds_de(particoes, "B"):
        outras = {s.id for s in segmentos
                  if s.binario == "falha" and s.rotulo != f.info["falha_de_fora"]}
        assert outras <= set(f.treino), f.nome


def test_b_contagens_de_treino(particoes, segmentos):
    """3 falhas inteiras (177) e a normal sem o bloco de teste e a faixa: 47–49."""
    for f in particao.folds_de(particoes, "B"):
        n_falha = sum(segmentos[i].binario == "falha" for i in f.treino)
        n_normal = sum(segmentos[i].binario == "normal" for i in f.treino)
        assert n_falha == 3 * N_SEG == 177, f.nome
        assert 47 <= n_normal <= 49, f.nome


def test_b_teste_tem_um_bloco_normal(particoes, segmentos):
    for f in particao.folds_de(particoes, "B"):
        normais = [segmentos[i] for i in f.teste if segmentos[i].binario == "normal"]
        assert {s.bloco for s in normais} == {f.info["bloco_normal"]}, f.nome


# --------------------------------------------------------------- a verificação acusa
def _corromper(particoes, fn):
    p = copy.deepcopy(particoes)
    fn(p)
    return particao.verificar(p)


def test_verificacao_acusa_segmento_em_treino_e_teste(particoes):
    def fn(p):
        f = p["folds"][0]
        f["treino"].append(f["teste"][0])
    assert any("treino E teste" in e for e in _corromper(particoes, fn))


def test_verificacao_acusa_faixa_de_descarte_violada(particoes):
    def fn(p):
        f = p["folds"][0]
        i = f["descartados"].pop()
        f["treino"].append(i)
    assert any("até" in e for e in _corromper(particoes, fn))


def test_verificacao_acusa_falha_de_fora_no_treino(particoes, segmentos):
    def fn(p):
        f = next(x for x in p["folds"] if x["protocolo"] == "B")
        i = next(s.id for s in segmentos if s.rotulo == f["info"]["falha_de_fora"])
        f["teste"].remove(i)
        f["treino"].append(i)
    erros = _corromper(particoes, fn)
    assert any("aparece no treino" in e for e in erros)


# --------------------------------------------------------------- arquivo
def test_ida_e_volta_pelo_arquivo(particoes, tmp_path):
    caminho = particao.salvar(particoes, tmp_path / "splits.json")
    lido = particao.carregar(caminho)
    assert lido == particoes
    assert particao.verificar(lido) == []


def test_arquivo_deterministico(clipes, tmp_path):
    a = particao.salvar(particao.gerar_particoes(clipes), tmp_path / "a.json")
    b = particao.salvar(particao.gerar_particoes(clipes), tmp_path / "b.json")
    assert particao.hash_arquivo(a) == particao.hash_arquivo(b)


def test_compatibilidade_acusa_gravacao_de_outro_tamanho(particoes):
    outro = [pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=config.FS_TRABALHO,
                         pcm=np.zeros(N_AMOSTRAS - 1, dtype=config.PCM_DTYPE))
             for r in config.CLASSES]
    assert particao.conferir_compatibilidade(particoes, outro)


# --------------------------------------------------------------- outra taxa
def _clipes_a(fs: int, n: int):
    return [pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=fs,
                        pcm=np.zeros(n, dtype=config.PCM_DTYPE))
            for r in config.CLASSES]


def test_outra_taxa_tem_os_mesmos_segmentos_em_segundos(particoes):
    """A 25,6 kHz (02 --fs 25600: 1.535.982 amostras), os segmentos e os folds são
    os mesmos da taxa de trabalho; só o número de amostras de cada um dobra."""
    fs = 25_600
    outra = particao.gerar_particoes(_clipes_a(fs, 1_535_982), config.amostras_por_segmento(fs))
    assert particao.verificar(outra) == []
    a, b = particao.segmentos_de(particoes), particao.segmentos_de(outra)
    assert [(s.rotulo, s.indice, s.bloco) for s in a] == [(s.rotulo, s.indice, s.bloco) for s in b]
    assert all(t.inicio == 2 * s.inicio and t.fim == 2 * s.fim for s, t in zip(a, b))
    assert [(f.nome, f.treino, f.teste, f.descartados) for f in particao.folds_de(particoes)] == \
        [(f.nome, f.treino, f.teste, f.descartados) for f in particao.folds_de(outra)]


def test_compatibilidade_confere_a_taxa_da_rodada(particoes):
    fs = 25_600
    clipes_25k = _clipes_a(fs, 1_535_982)
    outra = particao.gerar_particoes(clipes_25k, config.amostras_por_segmento(fs))
    assert particao.conferir_compatibilidade(outra, clipes_25k, fs) == []
    # a partição de 25,6 kHz não passa como se fosse da taxa de trabalho, nem o contrário
    assert particao.conferir_compatibilidade(outra, clipes_25k)
    assert particao.conferir_compatibilidade(particoes, clipes_25k, fs)


def test_caminhos_da_taxa_de_trabalho_nao_mudam():
    fs = config.FS_TRABALHO
    assert config.arquivo_splits(fs) == "data/processed/splits/splits.json"
    assert config.dir_features(fs) == "data/processed/features"
    assert config.arquivo_splits(25_600) == "data/processed/splits/splits_25600.json"
    assert config.dir_features(25_600) == "data/processed/features/25600"
    assert config.amostras_por_segmento(fs) == config.AMOSTRAS_POR_SEGMENTO
