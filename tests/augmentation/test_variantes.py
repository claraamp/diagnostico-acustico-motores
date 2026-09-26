"""
Testes das variantes de aumento e, principalmente, do filtro por fold.

O teste que importa é `test_variantes_aceitas_so_leem_treino`: para cada fold
dos protocolos A e B, nenhuma variante aceita pode ler uma amostra de segmento
de teste ou da faixa de descarte. É a garantia de que o aumento não reabre o
vazamento que o protocolo fechou.

Usam gravações sintéticas do tamanho das reais; não dependem de data/.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
import pcm_io
from augmentation import variantes as V
from validation import particao

N_AMOSTRAS = 767_982
N = config.AMOSTRAS_POR_SEGMENTO
N_COPIAS = 3


@pytest.fixture(scope="module")
def clipes():
    rng = np.random.default_rng(0)
    return [
        pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=config.FS_TRABALHO,
                    pcm=(rng.normal(0, 3000, N_AMOSTRAS)).astype(config.PCM_DTYPE))
        for r in config.CLASSES
    ]


@pytest.fixture(scope="module")
def particoes(clipes):
    return particao.gerar_particoes(clipes)


@pytest.fixture(scope="module")
def segmentos(particoes):
    return particao.segmentos_de(particoes)


@pytest.fixture(scope="module")
def todas(segmentos):
    """Variantes com as três técnicas para todos os segmentos (só parâmetros)."""
    return [V.sortear(s, k, V.TECNICAS, N_AMOSTRAS)
            for s in segmentos for k in range(N_COPIAS)]


@pytest.fixture(scope="module")
def lidos(segmentos, todas):
    return V.ids_lidos(segmentos,
                       np.array([v.origem for v in todas]),
                       np.array([v.inicio_lido for v in todas]),
                       np.array([v.fim_lido for v in todas]))


# --------------------------------------------------------------- sorteio
def test_sorteio_reprodutivel(segmentos):
    s = segmentos[17]
    assert V.sortear(s, 2, V.TECNICAS, N_AMOSTRAS) == V.sortear(s, 2, V.TECNICAS, N_AMOSTRAS)
    assert V.sortear(s, 1, V.TECNICAS, N_AMOSTRAS) != V.sortear(s, 2, V.TECNICAS, N_AMOSTRAS)


def test_parametros_dentro_das_faixas(todas):
    d_max = round(config.AUMENTO_DESLOC_MAX_S * config.FS_TRABALHO)
    lo, hi = config.AUMENTO_ESTIR_TAXAS
    s_lo, s_hi = config.AUMENTO_SNR_DB
    for v in todas:
        assert abs(v.deslocamento) <= d_max
        assert lo <= v.taxa <= hi
        assert s_lo <= v.snr_db <= s_hi
        assert 0 <= v.inicio_lido < v.fim_lido <= N_AMOSTRAS


def test_tecnica_desligada_fica_neutra_e_as_outras_nao_mudam(segmentos):
    """Ablação compara técnica, não sorte: o parâmetro ligado é o mesmo."""
    s = segmentos[30]
    tudo = V.sortear(s, 0, V.TECNICAS, N_AMOSTRAS)
    so_desloc = V.sortear(s, 0, ["deslocamento"], N_AMOSTRAS)
    so_ruido = V.sortear(s, 0, ["ruido"], N_AMOSTRAS)
    assert so_desloc.deslocamento == tudo.deslocamento
    assert so_desloc.taxa == 1.0 and so_desloc.snr_db is None
    assert so_ruido.snr_db == tudo.snr_db
    assert so_ruido.deslocamento == 0 and so_ruido.taxa == 1.0


def test_tecnicas_invalidas():
    with pytest.raises(ValueError):
        V.validar_tecnicas([])
    with pytest.raises(ValueError):
        V.validar_tecnicas(["reverb"])
    assert V.validar_tecnicas(["ruido", "deslocamento"]) == ("deslocamento", "ruido")


# --------------------------------------------------------------- sinal
@pytest.mark.parametrize("tecnicas", [
    ["deslocamento"], ["estiramento"], ["ruido"], list(V.TECNICAS)])
@pytest.mark.parametrize("modo", ["tempo", "velocidade"])
def test_aplicar_devolve_um_segmento(clipes, segmentos, tecnicas, modo):
    """Inclui o primeiro e o último segmento, onde a janela bate na borda."""
    x = clipes[0].x
    for s in (segmentos[0], segmentos[29], segmentos[58]):
        v = V.sortear(s, 0, tecnicas, N_AMOSTRAS)
        y = V.aplicar(x, v, N, modo=modo)
        assert y.shape == (N,)
        assert np.all(np.isfinite(y))


def test_so_deslocamento_le_o_trecho_deslocado(clipes, segmentos):
    x = clipes[0].x
    s = segmentos[20]
    v = V.sortear(s, 0, ["deslocamento"], N_AMOSTRAS)
    np.testing.assert_array_equal(V.aplicar(x, v, N), x[s.inicio + v.deslocamento:
                                                       s.fim + v.deslocamento])


# --------------------------------------------------------------- filtro por fold
def test_origem_sempre_entre_os_lidos(todas, lidos):
    for v, ids in zip(todas, lidos):
        assert v.origem in ids


def test_leitura_nunca_passa_do_vizinho(todas, lidos):
    """Deslocamento < faixa de descarte: no máximo o segmento vizinho é lido."""
    for v, ids in zip(todas, lidos):
        assert max(ids) - min(ids) <= 2 * config.SEGMENTOS_DESCARTE


@pytest.mark.parametrize("protocolo", ["A", "B"])
def test_variantes_aceitas_so_leem_treino(particoes, segmentos, todas, lidos, protocolo):
    faixa = {s.id: (s.rotulo, s.inicio, s.fim) for s in segmentos}
    for f in particao.folds_de(particoes, protocolo):
        mask = V.mascara_fold(f.treino, lidos)
        proibidos = [faixa[i] for i in f.teste + f.descartados]
        for v, ok in zip(todas, mask):
            if not ok:
                continue
            assert v.origem in f.treino
            rotulo = segmentos[v.origem].rotulo
            for r, a, b in proibidos:
                # nenhuma amostra lida pode estar num segmento de teste ou descarte
                assert not (r == rotulo and v.inicio_lido < b and a < v.fim_lido), (
                    f"{f.nome}: variante de {v.origem} lê [{v.inicio_lido}, {v.fim_lido}) "
                    f"e encosta em {r}[{a}, {b})")


@pytest.mark.parametrize("protocolo", ["A", "B"])
def test_filtro_nao_joga_fora_o_aumento(particoes, todas, lidos, protocolo):
    """Só as variantes de borda caem: a maior parte do treino continua aumentada."""
    for f in particao.folds_de(particoes, protocolo):
        mask = V.mascara_fold(f.treino, lidos)
        origens_treino = sum(1 for v in todas if v.origem in set(f.treino))
        assert mask.sum() >= 0.8 * origens_treino


def test_variante_que_encosta_no_teste_e_recusada(segmentos):
    """Caso construído: variante do segmento 9 lendo o 10, com o 10 no teste."""
    s9 = segmentos[9]
    a, b = V.intervalo_lido(s9, +3000, 1.0, N_AMOSTRAS)
    lidos = V.ids_lidos(segmentos, np.array([9]), np.array([a]), np.array([b]))
    assert lidos == [(9, 10)]
    treino_sem_10 = [i for i in range(0, 9)]
    assert not V.mascara_fold(treino_sem_10 + [9], lidos)[0]
    assert V.mascara_fold(treino_sem_10 + [9, 10], lidos)[0]
