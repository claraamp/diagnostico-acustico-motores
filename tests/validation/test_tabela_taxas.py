"""
Testes da tabela do Protocolo B em duas taxas (run_tabela_taxas.py).

Usam `metrics.json` mínimos e partições geradas de gravações sintéticas em
tmp_path; não dependem de data/. Conferem que (1) a tabela sai com a diferença
entre as taxas, (2) recusa rodadas não comparáveis e (3) recusa partições que
não descrevem os mesmos segmentos, ou que não são as que a rodada registrou.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import config
import pcm_io
from validation import particao
from validation import run_tabela_taxas as T

BASE = {"protocolo": "B", "tarefa": "binario", "modelo": "lda",
        "features": "mfcc_dsp_media_desvio", "segmento_s": 1.0,
        "segmentos_por_bloco": 10, "segmentos_descarte": 1,
        "mfcc_janela_ms": 25, "mfcc_hop_ms": 10, "mfcc_n_mels": 20, "mfcc_n_coefs": 13,
        "permutado": False, "sem_c0": False, "norm_clipe": False, "aumento": "nenhum"}
N = {12_800: 767_982, 25_600: 1_535_982}   # amostras por gravação em cada taxa


def _particao(raiz, fs, n=None):
    clipes = [pcm_io.Clip(rotulo=r, binario=config.BINARIO[r], fs=fs,
                          pcm=np.zeros(n or N[fs], dtype=config.PCM_DTYPE))
              for r in config.CLASSES]
    caminho = particao.salvar(particao.gerar_particoes(clipes, config.amostras_por_segmento(fs)),
                              raiz / config.arquivo_splits(fs))
    return particao.hash_arquivo(caminho)


def _rodada(pasta, exp_id, parametros, ab_bpfo03=0.5, ab_bpfi03=1.0):
    falhas = {"bpfi_0.3mm": ab_bpfi03, "bpfo_0.3mm": ab_bpfo03}
    media = sum(falhas.values()) / len(falhas)
    m = {
        "parametros": parametros,
        "resumo": {
            "sensibilidade_media": 2 * media - 1, "especificidade_media": 1.0,
            "acuracia_balanceada_media": media,
            "pior_falha": "bpfo_0.3mm", "pior_falha_acuracia_balanceada": ab_bpfo03,
            "pior_fold": "B_bpfo_0.3mm_bloco0", "pior_fold_acuracia_balanceada": ab_bpfo03,
            "por_falha": {k: {"acuracia_balanceada": v, "sensibilidade": 2 * v - 1}
                          for k, v in falhas.items()},
        },
        "folds": [],
    }
    d = pasta / f"{exp_id}_protocoloB"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")


@pytest.fixture
def raiz(tmp_path):
    h12, h25 = _particao(tmp_path, 12_800), _particao(tmp_path, 25_600)
    pasta = tmp_path / "reports"
    pasta.mkdir()
    _rodada(pasta, "exp001", {**BASE, "fs_hz": 12_800, "splits": h12})
    _rodada(pasta, "exp002", {**BASE, "fs_hz": 25_600, "splits": h25}, ab_bpfi03=0.98)
    _rodada(pasta, "exp003", {**BASE, "fs_hz": 25_600, "splits": h25, "mfcc_n_mels": 40})
    _rodada(pasta, "exp004", {**BASE, "fs_hz": 25_600, "splits": h25, "aumento": "ruido"})
    _rodada(pasta, "exp005", {**BASE, "fs_hz": 25_600, "splits": "outro"})
    _rodada(pasta, "exp006", {**BASE, "fs_hz": 12_800, "splits": h12, "permutado": True})
    return tmp_path


def test_tabela_com_a_diferenca_entre_taxas(raiz):
    texto = T.montar(["exp001", "exp002"], raiz / "reports", raiz)
    assert "| 12.8 kHz | exp001 | 6400 Hz |" in texto
    assert "| 25.6 kHz | exp002 | 12800 Hz |" in texto
    assert "diferença (25.6 − 12.8 kHz)" in texto
    assert "| diferença | -0.020 | 0 |" in texto     # acc. bal. da bpfi_0.3mm: 0,98 − 1


@pytest.mark.parametrize("ids", [
    ["exp001", "exp003"],   # outro MFCC
    ["exp001", "exp004"],   # com aumento
    ["exp001", "exp006"],   # controle permutado, e mesma taxa
    ["exp002", "exp005"],   # mesma taxa
])
def test_recusa_rodadas_nao_comparaveis(raiz, ids):
    with pytest.raises(SystemExit):
        T.montar(ids, raiz / "reports", raiz)


def test_recusa_hash_diferente_do_registrado(raiz):
    with pytest.raises(SystemExit, match="hash"):
        T.montar(["exp001", "exp005"], raiz / "reports", raiz)


def test_recusa_particoes_que_nao_descrevem_os_mesmos_segmentos(raiz):
    # gravação de 25,6 kHz mais curta: 58 segmentos em vez de 59
    h = _particao(raiz, 25_600, n=58 * 25_600)
    _rodada(raiz / "reports", "exp007", {**BASE, "fs_hz": 25_600, "splits": h})
    with pytest.raises(SystemExit, match="mesmos segmentos"):
        T.montar(["exp001", "exp007"], raiz / "reports", raiz)
